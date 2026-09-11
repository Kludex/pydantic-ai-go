// Package openai implements ai.Model against the OpenAI Chat Completions API.
package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// Model calls the OpenAI Chat Completions API. Create one with NewModel.
type Model struct {
	name                          string
	providerName                  string
	apiKey                        string
	baseURL                       string
	httpClient                    *http.Client
	providerHeaders               http.Header
	providerQuery                 url.Values
	prepareRequest                RequestPreparationFunc
	strictToolSupport             bool
	deferredToolSupport           bool
	defaultSettings               ai.ModelSettings
	background                    *bool
	backgroundPollInterval        time.Duration
	responsesPhaseSupport         *bool
	responsesCodeExecutionOutputs bool
	responsesFileSearchResults    bool
	chatWebSearchSupport          *bool
	chatCompatibility             ChatCompatibility
}

// Option configures a Model.
type Option func(*Model)

// RequestPreparationFunc prepares an HTTP request after provider defaults and
// before per-request extra headers are applied. It can add dynamic credentials.
// The function must be safe for concurrent calls and must not retain the request.
type RequestPreparationFunc func(*http.Request) error

// ProviderConfig configures an OpenAI-compatible endpoint. Name is persisted
// in message history and telemetry. Headers and Query are copied.
type ProviderConfig struct {
	// Name is persisted in responses, usage, and telemetry.
	Name string
	// BaseURL is the OpenAI-compatible API endpoint.
	BaseURL string
	// APIKey is sent as a bearer token.
	APIKey string
	// HTTPClient performs requests. Nil uses the shared default client.
	HTTPClient *http.Client
	// Headers contains detached provider-wide request headers.
	Headers http.Header
	// Query contains detached provider-wide query parameters.
	Query url.Values
	// PrepareRequest adds dynamic authentication or routing data.
	PrepareRequest RequestPreparationFunc
}

// ChatNativeTool is one provider-managed Chat Completions tool declaration.
type ChatNativeTool struct {
	// Type is the provider-native tool discriminator.
	Type string `json:"type"`
	// Parameters contains detached provider-specific options.
	Parameters map[string]any `json:"parameters,omitempty"`
}

// ChatNativeToolFunc renders one provider-managed tool. The bool reports
// whether the endpoint supports the tool. The function must be safe for concurrent calls.
type ChatNativeToolFunc func(ai.NativeTool) (ChatNativeTool, bool, error)

// ChatCompatibility configures documented extensions to the OpenAI Chat
// Completions wire format. The configuration is intended for provider packages;
// applications should prefer a dedicated provider model.
type ChatCompatibility struct {
	// ReasoningContent enables the reasoning_content extension.
	ReasoningContent bool
	// Reasoning enables the unified reasoning extension.
	Reasoning bool
	// ReasoningText enables GitHub Copilot's reasoning_text extension.
	ReasoningText bool
	// ReasoningFallback enables vLLM reasoning with reasoning_content fallback.
	ReasoningFallback bool
	// ReasoningDetails enables OpenRouter-style reasoning details.
	ReasoningDetails bool
	// LegacyMaxTokens sends max_tokens instead of max_completion_tokens.
	LegacyMaxTokens bool
	// ExtendedMetadata preserves routed-provider and server-tool metadata.
	ExtendedMetadata bool
	// ExecutedTools enables Groq-style provider-executed search results.
	ExecutedTools bool
	// VideoInput enables video_url user content.
	VideoInput bool
	// FileURLInput enables remote file user content.
	FileURLInput bool
	// AudioInputDataURI enables audio data URLs.
	AudioInputDataURI bool
	// DisableDocumentInput rejects document user content before transport.
	DisableDocumentInput bool
	// NativeToolFunc renders provider-native Chat Completions tools.
	NativeToolFunc ChatNativeToolFunc
	// FinishReasons maps provider-specific stop reasons.
	FinishReasons map[string]ai.FinishReason
	// RequireFinishReason rejects a clean stream end without a terminal finish reason.
	RequireFinishReason bool
	// DisableRequiredToolChoice prevents tool_choice=required on endpoints that reject it.
	DisableRequiredToolChoice bool
	// DisableForcedToolChoiceWithThinking prevents forced tools while reasoning is active.
	DisableForcedToolChoiceWithThinking bool
	// ReasoningEnabledByDefault marks models that reason without an explicit effort.
	ReasoningEnabledByDefault bool
	// ResponsesReasoningContent replays visible reasoning content in Responses history.
	ResponsesReasoningContent bool
}

// WithChatCompatibility configures OpenAI-compatible response and history
// extensions. The configuration is detached and safe to reuse.
func WithChatCompatibility(compatibility ChatCompatibility) Option {
	finishReasons := maps.Clone(compatibility.FinishReasons)
	for reason, normalized := range finishReasons {
		if reason == "" {
			panic("openai: compatibility finish reason must not be empty")
		}
		switch normalized {
		case ai.FinishReasonStop, ai.FinishReasonLength, ai.FinishReasonContentFilter,
			ai.FinishReasonToolCall, ai.FinishReasonError:
		default:
			panic(fmt.Sprintf("openai: invalid compatibility finish reason %q", normalized))
		}
	}
	return func(model *Model) {
		model.chatCompatibility = ChatCompatibility{
			ReasoningContent:                    compatibility.ReasoningContent,
			Reasoning:                           compatibility.Reasoning,
			ReasoningText:                       compatibility.ReasoningText,
			ReasoningFallback:                   compatibility.ReasoningFallback,
			ReasoningDetails:                    compatibility.ReasoningDetails,
			LegacyMaxTokens:                     compatibility.LegacyMaxTokens,
			ExtendedMetadata:                    compatibility.ExtendedMetadata,
			ExecutedTools:                       compatibility.ExecutedTools,
			VideoInput:                          compatibility.VideoInput,
			FileURLInput:                        compatibility.FileURLInput,
			AudioInputDataURI:                   compatibility.AudioInputDataURI,
			DisableDocumentInput:                compatibility.DisableDocumentInput,
			NativeToolFunc:                      compatibility.NativeToolFunc,
			FinishReasons:                       maps.Clone(finishReasons),
			RequireFinishReason:                 compatibility.RequireFinishReason,
			DisableRequiredToolChoice:           compatibility.DisableRequiredToolChoice,
			DisableForcedToolChoiceWithThinking: compatibility.DisableForcedToolChoiceWithThinking,
			ReasoningEnabledByDefault:           compatibility.ReasoningEnabledByDefault,
			ResponsesReasoningContent:           compatibility.ResponsesReasoningContent,
		}
	}
}

// WithChatDocumentInput controls document input for a compatible Chat Completions endpoint.
func WithChatDocumentInput(enabled bool) Option {
	return func(model *Model) { model.chatCompatibility.DisableDocumentInput = !enabled }
}

// WithChatWebSearchSupport overrides Chat Completions web-search support detection.
func WithChatWebSearchSupport(enabled bool) Option {
	return func(model *Model) {
		supported := enabled
		model.chatWebSearchSupport = &supported
	}
}

// WithResponsesCodeExecutionOutputs includes code-interpreter logs and image outputs in Responses results.
func WithResponsesCodeExecutionOutputs(enabled bool) Option {
	return func(model *Model) { model.responsesCodeExecutionOutputs = enabled }
}

// WithResponsesFileSearchResults includes retrieved file-search results in Responses results.
func WithResponsesFileSearchResults(enabled bool) Option {
	return func(model *Model) { model.responsesFileSearchResults = enabled }
}

// WithProvider configures an OpenAI-compatible provider in one option.
func WithProvider(provider ProviderConfig) Option {
	if provider.Name == "" {
		panic("openai: provider name must not be empty")
	}
	if provider.BaseURL == "" {
		panic("openai: provider base URL must not be empty")
	}
	headers := provider.Headers.Clone()
	query := cloneURLValues(provider.Query)
	return func(m *Model) {
		m.providerName = provider.Name
		m.baseURL = strings.TrimRight(provider.BaseURL, "/")
		m.apiKey = provider.APIKey
		if provider.HTTPClient != nil {
			m.httpClient = provider.HTTPClient
		}
		m.providerHeaders = headers.Clone()
		m.providerQuery = cloneURLValues(query)
		m.prepareRequest = provider.PrepareRequest
	}
}

// WithProviderName changes the provider identity persisted in message history
// and telemetry. Use it with WithBaseURL for OpenAI-compatible endpoints.
func WithProviderName(name string) Option {
	if name == "" {
		panic("openai: provider name must not be empty")
	}
	return func(m *Model) { m.providerName = name }
}

// WithAPIKey sets the API key. The default is the OPENAI_API_KEY environment variable.
// An empty key disables the default Authorization header.
func WithAPIKey(key string) Option { return func(m *Model) { m.apiKey = key } }

// WithBaseURL points the model at a different endpoint, such as a proxy or
// an OpenAI-compatible provider. The default is OPENAI_BASE_URL or https://api.openai.com/v1.
func WithBaseURL(url string) Option {
	return func(m *Model) { m.baseURL = strings.TrimRight(url, "/") }
}

// WithHTTPClient sets the HTTP client used for requests.
func WithHTTPClient(c *http.Client) Option { return func(m *Model) { m.httpClient = c } }

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	return func(m *Model) { m.defaultSettings = settings.Clone() }
}

// WithStrictToolSupport configures whether an OpenAI-compatible endpoint
// accepts strict function definitions. OpenAI supports them by default.
func WithStrictToolSupport(enabled bool) Option {
	return func(m *Model) { m.strictToolSupport = enabled }
}

// WithDeferredToolSupport configures native deferred-tool rendering for the
// Responses API. Disable it for OpenAI-compatible endpoints without tool_search.
func WithDeferredToolSupport(enabled bool) Option {
	return func(m *Model) { m.deferredToolSupport = enabled }
}

// WithResponsesPhaseSupport overrides whether Responses API assistant-message
// phases are replayed. By default, support follows the bundled model profile.
func WithResponsesPhaseSupport(enabled bool) Option {
	return func(m *Model) { m.responsesPhaseSupport = &enabled }
}

// WithBackgroundMode enables server-side execution for Responses API requests.
// Pending responses are polled automatically by the agent.
func WithBackgroundMode(enabled bool) Option {
	return func(m *Model) { m.background = &enabled }
}

// WithBackgroundPollInterval sets the delay between Responses API background
// retrievals. The default is two seconds. Zero polls immediately.
func WithBackgroundPollInterval(interval time.Duration) Option {
	if interval < 0 {
		panic("openai: background poll interval must not be negative")
	}
	return func(m *Model) { m.backgroundPollInterval = interval }
}

// NewModel creates a Model for the named OpenAI model, e.g. "gpt-5".
func NewModel(name string, opts ...Option) *Model {
	baseURL := os.Getenv("OPENAI_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}
	m := &Model{
		name:                   name,
		providerName:           "openai",
		apiKey:                 os.Getenv("OPENAI_API_KEY"),
		baseURL:                strings.TrimRight(baseURL, "/"),
		httpClient:             http.DefaultClient,
		strictToolSupport:      true,
		deferredToolSupport:    true,
		backgroundPollInterval: 2 * time.Second,
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// Name returns the model name.
func (m *Model) Name() string { return m.name }

// SupportsNativeTool reports Chat Completions native-tool support.
func (m *Model) SupportsNativeTool(tool ai.NativeTool) bool {
	if err := ai.ValidateNativeTools([]ai.NativeTool{tool}); err != nil {
		return false
	}
	tool = tool.CloneNativeTool()
	if m.chatCompatibility.NativeToolFunc != nil {
		_, include, err := m.chatCompatibility.NativeToolFunc(tool)
		return err == nil && include
	}
	_, webSearch := tool.(ai.WebSearchTool)
	return webSearch && supportsChatWebSearch(m.name, m.chatWebSearchSupport)
}

// ProviderName returns the durable provider identity.
func (m *Model) ProviderName() string { return m.providerName }

// ProviderURL returns the configured provider API URL.
func (m *Model) ProviderURL() string { return m.baseURL }

// DefaultModelSettings returns this model's request defaults.
func (m *Model) DefaultModelSettings() ai.ModelSettings { return m.defaultSettings.Clone() }

// PromptCacheRetention reports extended OpenAI prompt-cache retention.
func (m *Model) PromptCacheRetention(settings ai.ModelSettings) (time.Duration, bool) {
	_, cache, err := extractPromptCacheSettings(settings)
	if err != nil || cache.Retention != PromptCacheRetention24Hours {
		return 0, false
	}
	return 24 * time.Hour, true
}

// Request implements ai.Model.
func (m *Model) Request(ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
	payload, err := m.buildPayload(ctx, msgs, params)
	if err != nil {
		return nil, err
	}
	body, err := marshalRequest(payload, payload.ExtraBody)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if err := m.configureRequest(req, params.Settings.ExtraHeaders); err != nil {
		return nil, err
	}

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return nil, ai.NewModelTransportError(ctx, m, "request", err)
	}
	defer func() { _ = resp.Body.Close() }()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, ai.NewModelTransportError(ctx, m, "read response", err)
	}
	if resp.StatusCode != http.StatusOK {
		if response := m.azureContentFilterResponse(resp.StatusCode, data); response != nil {
			return response, nil
		}
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(data), ProviderName: m.providerName}
	}
	response, err := m.parseResponse(data)
	if response != nil {
		response.ProviderName = m.providerName
		response.ProviderURL = m.baseURL
	}
	return response, err
}

func (m *Model) azureContentFilterResponse(statusCode int, data []byte) *ai.ModelResponse {
	if m.providerName != "azure" || statusCode != http.StatusBadRequest {
		return nil
	}
	var envelope struct {
		Error *struct {
			Code       string         `json:"code"`
			InnerError map[string]any `json:"innererror"`
		} `json:"error"`
		Code       string         `json:"code"`
		InnerError map[string]any `json:"innererror"`
	}
	if json.Unmarshal(data, &envelope) != nil {
		return nil
	}
	code, inner := envelope.Code, envelope.InnerError
	if envelope.Error != nil {
		code, inner = envelope.Error.Code, envelope.Error.InnerError
	}
	if code != "content_filter" {
		return nil
	}
	details := map[string]any{"finish_reason": "content_filter"}
	if result, ok := inner["content_filter_result"]; ok {
		details["content_filter_result"] = result
	}
	return &ai.ModelResponse{
		ModelName: m.name, ProviderName: m.providerName, ProviderURL: m.baseURL,
		FinishReason: ai.FinishReasonContentFilter, State: ai.ModelResponseStateComplete,
		ProviderDetails: details,
	}
}

// APIError is a provider API failure. StatusCode is zero when an OpenAI-compatible
// endpoint returns no completion and no explicit error envelope.
type APIError struct {
	// StatusCode is the HTTP response status, or zero for an empty completion.
	StatusCode int
	// Body is the provider error or diagnostic body.
	Body string
	// ProviderName identifies the endpoint that returned the error.
	ProviderName string
}

// Error formats the provider status and body.
func (e *APIError) Error() string {
	providerName := e.ProviderName
	if providerName == "" {
		providerName = "openai"
	}
	if e.StatusCode == 0 {
		return fmt.Sprintf("%s: %s", providerName, e.Body)
	}
	return fmt.Sprintf("%s: API returned status %d: %s", providerName, e.StatusCode, e.Body)
}

// IsModelAPIError marks provider API responses as eligible for default model fallback.
func (*APIError) IsModelAPIError() bool { return true }

type chatRequest struct {
	Model                string                `json:"model"`
	Messages             []chatMessage         `json:"messages"`
	Tools                []any                 `json:"tools,omitempty"`
	ToolChoice           any                   `json:"tool_choice,omitempty"`
	ParallelToolCalls    *bool                 `json:"parallel_tool_calls,omitempty"`
	MaxTokens            int                   `json:"max_completion_tokens,omitempty"`
	LegacyMaxTokens      int                   `json:"max_tokens,omitempty"`
	Temperature          *float64              `json:"temperature,omitempty"`
	TopP                 *float64              `json:"top_p,omitempty"`
	Seed                 *int                  `json:"seed,omitempty"`
	Stop                 []string              `json:"stop,omitempty"`
	Stream               bool                  `json:"stream,omitempty"`
	StreamOptions        *streamOptions        `json:"stream_options,omitempty"`
	ResponseFormat       *responseFormat       `json:"response_format,omitempty"`
	ReasoningEffort      string                `json:"reasoning_effort,omitempty"`
	PresencePenalty      *float64              `json:"presence_penalty,omitempty"`
	FrequencyPenalty     *float64              `json:"frequency_penalty,omitempty"`
	LogitBias            map[string]int        `json:"logit_bias,omitempty"`
	Logprobs             *bool                 `json:"logprobs,omitempty"`
	TopLogprobs          *int                  `json:"top_logprobs,omitempty"`
	ServiceTier          ai.ServiceTier        `json:"service_tier,omitempty"`
	PromptCacheKey       string                `json:"prompt_cache_key,omitempty"`
	PromptCacheRetention PromptCacheRetention  `json:"prompt_cache_retention,omitempty"`
	PromptCacheOptions   *PromptCacheOptions   `json:"prompt_cache_options,omitempty"`
	Prediction           *chatPrediction       `json:"prediction,omitempty"`
	WebSearchOptions     *chatWebSearchOptions `json:"web_search_options,omitempty"`
	ExtraBody            map[string]any        `json:"-"`
}

type chatMessage struct {
	Role             string            `json:"role"`
	Content          any               `json:"content,omitempty"` // string or []contentPart
	ReasoningContent string            `json:"reasoning_content,omitempty"`
	Reasoning        string            `json:"reasoning,omitempty"`
	ReasoningText    string            `json:"reasoning_text,omitempty"`
	ReasoningDetails []reasoningDetail `json:"reasoning_details,omitempty"`
	ToolCalls        []toolCall        `json:"tool_calls,omitempty"`
	ToolCallID       string            `json:"tool_call_id,omitempty"`
}

type contentPart struct {
	Type                  string                       `json:"type"`
	Text                  string                       `json:"text,omitempty"`
	ImageURL              *imageURL                    `json:"image_url,omitempty"`
	VideoURL              *videoURL                    `json:"video_url,omitempty"`
	InputAudio            *inputAudio                  `json:"input_audio,omitempty"`
	File                  *chatFile                    `json:"file,omitempty"`
	CacheControl          *chatCacheControl            `json:"cache_control,omitempty"`
	PromptCacheBreakpoint *openAIPromptCacheBreakpoint `json:"prompt_cache_breakpoint,omitempty"`
}

type imageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type inputAudio struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

type chatFile struct {
	FileData string `json:"file_data,omitempty"`
	FileID   string `json:"file_id,omitempty"`
	Filename string `json:"filename,omitempty"`
}

type videoURL struct {
	URL string `json:"url"`
}

type toolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function functionCall `json:"function"`
}

type functionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatExecutedTool struct {
	Index         int            `json:"index"`
	Type          string         `json:"type"`
	Arguments     string         `json:"arguments"`
	Output        any            `json:"output"`
	SearchResults map[string]any `json:"search_results"`
}

type chatTool struct {
	Type         string            `json:"type"`
	Function     chatFunction      `json:"function"`
	CacheControl *chatCacheControl `json:"cache_control,omitempty"`
}

type chatFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
	Strict      *bool          `json:"strict,omitempty"`
}

func (m *Model) buildPayload(
	ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams,
) (*chatRequest, error) {
	params, err := ai.ResolveNativeToolPreferences(m, params)
	if err != nil {
		return nil, err
	}
	settings, promptCache, err := extractPromptCacheSettings(params.Settings)
	if err != nil {
		return nil, err
	}
	settings, prediction, err := extractPredictionSettings(settings)
	if err != nil {
		return nil, err
	}
	settings, responseSettings, err := extractRawAnnotationSettings(settings)
	if err != nil {
		return nil, err
	}
	if responseSettings.annotationsConfigured {
		return nil, fmt.Errorf("openai: include raw annotations is only supported by Responses")
	}
	params.Settings = settings
	var webSearchOptions *chatWebSearchOptions
	if m.chatCompatibility.NativeToolFunc == nil {
		for _, nativeTool := range params.NativeTools {
			webSearch, isWebSearch := nativeTool.CloneNativeTool().(ai.WebSearchTool)
			if !isWebSearch {
				if nativeTool.IsOptional() {
					continue
				}
				return nil, fmt.Errorf(
					"%s: Chat Completions does not support native tool %q", m.providerName, nativeTool.Kind(),
				)
			}
			if !supportsChatWebSearch(m.name, m.chatWebSearchSupport) {
				if webSearch.Optional {
					continue
				}
				return nil, fmt.Errorf(
					"%s: Chat Completions does not support native tool %q for model %q; use NewResponsesModel instead",
					m.providerName, webSearch.Kind(), m.name,
				)
			}
			contextSize := webSearch.SearchContextSize
			if contextSize == "" {
				contextSize = ai.WebSearchContextMedium
			}
			webSearchOptions = &chatWebSearchOptions{SearchContextSize: contextSize}
			if webSearch.UserLocation != nil {
				webSearchOptions.UserLocation = &chatWebSearchUserLocation{
					Type: "approximate",
					Approximate: chatWebSearchUserLocationApproximate{
						City: webSearch.UserLocation.City, Country: webSearch.UserLocation.Country,
						Region: webSearch.UserLocation.Region, Timezone: webSearch.UserLocation.Timezone,
					},
				}
			}
		}
	}
	reasoningEffort, err := openAIThinkingEffortForModel(m.name, params.Settings.Thinking)
	if err != nil {
		return nil, err
	}
	serviceTier, err := openAIServiceTier(params.Settings.ServiceTier)
	if err != nil {
		return nil, err
	}
	req := &chatRequest{
		Model:                m.name,
		MaxTokens:            params.Settings.MaxTokens,
		Temperature:          params.Settings.Temperature,
		TopP:                 params.Settings.TopP,
		Seed:                 params.Settings.Seed,
		Stop:                 params.Settings.StopSequences,
		ReasoningEffort:      reasoningEffort,
		PresencePenalty:      params.Settings.PresencePenalty,
		FrequencyPenalty:     params.Settings.FrequencyPenalty,
		LogitBias:            params.Settings.LogitBias,
		Logprobs:             params.Settings.Logprobs,
		TopLogprobs:          params.Settings.TopLogprobs,
		ServiceTier:          serviceTier,
		PromptCacheKey:       promptCache.Key,
		PromptCacheRetention: promptCache.Retention,
		PromptCacheOptions:   promptCache.Options,
		Prediction:           prediction,
		WebSearchOptions:     webSearchOptions,
		ExtraBody:            params.Settings.ExtraBody,
	}
	if m.chatCompatibility.LegacyMaxTokens {
		req.LegacyMaxTokens = req.MaxTokens
		req.MaxTokens = 0
	}
	if openAIModelReasoningActive(m.name, reasoningEffort) {
		req.Temperature = nil
		req.TopP = nil
	}
	cache := chatPromptCacheFromContext(ctx)
	if cache.ExplicitMarkerStyle == "" && m.providerName == "openai" && supportsOpenAIPromptCache(m.name) {
		cache.ExplicitMarkerStyle = ChatPromptCacheMarkerBreakpoint
	}
	if params.Instructions != "" {
		if cache.InstructionsTTL != "" && len(params.InstructionParts) > 0 {
			firstInstruction := len(req.Messages)
			lastStaticInstruction := -1
			hasDynamicInstructions := false
			for _, instruction := range params.InstructionParts {
				req.Messages = append(req.Messages, chatMessage{Role: m.chatSystemPromptRole(), Content: instruction.Content})
				if instruction.Dynamic {
					hasDynamicInstructions = true
				} else {
					lastStaticInstruction = len(req.Messages) - 1
				}
			}
			cacheIndex := len(req.Messages) - 1
			if hasDynamicInstructions {
				if cache.SupportsDynamicInstructions {
					cacheIndex = lastStaticInstruction
				} else {
					cacheIndex = -1
				}
			}
			if cacheIndex >= firstInstruction {
				addChatMessageCache(&req.Messages[cacheIndex], cache.InstructionsTTL, cache.IncludeTTL)
			}
		} else {
			req.Messages = append(req.Messages, chatMessage{Role: m.chatSystemPromptRole(), Content: params.Instructions})
			if cache.InstructionsTTL != "" {
				addChatMessageCache(&req.Messages[len(req.Messages)-1], cache.InstructionsTTL, cache.IncludeTTL)
			}
		}
	}
	for _, msg := range msgs {
		converted, err := m.convertMessage(ctx, msg, cache)
		if err != nil {
			return nil, err
		}
		req.Messages = append(req.Messages, converted...)
	}
	if cache.MessagesTTL != "" && len(req.Messages) > 0 {
		addChatMessageCache(&req.Messages[len(req.Messages)-1], cache.MessagesTTL, cache.IncludeTTL)
	}
	for _, tool := range params.Tools {
		converted, err := convertTool(tool, m.strictToolSupport)
		if err != nil {
			return nil, err
		}
		req.Tools = append(req.Tools, converted)
	}
	if params.OutputTool != nil {
		converted, err := convertTool(*params.OutputTool, m.strictToolSupport)
		if err != nil {
			return nil, err
		}
		req.Tools = append(req.Tools, converted)
		reasoningActive := reasoningEffort != "none" &&
			(reasoningEffort != "" || m.chatCompatibility.ReasoningEnabledByDefault)
		if !params.AllowText {
			if m.chatCompatibility.DisableRequiredToolChoice ||
				(m.chatCompatibility.DisableForcedToolChoiceWithThinking && reasoningActive) {
				req.ToolChoice = "auto"
			} else {
				req.ToolChoice = "required"
			}
		}
	}
	if cache.ToolsTTL != "" && len(req.Tools) > 0 {
		tool := req.Tools[len(req.Tools)-1].(chatTool)
		tool.CacheControl = newChatCacheControl(cache.ToolsTTL, cache.IncludeTTL)
		req.Tools[len(req.Tools)-1] = tool
	}
	if m.chatCompatibility.NativeToolFunc != nil {
		for _, nativeTool := range params.NativeTools {
			converted, supported, err := m.chatCompatibility.NativeToolFunc(nativeTool.CloneNativeTool())
			if err != nil {
				return nil, err
			}
			if !supported {
				if nativeTool.IsOptional() {
					continue
				}
				return nil, fmt.Errorf(
					"%s: Chat Completions does not support native tool %q", m.providerName, nativeTool.Kind(),
				)
			}
			if converted.Type == "" {
				return nil, fmt.Errorf("%s: native tool %q rendered an empty type", m.providerName, nativeTool.Kind())
			}
			req.Tools = append(req.Tools, converted)
		}
	}
	if len(req.Tools) > 0 {
		req.ParallelToolCalls = params.Settings.ParallelToolCalls
	}
	if err := limitChatCachePoints(req.Messages, req.Tools, cache.MaxPoints); err != nil {
		return nil, err
	}
	if params.OutputSchema != nil && params.OutputMode != ai.OutputModePrompted {
		strict := true
		schema, _, err := prepareOpenAISchema(params.OutputSchema, &strict)
		if err != nil {
			return nil, fmt.Errorf("openai: output schema: %w", err)
		}
		var strictFlag *bool
		if m.strictToolSupport {
			strictFlag = &strict
		}
		req.ResponseFormat = &responseFormat{
			Type:       "json_schema",
			JSONSchema: jsonSchemaFormat{Name: "final_result", Schema: schema, Strict: strictFlag},
		}
	}
	return req, nil
}

type responseFormat struct {
	Type       string           `json:"type"`
	JSONSchema jsonSchemaFormat `json:"json_schema"`
}

type jsonSchemaFormat struct {
	Name   string         `json:"name"`
	Schema map[string]any `json:"schema"`
	Strict *bool          `json:"strict,omitempty"`
}

func openAIServiceTier(tier ai.ServiceTier) (ai.ServiceTier, error) {
	switch tier {
	case "", ai.ServiceTierAuto, ai.ServiceTierDefault, ai.ServiceTierFlex, ai.ServiceTierPriority:
		return tier, nil
	default:
		return "", fmt.Errorf("openai: invalid service tier %q", tier)
	}
}

func openAIThinkingEffortForModel(modelName string, settings *ai.ThinkingSettings) (string, error) {
	effort, err := openAIThinkingEffort(settings)
	if err != nil {
		return "", err
	}
	modelName = strings.TrimPrefix(strings.ToLower(modelName), "openai.")
	if strings.HasPrefix(modelName, "gpt-6-astra") && effort == "none" {
		return "", nil
	}
	if effort == "minimal" && (strings.HasPrefix(modelName, "gpt-5.6") || strings.HasPrefix(modelName, "gpt-6-astra")) {
		return "low", nil
	}
	return effort, nil
}

func openAIThinkingEffort(settings *ai.ThinkingSettings) (string, error) {
	if settings == nil || settings.Level == "" {
		return "", nil
	}
	switch settings.Level {
	case ai.ThinkingLevelDisabled:
		return "none", nil
	case ai.ThinkingLevelEnabled:
		return "medium", nil
	case ai.ThinkingLevelMinimal, ai.ThinkingLevelLow, ai.ThinkingLevelMedium,
		ai.ThinkingLevelHigh, ai.ThinkingLevelXHigh:
		return string(settings.Level), nil
	default:
		return "", fmt.Errorf("openai: invalid thinking level %q", settings.Level)
	}
}

func openAIModelReasoningActive(modelName, effort string) bool {
	modelName = strings.TrimPrefix(strings.ToLower(modelName), "openai.")
	return effort != "none" && (effort != "" || strings.HasPrefix(modelName, "gpt-5.6") ||
		strings.HasPrefix(modelName, "gpt-6-astra"))
}

func (model *Model) convertMessage(
	ctx context.Context, msg ai.ModelMessage, cache ChatPromptCache,
) ([]chatMessage, error) {
	switch message := msg.(type) {
	case ai.ModelRequest:
		return model.convertRequest(ctx, message, cache)
	case ai.ModelResponse:
		return model.convertResponse(message)
	default:
		return nil, fmt.Errorf("openai: unknown message type %T", msg)
	}
}

func (model *Model) convertRequest(
	ctx context.Context, m ai.ModelRequest, cache ChatPromptCache,
) ([]chatMessage, error) {
	var out []chatMessage
	for _, part := range m.Parts {
		switch p := part.(type) {
		case ai.SpeechPart:
			return nil, ai.ErrUnpreparedSpeech
		case ai.SystemPromptPart:
			out = append(out, chatMessage{Role: model.chatSystemPromptRole(), Content: p.Content})
		case ai.UserPromptPart:
			msg, err := model.convertUserPrompt(ctx, p, cache)
			if err != nil {
				return nil, err
			}
			out = append(out, msg)
		case ai.ToolReturnPart:
			content, err := contentString(p.Content)
			if err != nil {
				return nil, err
			}
			out = append(out, chatMessage{Role: "tool", Content: content, ToolCallID: p.ToolCallID})
		case ai.ToolAvailabilityDeltaPart:
		case ai.RetryPromptPart:
			content := p.ModelResponse()
			if p.ToolCallID != "" {
				out = append(out, chatMessage{Role: "tool", Content: content, ToolCallID: p.ToolCallID})
			} else {
				out = append(out, chatMessage{Role: "user", Content: content})
			}
		default:
			return nil, fmt.Errorf("openai: unknown request part type %T", part)
		}
	}
	return out, nil
}

func (model *Model) convertResponse(m ai.ModelResponse) ([]chatMessage, error) {
	msg := chatMessage{Role: "assistant"}
	for _, part := range m.Parts {
		switch p := part.(type) {
		case ai.SpeechPart:
			return nil, ai.ErrUnpreparedSpeech
		case ai.TextPart:
			msg.Content = p.Content
		case ai.ThinkingPart:
			if p.ProviderName == "" || p.ProviderName == model.providerName {
				if model.chatCompatibility.ReasoningDetails {
					if detail, ok := reasoningDetailFromThinkingPart(p); ok {
						msg.ReasoningDetails = append(msg.ReasoningDetails, detail)
						continue
					}
				}
				if model.chatCompatibility.ReasoningContent {
					msg.ReasoningContent += p.Content
				}
				if model.chatCompatibility.Reasoning {
					msg.Reasoning += p.Content
				}
				if model.chatCompatibility.ReasoningText {
					msg.ReasoningText += p.Content
				}
				if model.chatCompatibility.ReasoningFallback {
					if p.ID == "reasoning_content" {
						msg.ReasoningContent += p.Content
					} else {
						msg.Reasoning += p.Content
					}
				}
			}
		case ai.ToolCallPart:
			msg.ToolCalls = append(msg.ToolCalls, toolCall{
				ID:       p.ToolCallID,
				Type:     "function",
				Function: functionCall{Name: p.ToolName, Arguments: string(p.Args)},
			})
		}
	}
	return []chatMessage{msg}, nil
}

func convertTool(def ai.ToolDefinition, supportsStrict bool) (chatTool, error) {
	def, err := ai.PrepareToolReturnSchema(def, false)
	if err != nil {
		return chatTool{}, err
	}
	schema, strict, err := prepareOpenAITool(def, supportsStrict)
	if err != nil {
		return chatTool{}, err
	}
	return chatTool{
		Type:     "function",
		Function: chatFunction{Name: def.Name, Description: def.Description, Parameters: schema, Strict: strict},
	}, nil
}

type chatError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type chatResponse struct {
	ID                string `json:"id"`
	Model             string `json:"model"`
	Provider          string `json:"provider"`
	Created           int64  `json:"created"`
	ServiceTier       string `json:"service_tier"`
	SystemFingerprint string `json:"system_fingerprint"`
	Choices           []struct {
		Message struct {
			Content          string             `json:"content"`
			Refusal          string             `json:"refusal"`
			ReasoningContent string             `json:"reasoning_content"`
			Reasoning        string             `json:"reasoning"`
			ReasoningText    string             `json:"reasoning_text"`
			ReasoningDetails []reasoningDetail  `json:"reasoning_details"`
			Annotations      []map[string]any   `json:"annotations"`
			ToolCalls        []toolCall         `json:"tool_calls"`
			ExecutedTools    []chatExecutedTool `json:"executed_tools"`
		} `json:"message"`
		FinishReason       string `json:"finish_reason"`
		NativeFinishReason string `json:"native_finish_reason"`
		Logprobs           *struct {
			Content []map[string]any `json:"content"`
		} `json:"logprobs"`
	} `json:"choices"`
	Usage      chatUsage      `json:"usage"`
	Moderation map[string]any `json:"moderation"`
	Error      *chatError     `json:"error"`
}

type chatUsage struct {
	PromptTokens        int      `json:"prompt_tokens"`
	CompletionTokens    int      `json:"completion_tokens"`
	Cost                *float64 `json:"cost"`
	IsBYOK              *bool    `json:"is_byok"`
	PromptTokensDetails struct {
		CachedTokens     int `json:"cached_tokens"`
		CacheWriteTokens int `json:"cache_write_tokens"`
		AudioTokens      int `json:"audio_tokens"`
	} `json:"prompt_tokens_details"`
	CostDetails struct {
		UpstreamInferenceCost            *float64 `json:"upstream_inference_cost"`
		UpstreamInferencePromptCost      *float64 `json:"upstream_inference_prompt_cost"`
		UpstreamInferenceCompletionsCost *float64 `json:"upstream_inference_completions_cost"`
	} `json:"cost_details"`
	ServerToolUseDetails struct {
		ToolCallsRequested *int `json:"tool_calls_requested"`
		ToolCallsExecuted  *int `json:"tool_calls_executed"`
		WebSearchRequests  *int `json:"web_search_requests"`
	} `json:"server_tool_use_details"`
	CompletionTokensDetails struct {
		ReasoningTokens          int `json:"reasoning_tokens"`
		AudioTokens              int `json:"audio_tokens"`
		AcceptedPredictionTokens int `json:"accepted_prediction_tokens"`
		RejectedPredictionTokens int `json:"rejected_prediction_tokens"`
	} `json:"completion_tokens_details"`
}

func (u chatUsage) usage() ai.Usage {
	return ai.Usage{
		Requests: 1, InputTokens: u.PromptTokens, OutputTokens: u.CompletionTokens,
		CacheWriteTokens:         u.PromptTokensDetails.CacheWriteTokens,
		CacheReadTokens:          u.PromptTokensDetails.CachedTokens,
		InputAudioTokens:         u.PromptTokensDetails.AudioTokens,
		OutputAudioTokens:        u.CompletionTokensDetails.AudioTokens,
		ReasoningTokens:          u.CompletionTokensDetails.ReasoningTokens,
		AcceptedPredictionTokens: u.CompletionTokensDetails.AcceptedPredictionTokens,
		RejectedPredictionTokens: u.CompletionTokensDetails.RejectedPredictionTokens,
		CostUSD:                  u.Cost,
		Details: map[string]int{
			"reasoning_tokens":           u.CompletionTokensDetails.ReasoningTokens,
			"audio_tokens":               u.CompletionTokensDetails.AudioTokens,
			"accepted_prediction_tokens": u.CompletionTokensDetails.AcceptedPredictionTokens,
			"rejected_prediction_tokens": u.CompletionTokensDetails.RejectedPredictionTokens,
		},
	}
}

func addExtendedChatUsageDetails(details map[string]any, usage chatUsage) {
	if usage.Cost != nil {
		details["cost"] = *usage.Cost
	}
	if usage.CostDetails.UpstreamInferenceCost != nil {
		details["upstream_inference_cost"] = *usage.CostDetails.UpstreamInferenceCost
	}
	if usage.CostDetails.UpstreamInferencePromptCost != nil {
		details["upstream_inference_prompt_cost"] = *usage.CostDetails.UpstreamInferencePromptCost
	}
	if usage.CostDetails.UpstreamInferenceCompletionsCost != nil {
		details["upstream_inference_completions_cost"] = *usage.CostDetails.UpstreamInferenceCompletionsCost
	}
	if usage.IsBYOK != nil {
		details["is_byok"] = *usage.IsBYOK
	}
	serverToolUse := map[string]int{}
	if usage.ServerToolUseDetails.ToolCallsRequested != nil {
		serverToolUse["tool_calls_requested"] = *usage.ServerToolUseDetails.ToolCallsRequested
	}
	if usage.ServerToolUseDetails.ToolCallsExecuted != nil {
		serverToolUse["tool_calls_executed"] = *usage.ServerToolUseDetails.ToolCallsExecuted
	}
	if usage.ServerToolUseDetails.WebSearchRequests != nil {
		serverToolUse["web_search_requests"] = *usage.ServerToolUseDetails.WebSearchRequests
	}
	if len(serverToolUse) > 0 {
		details["server_tool_use"] = serverToolUse
	}
}

func normalizeNestedChatResponse(data []byte) []byte {
	var envelope map[string]json.RawMessage
	if json.Unmarshal(data, &envelope) != nil {
		return data
	}
	provider := bytes.TrimSpace(envelope["provider"])
	if len(provider) == 0 || provider[0] != '{' {
		return data
	}
	var nested map[string]json.RawMessage
	_ = json.Unmarshal(provider, &nested)
	if len(nested["created"]) == 0 && len(envelope["created"]) > 0 {
		nested["created"] = envelope["created"]
	}
	if value := bytes.TrimSpace(nested["provider"]); len(value) == 0 || bytes.Equal(value, []byte("null")) {
		nested["provider"] = json.RawMessage(`"unknown"`)
	}
	normalized, _ := json.Marshal(nested)
	return normalized
}

func (model *Model) parseResponse(data []byte) (*ai.ModelResponse, error) {
	noCompletion := false
	if model.chatCompatibility.ExtendedMetadata {
		data = normalizeNestedChatResponse(data)
		var envelope struct {
			Choices json.RawMessage `json:"choices"`
		}
		_ = json.Unmarshal(data, &envelope)
		noCompletion = bytes.Equal(bytes.TrimSpace(envelope.Choices), []byte("null"))
	}
	var cr chatResponse
	if err := json.Unmarshal(data, &cr); err != nil {
		return nil, fmt.Errorf("openai: parse response: %w", err)
	}
	if model.chatCompatibility.ExtendedMetadata && cr.Error != nil {
		return nil, &APIError{
			StatusCode: cr.Error.Code, Body: cr.Error.Message, ProviderName: model.providerName,
		}
	}
	if len(cr.Choices) == 0 {
		if noCompletion {
			modelName := cr.Model
			if modelName == "" {
				modelName = model.name
			}
			return nil, &APIError{
				Body:         "returned a response with null choices and no error for model " + modelName,
				ProviderName: model.providerName,
			}
		}
		return nil, fmt.Errorf("openai: response has no choices")
	}
	providerDetails := map[string]any{}
	if cr.Choices[0].FinishReason != "" {
		providerDetails["finish_reason"] = cr.Choices[0].FinishReason
	}
	if model.chatCompatibility.ExtendedMetadata {
		providerDetails["downstream_provider"] = cr.Provider
		if cr.Choices[0].NativeFinishReason != "" {
			providerDetails["finish_reason"] = cr.Choices[0].NativeFinishReason
		}
		addExtendedChatUsageDetails(providerDetails, cr.Usage)
		if len(cr.Choices[0].Message.Annotations) > 0 {
			providerDetails["annotations"] = cr.Choices[0].Message.Annotations
		}
	}
	timestamp := time.Now().UTC()
	if cr.Created != 0 {
		timestamp = time.Unix(cr.Created, 0).UTC()
	}
	providerDetails["timestamp"] = timestamp
	if len(cr.Moderation) > 0 {
		providerDetails["moderation"] = cr.Moderation
	}
	if cr.ServiceTier != "" {
		providerDetails["service_tier"] = cr.ServiceTier
	}
	if cr.SystemFingerprint != "" {
		providerDetails["system_fingerprint"] = cr.SystemFingerprint
	}
	if cr.Choices[0].Logprobs != nil {
		providerDetails["logprobs"] = cr.Choices[0].Logprobs.Content
	}
	if cr.Choices[0].Message.Refusal != "" {
		delete(providerDetails, "finish_reason")
		providerDetails["refusal"] = cr.Choices[0].Message.Refusal
	}
	resp := &ai.ModelResponse{
		ModelName: cr.Model, Timestamp: timestamp, Usage: cr.Usage.usage(),
		ProviderDetails: providerDetails, ProviderResponseID: cr.ID,
		FinishReason: model.chatFinishReason(cr.Choices[0].FinishReason), State: ai.ModelResponseStateComplete,
	}
	msg := cr.Choices[0].Message
	if msg.Refusal != "" {
		resp.FinishReason = ai.FinishReasonContentFilter
		return resp, nil
	}
	if model.chatCompatibility.ReasoningDetails && len(msg.ReasoningDetails) > 0 {
		for _, detail := range msg.ReasoningDetails {
			part, err := thinkingPartFromReasoningDetail(detail, model.providerName)
			if err != nil {
				return nil, err
			}
			resp.Parts = append(resp.Parts, part)
		}
	} else {
		if model.chatCompatibility.ReasoningContent && msg.ReasoningContent != "" {
			resp.Parts = append(resp.Parts, ai.ThinkingPart{
				Content: msg.ReasoningContent, ID: "reasoning_content", ProviderName: model.providerName,
			})
		}
		if model.chatCompatibility.Reasoning && msg.Reasoning != "" {
			resp.Parts = append(resp.Parts, ai.ThinkingPart{
				Content: msg.Reasoning, ID: "reasoning", ProviderName: model.providerName,
			})
		}
		if model.chatCompatibility.ReasoningText && msg.ReasoningText != "" {
			resp.Parts = append(resp.Parts, ai.ThinkingPart{
				Content: msg.ReasoningText, ID: "reasoning_text", ProviderName: model.providerName,
			})
		}
		if model.chatCompatibility.ReasoningFallback {
			content, id := msg.Reasoning, "reasoning"
			if content == "" {
				content, id = msg.ReasoningContent, "reasoning_content"
			}
			if content != "" {
				resp.Parts = append(resp.Parts, ai.ThinkingPart{
					Content: content, ID: id, ProviderName: model.providerName,
				})
			}
		}
	}
	if model.chatCompatibility.ExecutedTools {
		for _, tool := range msg.ExecutedTools {
			call, result, ok, err := model.executedToolParts(tool)
			if err != nil {
				return nil, err
			}
			if ok {
				resp.Parts = append(resp.Parts, call, result)
			}
		}
	}
	if msg.Content != "" {
		resp.Parts = append(resp.Parts, ai.TextPart{Content: msg.Content, ProviderName: model.providerName})
	}
	for _, call := range msg.ToolCalls {
		if call.Type != "" && call.Type != "function" {
			return nil, fmt.Errorf("openai: unsupported chat tool call type %q", call.Type)
		}
		resp.Parts = append(resp.Parts, ai.ToolCallPart{
			ToolName:     call.Function.Name,
			Args:         json.RawMessage(call.Function.Arguments),
			ToolCallID:   call.ID,
			ProviderName: model.providerName,
		})
	}
	splitTaggedThinking(resp)
	return resp, nil
}

func (model *Model) executedToolParts(
	tool chatExecutedTool,
) (ai.NativeToolCallPart, ai.NativeToolReturnPart, bool, error) {
	if tool.Type != "search" {
		return ai.NativeToolCallPart{}, ai.NativeToolReturnPart{}, false, nil
	}
	arguments := json.RawMessage(tool.Arguments)
	if !json.Valid(arguments) {
		return ai.NativeToolCallPart{}, ai.NativeToolReturnPart{}, false,
			fmt.Errorf("openai: parse executed search arguments: invalid JSON")
	}
	content, _ := executedToolContent(tool)
	callID := fmt.Sprintf("groq-search:%d", tool.Index)
	details := map[string]any{"index": tool.Index, "type": tool.Type}
	call := ai.NativeToolCallPart{
		ToolName: "web_search", Args: arguments, ToolCallID: callID,
		ToolKind: ai.ToolPartKindWebSearch, ID: callID, ProviderName: model.providerName,
		ProviderDetails: details,
	}
	result := ai.NativeToolReturnPart{
		ToolName: "web_search", Content: content, ToolCallID: callID,
		ToolKind: ai.ToolPartKindWebSearch, Outcome: ai.ToolReturnOutcomeSuccess, ProviderName: model.providerName,
		ProviderDetails: maps.Clone(details),
	}
	return call, result, true, nil
}

func executedToolContent(tool chatExecutedTool) (any, bool) {
	for _, key := range []string{"images", "results"} {
		if values, ok := tool.SearchResults[key].([]any); ok && len(values) > 0 {
			return tool.SearchResults, true
		}
	}
	return tool.Output, tool.Output != nil
}

func (model *Model) chatSystemPromptRole() string {
	if strings.HasPrefix(model.name, "o1-mini") {
		return "user"
	}
	return "system"
}

func (model *Model) chatFinishReason(reason string) ai.FinishReason {
	if normalized, exists := model.chatCompatibility.FinishReasons[reason]; exists {
		return normalized
	}
	return map[string]ai.FinishReason{
		"": ai.FinishReasonStop, "stop": ai.FinishReasonStop, "length": ai.FinishReasonLength,
		"content_filter": ai.FinishReasonContentFilter,
		"tool_calls":     ai.FinishReasonToolCall, "function_call": ai.FinishReasonToolCall,
	}[reason]
}

func contentString(content any) (string, error) {
	if s, ok := content.(string); ok {
		return s, nil
	}
	b, err := json.Marshal(content)
	if err != nil {
		return "", fmt.Errorf("openai: marshal tool return: %w", err)
	}
	return string(b), nil
}

func (model *Model) convertUserPrompt(
	ctx context.Context, p ai.UserPromptPart, cache ChatPromptCache,
) (chatMessage, error) {
	if len(p.Contents) == 0 {
		return chatMessage{Role: "user", Content: p.Content}, nil
	}
	parts := make([]contentPart, 0, len(p.Contents))
	for _, c := range p.Contents {
		switch item := c.(type) {
		case ai.CachePoint:
			ttl, err := item.ResolvedTTL()
			if err != nil {
				return chatMessage{}, err
			}
			if cache.ExplicitMarkerStyle == "" {
				continue
			}
			if len(parts) == 0 {
				return chatMessage{}, fmt.Errorf("openai: cache point must follow user content")
			}
			switch cache.ExplicitMarkerStyle {
			case ChatPromptCacheMarkerBreakpoint:
				parts[len(parts)-1].PromptCacheBreakpoint = &openAIPromptCacheBreakpoint{Mode: "explicit"}
			case ChatPromptCacheMarkerControl:
				parts[len(parts)-1].CacheControl = newChatCacheControl(string(ttl), cache.IncludeTTL)
			default:
				return chatMessage{}, fmt.Errorf(
					"openai: invalid chat prompt cache marker style %q", cache.ExplicitMarkerStyle,
				)
			}
		case ai.TextContent:
			parts = append(parts, contentPart{Type: "text", Text: item.Text})
		case ai.ImageURL:
			if err := item.ForceDownload.Validate(); err != nil {
				return chatMessage{}, err
			}
			location := item.URL
			if item.ForceDownload != ai.FileDownloadNever {
				downloaded, err := downloadFileContent(
					ctx, item.URL, item.ResolvedMediaType, item.ForceDownload,
				)
				if err != nil {
					return chatMessage{}, err
				}
				location = downloaded.dataURI
			}
			parts = append(parts, contentPart{
				Type: "image_url", ImageURL: &imageURL{URL: location, Detail: chatImageDetail(item.VendorMetadata)},
			})
		case ai.VideoURL:
			if !model.chatCompatibility.VideoInput {
				return chatMessage{}, fmt.Errorf("%s: Chat Completions does not support video URL input", model.providerName)
			}
			if err := item.ForceDownload.Validate(); err != nil {
				return chatMessage{}, err
			}
			location := item.URL
			if item.ForceDownload != ai.FileDownloadNever {
				if item.IsYouTube() {
					return chatMessage{}, fmt.Errorf("%s: downloading YouTube videos is not supported", model.providerName)
				}
				downloaded, err := downloadFileContent(
					ctx, item.URL, item.ResolvedMediaType, item.ForceDownload,
				)
				if err != nil {
					return chatMessage{}, err
				}
				location = downloaded.dataURI
			}
			parts = append(parts, contentPart{Type: "video_url", VideoURL: &videoURL{URL: location}})
		case ai.AudioURL:
			downloaded, err := downloadFileContent(ctx, item.URL, item.ResolvedMediaType, item.ForceDownload)
			if err != nil {
				return chatMessage{}, err
			}
			audioPart, err := model.chatAudioPart(downloaded.dataURI, downloaded.mediaType)
			if err != nil {
				return chatMessage{}, err
			}
			parts = append(parts, audioPart)
		case ai.DocumentURL:
			if err := item.ForceDownload.Validate(); err != nil {
				return chatMessage{}, err
			}
			mediaType, err := item.ResolvedMediaType()
			if err != nil {
				return chatMessage{}, err
			}
			if item.ForceDownload == ai.FileDownloadNever && model.chatCompatibility.FileURLInput {
				extension, err := fileExtension(mediaType)
				if err != nil {
					return chatMessage{}, err
				}
				parts = append(parts, contentPart{
					Type: "file", File: &chatFile{FileData: item.URL, Filename: "filename." + extension},
				})
				continue
			}
			downloaded, err := downloadFileContent(ctx, item.URL, item.ResolvedMediaType, item.ForceDownload)
			if err != nil {
				return chatMessage{}, err
			}
			documentPart, err := model.chatDocumentPart(
				downloaded.data, downloaded.dataURI, downloaded.mediaType, item.ResolvedIdentifier(),
			)
			if err != nil {
				return chatMessage{}, err
			}
			parts = append(parts, documentPart)
		case ai.UploadedFile:
			if item.ProviderName != model.providerName {
				return chatMessage{}, fmt.Errorf(
					"%s: uploaded file %q belongs to provider %q", model.providerName, item.FileID, item.ProviderName,
				)
			}
			if isImageMediaType(item.MediaType) {
				return chatMessage{}, fmt.Errorf(
					"%s: referencing uploaded images by file ID is not supported by Chat Completions", model.providerName,
				)
			}
			parts = append(parts, contentPart{Type: "file", File: &chatFile{FileID: item.FileID}})
		case ai.BinaryContent:
			location := fmt.Sprintf("data:%s;base64,%s", item.MediaType, base64.StdEncoding.EncodeToString(item.Data))
			switch {
			case isImageMediaType(item.MediaType):
				parts = append(parts, contentPart{
					Type: "image_url", ImageURL: &imageURL{
						URL: location, Detail: chatImageDetail(item.VendorMetadata),
					},
				})
			case isAudioMediaType(item.MediaType):
				audioPart, err := model.chatAudioPart(location, item.MediaType)
				if err != nil {
					return chatMessage{}, err
				}
				parts = append(parts, audioPart)
			case isVideoMediaType(item.MediaType):
				if !model.chatCompatibility.VideoInput {
					return chatMessage{}, fmt.Errorf(
						"%s: Chat Completions does not support inline video input", model.providerName,
					)
				}
				parts = append(parts, contentPart{Type: "video_url", VideoURL: &videoURL{URL: location}})
			default:
				if _, err := fileExtension(item.MediaType); err != nil {
					return chatMessage{}, err
				}
				documentPart, err := model.chatDocumentPart(
					item.Data, location, item.MediaType, item.ResolvedIdentifier(),
				)
				if err != nil {
					return chatMessage{}, err
				}
				parts = append(parts, documentPart)
			}
		default:
			return chatMessage{}, fmt.Errorf("openai: unknown user content type %T", c)
		}
	}
	return chatMessage{Role: "user", Content: parts}, nil
}
