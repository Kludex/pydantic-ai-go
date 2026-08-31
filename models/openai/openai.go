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
	"reflect"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
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
	Name           string
	BaseURL        string
	APIKey         string
	HTTPClient     *http.Client
	Headers        http.Header
	Query          url.Values
	PrepareRequest RequestPreparationFunc
}

// ChatCompatibility configures documented extensions to the OpenAI Chat
// Completions wire format. ReasoningContent preserves the Z.AI-compatible
// reasoning_content field. FinishReasons extends or overrides normalization.
type ChatCompatibility struct {
	ReasoningContent bool
	FinishReasons    map[string]ai.FinishReason
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
			ReasoningContent: compatibility.ReasoningContent,
			FinishReasons:    maps.Clone(finishReasons),
		}
	}
}

// WithResponsesCodeExecutionOutputs includes code-interpreter logs and image outputs in Responses results.
func WithResponsesCodeExecutionOutputs(enabled bool) Option {
	return func(model *Model) { model.responsesCodeExecutionOutputs = enabled }
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

// ProviderName returns the durable provider identity.
func (m *Model) ProviderName() string { return m.providerName }

// ProviderURL returns the configured provider API URL.
func (m *Model) ProviderURL() string { return m.baseURL }

// DefaultModelSettings returns this model's request defaults.
func (m *Model) DefaultModelSettings() ai.ModelSettings { return m.defaultSettings.Clone() }

// Request implements ai.Model.
func (m *Model) Request(ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
	payload, err := m.buildPayload(msgs, params)
	if err != nil {
		return nil, err
	}
	body, err := marshalRequest(payload, params.Settings.ExtraBody)
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
		return nil, fmt.Errorf("openai: request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(data)}
	}
	response, err := m.parseResponse(data)
	if response != nil {
		response.ProviderName = m.providerName
		response.ProviderURL = m.baseURL
	}
	return response, err
}

// APIError is a non-200 response from the OpenAI API.
type APIError struct {
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("openai: API returned status %d: %s", e.StatusCode, e.Body)
}

// IsModelAPIError marks provider API responses as eligible for default model fallback.
func (*APIError) IsModelAPIError() bool { return true }

type chatRequest struct {
	Model             string          `json:"model"`
	Messages          []chatMessage   `json:"messages"`
	Tools             []chatTool      `json:"tools,omitempty"`
	ToolChoice        any             `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool           `json:"parallel_tool_calls,omitempty"`
	MaxTokens         int             `json:"max_completion_tokens,omitempty"`
	Temperature       *float64        `json:"temperature,omitempty"`
	TopP              *float64        `json:"top_p,omitempty"`
	Seed              *int            `json:"seed,omitempty"`
	Stop              []string        `json:"stop,omitempty"`
	Stream            bool            `json:"stream,omitempty"`
	StreamOptions     *streamOptions  `json:"stream_options,omitempty"`
	ResponseFormat    *responseFormat `json:"response_format,omitempty"`
	ReasoningEffort   string          `json:"reasoning_effort,omitempty"`
	PresencePenalty   *float64        `json:"presence_penalty,omitempty"`
	FrequencyPenalty  *float64        `json:"frequency_penalty,omitempty"`
	LogitBias         map[string]int  `json:"logit_bias,omitempty"`
	Logprobs          *bool           `json:"logprobs,omitempty"`
	TopLogprobs       *int            `json:"top_logprobs,omitempty"`
	ServiceTier       ai.ServiceTier  `json:"service_tier,omitempty"`
}

type chatMessage struct {
	Role             string     `json:"role"`
	Content          any        `json:"content,omitempty"` // string or []contentPart
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []toolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
}

type contentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

type imageURL struct {
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

type chatTool struct {
	Type     string       `json:"type"`
	Function chatFunction `json:"function"`
}

type chatFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
	Strict      *bool          `json:"strict,omitempty"`
}

func (m *Model) buildPayload(msgs []ai.ModelMessage, params ai.ModelRequestParams) (*chatRequest, error) {
	for _, nativeTool := range params.NativeTools {
		if nativeToolIsNil(nativeTool) {
			return nil, fmt.Errorf("openai: native tool must not be nil")
		}
		if !nativeTool.IsOptional() {
			return nil, fmt.Errorf("openai: Chat Completions does not support native tool %q", nativeTool.Kind())
		}
	}
	reasoningEffort, err := openAIThinkingEffort(params.Settings.Thinking)
	if err != nil {
		return nil, err
	}
	serviceTier, err := openAIServiceTier(params.Settings.ServiceTier)
	if err != nil {
		return nil, err
	}
	req := &chatRequest{
		Model:            m.name,
		MaxTokens:        params.Settings.MaxTokens,
		Temperature:      params.Settings.Temperature,
		TopP:             params.Settings.TopP,
		Seed:             params.Settings.Seed,
		Stop:             params.Settings.StopSequences,
		ReasoningEffort:  reasoningEffort,
		PresencePenalty:  params.Settings.PresencePenalty,
		FrequencyPenalty: params.Settings.FrequencyPenalty,
		LogitBias:        params.Settings.LogitBias,
		Logprobs:         params.Settings.Logprobs,
		TopLogprobs:      params.Settings.TopLogprobs,
		ServiceTier:      serviceTier,
	}
	if openAIReasoningActive(reasoningEffort) {
		req.Temperature = nil
		req.TopP = nil
	}
	if params.Instructions != "" {
		req.Messages = append(req.Messages, chatMessage{Role: "system", Content: params.Instructions})
	}
	for _, msg := range msgs {
		converted, err := m.convertMessage(msg)
		if err != nil {
			return nil, err
		}
		req.Messages = append(req.Messages, converted...)
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
		if !params.AllowText {
			req.ToolChoice = "required"
		}
	}
	if len(req.Tools) > 0 {
		req.ParallelToolCalls = params.Settings.ParallelToolCalls
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

func nativeToolIsNil(tool ai.NativeTool) bool {
	if tool == nil {
		return true
	}
	value := reflect.ValueOf(tool)
	return value.Kind() == reflect.Pointer && value.IsNil()
}

func openAIServiceTier(tier ai.ServiceTier) (ai.ServiceTier, error) {
	switch tier {
	case "", ai.ServiceTierAuto, ai.ServiceTierDefault, ai.ServiceTierFlex, ai.ServiceTierPriority:
		return tier, nil
	default:
		return "", fmt.Errorf("openai: invalid service tier %q", tier)
	}
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

func openAIReasoningActive(effort string) bool {
	return effort != "" && effort != "none"
}

func (model *Model) convertMessage(msg ai.ModelMessage) ([]chatMessage, error) {
	switch message := msg.(type) {
	case ai.ModelRequest:
		return convertRequest(message)
	case ai.ModelResponse:
		return model.convertResponse(message), nil
	default:
		return nil, fmt.Errorf("openai: unknown message type %T", msg)
	}
}

func convertRequest(m ai.ModelRequest) ([]chatMessage, error) {
	var out []chatMessage
	for _, part := range m.Parts {
		switch p := part.(type) {
		case ai.SystemPromptPart:
			out = append(out, chatMessage{Role: "system", Content: p.Content})
		case ai.UserPromptPart:
			msg, err := convertUserPrompt(p)
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

func (model *Model) convertResponse(m ai.ModelResponse) []chatMessage {
	msg := chatMessage{Role: "assistant"}
	for _, part := range m.Parts {
		switch p := part.(type) {
		case ai.TextPart:
			msg.Content = p.Content
		case ai.ThinkingPart:
			if model.chatCompatibility.ReasoningContent &&
				(p.ProviderName == "" || p.ProviderName == model.providerName) {
				msg.ReasoningContent += p.Content
			}
		case ai.ToolCallPart:
			msg.ToolCalls = append(msg.ToolCalls, toolCall{
				ID:       p.ToolCallID,
				Type:     "function",
				Function: functionCall{Name: p.ToolName, Arguments: string(p.Args)},
			})
		}
	}
	return []chatMessage{msg}
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

type chatResponse struct {
	ID                string `json:"id"`
	Model             string `json:"model"`
	Created           int64  `json:"created"`
	ServiceTier       string `json:"service_tier"`
	SystemFingerprint string `json:"system_fingerprint"`
	Choices           []struct {
		Message struct {
			Content          string     `json:"content"`
			Refusal          string     `json:"refusal"`
			ReasoningContent string     `json:"reasoning_content"`
			ToolCalls        []toolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
		Logprobs     *struct {
			Content []map[string]any `json:"content"`
		} `json:"logprobs"`
	} `json:"choices"`
	Usage chatUsage `json:"usage"`
}

type chatUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
		AudioTokens  int `json:"audio_tokens"`
	} `json:"prompt_tokens_details"`
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
		CacheReadTokens:          u.PromptTokensDetails.CachedTokens,
		InputAudioTokens:         u.PromptTokensDetails.AudioTokens,
		OutputAudioTokens:        u.CompletionTokensDetails.AudioTokens,
		ReasoningTokens:          u.CompletionTokensDetails.ReasoningTokens,
		AcceptedPredictionTokens: u.CompletionTokensDetails.AcceptedPredictionTokens,
		RejectedPredictionTokens: u.CompletionTokensDetails.RejectedPredictionTokens,
		Details: map[string]int{
			"reasoning_tokens":           u.CompletionTokensDetails.ReasoningTokens,
			"audio_tokens":               u.CompletionTokensDetails.AudioTokens,
			"accepted_prediction_tokens": u.CompletionTokensDetails.AcceptedPredictionTokens,
			"rejected_prediction_tokens": u.CompletionTokensDetails.RejectedPredictionTokens,
		},
	}
}

func (model *Model) parseResponse(data []byte) (*ai.ModelResponse, error) {
	var cr chatResponse
	if err := json.Unmarshal(data, &cr); err != nil {
		return nil, fmt.Errorf("openai: parse response: %w", err)
	}
	if len(cr.Choices) == 0 {
		return nil, fmt.Errorf("openai: response has no choices")
	}
	providerDetails := map[string]any{}
	if cr.Choices[0].FinishReason != "" {
		providerDetails["finish_reason"] = cr.Choices[0].FinishReason
	}
	if cr.Created != 0 {
		providerDetails["timestamp"] = time.Unix(cr.Created, 0).UTC()
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
	if len(providerDetails) == 0 {
		providerDetails = nil
	}
	resp := &ai.ModelResponse{
		ModelName: cr.Model, Timestamp: time.Unix(cr.Created, 0).UTC(), Usage: cr.Usage.usage(),
		ProviderDetails: providerDetails, ProviderResponseID: cr.ID,
		FinishReason: model.chatFinishReason(cr.Choices[0].FinishReason), State: ai.ModelResponseStateComplete,
	}
	msg := cr.Choices[0].Message
	if msg.Refusal != "" {
		resp.FinishReason = ai.FinishReasonContentFilter
		return resp, nil
	}
	if model.chatCompatibility.ReasoningContent && msg.ReasoningContent != "" {
		resp.Parts = append(resp.Parts, ai.ThinkingPart{
			Content: msg.ReasoningContent, ProviderName: model.providerName,
		})
	}
	if msg.Content != "" {
		resp.Parts = append(resp.Parts, ai.TextPart{Content: msg.Content, ProviderName: model.providerName})
	}
	for _, call := range msg.ToolCalls {
		resp.Parts = append(resp.Parts, ai.ToolCallPart{
			ToolName:     call.Function.Name,
			Args:         json.RawMessage(call.Function.Arguments),
			ToolCallID:   call.ID,
			ProviderName: model.providerName,
		})
	}
	return resp, nil
}

func (model *Model) chatFinishReason(reason string) ai.FinishReason {
	if normalized, exists := model.chatCompatibility.FinishReasons[reason]; exists {
		return normalized
	}
	return map[string]ai.FinishReason{
		"stop": ai.FinishReasonStop, "length": ai.FinishReasonLength,
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

func convertUserPrompt(p ai.UserPromptPart) (chatMessage, error) {
	if len(p.Contents) == 0 {
		return chatMessage{Role: "user", Content: p.Content}, nil
	}
	parts := make([]contentPart, 0, len(p.Contents))
	for _, c := range p.Contents {
		switch item := c.(type) {
		case ai.TextContent:
			parts = append(parts, contentPart{Type: "text", Text: item.Text})
		case ai.ImageURL:
			parts = append(parts, contentPart{Type: "image_url", ImageURL: &imageURL{URL: item.URL}})
		case ai.BinaryContent:
			url := fmt.Sprintf("data:%s;base64,%s", item.MediaType, base64.StdEncoding.EncodeToString(item.Data))
			parts = append(parts, contentPart{Type: "image_url", ImageURL: &imageURL{URL: url}})
		default:
			return chatMessage{}, fmt.Errorf("openai: unknown user content type %T", c)
		}
	}
	return chatMessage{Role: "user", Content: parts}, nil
}
