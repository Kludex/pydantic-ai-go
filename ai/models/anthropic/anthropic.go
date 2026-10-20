// Package anthropic implements ai.Model against the Anthropic Messages API.
package anthropic

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/contextwindow"
	"github.com/Kludex/pydantic-ai-go/ai/internal/download"
	"github.com/Kludex/pydantic-ai-go/ai/internal/promptcache"
	"github.com/Kludex/pydantic-ai-go/ai/internal/thinking"
)

const defaultMaxTokens = 16384

// Model calls the Anthropic Messages API. Create one with NewModel.
type Model struct {
	name                string
	apiKey              string
	baseURL             string
	httpClient          *http.Client
	strictToolSupport   bool
	deferredToolSupport bool
	carryThinking       bool
	requestScoped       bool
	schemaWarning       func(SchemaWarning)
	defaultSettings     ai.ModelSettings
	legacyBedrockClient LegacyBedrockClient
	automaticCaching    *bool
	contextWindow       int
}

// ModelProfile reports model behavior and the bundled context window when known.
func (m *Model) ModelProfile() ai.ModelProfile {
	tiers := []ai.CacheRetention{ai.CacheRetention5Minutes, ai.CacheRetention1Hour}
	if m.legacyBedrockClient != nil {
		tiers = promptcache.BedrockRetentions(m.name)
	}
	return ai.ModelProfile{
		SupportsCache: true, SupportsAutoCache: m.supportsAutomaticCaching(), SupportedCacheRetentions: tiers,
		DefaultOutputMode:                  ai.OutputModeTool,
		SupportsToolAvailabilityDelta:      m.deferredToolSupport,
		MimicsAssistantMessageFormatting:   m.carryThinking,
		SupportsRequestScopedSystemPrompts: m.clearsRequestScopedPrompts(),
		ContextWindow:                      m.contextWindow,
		DefaultCacheRetention:              5 * time.Minute,
	}
}

// supportsAutomaticCaching reports whether the API takes the top-level
// cache_control parameter. The legacy Bedrock InvokeModel API rejects it on the
// models it does not serve from the Messages API infrastructure.
func (m *Model) supportsAutomaticCaching() bool {
	if m.automaticCaching != nil {
		return *m.automaticCaching
	}
	return m.legacyBedrockClient == nil || promptcache.BedrockSupportsAutomaticCaching(m.name)
}

// clearsRequestScopedPrompts excludes the legacy Bedrock client, which sends no
// anthropic-beta, and the API rejects clear_at without it.
func (m *Model) clearsRequestScopedPrompts() bool {
	return m.requestScoped && m.legacyBedrockClient == nil
}

// SupportsToolAvailabilityDelta reports whether this request can render native tool additions.
func (m *Model) SupportsToolAvailabilityDelta(params ai.ModelRequestParams) bool {
	return m.deferredToolSupport && len(params.DeferredTools) > 0 && hasStableAnthropicTool(params)
}

// Option configures a Model.
type Option func(*Model)

// WithAPIKey sets the API key. The default is the ANTHROPIC_API_KEY environment variable.
func WithAPIKey(key string) Option { return func(m *Model) { m.apiKey = key } }

// WithBaseURL points the model at a different endpoint. The default is
// https://api.anthropic.com/v1.
func WithBaseURL(url string) Option { return func(m *Model) { m.baseURL = url } }

// WithHTTPClient sets the HTTP client used for requests.
func WithHTTPClient(c *http.Client) Option { return func(m *Model) { m.httpClient = c } }

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	return func(m *Model) { m.defaultSettings = settings.Clone() }
}

// WithStrictToolSupport overrides whether the model supports Anthropic strict
// tool definitions. Use it for aliases and newly released model versions.
func WithStrictToolSupport(enabled bool) Option {
	return func(m *Model) { m.strictToolSupport = enabled }
}

// WithDeferredToolSupport overrides native deferred-tool rendering. Disable
// it for Anthropic-compatible endpoints that do not support defer_loading.
func WithDeferredToolSupport(enabled bool) Option {
	return func(m *Model) { m.deferredToolSupport = enabled }
}

// WithMimicsAssistantMessageFormatting overrides whether Claude copies markup from
// earlier assistant turns into its answers. It is enabled by default, so reasoning
// without a valid Anthropic signature is sent in a user turn ahead of the assistant
// turn. Disable it to send that reasoning in <thinking> tags in the assistant turn.
func WithMimicsAssistantMessageFormatting(enabled bool) Option {
	return func(m *Model) { m.carryThinking = enabled }
}

// WithRequestScopedSystemPrompts overrides whether request-scoped system
// prompts are sent as system entries the API clears itself, under the
// mid-conversation-system-clear-at-2026-08-21 beta. Disable it for an account
// without access to the beta: each prompt is then sent only with its own request.
func WithRequestScopedSystemPrompts(enabled bool) Option {
	return func(m *Model) { m.requestScoped = enabled }
}

// WithAutomaticCaching overrides whether the API takes Anthropic's top-level
// automatic cache_control parameter. Without it, Settings.Cache and the portable
// cache setting place a breakpoint on the last message block instead. Enable it
// for a legacy Bedrock model this package does not know supports automatic
// caching yet, or disable it for an Anthropic-compatible gateway that rejects it.
func WithAutomaticCaching(enabled bool) Option {
	return func(m *Model) { m.automaticCaching = &enabled }
}

// SchemaWarning describes a lossy strict-schema conversion.
type SchemaWarning struct {
	// ToolName identifies the transformed definition.
	ToolName string
	// Message explains which schema behavior was not preserved.
	Message string
}

// WithSchemaWarningHandler receives inspectable warnings when Anthropic's
// strict-schema conversion cannot preserve a tool schema. The handler may be
// called concurrently when the model is shared by concurrent runs.
func WithSchemaWarningHandler(handler func(SchemaWarning)) Option {
	return func(m *Model) { m.schemaWarning = handler }
}

// NewModel creates a Model for the named Anthropic model, e.g. "claude-sonnet-4-5".
func NewModel(name string, opts ...Option) *Model {
	m := &Model{
		name:                name,
		apiKey:              os.Getenv("ANTHROPIC_API_KEY"),
		baseURL:             "https://api.anthropic.com/v1",
		httpClient:          http.DefaultClient,
		strictToolSupport:   supportsStrictTools(name),
		deferredToolSupport: supportsDeferredTools(name),
		carryThinking:       true,
		requestScoped:       supportsRequestScopedSystemPrompts(name),
	}
	for _, opt := range opts {
		opt(m)
	}
	m.contextWindow = contextwindow.Lookup(m.name, "anthropic", m.baseURL)
	return m
}

// Name returns the model name.
func (m *Model) Name() string { return m.name }

// ContextWindow returns the bundled context window. Zero means unknown.
func (m *Model) ContextWindow() int { return m.contextWindow }

// SupportsNativeTool reports Anthropic native-tool support.
func (m *Model) SupportsNativeTool(tool ai.NativeTool) bool {
	if err := ai.ValidateNativeTools([]ai.NativeTool{tool}); err != nil {
		return false
	}
	switch tool := tool.CloneNativeTool().(type) {
	case ai.WebSearchTool:
		return tool.ExternalWebAccess == nil || *tool.ExternalWebAccess
	case ai.WebFetchTool, ai.CodeExecutionTool, ai.MemoryTool, ai.MCPServerTool:
		return true
	case ai.AdvisorTool:
		return anthropicSupportsAdvisor(m.name)
	default:
		return false
	}
}

// ProviderName returns Anthropic's durable provider identity.
func (*Model) ProviderName() string { return "anthropic" }

// ProviderURL returns the configured provider API URL.
func (m *Model) ProviderURL() string { return m.baseURL }

// DefaultModelSettings returns this model's request defaults.
func (m *Model) DefaultModelSettings() ai.ModelSettings { return m.defaultSettings.Clone() }

// PromptCacheRetention reports the longest requested Anthropic cache lifetime.
func (m *Model) PromptCacheRetention(settings ai.ModelSettings) (time.Duration, bool) {
	settings, err := m.translateCache(settings)
	if err != nil {
		return 0, false
	}
	_, cache, err := extractCacheSettings(settings)
	if err != nil {
		return 0, false
	}
	for _, ttl := range []CacheTTL{cache.Automatic, cache.Instructions, cache.Messages, cache.ToolDefinitions} {
		if ttl == CacheTTL1Hour {
			return time.Hour, true
		}
	}
	for _, ttl := range []CacheTTL{cache.Automatic, cache.Instructions, cache.Messages, cache.ToolDefinitions} {
		if ttl == CacheTTL5Minutes {
			return 5 * time.Minute, true
		}
	}
	return 0, false
}

// Request implements ai.Model.
func (m *Model) Request(ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
	payload, err := m.buildPayload(ctx, msgs, params)
	if err != nil {
		return nil, err
	}
	if m.legacyBedrockClient != nil {
		return m.requestLegacyBedrock(ctx, payload, params.Settings.ExtraHeaders)
	}
	send := func() (int, []byte, error) {
		body, err := marshalRequest(payload, payload.ExtraBody)
		if err != nil {
			return 0, nil, fmt.Errorf("anthropic: marshal request: %w", err)
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/messages", bytes.NewReader(body))
		if err != nil {
			return 0, nil, err
		}
		setExtraHeaders(req, params.Settings.ExtraHeaders)
		m.setRequestHeaders(req, payload, false)
		resp, err := m.httpClient.Do(req)
		if err != nil {
			return 0, nil, ai.NewModelTransportError(ctx, m, "request", err)
		}
		data, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if readErr != nil {
			return 0, nil, ai.NewModelTransportError(ctx, m, "read response", readErr)
		}
		return resp.StatusCode, data, nil
	}
	statusCode, data, err := send()
	if err != nil {
		return nil, err
	}
	if anthropicExpiredContainer(statusCode, data) && payload.ContainerFromHistory &&
		messagesHaveContainerUploads(payload.Messages) {
		payload.Container = nil
		statusCode, data, err = send()
		if err != nil {
			return nil, err
		}
	}
	if anthropicStaleThinkingError(m.name, statusCode, anthropicErrorMessage(data), payload) {
		enableAnthropicStaleThinkingDrop(payload)
		statusCode, data, err = send()
		if err != nil {
			return nil, err
		}
	}
	if statusCode != http.StatusOK {
		return nil, &APIError{StatusCode: statusCode, Body: string(data)}
	}
	response, err := parseResponse(data)
	if response != nil {
		response.ProviderName = "anthropic"
		response.ProviderURL = m.baseURL
	}
	return response, err
}

// CountTokens counts prospective input tokens through Anthropic's count endpoint.
func (m *Model) CountTokens(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (ai.Usage, error) {
	payload, err := m.buildPayload(ctx, messages, params)
	if err != nil {
		return ai.Usage{}, err
	}
	if m.legacyBedrockClient != nil {
		return m.countLegacyBedrock(ctx, payload, params.Settings.ExtraHeaders)
	}
	countTools := make([]toolParam, 0, len(payload.Tools))
	for _, tool := range payload.Tools {
		if tool.Type == "" || tool.Type == "memory_20250818" {
			countTools = append(countTools, tool)
		}
	}
	countMessages := make([]messageParam, len(payload.Messages))
	for index, message := range payload.Messages {
		countMessages[index] = message
		countMessages[index].Content = make([]contentBlock, 0, len(message.Content))
		for _, block := range message.Content {
			if block.Type == "advisor_tool_result" || block.Type == "server_tool_use" && block.Name == "advisor" {
				continue
			}
			countMessages[index].Content = append(countMessages[index].Content, block)
		}
	}
	countPayload := struct {
		Model             string                       `json:"model"`
		System            any                          `json:"system,omitempty"`
		Messages          []messageParam               `json:"messages"`
		Tools             []toolParam                  `json:"tools,omitempty"`
		ToolChoice        *toolChoiceParam             `json:"tool_choice,omitempty"`
		Thinking          *thinkingParam               `json:"thinking,omitempty"`
		ContextManagement map[string]any               `json:"context_management,omitempty"`
		CacheControl      *anthropicPromptCacheControl `json:"cache_control,omitempty"`
	}{
		Model: payload.Model, System: payload.System, Messages: countMessages, Tools: countTools,
		ToolChoice: payload.ToolChoice, Thinking: payload.Thinking, ContextManagement: payload.ContextManagement,
		CacheControl: payload.CacheControl,
	}
	send := func() (int, []byte, error) {
		countPayload.Thinking = payload.Thinking
		body, err := marshalRequest(countPayload, payload.ExtraBody)
		if err != nil {
			return 0, nil, fmt.Errorf("anthropic: marshal token count request: %w", err)
		}
		req, err := http.NewRequestWithContext(
			ctx, http.MethodPost, m.baseURL+"/messages/count_tokens?beta=true", bytes.NewReader(body),
		)
		if err != nil {
			return 0, nil, err
		}
		setExtraHeaders(req, params.Settings.ExtraHeaders)
		headerPayload := *payload
		headerPayload.Betas = slices.DeleteFunc(slices.Clone(payload.Betas), func(beta string) bool {
			return beta != "context-management-2025-06-27" && beta != "thinking-binding-controls-2026-08-01"
		})
		m.setRequestHeaders(req, &headerPayload, false)
		resp, err := m.httpClient.Do(req)
		if err != nil {
			return 0, nil, ai.NewModelTransportError(ctx, m, "token count request", err)
		}
		data, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if readErr != nil {
			return 0, nil, ai.NewModelTransportError(ctx, m, "read token count response", readErr)
		}
		return resp.StatusCode, data, nil
	}
	statusCode, data, err := send()
	if err != nil {
		return ai.Usage{}, err
	}
	if anthropicStaleThinkingError(m.name, statusCode, anthropicErrorMessage(data), payload) {
		enableAnthropicStaleThinkingDrop(payload)
		statusCode, data, err = send()
		if err != nil {
			return ai.Usage{}, err
		}
	}
	if statusCode != http.StatusOK {
		return ai.Usage{}, &APIError{StatusCode: statusCode, Body: string(data)}
	}
	var counted struct {
		InputTokens *int `json:"input_tokens"`
	}
	if err := json.Unmarshal(data, &counted); err != nil {
		return ai.Usage{}, ai.NewModelTransportError(ctx, m, "decode token count response", err)
	}
	if counted.InputTokens == nil {
		return ai.Usage{}, fmt.Errorf("anthropic: token count response omitted input_tokens")
	}
	return ai.Usage{InputTokens: *counted.InputTokens}, nil
}

func (m *Model) setRequestHeaders(req *http.Request, payload *messagesRequest, stream bool) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", m.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	if betas := anthropicBetas(req.Header.Get("anthropic-beta"), payload); len(betas) > 0 {
		req.Header.Set("anthropic-beta", strings.Join(betas, ","))
	}
	if stream {
		req.Header.Set("Accept", "text/event-stream")
	}
}

func anthropicBetas(requested string, payload *messagesRequest) []string {
	betas := strings.FieldsFunc(requested, func(r rune) bool { return r == ',' || r == ' ' })
	if payload.ToolAdditions {
		betas = append(betas, "mid-conversation-tool-changes-2026-07-01")
	}
	if payload.Compaction {
		betas = append(betas, "compact-2026-01-12")
	}
	betas = append(betas, payload.Betas...)
	unique := make([]string, 0, len(betas))
	for _, beta := range betas {
		if !slices.Contains(unique, beta) {
			unique = append(unique, beta)
		}
	}
	return unique
}

// APIError is a non-200 response from the Anthropic API.
type APIError struct {
	// StatusCode is the HTTP response status.
	StatusCode int
	// Body is the provider response body.
	Body string
}

// Error formats the provider status and body.
func (e *APIError) Error() string {
	return fmt.Sprintf("anthropic: API returned status %d: %s", e.StatusCode, e.Body)
}

// IsModelAPIError marks provider API responses as eligible for default model fallback.
func (*APIError) IsModelAPIError() bool { return true }

type messagesRequest struct {
	Model                string                       `json:"model"`
	MaxTokens            int                          `json:"max_tokens"`
	System               any                          `json:"system,omitempty"`
	Messages             []messageParam               `json:"messages"`
	Tools                []toolParam                  `json:"tools,omitempty"`
	ToolChoice           *toolChoiceParam             `json:"tool_choice,omitempty"`
	Temperature          *float64                     `json:"temperature,omitempty"`
	TopP                 *float64                     `json:"top_p,omitempty"`
	Stop                 []string                     `json:"stop_sequences,omitempty"`
	Stream               bool                         `json:"stream,omitempty"`
	ToolAdditions        bool                         `json:"-"`
	Compaction           bool                         `json:"-"`
	ContainerFromHistory bool                         `json:"-"`
	Thinking             *thinkingParam               `json:"thinking,omitempty"`
	ServiceTier          string                       `json:"service_tier,omitempty"`
	ContextManagement    map[string]any               `json:"context_management,omitempty"`
	Container            any                          `json:"container,omitempty"`
	MCPServers           []anthropicMCPServer         `json:"mcp_servers,omitempty"`
	Betas                []string                     `json:"-"`
	CacheControl         *anthropicPromptCacheControl `json:"cache_control,omitempty"`
	OutputConfig         *anthropicOutputConfig       `json:"output_config,omitempty"`
	ExtraBody            map[string]any               `json:"-"`
	Diagnostics          *struct {
		PreviousMessageID *string `json:"previous_message_id"`
	} `json:"diagnostics,omitempty"`
}

type anthropicOutputConfig struct {
	Format *anthropicOutputFormat `json:"format,omitempty"`
	Effort Effort                 `json:"effort,omitempty"`
}

type anthropicOutputFormat struct {
	Type   string         `json:"type"`
	Schema map[string]any `json:"schema"`
}

type anthropicMCPServer struct {
	Type               string                         `json:"type"`
	Name               string                         `json:"name"`
	URL                string                         `json:"url"`
	AuthorizationToken string                         `json:"authorization_token,omitempty"`
	ToolConfiguration  *anthropicMCPToolConfiguration `json:"tool_configuration,omitempty"`
}

type anthropicMCPToolConfiguration struct {
	Enabled      bool     `json:"enabled"`
	AllowedTools []string `json:"allowed_tools"`
}

type thinkingParam struct {
	Type         string                `json:"type,omitempty"`
	BudgetTokens int                   `json:"budget_tokens,omitempty"`
	BlockBinding *thinkingBlockBinding `json:"block_binding,omitempty"`
}

type thinkingBlockBinding struct {
	PrefixMismatchBehavior string `json:"prefix_mismatch_behavior"`
}

type messageParam struct {
	Role    string         `json:"role"`
	Content []contentBlock `json:"content"`
	ClearAt string         `json:"clear_at,omitempty"`
}

type contentBlock struct {
	Type string `json:"type"`
	// text and thinking
	Text      string  `json:"text,omitempty"`
	Thinking  *string `json:"thinking,omitempty"`
	Signature string  `json:"signature,omitempty"`
	// image
	Source *imageSource `json:"source,omitempty"`
	// tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
	// tool_result
	ToolUseID        string                       `json:"tool_use_id,omitempty"`
	Content          any                          `json:"content,omitempty"`
	EncryptedContent string                       `json:"encrypted_content,omitempty"`
	Caller           map[string]any               `json:"caller,omitempty"`
	IsError          bool                         `json:"is_error,omitempty"`
	Tool             *toolReferenceParam          `json:"tool,omitempty"`
	FileID           string                       `json:"file_id,omitempty"`
	ServerName       string                       `json:"server_name,omitempty"`
	CacheControl     *anthropicPromptCacheControl `json:"cache_control,omitempty"`
}

type toolReferenceParam struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

type toolReferenceContent struct {
	Type     string `json:"type"`
	ToolName string `json:"tool_name"`
}

type toolParam struct {
	Type             string                       `json:"type,omitempty"`
	Name             string                       `json:"name"`
	Description      string                       `json:"description,omitempty"`
	InputSchema      map[string]any               `json:"input_schema,omitempty"`
	Strict           *bool                        `json:"strict,omitempty"`
	CacheControl     *anthropicPromptCacheControl `json:"cache_control,omitempty"`
	DeferLoading     bool                         `json:"defer_loading,omitempty"`
	MaxUses          *int                         `json:"max_uses,omitempty"`
	AllowedDomains   []string                     `json:"allowed_domains,omitempty"`
	BlockedDomains   []string                     `json:"blocked_domains,omitempty"`
	UserLocation     *anthropicWebSearchLocation  `json:"user_location,omitempty"`
	Citations        *anthropicCitations          `json:"citations,omitempty"`
	MaxContentTokens int                          `json:"max_content_tokens,omitempty"`
	Model            string                       `json:"model,omitempty"`
	MaxTokens        *int                         `json:"max_tokens,omitempty"`
	Caching          *anthropicCacheControl       `json:"caching,omitempty"`
}

type anthropicCacheControl struct {
	Type string               `json:"type"`
	TTL  ai.AdvisorCachingTTL `json:"ttl"`
}

type anthropicPromptCacheControl struct {
	Type string           `json:"type"`
	TTL  ai.CachePointTTL `json:"ttl"`
}

type anthropicCitations struct {
	Enabled bool `json:"enabled"`
}

type anthropicWebSearchLocation struct {
	Type     string `json:"type"`
	City     string `json:"city,omitempty"`
	Country  string `json:"country,omitempty"`
	Region   string `json:"region,omitempty"`
	Timezone string `json:"timezone,omitempty"`
}

type toolChoiceParam struct {
	Type                   string `json:"type"`
	DisableParallelToolUse *bool  `json:"disable_parallel_tool_use,omitempty"`
}

type imageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
	FileID    string `json:"file_id,omitempty"`
}

func convertUserPrompt(ctx context.Context, p ai.UserPromptPart) ([]contentBlock, error) {
	if len(p.Contents) == 0 {
		return []contentBlock{{Type: "text", Text: p.Content}}, nil
	}
	blocks := make([]contentBlock, 0, len(p.Contents))
	for _, c := range p.Contents {
		switch item := c.(type) {
		case ai.CachePoint:
			ttl, err := item.ResolvedTTL()
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, contentBlock{
				Type:         "cache_point_marker",
				CacheControl: &anthropicPromptCacheControl{Type: "ephemeral", TTL: ttl},
			})
		case ai.TextContent:
			blocks = append(blocks, contentBlock{Type: "text", Text: item.Text})
		case ai.BinaryContent:
			block, err := anthropicBinaryBlock(item.Data, item.MediaType)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, block)
		case ai.ImageURL:
			if err := item.ForceDownload.Validate(); err != nil {
				return nil, err
			}
			if item.ForceDownload == ai.FileDownloadNever {
				blocks = append(blocks, contentBlock{
					Type: "image", Source: &imageSource{Type: "url", URL: item.URL},
				})
				continue
			}
			mediaType, err := item.ResolvedMediaType()
			if err != nil {
				return nil, err
			}
			downloaded, err := download.Fetch(
				ctx, item.URL, item.ForceDownload == ai.FileDownloadAllowLocal,
			)
			if err != nil {
				return nil, err
			}
			block, err := anthropicBinaryBlock(downloaded.Data, mediaType)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, block)
		case ai.DocumentURL:
			if err := item.ForceDownload.Validate(); err != nil {
				return nil, err
			}
			mediaType, err := item.ResolvedMediaType()
			if err != nil {
				return nil, err
			}
			if mediaType == "application/pdf" && item.ForceDownload == ai.FileDownloadNever {
				blocks = append(blocks, contentBlock{
					Type: "document", Source: &imageSource{Type: "url", URL: item.URL},
				})
				continue
			}
			if mediaType != "application/pdf" && mediaType != "text/plain" {
				return nil, fmt.Errorf("anthropic: unsupported document media type %q", mediaType)
			}
			downloaded, err := download.Fetch(
				ctx, item.URL, item.ForceDownload == ai.FileDownloadAllowLocal,
			)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, anthropicDocumentBlock(downloaded.Data, mediaType))
		case ai.AudioURL:
			return nil, fmt.Errorf("anthropic: audio URL input is not supported")
		case ai.VideoURL:
			return nil, fmt.Errorf("anthropic: video URL input is not supported")
		case ai.UploadedFile:
			if item.ProviderName != "anthropic" {
				return nil, fmt.Errorf(
					"anthropic: uploaded file %q belongs to provider %q", item.FileID, item.ProviderName,
				)
			}
			blockType := "document"
			if strings.HasPrefix(item.MediaType, "image/") {
				blockType = "image"
			} else if item.MediaType != "" && !strings.HasPrefix(item.MediaType, "text/") &&
				!strings.HasPrefix(item.MediaType, "application/") {
				return nil, fmt.Errorf("anthropic: unsupported uploaded file media type %q", item.MediaType)
			}
			blocks = append(blocks, contentBlock{
				Type: blockType, Source: &imageSource{Type: "file", FileID: item.FileID},
			})
		default:
			return nil, fmt.Errorf("anthropic: unsupported user content type %T", c)
		}
	}
	return blocks, nil
}

func anthropicBinaryBlock(data []byte, mediaType string) (contentBlock, error) {
	switch {
	case strings.HasPrefix(mediaType, "image/"):
		return contentBlock{Type: "image", Source: &imageSource{
			Type: "base64", MediaType: mediaType, Data: base64.StdEncoding.EncodeToString(data),
		}}, nil
	case mediaType == "application/pdf", mediaType == "text/plain":
		return anthropicDocumentBlock(data, mediaType), nil
	default:
		return contentBlock{}, fmt.Errorf("anthropic: unsupported binary content media type %q", mediaType)
	}
}

func anthropicDocumentBlock(data []byte, mediaType string) contentBlock {
	if mediaType == "application/pdf" {
		return contentBlock{Type: "document", Source: &imageSource{
			Type: "base64", MediaType: mediaType, Data: base64.StdEncoding.EncodeToString(data),
		}}
	}
	return contentBlock{Type: "document", Source: &imageSource{
		Type: "text", MediaType: mediaType, Data: string(data),
	}}
}

func resolveAnthropicCachePoints(messages *[]messageParam) error {
	var lastBlock *contentBlock
	resolved := make([]messageParam, 0, len(*messages))
	for _, message := range *messages {
		if message.ClearAt != "" {
			resolved = append(resolved, message)
			continue
		}
		content := make([]contentBlock, 0, len(message.Content))
		hadMarker := false
		for _, block := range message.Content {
			if block.Type == "cache_point_marker" {
				hadMarker = true
				if lastBlock == nil {
					return fmt.Errorf("anthropic: cache point must follow user content")
				}
				if !anthropicCacheable(lastBlock.Type) {
					return fmt.Errorf("anthropic: cache control not supported for content type %q", lastBlock.Type)
				}
				lastBlock.CacheControl = block.CacheControl
				continue
			}
			content = append(content, block)
			lastBlock = &content[len(content)-1]
		}
		if len(content) > 0 || !hadMarker {
			message.Content = content
			resolved = append(resolved, message)
			if len(content) > 0 {
				lastBlock = &resolved[len(resolved)-1].Content[len(content)-1]
			}
		}
	}
	*messages = resolved
	return nil
}

// addAnthropicMessageCachePoint marks the last block the next request still
// sends: it skips clear_at entries, which cannot carry one, and the trailing
// text blocks rendering scopedTail, which the next request leaves out.
func addAnthropicMessageCachePoint(messages []messageParam, ttl CacheTTL, scopedTail []string) {
	last := lastLastingMessage(messages)
	if last < 0 {
		return
	}
	blocks := messages[last].Content
	for len(blocks) > 0 && len(scopedTail) > 0 {
		block := blocks[len(blocks)-1]
		if block.Type != "text" || block.Text != scopedTail[len(scopedTail)-1] {
			break
		}
		blocks = blocks[:len(blocks)-1]
		scopedTail = scopedTail[:len(scopedTail)-1]
	}
	for index := len(blocks) - 1; index >= 0; index-- {
		if blocks[index].CacheControl != nil {
			return
		}
		if anthropicCacheable(blocks[index].Type) {
			blocks[index].CacheControl = promptCacheControl(ttl)
			return
		}
	}
}

func anthropicCacheable(blockType string) bool {
	switch blockType {
	case "text", "tool_use", "server_tool_use", "image", "tool_result", "document", "tool_addition", "mcp_tool_result":
		return true
	}
	return false
}

func lastLastingMessage(messages []messageParam) int {
	for index := len(messages) - 1; index >= 0; index-- {
		if messages[index].ClearAt == "" {
			return index
		}
	}
	return -1
}

func isRequestScoped(part ai.RequestPart) bool {
	system, ok := part.(ai.SystemPromptPart)
	return ok && system.Scope == ai.SystemPromptScopeRequest
}

// requestScopedTail returns the request-scoped prompts ending the last request,
// where preparation moves them for a model that cannot clear them itself.
func requestScopedTail(messages []ai.ModelMessage) []string {
	for index := len(messages) - 1; index >= 0; index-- {
		request, ok := messages[index].(ai.ModelRequest)
		if !ok {
			continue
		}
		start := len(request.Parts)
		for start > 0 && isRequestScoped(request.Parts[start-1]) {
			start--
		}
		tail := make([]string, 0, len(request.Parts)-start)
		for _, part := range request.Parts[start:] {
			tail = append(tail, part.(ai.SystemPromptPart).Content)
		}
		return tail
	}
	return nil
}

// placeAnthropicSystemMessages moves each system entry past the user turns
// after it, since the API accepts one only right before an assistant turn or
// at the end, and gives an entry that follows no user turn one to follow.
func placeAnthropicSystemMessages(messages []messageParam) []messageParam {
	for index := len(messages) - 1; index >= 0; index-- {
		if messages[index].Role != "system" {
			continue
		}
		target := index + 1
		for target < len(messages) && messages[target].Role == "user" {
			target++
		}
		entry := messages[index]
		copy(messages[index:], messages[index+1:target])
		messages[target-1] = entry
	}
	for index := len(messages) - 1; index >= 0; index-- {
		if messages[index].Role != "system" || index > 0 && messages[index-1].Role != "assistant" {
			continue
		}
		messages = slices.Insert(messages, index, messageParam{
			Role: "user", Content: []contentBlock{{Type: "text", Text: "."}},
		})
	}
	return messages
}

func raiseAnthropicCacheTTLs(request *messagesRequest) {
	controls := make([]*anthropicPromptCacheControl, 0, 5)
	for index := range request.Tools {
		if request.Tools[index].CacheControl != nil {
			controls = append(controls, request.Tools[index].CacheControl)
		}
	}
	if blocks, ok := request.System.([]contentBlock); ok {
		for index := range blocks {
			if blocks[index].CacheControl != nil {
				controls = append(controls, blocks[index].CacheControl)
			}
		}
	}
	for messageIndex := range request.Messages {
		for blockIndex := range request.Messages[messageIndex].Content {
			if control := request.Messages[messageIndex].Content[blockIndex].CacheControl; control != nil {
				controls = append(controls, control)
			}
		}
	}
	if request.CacheControl != nil {
		controls = append(controls, request.CacheControl)
	}
	ttls := make([]string, len(controls))
	for index, control := range controls {
		ttls[index] = string(control.TTL)
	}
	for index, ttl := range promptcache.RaiseEarlierTTLs(ttls) {
		controls[index].TTL = ai.CachePointTTL(ttl)
	}
}

func limitAnthropicCachePoints(request *messagesRequest, automatic bool) error {
	maximum := 4
	if automatic {
		maximum--
	}
	used := 0
	if blocks, ok := request.System.([]contentBlock); ok {
		for _, block := range blocks {
			if block.CacheControl != nil {
				used++
			}
		}
	}
	for _, tool := range request.Tools {
		if tool.CacheControl != nil {
			used++
		}
	}
	if used > maximum {
		return fmt.Errorf(
			"anthropic: system and tool definitions use %d cache points, exceeding the maximum of %d",
			used, maximum,
		)
	}
	remaining := maximum - used
	for messageIndex := len(request.Messages) - 1; messageIndex >= 0; messageIndex-- {
		for blockIndex := len(request.Messages[messageIndex].Content) - 1; blockIndex >= 0; blockIndex-- {
			block := &request.Messages[messageIndex].Content[blockIndex]
			if block.CacheControl == nil {
				continue
			}
			if remaining > 0 {
				remaining--
			} else {
				block.CacheControl = nil
			}
		}
	}
	return nil
}

func anthropicNativeTools(
	modelName string, nativeTools []ai.NativeTool, codeExecutionVersion CodeExecutionToolVersion,
) ([]toolParam, error) {
	var tools []toolParam
	for _, nativeTool := range nativeTools {
		switch nativeTool := nativeTool.(type) {
		case ai.WebSearchTool:
			tools = append(tools, anthropicWebSearchTool(modelName, nativeTool))
		case *ai.WebSearchTool:
			tools = append(tools, anthropicWebSearchTool(modelName, *nativeTool))
		case ai.WebFetchTool:
			tools = append(tools, anthropicWebFetchTool(modelName, nativeTool))
		case *ai.WebFetchTool:
			tools = append(tools, anthropicWebFetchTool(modelName, *nativeTool))
		case ai.CodeExecutionTool, *ai.CodeExecutionTool:
			version, err := anthropicCodeExecutionToolVersion(modelName, codeExecutionVersion)
			if err != nil {
				return nil, err
			}
			tools = append(tools, toolParam{Type: "code_execution_" + string(version), Name: "code_execution"})
		case ai.MemoryTool, *ai.MemoryTool:
			tools = append(tools, toolParam{Type: "memory_20250818", Name: "memory"})
		case ai.AdvisorTool:
			if anthropicSupportsAdvisor(modelName) {
				tools = append(tools, anthropicAdvisorTool(nativeTool))
			} else if !nativeTool.Optional {
				return nil, fmt.Errorf("anthropic: model %q does not support the advisor tool", modelName)
			}
		case *ai.AdvisorTool:
			if anthropicSupportsAdvisor(modelName) {
				tools = append(tools, anthropicAdvisorTool(*nativeTool))
			} else if !nativeTool.Optional {
				return nil, fmt.Errorf("anthropic: model %q does not support the advisor tool", modelName)
			}
		case ai.MCPServerTool, *ai.MCPServerTool:
		default:
			if nativeTool.IsOptional() {
				continue
			}
			return nil, fmt.Errorf("anthropic: native tool %q is not implemented", nativeTool.Kind())
		}
	}
	return tools, nil
}

func anthropicAdvisorTool(advisor ai.AdvisorTool) toolParam {
	tool := toolParam{
		Type: "advisor_20260301", Name: "advisor", Model: advisor.Model,
	}
	if advisor.MaxUses != nil {
		maximum := *advisor.MaxUses
		tool.MaxUses = &maximum
	}
	if advisor.MaxTokens != nil {
		maximum := *advisor.MaxTokens
		tool.MaxTokens = &maximum
	}
	if advisor.Caching != "" {
		tool.Caching = &anthropicCacheControl{Type: "ephemeral", TTL: advisor.Caching}
	}
	return tool
}

func anthropicWebSearchTool(modelName string, webSearch ai.WebSearchTool) toolParam {
	version := "web_search_20250305"
	if anthropicSupportsDynamicFiltering(modelName) {
		version = "web_search_20260209"
	}
	tool := toolParam{
		Type: version, Name: "web_search",
		AllowedDomains: slices.Clone(webSearch.AllowedDomains),
		BlockedDomains: slices.Clone(webSearch.BlockedDomains),
	}
	if webSearch.MaxUses != 0 {
		maximum := webSearch.MaxUses
		tool.MaxUses = &maximum
	}
	if webSearch.UserLocation != nil {
		tool.UserLocation = &anthropicWebSearchLocation{
			Type: "approximate", City: webSearch.UserLocation.City, Country: webSearch.UserLocation.Country,
			Region: webSearch.UserLocation.Region, Timezone: webSearch.UserLocation.Timezone,
		}
	}
	return tool
}

func anthropicWebFetchTool(modelName string, webFetch ai.WebFetchTool) toolParam {
	version := "web_fetch_20250910"
	if anthropicSupportsDynamicFiltering(modelName) {
		version = "web_fetch_20260209"
	}
	tool := toolParam{
		Type: version, Name: "web_fetch",
		AllowedDomains: slices.Clone(webFetch.AllowedDomains),
		BlockedDomains: slices.Clone(webFetch.BlockedDomains), MaxContentTokens: webFetch.MaxContentTokens,
	}
	if webFetch.MaxUses != 0 {
		maximum := webFetch.MaxUses
		tool.MaxUses = &maximum
	}
	if webFetch.EnableCitations {
		tool.Citations = &anthropicCitations{Enabled: true}
	}
	return tool
}

func anthropicEffort(modelName string, level ai.ThinkingLevel) Effort {
	switch level {
	case ai.ThinkingLevelMinimal, ai.ThinkingLevelLow:
		return EffortLow
	case ai.ThinkingLevelMedium:
		return EffortMedium
	case ai.ThinkingLevelHigh:
		return EffortHigh
	case ai.ThinkingLevelXHigh:
		if anthropicSupportsXHighEffort(modelName) {
			return EffortXHigh
		}
		return EffortMax
	default:
		return ""
	}
}

func anthropicSupportsAdaptiveThinking(modelName string) bool {
	for _, prefix := range []string{
		"claude-fable-5", "claude-mythos-5", "claude-sonnet-4-6", "claude-sonnet-5",
		"claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-haiku-5-5",
	} {
		if strings.HasPrefix(modelName, prefix) {
			return true
		}
	}
	return false
}

func anthropicDisallowsBudgetThinking(modelName string) bool {
	for _, prefix := range []string{
		"claude-fable-5", "claude-mythos-5", "claude-opus-4-7", "claude-opus-4-8",
		"claude-opus-5", "claude-sonnet-5", "claude-haiku-5-5",
	} {
		if strings.HasPrefix(modelName, prefix) {
			return true
		}
	}
	return false
}

func anthropicDisallowsSamplingSettings(modelName string) bool {
	return anthropicDisallowsBudgetThinking(modelName)
}

func defaultAnthropicMaxTokens(modelName string) int {
	for _, prefix := range []string{
		"claude-fable-5", "claude-mythos-5", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8",
		"claude-opus-5", "claude-sonnet-4-6", "claude-sonnet-5", "claude-haiku-5-5",
	} {
		if strings.HasPrefix(modelName, prefix) {
			return 128000
		}
	}
	for _, prefix := range []string{"claude-haiku-4-5", "claude-opus-4-5", "claude-sonnet-4-5"} {
		if strings.HasPrefix(modelName, prefix) {
			return 64000
		}
	}
	if modelName == "claude-opus-4" || modelName == "claude-sonnet-4" {
		return 4096
	}
	for _, prefix := range []string{
		"claude-3", "claude-4-", "claude-opus-4-0", "claude-opus-4-1", "claude-opus-4-2",
		"claude-opus-4@", "claude-sonnet-4-0", "claude-sonnet-4-2", "claude-sonnet-4@",
	} {
		if strings.HasPrefix(modelName, prefix) {
			return 4096
		}
	}
	return defaultMaxTokens
}

func anthropicSupportsForcedToolChoice(modelName string) bool {
	for _, prefix := range []string{"claude-fable-5-1", "claude-mythos-5-1", "claude-opus-5-5"} {
		if strings.HasPrefix(modelName, prefix) {
			return false
		}
	}
	return true
}

func anthropicBindsThinkingBlocks(modelName string) bool {
	modelName = strings.ToLower(modelName)
	if _, bedrockName, found := strings.Cut(modelName, "anthropic."); found {
		modelName = bedrockName
	}
	for _, prefix := range []string{"claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"} {
		if strings.HasPrefix(modelName, prefix) {
			return true
		}
	}
	return false
}

func supportsRequestScopedSystemPrompts(modelName string) bool {
	modelName = strings.ToLower(modelName)
	for _, prefix := range []string{
		"claude-fable-5", "claude-mythos-5", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5-5", "claude-haiku-5-5",
	} {
		if strings.Contains(modelName, prefix) {
			return true
		}
	}
	return false
}

func anthropicSupportsXHighEffort(modelName string) bool {
	for _, prefix := range []string{
		"claude-fable-5", "claude-mythos-5", "claude-opus-4-7", "claude-opus-4-8",
		"claude-opus-5", "claude-sonnet-5", "claude-haiku-5-5",
	} {
		if strings.HasPrefix(modelName, prefix) {
			return true
		}
	}
	return false
}

func anthropicSupportsAdvisor(modelName string) bool {
	for _, prefix := range []string{
		"claude-fable-5", "claude-mythos-5", "claude-opus-4-6", "claude-opus-4-7",
		"claude-opus-4-8", "claude-opus-5", "claude-sonnet-4-6", "claude-sonnet-5", "claude-haiku-4-5",
		"claude-haiku-5-5",
	} {
		if strings.HasPrefix(modelName, prefix) {
			return true
		}
	}
	return false
}

func anthropicCodeExecutionToolVersion(
	modelName string, configured CodeExecutionToolVersion,
) (CodeExecutionToolVersion, error) {
	if configured == "" || configured == CodeExecutionToolVersionAuto {
		if anthropicSupportsLatestCodeExecution(modelName) {
			return CodeExecutionToolVersion20260120, nil
		}
		return CodeExecutionToolVersion20250825, nil
	}
	if configured == CodeExecutionToolVersion20260120 && !anthropicSupportsLatestCodeExecution(modelName) {
		return "", fmt.Errorf(
			"anthropic: code execution tool version %q is not supported by model %q", configured, modelName,
		)
	}
	return configured, nil
}

func anthropicSupportsLatestCodeExecution(modelName string) bool {
	for _, prefix := range []string{
		"claude-fable-5", "claude-mythos-5", "claude-haiku-5-5", "claude-sonnet-4-5",
		"claude-sonnet-4-6", "claude-sonnet-5", "claude-opus-4-5", "claude-opus-4-6",
		"claude-opus-4-7", "claude-opus-4-8", "claude-opus-5",
	} {
		if strings.HasPrefix(modelName, prefix) {
			return true
		}
	}
	return false
}

func anthropicSupportsDynamicFiltering(modelName string) bool {
	for _, prefix := range []string{
		"claude-fable-5", "claude-mythos-5", "claude-mythos-preview", "claude-sonnet-4-6",
		"claude-sonnet-5", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5",
		"claude-haiku-5-5",
	} {
		if strings.HasPrefix(modelName, prefix) {
			return true
		}
	}
	return false
}

func (m *Model) buildPayload(
	ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams,
) (*messagesRequest, error) {
	params, err := ai.ResolveNativeToolPreferences(m, params)
	if err != nil {
		return nil, err
	}
	translated, err := m.translateCache(params.Settings)
	if err != nil {
		return nil, err
	}
	settings, cache, err := extractCacheSettings(translated)
	if err != nil {
		return nil, err
	}
	if cache.Automatic != "" && !m.supportsAutomaticCaching() {
		cache.Messages, cache.Automatic = cache.Automatic, ""
	}
	settings, providerSettings, err := extractProviderSettings(settings)
	if err != nil {
		return nil, err
	}
	params.Settings = settings
	memoryEnabled := hasAnthropicMemoryTool(params.NativeTools)
	if memoryEnabled {
		memoryDefined := false
		for _, tool := range params.Tools {
			if tool.Name == "memory" {
				memoryDefined = true
				break
			}
		}
		if !memoryDefined {
			return nil, fmt.Errorf("anthropic: native memory requires a function tool named %q", "memory")
		}
	}
	nativeTools, err := anthropicNativeTools(
		m.name, params.NativeTools, providerSettings.CodeExecutionToolVersion,
	)
	if err != nil {
		return nil, err
	}
	advisorActive := slices.ContainsFunc(nativeTools, func(tool toolParam) bool { return tool.Type == "advisor_20260301" })
	var mcpServers []anthropicMCPServer
	var nativeBetas []string
	for _, nativeTool := range params.NativeTools {
		var mcpServer *ai.MCPServerTool
		var codeFiles []ai.UploadedFile
		switch tool := nativeTool.(type) {
		case ai.MCPServerTool:
			copy := tool
			mcpServer = &copy
		case *ai.MCPServerTool:
			mcpServer = tool
		case ai.AdvisorTool, *ai.AdvisorTool:
			if advisorActive && !slices.Contains(nativeBetas, "advisor-tool-2026-03-01") {
				nativeBetas = append(nativeBetas, "advisor-tool-2026-03-01")
			}
		case ai.MemoryTool, *ai.MemoryTool:
			if !slices.Contains(nativeBetas, "context-management-2025-06-27") {
				nativeBetas = append(nativeBetas, "context-management-2025-06-27")
			}
		case ai.WebFetchTool, *ai.WebFetchTool:
			if !anthropicSupportsDynamicFiltering(m.name) &&
				!slices.Contains(nativeBetas, "web-fetch-2025-09-10") {
				nativeBetas = append(nativeBetas, "web-fetch-2025-09-10")
			}
		case ai.CodeExecutionTool:
			codeFiles = tool.Files
		case *ai.CodeExecutionTool:
			codeFiles = tool.Files
		}
		if mcpServer != nil {
			if strings.HasPrefix(mcpServer.URL, "x-openai-connector:") {
				if mcpServer.Optional {
					continue
				}
				return nil, fmt.Errorf("anthropic: OpenAI MCP connector URLs are not supported")
			}
			server := anthropicMCPServer{
				Type: "url", Name: mcpServer.ID, URL: mcpServer.URL,
				AuthorizationToken: mcpServer.AuthorizationToken,
			}
			if mcpServer.AllowedTools != nil {
				server.ToolConfiguration = &anthropicMCPToolConfiguration{
					Enabled:      len(mcpServer.AllowedTools) > 0,
					AllowedTools: slices.Clone(mcpServer.AllowedTools),
				}
			}
			mcpServers = append(mcpServers, server)
			if !slices.Contains(nativeBetas, "mcp-client-2025-04-04") {
				nativeBetas = append(nativeBetas, "mcp-client-2025-04-04")
			}
		}
		for _, file := range codeFiles {
			if file.ProviderName == "anthropic" && !slices.Contains(nativeBetas, "files-api-2025-04-14") {
				nativeBetas = append(nativeBetas, "files-api-2025-04-14")
			}
		}
	}
	thinking, effort, err := anthropicThinking(m.name, params.Settings.Thinking, providerSettings.Effort)
	if err != nil {
		return nil, err
	}
	if anthropicHistoryNeedsStaleThinkingDrop(m.name, msgs) {
		if rawThinking, ok := params.Settings.ExtraBody["thinking"].(map[string]any); ok {
			params.Settings.ExtraBody = maps.Clone(params.Settings.ExtraBody)
			rawThinking = maps.Clone(rawThinking)
			if _, configured := rawThinking["block_binding"]; !configured {
				rawThinking["block_binding"] = map[string]any{"prefix_mismatch_behavior": "drop_block"}
				params.Settings.ExtraBody["thinking"] = rawThinking
			}
		} else {
			if thinking == nil {
				thinking = &thinkingParam{}
			}
			thinking.BlockBinding = &thinkingBlockBinding{PrefixMismatchBehavior: "drop_block"}
		}
	}
	serviceTier, err := anthropicServiceTier(params.Settings.ServiceTier)
	if err != nil {
		return nil, err
	}
	container := anthropicContainerFromHistory(msgs)
	containerFromHistory := container != nil
	if container != nil && !hasAnthropicCodeExecutionTool(params.NativeTools) {
		container = nil
		containerFromHistory = false
	}
	if len(msgs) > 0 {
		if response, ok := msgs[len(msgs)-1].(ai.ModelResponse); ok && response.State == ai.ModelResponseStateSuspended {
			containerFromHistory = false
		}
	}
	if providerSettings.ContainerSet {
		container = providerSettings.Container
		containerFromHistory = false
	}
	temperature := params.Settings.Temperature
	topP := params.Settings.TopP
	if anthropicDisallowsSamplingSettings(m.name) {
		temperature = nil
		topP = nil
	}
	rawThinkingBinding := false
	if rawThinking, ok := params.Settings.ExtraBody["thinking"].(map[string]any); ok {
		_, rawThinkingBinding = rawThinking["block_binding"]
	}
	if (thinking != nil && thinking.BlockBinding != nil || rawThinkingBinding) &&
		!slices.Contains(nativeBetas, "thinking-binding-controls-2026-08-01") {
		nativeBetas = append(nativeBetas, "thinking-binding-controls-2026-08-01")
	}
	requestScoped := m.clearsRequestScopedPrompts()
	var scopedTail []string
	if requestScoped {
		if slices.ContainsFunc(msgs, func(message ai.ModelMessage) bool {
			request, ok := message.(ai.ModelRequest)
			return ok && slices.ContainsFunc(request.Parts, isRequestScoped)
		}) {
			nativeBetas = append(nativeBetas, "mid-conversation-system-clear-at-2026-08-21")
		}
	} else {
		scopedTail = requestScopedTail(msgs)
	}
	req := &messagesRequest{
		Model:                m.name,
		Tools:                nativeTools,
		MCPServers:           mcpServers,
		Betas:                nativeBetas,
		MaxTokens:            params.Settings.MaxTokens,
		Temperature:          temperature,
		TopP:                 topP,
		Stop:                 params.Settings.StopSequences,
		Thinking:             thinking,
		ServiceTier:          serviceTier,
		Container:            container,
		ContainerFromHistory: containerFromHistory,
		CacheControl:         promptCacheControl(cache.Automatic),
		ExtraBody:            params.Settings.ExtraBody,
	}
	if cache.Diagnostics && m.legacyBedrockClient == nil {
		req.Diagnostics = &struct {
			PreviousMessageID *string `json:"previous_message_id"`
		}{}
		for index := len(msgs) - 1; index >= 0; index-- {
			response, ok := msgs[index].(ai.ModelResponse)
			if !ok || response.ProviderName != "anthropic" {
				continue
			}
			if strings.HasPrefix(response.ProviderResponseID, "msg_") {
				id := response.ProviderResponseID
				req.Diagnostics.PreviousMessageID = &id
			}
			break
		}
	}
	if params.Instructions != "" {
		req.System = params.Instructions
	}
	if cache.Instructions != "" && params.Instructions != "" {
		blocks := make([]contentBlock, 0, max(1, len(params.InstructionParts)))
		cacheIndex := -1
		if len(params.InstructionParts) == 0 {
			blocks = append(blocks, contentBlock{Type: "text", Text: params.Instructions})
			cacheIndex = 0
		} else {
			for _, instruction := range params.InstructionParts {
				blocks = append(blocks, contentBlock{Type: "text", Text: instruction.Content})
				if !instruction.Dynamic {
					cacheIndex = len(blocks) - 1
				}
			}
		}
		if cacheIndex >= 0 {
			blocks[cacheIndex].CacheControl = promptCacheControl(cache.Instructions)
		}
		req.System = blocks
	}
	if req.MaxTokens == 0 {
		req.MaxTokens = defaultAnthropicMaxTokens(m.name)
		if thinking != nil && thinking.Type == "enabled" && thinking.BudgetTokens+4096 > req.MaxTokens {
			req.MaxTokens = thinking.BudgetTokens + 4096
		}
	}
	trimmedMessages, compaction := trimAnthropicCompactionMessages(msgs)
	var searchTool *ai.ToolDefinition
	for _, tool := range params.Tools {
		if tool.ToolKind == ai.ToolPartKindToolSearch {
			definition := tool
			searchTool = &definition
			break
		}
	}
	if m.legacyBedrockClient != nil && searchTool != nil &&
		searchTool.ToolSearchStrategy == ai.ToolSearchStrategyBM25 {
		return nil, fmt.Errorf(
			"anthropic: tool search strategy %q is not supported by legacy Bedrock; use %q",
			ai.ToolSearchStrategyBM25, ai.ToolSearchStrategyRegex,
		)
	}
	nativeDeferred := m.deferredToolSupport && len(params.DeferredTools) > 0 && hasStableAnthropicTool(params)
	serverToolSearch := searchTool != nil && nativeDeferred &&
		(searchTool.ToolSearchStrategy == ai.ToolSearchStrategyAuto ||
			searchTool.ToolSearchStrategy == ai.ToolSearchStrategyBM25 ||
			searchTool.ToolSearchStrategy == ai.ToolSearchStrategyRegex)
	if searchTool != nil &&
		(searchTool.ToolSearchStrategy == ai.ToolSearchStrategyBM25 ||
			searchTool.ToolSearchStrategy == ai.ToolSearchStrategyRegex) && !serverToolSearch {
		return nil, fmt.Errorf(
			"anthropic: tool search strategy %q requires deferred-tool support",
			searchTool.ToolSearchStrategy,
		)
	}
	deferredNames := make(map[string]struct{}, len(params.DeferredTools))
	if nativeDeferred {
		for _, tool := range params.DeferredTools {
			deferredNames[tool.Name] = struct{}{}
		}
		req.ToolAdditions = hasAnthropicToolAdditions(trimmedMessages, deferredNames)
	}
	req.Compaction = compaction || hasCompactionEdit(params.Settings.ExtraBody)
	if _, overridden := params.Settings.ExtraBody["context_management"]; compaction && !overridden {
		req.ContextManagement = map[string]any{
			"edits": []any{map[string]any{"type": "compact_20260112"}},
		}
	}
	for _, msg := range trimmedMessages {
		converted, err := convertMessage(ctx, msg, deferredNames, advisorActive, m.carryThinking, requestScoped)
		if err != nil {
			return nil, err
		}
		req.Messages = append(req.Messages, converted...)
	}
	req.Messages = placeAnthropicSystemMessages(req.Messages)
	if uploads := anthropicContainerUploads(params.NativeTools); len(uploads) > 0 {
		for index := range req.Messages {
			if req.Messages[index].Role != "user" || anthropicToolResultsOnly(req.Messages[index].Content) {
				continue
			}
			for _, fileID := range uploads {
				req.Messages[index].Content = append(req.Messages[index].Content, contentBlock{
					Type: "container_upload", FileID: fileID,
				})
			}
		}
	}
	if err := resolveAnthropicCachePoints(&req.Messages); err != nil {
		return nil, err
	}
	if cache.Messages != "" {
		if m.legacyBedrockClient != nil {
			roles := make([]string, len(req.Messages))
			counts := make([]int, len(req.Messages))
			for index, message := range req.Messages {
				roles[index], counts[index] = message.Role, len(message.Content)
			}
			if previous := promptcache.PreviousTail(roles, counts); previous >= 0 {
				addAnthropicMessageCachePoint(req.Messages[:previous+1], cache.Messages, nil)
			}
		}
		addAnthropicMessageCachePoint(req.Messages, cache.Messages, scopedTail)
	} else if cache.Automatic != "" && len(scopedTail) > 0 {
		addAnthropicMessageCachePoint(req.Messages, cache.Automatic, scopedTail)
	}
	if last := lastLastingMessage(req.Messages); last >= 0 {
		blocks := req.Messages[last].Content
		for index := len(blocks) - 1; index >= 0; index-- {
			if blocks[index].CacheControl != nil {
				req.CacheControl = nil
				break
			}
			if anthropicCacheable(blocks[index].Type) {
				break
			}
		}
	}
	lastFunctionToolIndex := -1
	for _, tool := range params.Tools {
		if memoryEnabled && tool.Name == "memory" {
			continue
		}
		if serverToolSearch && tool.ToolKind == ai.ToolPartKindToolSearch {
			continue
		}
		if nativeDeferred && tool.DeferLoading {
			continue
		}
		converted, err := prepareAnthropicTool(tool, m.strictToolSupport, m.schemaWarning)
		if err != nil {
			return nil, err
		}
		req.Tools = append(req.Tools, converted)
		lastFunctionToolIndex = len(req.Tools) - 1
	}
	if nativeDeferred {
		for _, tool := range params.DeferredTools {
			converted, err := prepareAnthropicTool(tool, m.strictToolSupport, m.schemaWarning)
			if err != nil {
				return nil, err
			}
			converted.DeferLoading = true
			req.Tools = append(req.Tools, converted)
			lastFunctionToolIndex = len(req.Tools) - 1
		}
	}
	if serverToolSearch {
		if searchTool.ToolSearchStrategy == ai.ToolSearchStrategyRegex || m.legacyBedrockClient != nil {
			req.Tools = append(req.Tools, toolParam{
				Type: "tool_search_tool_regex_20251119", Name: "tool_search_tool_regex",
			})
		} else {
			req.Tools = append(req.Tools, toolParam{
				Type: "tool_search_tool_bm25_20251119", Name: "tool_search_tool_bm25",
			})
		}
	}
	if effort != "" {
		req.OutputConfig = &anthropicOutputConfig{Effort: effort}
	}
	if params.OutputTool != nil {
		converted, err := prepareAnthropicTool(*params.OutputTool, m.strictToolSupport, m.schemaWarning)
		if err != nil {
			return nil, err
		}
		req.Tools = append(req.Tools, converted)
		lastFunctionToolIndex = len(req.Tools) - 1
		if !params.AllowText {
			choice := "any"
			if (thinking != nil && thinking.Type == "enabled") || !anthropicSupportsForcedToolChoice(m.name) {
				choice = "auto"
			}
			req.ToolChoice = &toolChoiceParam{Type: choice}
		}
	}
	if cache.ToolDefinitions != "" && lastFunctionToolIndex >= 0 {
		req.Tools[lastFunctionToolIndex].CacheControl = promptCacheControl(cache.ToolDefinitions)
	}
	if len(req.Tools) > 0 && params.Settings.ParallelToolCalls != nil {
		if req.ToolChoice == nil {
			req.ToolChoice = &toolChoiceParam{Type: "auto"}
		}
		disable := !*params.Settings.ParallelToolCalls
		req.ToolChoice.DisableParallelToolUse = &disable
	}
	if params.OutputSchema != nil && params.OutputMode == ai.OutputModeNative {
		supported := supportsAnthropicNativeOutput(m.name)
		if m.legacyBedrockClient != nil {
			supported = legacyBedrockSupportsStructuredOutput(m.name)
		}
		if !supported {
			return nil, fmt.Errorf("anthropic: model %q does not support native JSON output", m.name)
		}
		if req.OutputConfig == nil {
			req.OutputConfig = &anthropicOutputConfig{}
		}
		req.OutputConfig.Format = &anthropicOutputFormat{Type: "json_schema", Schema: params.OutputSchema}
	}
	if err := limitAnthropicCachePoints(req, req.CacheControl != nil); err != nil { // pragma: no cover - fixed boundaries cannot exceed four slots.
		return nil, err
	}
	raiseAnthropicCacheTTLs(req)
	return req, nil
}

func supportsAnthropicNativeOutput(modelName string) bool {
	for _, prefix := range []string{
		"claude-fable-5", "claude-mythos-5", "claude-haiku-4-5", "claude-haiku-5-5", "claude-sonnet-4-5",
		"claude-sonnet-4-6", "claude-opus-4-1", "claude-opus-4-5", "claude-opus-4-6",
		"claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
	} {
		if strings.HasPrefix(modelName, prefix) {
			return true
		}
	}
	return false
}

func hasAnthropicMemoryTool(nativeTools []ai.NativeTool) bool {
	for _, nativeTool := range nativeTools {
		switch nativeTool.(type) {
		case ai.MemoryTool, *ai.MemoryTool:
			return true
		}
	}
	return false
}

func hasAnthropicCodeExecutionTool(nativeTools []ai.NativeTool) bool {
	for _, nativeTool := range nativeTools {
		switch nativeTool.(type) {
		case ai.CodeExecutionTool, *ai.CodeExecutionTool:
			return true
		}
	}
	return false
}

func anthropicContainerUploads(nativeTools []ai.NativeTool) []string {
	var uploads []string
	for _, nativeTool := range nativeTools {
		var files []ai.UploadedFile
		switch tool := nativeTool.(type) {
		case ai.CodeExecutionTool:
			files = tool.Files
		case *ai.CodeExecutionTool:
			if tool != nil {
				files = tool.Files
			}
		}
		for _, file := range files {
			if file.ProviderName == "anthropic" {
				uploads = append(uploads, file.FileID)
			}
		}
	}
	return uploads
}

func anthropicToolResultsOnly(content []contentBlock) bool {
	onlyToolResults := len(content) > 0
	for _, block := range content {
		if block.Type != "tool_result" {
			return false
		}
	}
	return onlyToolResults
}

func messagesHaveContainerUploads(messages []messageParam) bool {
	for _, message := range messages {
		for _, block := range message.Content {
			if block.Type == "container_upload" {
				return true
			}
		}
	}
	return false
}

func anthropicExpiredContainer(statusCode int, data []byte) bool {
	return statusCode == http.StatusInternalServerError ||
		statusCode == http.StatusNotFound && strings.HasPrefix(anthropicErrorMessage(data), "Container not found")
}

// anthropicErrorMessage reads the direct API's nested error message or Bedrock's top-level one.
func anthropicErrorMessage(data []byte) string {
	var body struct {
		Message string `json:"message"`
		Error   *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(data, &body) != nil {
		return ""
	}
	if body.Error != nil {
		return body.Error.Message
	}
	return body.Message
}

func anthropicStaleThinkingError(modelName string, statusCode int, message string, payload *messagesRequest) bool {
	if !anthropicBindsThinkingBlocks(modelName) || statusCode != http.StatusBadRequest ||
		payload.Thinking != nil && payload.Thinking.BlockBinding != nil {
		return false
	}
	if thinking, ok := payload.ExtraBody["thinking"].(map[string]any); ok {
		if _, configured := thinking["block_binding"]; configured {
			return false
		}
	}
	return strings.Contains(message, "The block is bound to a different conversation")
}

func enableAnthropicStaleThinkingDrop(payload *messagesRequest) {
	if thinking, ok := payload.ExtraBody["thinking"].(map[string]any); ok {
		payload.ExtraBody = maps.Clone(payload.ExtraBody)
		thinking = maps.Clone(thinking)
		thinking["block_binding"] = map[string]any{"prefix_mismatch_behavior": "drop_block"}
		payload.ExtraBody["thinking"] = thinking
	} else {
		if payload.Thinking == nil {
			payload.Thinking = &thinkingParam{}
		}
		payload.Thinking.BlockBinding = &thinkingBlockBinding{PrefixMismatchBehavior: "drop_block"}
	}
	if !slices.Contains(payload.Betas, "thinking-binding-controls-2026-08-01") {
		payload.Betas = append(payload.Betas, "thinking-binding-controls-2026-08-01")
	}
}

func anthropicHistoryNeedsStaleThinkingDrop(modelName string, messages []ai.ModelMessage) bool {
	if !anthropicBindsThinkingBlocks(modelName) {
		return false
	}
	for index := len(messages) - 1; index >= 0; index-- {
		response, ok := messages[index].(ai.ModelResponse)
		if !ok {
			continue
		}
		compactionBoundary := false
		for _, part := range response.Parts {
			if _, compacted := part.(ai.CompactionPart); compacted {
				compactionBoundary = true
				break
			}
		}
		if compactionBoundary {
			return false
		}
		switch transformations := response.ProviderDetails["input_transformations"].(type) {
		case []map[string]any:
			for _, transformation := range transformations {
				if anthropicThinkingWasDropped(transformation) {
					return true
				}
			}
		case []any:
			for _, value := range transformations {
				if transformation, ok := value.(map[string]any); ok && anthropicThinkingWasDropped(transformation) {
					return true
				}
			}
		}
	}
	return false
}

func anthropicThinkingWasDropped(transformation map[string]any) bool {
	return transformation["type"] == "thinking_dropped" &&
		transformation["reason"] == "prefix_binding_mismatch"
}

func anthropicContainerFromHistory(messages []ai.ModelMessage) any {
	for index := len(messages) - 1; index >= 0; index-- {
		response, ok := messages[index].(ai.ModelResponse)
		if !ok || response.ProviderName != "anthropic" {
			continue
		}
		if containerID, _ := response.ProviderDetails["container_id"].(string); containerID != "" {
			return containerID
		}
	}
	return nil
}

func anthropicServiceTier(tier ai.ServiceTier) (string, error) {
	switch tier {
	case "", ai.ServiceTierFlex, ai.ServiceTierPriority:
		return "", nil
	case ai.ServiceTierAuto:
		return "auto", nil
	case ai.ServiceTierDefault:
		return "standard_only", nil
	default:
		return "", fmt.Errorf("anthropic: invalid service tier %q", tier)
	}
}

func anthropicThinking(
	modelName string, settings *ai.ThinkingSettings, configuredEffort Effort,
) (*thinkingParam, Effort, error) {
	if settings == nil || settings.Level == ai.ThinkingLevelDisabled ||
		settings.Level == "" && settings.TokenBudget == nil {
		if settings != nil && settings.Level == ai.ThinkingLevelDisabled &&
			(configuredEffort == EffortXHigh || configuredEffort == EffortMax) &&
			(strings.HasPrefix(modelName, "claude-haiku-5-5") ||
				strings.HasPrefix(modelName, "claude-opus-5") && !strings.HasPrefix(modelName, "claude-opus-5-5")) {
			return nil, "", fmt.Errorf(
				"anthropic: model %q does not support effort %q while thinking is disabled", modelName, configuredEffort,
			)
		}
		return nil, configuredEffort, nil
	}
	if settings.TokenBudget != nil && anthropicDisallowsBudgetThinking(modelName) {
		return nil, "", fmt.Errorf("anthropic: model %q does not support budget-based thinking", modelName)
	}
	if settings.TokenBudget == nil && anthropicSupportsAdaptiveThinking(modelName) {
		effort := configuredEffort
		if effort == "" {
			effort = anthropicEffort(modelName, settings.Level)
		}
		return &thinkingParam{Type: "adaptive"}, effort, nil
	}
	budget := 0
	if settings.TokenBudget != nil {
		budget = *settings.TokenBudget
	} else {
		switch settings.Level {
		case "", ai.ThinkingLevelEnabled, ai.ThinkingLevelMedium:
			budget = 10000
		case ai.ThinkingLevelMinimal:
			budget = 1024
		case ai.ThinkingLevelLow:
			budget = 2048
		case ai.ThinkingLevelHigh:
			budget = 16384
		case ai.ThinkingLevelXHigh:
			budget = 32768
		default:
			return nil, "", fmt.Errorf("anthropic: invalid thinking level %q", settings.Level)
		}
	}
	if budget <= 0 {
		return nil, "", fmt.Errorf("anthropic: thinking token budget must be positive, got %d", budget)
	}
	return &thinkingParam{Type: "enabled", BudgetTokens: budget}, configuredEffort, nil
}

func hasAnthropicToolAdditions(msgs []ai.ModelMessage, deferredNames map[string]struct{}) bool {
	for _, message := range msgs {
		request, ok := message.(ai.ModelRequest)
		if !ok {
			continue
		}
		for _, part := range request.Parts {
			delta, ok := part.(ai.ToolAvailabilityDeltaPart)
			if !ok {
				continue
			}
			for _, name := range delta.ToolsAdded {
				if _, deferred := deferredNames[name]; deferred {
					return true
				}
			}
		}
	}
	return false
}

func hasStableAnthropicTool(params ai.ModelRequestParams) bool {
	if params.OutputTool != nil {
		return true
	}
	for _, tool := range params.Tools {
		if !tool.DeferLoading {
			return true
		}
	}
	return false
}

func convertMessage(
	ctx context.Context, msg ai.ModelMessage, deferredNames map[string]struct{},
	advisorActive, carryThinking, requestScoped bool,
) ([]messageParam, error) {
	switch m := msg.(type) {
	case ai.ModelRequest:
		return convertRequest(ctx, m, deferredNames, requestScoped)
	case ai.ModelResponse:
		return convertResponse(m, deferredNames, advisorActive, carryThinking)
	default:
		return nil, fmt.Errorf("anthropic: unknown message type %T", msg)
	}
}

func convertRequest(
	ctx context.Context, m ai.ModelRequest, deferredNames map[string]struct{}, requestScoped bool,
) ([]messageParam, error) {
	var blocks, scoped []contentBlock
	searchReveals := make(map[string]struct{})
	for _, part := range m.Parts {
		switch p := part.(type) {
		case ai.SpeechPart:
			return nil, ai.ErrUnpreparedSpeech
		case ai.SystemPromptPart:
			if requestScoped && p.Scope == ai.SystemPromptScopeRequest {
				scoped = append(scoped, contentBlock{Type: "text", Text: p.Content})
				continue
			}
			// Anthropic takes the system prompt at the top level; a
			// system part in history becomes user-visible context.
			blocks = append(blocks, contentBlock{Type: "text", Text: p.Content})
		case ai.UserPromptPart:
			converted, err := convertUserPrompt(ctx, p)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, converted...)
		case ai.ToolReturnPart:
			if p.ToolKind == ai.ToolPartKindToolSearch && len(deferredNames) > 0 {
				names, message, err := anthropicToolSearchResult(p.Content, deferredNames)
				if err != nil {
					return nil, err
				}
				var content any
				if len(names) == 0 {
					content = []contentBlock{{Type: "text", Text: message}}
				} else {
					references := make([]toolReferenceContent, len(names))
					for index, name := range names {
						references[index] = toolReferenceContent{Type: "tool_reference", ToolName: name}
						searchReveals[name] = struct{}{}
					}
					content = references
				}
				blocks = append(blocks, contentBlock{
					Type: "tool_result", ToolUseID: p.ToolCallID, Content: content,
				})
				continue
			}
			content, err := contentString(p.Content)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, contentBlock{
				Type: "tool_result", ToolUseID: p.ToolCallID, Content: content,
				IsError: p.Outcome == ai.ToolReturnOutcomeFailed || p.Outcome == ai.ToolReturnOutcomeInterrupted,
			})
		case ai.ToolAvailabilityDeltaPart:
			for _, name := range p.ToolsAdded {
				if _, deferred := deferredNames[name]; !deferred {
					continue
				}
				if _, fromSearch := searchReveals[name]; fromSearch {
					continue
				}
				blocks = append(blocks, contentBlock{
					Type: "tool_addition", Tool: &toolReferenceParam{Type: "tool_reference", Name: name},
				})
			}
		case ai.RetryPromptPart:
			content := p.ModelResponse()
			if p.ToolCallID != "" {
				blocks = append(blocks, contentBlock{Type: "tool_result", ToolUseID: p.ToolCallID, Content: content, IsError: true})
			} else {
				blocks = append(blocks, contentBlock{Type: "text", Text: content})
			}
		default:
			return nil, fmt.Errorf("anthropic: unknown request part type %T", part)
		}
	}
	if len(scoped) == 0 {
		return []messageParam{{Role: "user", Content: blocks}}, nil
	}
	system := messageParam{Role: "system", Content: scoped, ClearAt: "next_user_message"}
	if len(blocks) == 0 {
		return []messageParam{system}, nil
	}
	return []messageParam{{Role: "user", Content: blocks}, system}, nil
}

func anthropicToolSearchResult(
	content any, deferredNames map[string]struct{},
) ([]string, string, error) {
	encoded, err := json.Marshal(content)
	if err != nil {
		return nil, "", fmt.Errorf("anthropic: marshal tool search return: %w", err)
	}
	var result ai.ToolSearchResult
	if err := json.Unmarshal(encoded, &result); err != nil {
		return nil, "", fmt.Errorf("anthropic: parse tool search return: %w", err)
	}
	names := make([]string, 0, len(result.DiscoveredTools))
	seen := make(map[string]struct{}, len(result.DiscoveredTools))
	for _, match := range result.DiscoveredTools {
		if _, deferred := deferredNames[match.Name]; !deferred {
			continue
		}
		if _, duplicate := seen[match.Name]; duplicate {
			continue
		}
		seen[match.Name] = struct{}{}
		names = append(names, match.Name)
	}
	message := result.Message
	if message == "" {
		message = "No matching tools found. The tools you need may not be available."
	}
	return names, message, nil
}

func trimAnthropicCompactionMessages(messages []ai.ModelMessage) ([]ai.ModelMessage, bool) {
	for messageIndex := len(messages) - 1; messageIndex >= 0; messageIndex-- {
		response, ok := messages[messageIndex].(ai.ModelResponse)
		if !ok {
			continue
		}
		for partIndex := len(response.Parts) - 1; partIndex >= 0; partIndex-- {
			part, ok := response.Parts[partIndex].(ai.CompactionPart)
			if !ok || part.ProviderName != "anthropic" || !part.HasContent() {
				continue
			}
			response.Parts = slices.Clone(response.Parts[partIndex:])
			trimmed := make([]ai.ModelMessage, 1, len(messages)-messageIndex)
			trimmed[0] = response
			return append(trimmed, messages[messageIndex+1:]...), true
		}
	}
	return messages, false
}

func convertResponse(
	m ai.ModelResponse, deferredNames map[string]struct{}, advisorActive, carryThinking bool,
) ([]messageParam, error) {
	var blocks, carried []contentBlock
	for _, part := range m.Parts {
		switch p := part.(type) {
		case ai.SpeechPart:
			return nil, ai.ErrUnpreparedSpeech
		case ai.TextPart:
			blocks = append(blocks, contentBlock{Type: "text", Text: p.Content})
		case ai.CompactionPart:
			if p.ProviderName == "anthropic" && p.HasContent() {
				encryptedContent, _ := p.ProviderDetails["encrypted_content"].(string)
				blocks = append(blocks, contentBlock{
					Type: "compaction", Content: p.Content, EncryptedContent: encryptedContent,
				})
			}
		case ai.ThinkingPart:
			switch {
			case p.Signature != "" && (p.ProviderName == "" || p.ProviderName == "anthropic"):
				blocks = append(blocks, contentBlock{Type: "thinking", Thinking: &p.Content, Signature: p.Signature})
			case p.Content == "":
			case carryThinking:
				// Claude copies thinking tags it sees in assistant turns into the answers users read.
				carried = append(carried, contentBlock{
					Type: "text", Text: thinking.Carried(p.Content, m.ModelName, m.ProviderName),
				})
			default:
				blocks = append(blocks, contentBlock{Type: "text", Text: "<thinking>\n" + p.Content + "\n</thinking>"})
			}
		case ai.ToolCallPart:
			blocks = append(blocks, contentBlock{Type: "tool_use", ID: p.ToolCallID, Name: p.ToolName, Input: p.Args})
		case ai.NativeToolCallPart:
			if p.ProviderName != "anthropic" {
				continue
			}
			if p.ToolKind == ai.ToolPartKindAdvisor {
				if !advisorActive {
					continue
				}
				input := slices.Clone(p.Args)
				if len(input) == 0 {
					input = json.RawMessage(`{}`)
				}
				block := contentBlock{Type: "server_tool_use", ID: p.ToolCallID, Name: "advisor", Input: input}
				if caller, ok := p.ProviderDetails["anthropic_caller"].(map[string]any); ok {
					block.Caller = caller
				}
				blocks = append(blocks, block)
				continue
			}
			if p.ToolKind == ai.ToolPartKindMCPServer {
				serverName, ok := strings.CutPrefix(p.ToolName, "mcp_server:")
				if !ok || serverName == "" {
					return nil, fmt.Errorf("anthropic: invalid MCP server tool name %q", p.ToolName)
				}
				var arguments struct {
					Action   string          `json:"action"`
					ToolName string          `json:"tool_name"`
					ToolArgs json.RawMessage `json:"tool_args"`
				}
				if err := json.Unmarshal(p.Args, &arguments); err != nil {
					return nil, fmt.Errorf("anthropic: parse MCP server arguments: %w", err)
				}
				if arguments.Action != "call_tool" {
					return nil, fmt.Errorf("anthropic: invalid MCP server action %q", arguments.Action)
				}
				if arguments.ToolName == "" {
					return nil, fmt.Errorf("anthropic: MCP call tool name must not be empty")
				}
				if len(arguments.ToolArgs) == 0 {
					arguments.ToolArgs = json.RawMessage(`{}`)
				}
				blocks = append(blocks, contentBlock{
					Type: "mcp_tool_use", ID: p.ToolCallID, ServerName: serverName,
					Name: arguments.ToolName, Input: arguments.ToolArgs,
				})
				continue
			}
			if p.ToolKind == ai.ToolPartKindCodeExecution {
				wireName, _ := p.ProviderDetails["anthropic_tool_name"].(string)
				if wireName != "bash_code_execution" && wireName != "text_editor_code_execution" {
					wireName = "code_execution"
				}
				input := slices.Clone(p.Args)
				if len(input) == 0 {
					input = json.RawMessage(`{}`)
				}
				block := contentBlock{Type: "server_tool_use", ID: p.ToolCallID, Name: wireName, Input: input}
				if caller, ok := p.ProviderDetails["anthropic_caller"].(map[string]any); ok {
					block.Caller = caller
				}
				blocks = append(blocks, block)
				continue
			}
			if p.ToolKind == ai.ToolPartKindWebSearch || p.ToolKind == ai.ToolPartKindWebFetch {
				input := slices.Clone(p.Args)
				if len(input) == 0 {
					input = json.RawMessage(`{}`)
				}
				block := contentBlock{
					Type: "server_tool_use", ID: p.ToolCallID, Name: p.ToolName, Input: input,
				}
				if caller, ok := p.ProviderDetails["anthropic_caller"].(map[string]any); ok {
					block.Caller = caller
				}
				blocks = append(blocks, block)
				continue
			}
			if p.ToolKind != ai.ToolPartKindToolSearch || len(deferredNames) == 0 {
				continue
			}
			strategy, _ := p.ProviderDetails["strategy"].(string)
			wireName, wireKey := "tool_search_tool_bm25", "query"
			if strategy == "regex" {
				wireName, wireKey = "tool_search_tool_regex", "pattern"
			}
			input, err := anthropicToolSearchReplayInput(p.Args, wireKey)
			if err != nil {
				return nil, err
			}
			block := contentBlock{
				Type: "server_tool_use", ID: p.ToolCallID, Name: wireName, Input: input,
			}
			if caller, ok := p.ProviderDetails["anthropic_caller"].(map[string]any); ok {
				block.Caller = caller
			}
			blocks = append(blocks, block)
		case ai.NativeToolReturnPart:
			if p.ProviderName != "anthropic" {
				continue
			}
			if p.ToolKind == ai.ToolPartKindAdvisor {
				if !advisorActive {
					continue
				}
				if _, ok := p.Content.(map[string]any); !ok {
					continue
				}
				blocks = append(blocks, contentBlock{
					Type: "advisor_tool_result", ToolUseID: p.ToolCallID, Content: p.Content,
				})
				continue
			}
			if p.ToolKind == ai.ToolPartKindMCPServer {
				encoded, err := json.Marshal(p.Content)
				if err != nil {
					return nil, fmt.Errorf("anthropic: marshal MCP server result: %w", err)
				}
				var result struct {
					Content json.RawMessage `json:"content"`
					IsError bool            `json:"is_error"`
				}
				if err := json.Unmarshal(encoded, &result); err != nil {
					return nil, fmt.Errorf("anthropic: parse MCP server result: %w", err)
				}
				content := result.Content
				if len(content) == 0 {
					content = json.RawMessage(`null`)
				}
				blocks = append(blocks, contentBlock{
					Type: "mcp_tool_result", ToolUseID: p.ToolCallID, Content: content, IsError: result.IsError,
				})
				continue
			}
			if p.ToolKind == ai.ToolPartKindCodeExecution {
				wireName, _ := p.ProviderDetails["anthropic_tool_name"].(string)
				resultType := "code_execution_tool_result"
				if wireName == "bash_code_execution" || wireName == "text_editor_code_execution" {
					resultType = wireName + "_tool_result"
				}
				blocks = append(blocks, contentBlock{
					Type: resultType, ToolUseID: p.ToolCallID, Content: p.Content,
				})
				continue
			}
			if p.ToolKind == ai.ToolPartKindWebSearch || p.ToolKind == ai.ToolPartKindWebFetch {
				block := contentBlock{
					Type: p.ToolName + "_tool_result", ToolUseID: p.ToolCallID, Content: p.Content,
				}
				if caller, ok := p.ProviderDetails["anthropic_caller"].(map[string]any); ok {
					block.Caller = caller
				}
				blocks = append(blocks, block)
				continue
			}
			if p.ToolKind != ai.ToolPartKindToolSearch || len(deferredNames) == 0 {
				continue
			}
			content, err := anthropicToolSearchReplayResult(p, deferredNames)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, contentBlock{
				Type: "tool_search_tool_result", ToolUseID: p.ToolCallID, Content: content,
			})
		}
	}
	if len(carried) > 0 {
		messages := []messageParam{{Role: "user", Content: carried}}
		if len(blocks) > 0 {
			messages = append(messages, messageParam{Role: "assistant", Content: blocks})
		}
		return messages, nil
	}
	return []messageParam{{Role: "assistant", Content: blocks}}, nil
}

func anthropicToolSearchReplayInput(args json.RawMessage, wireKey string) (json.RawMessage, error) {
	if len(args) == 0 {
		return json.RawMessage(`{}`), nil
	}
	var value map[string]any
	if err := json.Unmarshal(args, &value); err != nil {
		return nil, fmt.Errorf("anthropic: parse native tool search arguments: %w", err)
	}
	rawQueries, exists := value["queries"]
	if !exists {
		return slices.Clone(args), nil
	}
	queries, _ := rawQueries.([]any)
	stringsOnly := make([]string, 0, len(queries))
	for _, query := range queries {
		if query, ok := query.(string); ok {
			stringsOnly = append(stringsOnly, query)
		}
	}
	return json.Marshal(map[string]any{wireKey: strings.Join(stringsOnly, " ")})
}

func anthropicToolSearchReplayResult(
	part ai.NativeToolReturnPart, deferredNames map[string]struct{},
) (map[string]any, error) {
	if errorCode, exists := part.ProviderDetails["error_code"]; exists {
		return map[string]any{"type": "tool_search_tool_result_error", "error_code": errorCode}, nil
	}
	names, _, err := anthropicToolSearchResult(part.Content, deferredNames)
	if err != nil {
		return nil, err
	}
	references := make([]toolReferenceContent, len(names))
	for index, name := range names {
		references[index] = toolReferenceContent{Type: "tool_reference", ToolName: name}
	}
	return map[string]any{
		"type": "tool_search_tool_search_result", "tool_references": references,
	}, nil
}

func supportsDeferredTools(name string) bool {
	for _, prefix := range []string{
		"claude-haiku-4-5", "claude-haiku-5-5", "claude-sonnet-4-5", "claude-sonnet-4-6", "claude-sonnet-5",
		"claude-opus-4-5", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5",
	} {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

func supportsStrictTools(name string) bool {
	for _, prefix := range []string{
		"claude-fable-5", "claude-mythos-5", "claude-haiku-4-5", "claude-haiku-5-5", "claude-sonnet-4-5",
		"claude-sonnet-4-6", "claude-opus-4-1", "claude-opus-4-5", "claude-opus-4-6",
		"claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
	} {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

type messagesResponse struct {
	ID                   string                 `json:"id"`
	Model                string                 `json:"model"`
	StopReason           string                 `json:"stop_reason"`
	ServiceTier          string                 `json:"service_tier"`
	Content              []responseContentBlock `json:"content"`
	Usage                anthropicUsage         `json:"usage"`
	InputTransformations []map[string]any       `json:"input_transformations"`
	Diagnostics          map[string]any         `json:"diagnostics"`
	Container            *struct {
		ID string `json:"id"`
	} `json:"container"`
}

type responseContentBlock struct {
	Type             string           `json:"type"`
	Text             string           `json:"text"`
	Citations        []map[string]any `json:"citations"`
	Thinking         string           `json:"thinking"`
	Signature        string           `json:"signature"`
	ID               string           `json:"id"`
	Name             string           `json:"name"`
	Input            json.RawMessage  `json:"input"`
	ToolUseID        string           `json:"tool_use_id"`
	Content          json.RawMessage  `json:"content"`
	EncryptedContent string           `json:"encrypted_content"`
	Caller           map[string]any   `json:"caller"`
	ServerName       string           `json:"server_name"`
	IsError          bool             `json:"is_error"`
}

type anthropicUsage struct {
	InputTokens              int                       `json:"input_tokens"`
	OutputTokens             int                       `json:"output_tokens"`
	CacheCreationInputTokens int                       `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int                       `json:"cache_read_input_tokens"`
	CacheCreation            *anthropicCacheCreation   `json:"cache_creation"`
	Iterations               []anthropicIterationUsage `json:"iterations"`
	ServerToolUse            *struct {
		WebSearchRequests int `json:"web_search_requests"`
	} `json:"server_tool_use"`
	OutputTokensDetails *struct {
		ThinkingTokens int `json:"thinking_tokens"`
	} `json:"output_tokens_details"`
}

func (u anthropicUsage) thinkingTokens() int {
	if u.OutputTokensDetails == nil {
		return 0
	}
	return u.OutputTokensDetails.ThinkingTokens
}

type anthropicIterationUsage struct {
	Type                     string                  `json:"type"`
	InputTokens              int                     `json:"input_tokens"`
	OutputTokens             int                     `json:"output_tokens"`
	CacheCreationInputTokens int                     `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int                     `json:"cache_read_input_tokens"`
	CacheCreation            *anthropicCacheCreation `json:"cache_creation"`
}

type anthropicCacheCreation struct {
	Ephemeral1HInputTokens int `json:"ephemeral_1h_input_tokens"`
}

func (u anthropicUsage) usage() ai.Usage {
	details := map[string]int{
		"input_tokens":                u.InputTokens,
		"output_tokens":               u.OutputTokens,
		"cache_creation_input_tokens": u.CacheCreationInputTokens,
		"cache_read_input_tokens":     u.CacheReadInputTokens,
	}
	oneHourCacheWrites := 0
	compactionInputTokens := 0
	compactionOutputTokens := 0
	compactionCacheWriteTokens := 0
	compactionCacheReadTokens := 0
	if u.CacheCreation != nil && u.CacheCreation.Ephemeral1HInputTokens > 0 {
		oneHourCacheWrites = u.CacheCreation.Ephemeral1HInputTokens
		details["ephemeral_1h_input_tokens"] = oneHourCacheWrites
	}
	if u.ServerToolUse != nil && u.ServerToolUse.WebSearchRequests > 0 {
		details["web_search_requests"] = u.ServerToolUse.WebSearchRequests
	}
	// Billed inside output_tokens, so a readable subset of it rather than an addition.
	if thinking := u.thinkingTokens(); thinking > 0 {
		details["thinking_tokens"] = thinking
	}
	if len(u.Iterations) > 0 {
		for _, iteration := range u.Iterations {
			prefix := "message"
			switch iteration.Type {
			case "advisor_message":
				prefix = "advisor"
			case "compaction":
				prefix = "compaction"
			}
			details[prefix+"_iterations"]++
			for name, value := range map[string]int{
				"input_tokens": iteration.InputTokens, "output_tokens": iteration.OutputTokens,
				"cache_creation_input_tokens": iteration.CacheCreationInputTokens,
				"cache_read_input_tokens":     iteration.CacheReadInputTokens,
			} {
				if value > 0 {
					details[prefix+"_"+name] += value
				}
			}
			if prefix == "compaction" {
				compactionInputTokens += iteration.InputTokens
				compactionOutputTokens += iteration.OutputTokens
				compactionCacheWriteTokens += iteration.CacheCreationInputTokens
				compactionCacheReadTokens += iteration.CacheReadInputTokens
			}
			if iteration.CacheCreation != nil && iteration.CacheCreation.Ephemeral1HInputTokens > 0 {
				value := iteration.CacheCreation.Ephemeral1HInputTokens
				details[prefix+"_ephemeral_1h_input_tokens"] += value
				if prefix == "compaction" {
					oneHourCacheWrites += value
				}
			}
		}
	}
	return ai.Usage{
		Requests: 1,
		InputTokens: u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens +
			compactionInputTokens + compactionCacheWriteTokens + compactionCacheReadTokens,
		CacheWriteTokens:   u.CacheCreationInputTokens + compactionCacheWriteTokens,
		CacheWrite1HTokens: oneHourCacheWrites,
		CacheReadTokens:    u.CacheReadInputTokens + compactionCacheReadTokens,
		OutputTokens:       u.OutputTokens + compactionOutputTokens,
		ReasoningTokens:    u.thinkingTokens(),
		Details:            details,
	}
}

func parseResponse(data []byte) (*ai.ModelResponse, error) {
	var mr messagesResponse
	if err := json.Unmarshal(data, &mr); err != nil {
		return nil, &ai.ModelTransportError{ProviderName: "anthropic", Operation: "parse response", Err: err}
	}
	providerDetails := map[string]any{}
	if mr.StopReason != "" {
		providerDetails["finish_reason"] = mr.StopReason
	}
	if mr.ServiceTier != "" {
		providerDetails["service_tier"] = mr.ServiceTier
	}
	if mr.Diagnostics != nil {
		providerDetails["cache_diagnostics"] = mr.Diagnostics
	}
	if mr.Container != nil && mr.Container.ID != "" {
		providerDetails["container_id"] = mr.Container.ID
	}
	if len(mr.InputTransformations) > 0 {
		providerDetails["input_transformations"] = slices.Clone(mr.InputTransformations)
	}
	if len(providerDetails) == 0 {
		providerDetails = nil
	}
	state := ai.ModelResponseStateComplete
	if mr.StopReason == "pause_turn" {
		state = ai.ModelResponseStateSuspended
	}
	resp := &ai.ModelResponse{
		ModelName: mr.Model, Usage: mr.Usage.usage(), ProviderDetails: providerDetails,
		ProviderResponseID: mr.ID, FinishReason: anthropicFinishReason(mr.StopReason), State: state,
	}
	mcpToolNames := make(map[string]string)
	for _, block := range mr.Content {
		switch block.Type {
		case "text":
			var details map[string]any
			if len(block.Citations) > 0 {
				details = map[string]any{"citations": cloneAnthropicCitations(block.Citations)}
			}
			resp.Parts = append(resp.Parts, ai.TextPart{Content: block.Text, ProviderDetails: details})
		case "compaction":
			var details map[string]any
			if block.EncryptedContent != "" {
				details = map[string]any{"encrypted_content": block.EncryptedContent}
			}
			resp.Parts = append(resp.Parts, ai.CompactionPart{
				Content: rawJSONString(block.Content), ProviderName: "anthropic", ProviderDetails: details,
			})
		case "thinking":
			resp.Parts = append(resp.Parts, ai.ThinkingPart{
				Content: block.Thinking, Signature: block.Signature, ProviderName: "anthropic",
			})
		case "tool_use":
			resp.Parts = append(resp.Parts, ai.ToolCallPart{ToolName: block.Name, Args: block.Input, ToolCallID: block.ID})
		case "mcp_tool_use":
			args, _ := anthropicMCPCallArgs(block.Name, block.Input)
			toolName := "mcp_server:" + block.ServerName
			mcpToolNames[block.ID] = toolName
			resp.Parts = append(resp.Parts, ai.NativeToolCallPart{
				ToolName: toolName, Args: args, ToolCallID: block.ID,
				ToolKind: ai.ToolPartKindMCPServer, ProviderName: "anthropic",
			})
		case "mcp_tool_result":
			resp.Parts = append(resp.Parts, anthropicMCPResult(block, mcpToolNames[block.ToolUseID]))
		case "server_tool_use":
			if block.Name == "advisor" {
				args := slices.Clone(block.Input)
				if string(args) == "{}" || string(args) == "null" {
					args = nil
				}
				var details map[string]any
				if callerType, _ := block.Caller["type"].(string); callerType != "" && callerType != "direct" {
					details = map[string]any{"anthropic_caller": block.Caller}
				}
				resp.Parts = append(resp.Parts, ai.NativeToolCallPart{
					ToolName: "advisor", Args: args, ToolCallID: block.ID,
					ToolKind: ai.ToolPartKindAdvisor, ProviderName: "anthropic", ProviderDetails: details,
				})
				continue
			}
			if block.Name == "code_execution" || block.Name == "bash_code_execution" ||
				block.Name == "text_editor_code_execution" {
				args := slices.Clone(block.Input)
				if len(args) == 0 || string(args) == "null" {
					args = json.RawMessage(`{}`)
				}
				var details map[string]any
				if block.Name != "code_execution" {
					details = map[string]any{"anthropic_tool_name": block.Name}
				}
				if callerType, _ := block.Caller["type"].(string); callerType != "" && callerType != "direct" {
					if details == nil {
						details = map[string]any{}
					}
					details["anthropic_caller"] = block.Caller
				}
				resp.Parts = append(resp.Parts, ai.NativeToolCallPart{
					ToolName: "code_execution", Args: args, ToolCallID: block.ID,
					ToolKind: ai.ToolPartKindCodeExecution, ProviderName: "anthropic", ProviderDetails: details,
				})
				continue
			}
			if block.Name == "web_search" || block.Name == "web_fetch" {
				args := slices.Clone(block.Input)
				if len(args) == 0 || string(args) == "null" {
					args = json.RawMessage(`{}`)
				}
				var details map[string]any
				if callerType, _ := block.Caller["type"].(string); callerType != "" && callerType != "direct" {
					details = map[string]any{"anthropic_caller": block.Caller}
				}
				kind := ai.ToolPartKindWebSearch
				if block.Name == "web_fetch" {
					kind = ai.ToolPartKindWebFetch
				}
				resp.Parts = append(resp.Parts, ai.NativeToolCallPart{
					ToolName: block.Name, Args: args, ToolCallID: block.ID,
					ToolKind: kind, ProviderName: "anthropic", ProviderDetails: details,
				})
				continue
			}
			if block.Name != "tool_search_tool_bm25" && block.Name != "tool_search_tool_regex" {
				return nil, fmt.Errorf("anthropic: unsupported server tool %q", block.Name)
			}
			strategy := "bm25"
			if block.Name == "tool_search_tool_regex" {
				strategy = "regex"
			}
			args, err := normalizeAnthropicToolSearchArguments(block.Input, strategy)
			if err != nil {
				return nil, err
			}
			details := map[string]any{"strategy": strategy}
			if callerType, _ := block.Caller["type"].(string); callerType != "" && callerType != "direct" {
				details["anthropic_caller"] = block.Caller
			}
			resp.Parts = append(resp.Parts, ai.NativeToolCallPart{
				ToolName: ai.ToolSearchName, Args: args, ToolCallID: block.ID,
				ToolKind: ai.ToolPartKindToolSearch, ProviderName: "anthropic", ProviderDetails: details,
			})
		case "web_search_tool_result":
			resp.Parts = append(resp.Parts, parseAnthropicWebResult(block, "web_search", ai.ToolPartKindWebSearch))
		case "web_fetch_tool_result":
			resp.Parts = append(resp.Parts, parseAnthropicWebResult(block, "web_fetch", ai.ToolPartKindWebFetch))
		case "code_execution_tool_result", "bash_code_execution_tool_result", "text_editor_code_execution_tool_result":
			resp.Parts = append(resp.Parts, parseAnthropicCodeExecutionResult(block))
		case "advisor_tool_result":
			var content any
			if len(block.Content) > 0 {
				_ = json.Unmarshal(block.Content, &content)
			}
			resp.Parts = append(resp.Parts, ai.NativeToolReturnPart{
				ToolName: "advisor", ToolCallID: block.ToolUseID, ToolKind: ai.ToolPartKindAdvisor,
				Content: content, ProviderName: "anthropic",
			})
		case "tool_search_tool_result":
			part, err := parseAnthropicToolSearchResult(block)
			if err != nil {
				return nil, err
			}
			resp.Parts = append(resp.Parts, part)
		default:
			return nil, fmt.Errorf("anthropic: unknown content block type %q", block.Type)
		}
	}
	return resp, nil
}

func cloneAnthropicCitations(citations []map[string]any) []map[string]any {
	cloned := make([]map[string]any, len(citations))
	for index, citation := range citations {
		encoded, _ := json.Marshal(citation)
		_ = json.Unmarshal(encoded, &cloned[index])
	}
	return cloned
}

func rawJSONString(raw json.RawMessage) string {
	var value string
	_ = json.Unmarshal(raw, &value)
	return value
}

func anthropicToolSearchStrategy(name string) (string, bool) {
	switch name {
	case "tool_search_tool_bm25":
		return "bm25", true
	case "tool_search_tool_regex":
		return "regex", true
	default:
		return "", false
	}
}

func normalizeAnthropicToolSearchArguments(raw json.RawMessage, strategy string) (json.RawMessage, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return json.RawMessage(`{"queries":[]}`), nil
	}
	var input map[string]any
	if err := json.Unmarshal(raw, &input); err != nil {
		return nil, fmt.Errorf("anthropic: parse tool search arguments: %w", err)
	}
	key := "query"
	if strategy == "regex" {
		key = "pattern"
	}
	query, _ := input[key].(string)
	queries := make([]string, 0, 1)
	if query != "" {
		queries = append(queries, query)
	}
	return json.Marshal(map[string]any{"queries": queries})
}

func anthropicMCPCallArgs(toolName string, raw json.RawMessage) (json.RawMessage, error) {
	var toolArgs any = map[string]any{}
	if len(raw) > 0 && string(raw) != "null" {
		if err := json.Unmarshal(raw, &toolArgs); err != nil {
			return nil, fmt.Errorf("anthropic: parse MCP tool arguments: %w", err)
		}
	}
	args, _ := json.Marshal(map[string]any{
		"action": "call_tool", "tool_name": toolName, "tool_args": toolArgs,
	})
	return args, nil
}

func anthropicMCPResult(block responseContentBlock, toolName string) ai.NativeToolReturnPart {
	if toolName == "" {
		toolName = "mcp_server"
	}
	var content any
	if len(block.Content) > 0 {
		_ = json.Unmarshal(block.Content, &content)
	}
	return ai.NativeToolReturnPart{
		ToolName: toolName, ToolCallID: block.ToolUseID, ToolKind: ai.ToolPartKindMCPServer,
		Content: map[string]any{"content": content, "is_error": block.IsError}, ProviderName: "anthropic",
	}
}

func parseAnthropicCodeExecutionResult(block responseContentBlock) ai.NativeToolReturnPart {
	var content any
	if len(block.Content) > 0 {
		_ = json.Unmarshal(block.Content, &content)
	}
	var details map[string]any
	if block.Type == "bash_code_execution_tool_result" || block.Type == "text_editor_code_execution_tool_result" {
		details = map[string]any{"anthropic_tool_name": strings.TrimSuffix(block.Type, "_tool_result")}
	}
	return ai.NativeToolReturnPart{
		ToolName: "code_execution", ToolCallID: block.ToolUseID, ToolKind: ai.ToolPartKindCodeExecution,
		Content: content, ProviderName: "anthropic", ProviderDetails: details,
	}
}

func parseAnthropicWebResult(
	block responseContentBlock, toolName string, kind ai.ToolPartKind,
) ai.NativeToolReturnPart {
	var content any
	if len(block.Content) > 0 {
		_ = json.Unmarshal(block.Content, &content)
	}
	var details map[string]any
	if callerType, _ := block.Caller["type"].(string); callerType != "" && callerType != "direct" {
		details = map[string]any{"anthropic_caller": block.Caller}
	}
	return ai.NativeToolReturnPart{
		ToolName: toolName, ToolCallID: block.ToolUseID, ToolKind: kind,
		Content: content, ProviderName: "anthropic", ProviderDetails: details,
	}
}

func parseAnthropicToolSearchResult(block responseContentBlock) (ai.NativeToolReturnPart, error) {
	var content struct {
		Type           string `json:"type"`
		ErrorCode      string `json:"error_code"`
		ErrorMessage   string `json:"error_message"`
		ToolReferences []struct {
			ToolName string `json:"tool_name"`
		} `json:"tool_references"`
	}
	if err := json.Unmarshal(block.Content, &content); err != nil {
		return ai.NativeToolReturnPart{}, fmt.Errorf("anthropic: parse tool search result: %w", err)
	}
	var details map[string]any
	switch content.Type {
	case "tool_search_tool_search_result":
	case "tool_search_tool_result_error":
		details = map[string]any{"error_code": content.ErrorCode, "error_message": content.ErrorMessage}
	default:
		return ai.NativeToolReturnPart{}, fmt.Errorf("anthropic: unknown tool search result type %q", content.Type)
	}
	matches := make([]ai.ToolSearchMatch, len(content.ToolReferences))
	for index, reference := range content.ToolReferences {
		matches[index] = ai.ToolSearchMatch{Name: reference.ToolName}
	}
	return ai.NativeToolReturnPart{
		ToolName: ai.ToolSearchName, ToolCallID: block.ToolUseID, ToolKind: ai.ToolPartKindToolSearch,
		Content: ai.ToolSearchResult{DiscoveredTools: matches}, Outcome: ai.ToolReturnOutcomeSuccess,
		ProviderName: "anthropic", ProviderDetails: details,
	}, nil
}

// SupportsToolSearchStrategy reports Anthropic's named hosted search variants.
func (m *Model) SupportsToolSearchStrategy(strategy ai.ToolSearchStrategy) bool {
	if !m.deferredToolSupport {
		return false
	}
	if m.legacyBedrockClient != nil {
		return strategy == ai.ToolSearchStrategyRegex
	}
	return strategy == ai.ToolSearchStrategyBM25 || strategy == ai.ToolSearchStrategyRegex
}

// NativeToolSearchProvider identifies histories this model can replay natively.
func (*Model) NativeToolSearchProvider() string { return "anthropic" }

func anthropicFinishReason(reason string) ai.FinishReason {
	return map[string]ai.FinishReason{
		"compaction": ai.FinishReasonStop, "end_turn": ai.FinishReasonStop,
		"stop_sequence": ai.FinishReasonStop, "max_tokens": ai.FinishReasonLength,
		"model_context_window_exceeded": ai.FinishReasonLength, "tool_use": ai.FinishReasonToolCall,
		"refusal": ai.FinishReasonContentFilter,
	}[reason]
}

func contentString(content any) (string, error) {
	if s, ok := content.(string); ok {
		return s, nil
	}
	b, err := json.Marshal(content)
	if err != nil {
		return "", fmt.Errorf("anthropic: marshal tool return: %w", err)
	}
	return string(b), nil
}
