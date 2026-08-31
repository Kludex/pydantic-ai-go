// Package anthropic implements ai.Model against the Anthropic Messages API.
package anthropic

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
)

const defaultMaxTokens = 4096

// Model calls the Anthropic Messages API. Create one with NewModel.
type Model struct {
	name                string
	apiKey              string
	baseURL             string
	httpClient          *http.Client
	strictToolSupport   bool
	deferredToolSupport bool
	schemaWarning       func(SchemaWarning)
	defaultSettings     ai.ModelSettings
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

// SchemaWarning describes a lossy strict-schema conversion.
type SchemaWarning struct {
	ToolName string
	Message  string
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
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// Name returns the model name.
func (m *Model) Name() string { return m.name }

// ProviderName returns Anthropic's durable provider identity.
func (*Model) ProviderName() string { return "anthropic" }

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
		return nil, fmt.Errorf("anthropic: marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/messages", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	setExtraHeaders(req, params.Settings.ExtraHeaders)
	m.setRequestHeaders(req, payload, false)

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("anthropic: request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("anthropic: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(data)}
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
	payload, err := m.buildPayload(messages, params)
	if err != nil {
		return ai.Usage{}, err
	}
	countPayload := struct {
		Model             string           `json:"model"`
		System            string           `json:"system,omitempty"`
		Messages          []messageParam   `json:"messages"`
		Tools             []toolParam      `json:"tools,omitempty"`
		ToolChoice        *toolChoiceParam `json:"tool_choice,omitempty"`
		Thinking          *thinkingParam   `json:"thinking,omitempty"`
		ContextManagement map[string]any   `json:"context_management,omitempty"`
	}{
		Model: payload.Model, System: payload.System, Messages: payload.Messages, Tools: payload.Tools,
		ToolChoice: payload.ToolChoice, Thinking: payload.Thinking, ContextManagement: payload.ContextManagement,
	}
	body, err := marshalRequest(countPayload, params.Settings.ExtraBody)
	if err != nil {
		return ai.Usage{}, fmt.Errorf("anthropic: marshal token count request: %w", err)
	}
	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost, m.baseURL+"/messages/count_tokens?beta=true", bytes.NewReader(body),
	)
	if err != nil {
		return ai.Usage{}, err
	}
	setExtraHeaders(req, params.Settings.ExtraHeaders)
	m.setRequestHeaders(req, payload, false)
	resp, err := m.httpClient.Do(req)
	if err != nil {
		return ai.Usage{}, fmt.Errorf("anthropic: token count request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return ai.Usage{}, fmt.Errorf("anthropic: read token count response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return ai.Usage{}, &APIError{StatusCode: resp.StatusCode, Body: string(data)}
	}
	var counted struct {
		InputTokens *int `json:"input_tokens"`
	}
	if err := json.Unmarshal(data, &counted); err != nil {
		return ai.Usage{}, fmt.Errorf("anthropic: decode token count response: %w", err)
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
	betas := strings.FieldsFunc(req.Header.Get("anthropic-beta"), func(r rune) bool {
		return r == ',' || r == ' '
	})
	if payload.ToolAdditions && !slices.Contains(betas, "mid-conversation-tool-changes-2026-07-01") {
		betas = append(betas, "mid-conversation-tool-changes-2026-07-01")
	}
	if payload.Compaction && !slices.Contains(betas, "compact-2026-01-12") {
		betas = append(betas, "compact-2026-01-12")
	}
	if len(betas) > 0 {
		unique := make([]string, 0, len(betas))
		for _, beta := range betas {
			if !slices.Contains(unique, beta) {
				unique = append(unique, beta)
			}
		}
		req.Header.Set("anthropic-beta", strings.Join(unique, ","))
	}
	if stream {
		req.Header.Set("Accept", "text/event-stream")
	}
}

// APIError is a non-200 response from the Anthropic API.
type APIError struct {
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("anthropic: API returned status %d: %s", e.StatusCode, e.Body)
}

// IsModelAPIError marks provider API responses as eligible for default model fallback.
func (*APIError) IsModelAPIError() bool { return true }

type messagesRequest struct {
	Model             string           `json:"model"`
	MaxTokens         int              `json:"max_tokens"`
	System            string           `json:"system,omitempty"`
	Messages          []messageParam   `json:"messages"`
	Tools             []toolParam      `json:"tools,omitempty"`
	ToolChoice        *toolChoiceParam `json:"tool_choice,omitempty"`
	Temperature       *float64         `json:"temperature,omitempty"`
	TopP              *float64         `json:"top_p,omitempty"`
	Stop              []string         `json:"stop_sequences,omitempty"`
	Stream            bool             `json:"stream,omitempty"`
	ToolAdditions     bool             `json:"-"`
	Compaction        bool             `json:"-"`
	Thinking          *thinkingParam   `json:"thinking,omitempty"`
	ServiceTier       string           `json:"service_tier,omitempty"`
	ContextManagement map[string]any   `json:"context_management,omitempty"`
}

type thinkingParam struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens"`
}

type messageParam struct {
	Role    string         `json:"role"`
	Content []contentBlock `json:"content"`
}

type contentBlock struct {
	Type string `json:"type"`
	// text and thinking
	Text      string `json:"text,omitempty"`
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
	// image
	Source *imageSource `json:"source,omitempty"`
	// tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
	// tool_result
	ToolUseID        string              `json:"tool_use_id,omitempty"`
	Content          any                 `json:"content,omitempty"`
	EncryptedContent string              `json:"encrypted_content,omitempty"`
	Caller           map[string]any      `json:"caller,omitempty"`
	IsError          bool                `json:"is_error,omitempty"`
	Tool             *toolReferenceParam `json:"tool,omitempty"`
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
	Type         string         `json:"type,omitempty"`
	Name         string         `json:"name"`
	Description  string         `json:"description,omitempty"`
	InputSchema  map[string]any `json:"input_schema,omitempty"`
	Strict       *bool          `json:"strict,omitempty"`
	DeferLoading bool           `json:"defer_loading,omitempty"`
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
}

func convertUserPrompt(p ai.UserPromptPart) ([]contentBlock, error) {
	if len(p.Contents) == 0 {
		return []contentBlock{{Type: "text", Text: p.Content}}, nil
	}
	blocks := make([]contentBlock, 0, len(p.Contents))
	for _, c := range p.Contents {
		switch item := c.(type) {
		case ai.TextContent:
			blocks = append(blocks, contentBlock{Type: "text", Text: item.Text})
		case ai.BinaryContent:
			blocks = append(blocks, contentBlock{Type: "image", Source: &imageSource{
				Type:      "base64",
				MediaType: item.MediaType,
				Data:      base64.StdEncoding.EncodeToString(item.Data),
			}})
		case ai.ImageURL:
			blocks = append(blocks, contentBlock{Type: "image", Source: &imageSource{Type: "url", URL: item.URL}})
		default:
			return nil, fmt.Errorf("anthropic: unsupported user content type %T", c)
		}
	}
	return blocks, nil
}

func (m *Model) buildPayload(msgs []ai.ModelMessage, params ai.ModelRequestParams) (*messagesRequest, error) {
	thinking, err := anthropicThinking(params.Settings.Thinking)
	if err != nil {
		return nil, err
	}
	serviceTier, err := anthropicServiceTier(params.Settings.ServiceTier)
	if err != nil {
		return nil, err
	}
	req := &messagesRequest{
		Model:       m.name,
		MaxTokens:   params.Settings.MaxTokens,
		System:      params.Instructions,
		Temperature: params.Settings.Temperature,
		TopP:        params.Settings.TopP,
		Stop:        params.Settings.StopSequences,
		Thinking:    thinking,
		ServiceTier: serviceTier,
	}
	if req.MaxTokens == 0 {
		req.MaxTokens = defaultMaxTokens
	}
	trimmedMessages, compaction := trimAnthropicCompactionMessages(msgs)
	var searchTool *ai.ToolDefinition
	for _, tool := range params.Tools {
		if tool.Name == ai.ToolSearchName && tool.ToolKind == ai.ToolPartKindToolSearch {
			definition := tool
			searchTool = &definition
			break
		}
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
		converted, err := convertMessage(msg, deferredNames)
		if err != nil {
			return nil, err
		}
		req.Messages = append(req.Messages, converted...)
	}
	for _, tool := range params.Tools {
		if serverToolSearch && tool.Name == ai.ToolSearchName {
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
	}
	if nativeDeferred {
		for _, tool := range params.DeferredTools {
			converted, err := prepareAnthropicTool(tool, m.strictToolSupport, m.schemaWarning)
			if err != nil {
				return nil, err
			}
			converted.DeferLoading = true
			req.Tools = append(req.Tools, converted)
		}
	}
	if serverToolSearch {
		if searchTool.ToolSearchStrategy == ai.ToolSearchStrategyRegex {
			req.Tools = append(req.Tools, toolParam{
				Type: "tool_search_tool_regex_20251119", Name: "tool_search_tool_regex",
			})
		} else {
			req.Tools = append(req.Tools, toolParam{
				Type: "tool_search_tool_bm25_20251119", Name: "tool_search_tool_bm25",
			})
		}
	}
	if params.OutputTool != nil {
		if thinking != nil && !params.AllowText {
			return nil, fmt.Errorf("anthropic: extended thinking and forced output tools cannot be used together")
		}
		converted, err := prepareAnthropicTool(*params.OutputTool, m.strictToolSupport, m.schemaWarning)
		if err != nil {
			return nil, err
		}
		req.Tools = append(req.Tools, converted)
		if !params.AllowText {
			req.ToolChoice = &toolChoiceParam{Type: "any"}
		}
	}
	if len(req.Tools) > 0 && params.Settings.ParallelToolCalls != nil {
		if req.ToolChoice == nil {
			req.ToolChoice = &toolChoiceParam{Type: "auto"}
		}
		disable := !*params.Settings.ParallelToolCalls
		req.ToolChoice.DisableParallelToolUse = &disable
	}
	if params.OutputSchema != nil && params.OutputMode != ai.OutputModePrompted {
		return nil, fmt.Errorf("anthropic: native JSON output mode is not supported; use OutputModeTool")
	}
	return req, nil
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

func anthropicThinking(settings *ai.ThinkingSettings) (*thinkingParam, error) {
	if settings == nil || settings.Level == ai.ThinkingLevelDisabled ||
		settings.Level == "" && settings.TokenBudget == nil {
		return nil, nil
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
			return nil, fmt.Errorf("anthropic: invalid thinking level %q", settings.Level)
		}
	}
	if budget <= 0 {
		return nil, fmt.Errorf("anthropic: thinking token budget must be positive, got %d", budget)
	}
	return &thinkingParam{Type: "enabled", BudgetTokens: budget}, nil
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

func convertMessage(msg ai.ModelMessage, deferredNames map[string]struct{}) ([]messageParam, error) {
	switch m := msg.(type) {
	case ai.ModelRequest:
		return convertRequest(m, deferredNames)
	case ai.ModelResponse:
		return convertResponse(m, deferredNames)
	default:
		return nil, fmt.Errorf("anthropic: unknown message type %T", msg)
	}
}

func convertRequest(m ai.ModelRequest, deferredNames map[string]struct{}) ([]messageParam, error) {
	var blocks []contentBlock
	searchReveals := make(map[string]struct{})
	for _, part := range m.Parts {
		switch p := part.(type) {
		case ai.SystemPromptPart:
			// Anthropic takes the system prompt at the top level; a
			// system part in history becomes user-visible context.
			blocks = append(blocks, contentBlock{Type: "text", Text: p.Content})
		case ai.UserPromptPart:
			converted, err := convertUserPrompt(p)
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
	return []messageParam{{Role: "user", Content: blocks}}, nil
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

func convertResponse(m ai.ModelResponse, deferredNames map[string]struct{}) ([]messageParam, error) {
	var blocks []contentBlock
	for _, part := range m.Parts {
		switch p := part.(type) {
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
			if p.Signature != "" && (p.ProviderName == "" || p.ProviderName == "anthropic") {
				blocks = append(blocks, contentBlock{Type: "thinking", Thinking: p.Content, Signature: p.Signature})
			}
		case ai.ToolCallPart:
			blocks = append(blocks, contentBlock{Type: "tool_use", ID: p.ToolCallID, Name: p.ToolName, Input: p.Args})
		case ai.NativeToolCallPart:
			if p.ProviderName != "anthropic" || p.ToolKind != ai.ToolPartKindToolSearch || len(deferredNames) == 0 {
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
			if p.ProviderName != "anthropic" || p.ToolKind != ai.ToolPartKindToolSearch || len(deferredNames) == 0 {
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
		"claude-haiku-4-5", "claude-sonnet-4-5", "claude-sonnet-4-6", "claude-sonnet-5",
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
		"claude-fable-5", "claude-mythos-5", "claude-haiku-4-5", "claude-sonnet-4-5",
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
	ID          string                 `json:"id"`
	Model       string                 `json:"model"`
	StopReason  string                 `json:"stop_reason"`
	ServiceTier string                 `json:"service_tier"`
	Content     []responseContentBlock `json:"content"`
	Usage       anthropicUsage         `json:"usage"`
}

type responseContentBlock struct {
	Type             string          `json:"type"`
	Text             string          `json:"text"`
	Thinking         string          `json:"thinking"`
	Signature        string          `json:"signature"`
	ID               string          `json:"id"`
	Name             string          `json:"name"`
	Input            json.RawMessage `json:"input"`
	ToolUseID        string          `json:"tool_use_id"`
	Content          json.RawMessage `json:"content"`
	EncryptedContent string          `json:"encrypted_content"`
	Caller           map[string]any  `json:"caller"`
}

type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

func (u anthropicUsage) usage() ai.Usage {
	return ai.Usage{
		Requests:         1,
		InputTokens:      u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens,
		CacheWriteTokens: u.CacheCreationInputTokens, CacheReadTokens: u.CacheReadInputTokens,
		OutputTokens: u.OutputTokens,
		Details: map[string]int{
			"input_tokens":                u.InputTokens,
			"output_tokens":               u.OutputTokens,
			"cache_creation_input_tokens": u.CacheCreationInputTokens,
			"cache_read_input_tokens":     u.CacheReadInputTokens,
		},
	}
}

func parseResponse(data []byte) (*ai.ModelResponse, error) {
	var mr messagesResponse
	if err := json.Unmarshal(data, &mr); err != nil {
		return nil, fmt.Errorf("anthropic: parse response: %w", err)
	}
	providerDetails := map[string]any{}
	if mr.StopReason != "" {
		providerDetails["finish_reason"] = mr.StopReason
	}
	if mr.ServiceTier != "" {
		providerDetails["service_tier"] = mr.ServiceTier
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
	for _, block := range mr.Content {
		switch block.Type {
		case "text":
			resp.Parts = append(resp.Parts, ai.TextPart{Content: block.Text})
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
		case "server_tool_use":
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
	return m.deferredToolSupport &&
		(strategy == ai.ToolSearchStrategyBM25 || strategy == ai.ToolSearchStrategyRegex)
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
