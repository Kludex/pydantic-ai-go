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
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
)

const defaultMaxTokens = 4096

// Model calls the Anthropic Messages API. Create one with NewModel.
type Model struct {
	name              string
	apiKey            string
	baseURL           string
	httpClient        *http.Client
	strictToolSupport bool
	schemaWarning     func(SchemaWarning)
	defaultSettings   ai.ModelSettings
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
		name:              name,
		apiKey:            os.Getenv("ANTHROPIC_API_KEY"),
		baseURL:           "https://api.anthropic.com/v1",
		httpClient:        http.DefaultClient,
		strictToolSupport: supportsStrictTools(name),
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// Name returns the model name.
func (m *Model) Name() string { return m.name }

// DefaultModelSettings returns this model's request defaults.
func (m *Model) DefaultModelSettings() ai.ModelSettings { return m.defaultSettings.Clone() }

// Request implements ai.Model.
func (m *Model) Request(ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
	payload, err := m.buildPayload(msgs, params)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/messages", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", m.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

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

// APIError is a non-200 response from the Anthropic API.
type APIError struct {
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("anthropic: API returned status %d: %s", e.StatusCode, e.Body)
}

type messagesRequest struct {
	Model       string           `json:"model"`
	MaxTokens   int              `json:"max_tokens"`
	System      string           `json:"system,omitempty"`
	Messages    []messageParam   `json:"messages"`
	Tools       []toolParam      `json:"tools,omitempty"`
	ToolChoice  *toolChoiceParam `json:"tool_choice,omitempty"`
	Temperature *float64         `json:"temperature,omitempty"`
	TopP        *float64         `json:"top_p,omitempty"`
	Stop        []string         `json:"stop_sequences,omitempty"`
	Stream      bool             `json:"stream,omitempty"`
}

type messageParam struct {
	Role    string         `json:"role"`
	Content []contentBlock `json:"content"`
}

type contentBlock struct {
	Type string `json:"type"`
	// text
	Text string `json:"text,omitempty"`
	// image
	Source *imageSource `json:"source,omitempty"`
	// tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
	// tool_result
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   string `json:"content,omitempty"`
	IsError   bool   `json:"is_error,omitempty"`
}

type toolParam struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
	Strict      *bool          `json:"strict,omitempty"`
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
	req := &messagesRequest{
		Model:       m.name,
		MaxTokens:   params.Settings.MaxTokens,
		System:      params.Instructions,
		Temperature: params.Settings.Temperature,
		TopP:        params.Settings.TopP,
		Stop:        params.Settings.StopSequences,
	}
	if req.MaxTokens == 0 {
		req.MaxTokens = defaultMaxTokens
	}
	for _, msg := range msgs {
		converted, err := convertMessage(msg)
		if err != nil {
			return nil, err
		}
		req.Messages = append(req.Messages, converted...)
	}
	for _, tool := range params.Tools {
		converted, err := prepareAnthropicTool(tool, m.strictToolSupport, m.schemaWarning)
		if err != nil {
			return nil, err
		}
		req.Tools = append(req.Tools, converted)
	}
	if params.OutputTool != nil {
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
	if params.OutputSchema != nil {
		return nil, fmt.Errorf("anthropic: native JSON output mode is not supported; use OutputModeTool")
	}
	return req, nil
}

func convertMessage(msg ai.ModelMessage) ([]messageParam, error) {
	switch m := msg.(type) {
	case ai.ModelRequest:
		return convertRequest(m)
	case ai.ModelResponse:
		return convertResponse(m), nil
	default:
		return nil, fmt.Errorf("anthropic: unknown message type %T", msg)
	}
}

func convertRequest(m ai.ModelRequest) ([]messageParam, error) {
	var blocks []contentBlock
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
			content, err := contentString(p.Content)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, contentBlock{
				Type: "tool_result", ToolUseID: p.ToolCallID, Content: content,
				IsError: p.Outcome == ai.ToolReturnOutcomeFailed || p.Outcome == ai.ToolReturnOutcomeInterrupted,
			})
		case ai.RetryPromptPart:
			if p.ToolCallID != "" {
				blocks = append(blocks, contentBlock{Type: "tool_result", ToolUseID: p.ToolCallID, Content: p.Content, IsError: true})
			} else {
				blocks = append(blocks, contentBlock{Type: "text", Text: p.Content})
			}
		default:
			return nil, fmt.Errorf("anthropic: unknown request part type %T", part)
		}
	}
	return []messageParam{{Role: "user", Content: blocks}}, nil
}

func convertResponse(m ai.ModelResponse) []messageParam {
	var blocks []contentBlock
	for _, part := range m.Parts {
		switch p := part.(type) {
		case ai.TextPart:
			blocks = append(blocks, contentBlock{Type: "text", Text: p.Content})
		case ai.ToolCallPart:
			blocks = append(blocks, contentBlock{Type: "tool_use", ID: p.ToolCallID, Name: p.ToolName, Input: p.Args})
		}
	}
	return []messageParam{{Role: "assistant", Content: blocks}}
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
	ID         string `json:"id"`
	Model      string `json:"model"`
	StopReason string `json:"stop_reason"`
	Content    []struct {
		Type     string          `json:"type"`
		Text     string          `json:"text"`
		Thinking string          `json:"thinking"`
		ID       string          `json:"id"`
		Name     string          `json:"name"`
		Input    json.RawMessage `json:"input"`
	} `json:"content"`
	Usage anthropicUsage `json:"usage"`
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
		case "thinking":
			resp.Parts = append(resp.Parts, ai.ThinkingPart{Content: block.Thinking})
		case "tool_use":
			resp.Parts = append(resp.Parts, ai.ToolCallPart{ToolName: block.Name, Args: block.Input, ToolCallID: block.ID})
		default:
			return nil, fmt.Errorf("anthropic: unknown content block type %q", block.Type)
		}
	}
	return resp, nil
}

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
