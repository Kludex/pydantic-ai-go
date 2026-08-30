// Package anthropic implements ai.Model against the Anthropic Messages API.
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	ai "github.com/Kludex/pydantic-ai-go"
)

const defaultMaxTokens = 4096

// Model calls the Anthropic Messages API. Create one with NewModel.
type Model struct {
	name       string
	apiKey     string
	baseURL    string
	httpClient *http.Client
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

// NewModel creates a Model for the named Anthropic model, e.g. "claude-sonnet-4-5".
func NewModel(name string, opts ...Option) *Model {
	m := &Model{
		name:       name,
		apiKey:     os.Getenv("ANTHROPIC_API_KEY"),
		baseURL:    "https://api.anthropic.com/v1",
		httpClient: http.DefaultClient,
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// Name returns the model name.
func (m *Model) Name() string { return m.name }

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
	return parseResponse(data)
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
}

type messageParam struct {
	Role    string         `json:"role"`
	Content []contentBlock `json:"content"`
}

type contentBlock struct {
	Type string `json:"type"`
	// text
	Text string `json:"text,omitempty"`
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
}

type toolChoiceParam struct {
	Type string `json:"type"`
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
		req.Tools = append(req.Tools, convertTool(tool))
	}
	if params.OutputTool != nil {
		req.Tools = append(req.Tools, convertTool(*params.OutputTool))
		if !params.AllowText {
			req.ToolChoice = &toolChoiceParam{Type: "any"}
		}
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
			blocks = append(blocks, contentBlock{Type: "text", Text: p.Content})
		case ai.ToolReturnPart:
			content, err := contentString(p.Content)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, contentBlock{Type: "tool_result", ToolUseID: p.ToolCallID, Content: content})
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

func convertTool(def ai.ToolDefinition) toolParam {
	return toolParam{Name: def.Name, Description: def.Description, InputSchema: def.Schema}
}

type messagesResponse struct {
	Model   string `json:"model"`
	Content []struct {
		Type     string          `json:"type"`
		Text     string          `json:"text"`
		Thinking string          `json:"thinking"`
		ID       string          `json:"id"`
		Name     string          `json:"name"`
		Input    json.RawMessage `json:"input"`
	} `json:"content"`
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

func parseResponse(data []byte) (*ai.ModelResponse, error) {
	var mr messagesResponse
	if err := json.Unmarshal(data, &mr); err != nil {
		return nil, fmt.Errorf("anthropic: parse response: %w", err)
	}
	resp := &ai.ModelResponse{
		ModelName: mr.Model,
		Usage: ai.Usage{
			Requests:     1,
			InputTokens:  mr.Usage.InputTokens,
			OutputTokens: mr.Usage.OutputTokens,
		},
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
