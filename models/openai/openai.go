// Package openai implements ai.Model against the OpenAI Chat Completions API.
package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
)

// Model calls the OpenAI Chat Completions API. Create one with NewModel.
type Model struct {
	name       string
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// Option configures a Model.
type Option func(*Model)

// WithAPIKey sets the API key. The default is the OPENAI_API_KEY environment variable.
func WithAPIKey(key string) Option { return func(m *Model) { m.apiKey = key } }

// WithBaseURL points the model at a different endpoint, such as a proxy or
// an OpenAI-compatible provider. The default is https://api.openai.com/v1.
func WithBaseURL(url string) Option { return func(m *Model) { m.baseURL = url } }

// WithHTTPClient sets the HTTP client used for requests.
func WithHTTPClient(c *http.Client) Option { return func(m *Model) { m.httpClient = c } }

// NewModel creates a Model for the named OpenAI model, e.g. "gpt-5".
func NewModel(name string, opts ...Option) *Model {
	m := &Model{
		name:       name,
		apiKey:     os.Getenv("OPENAI_API_KEY"),
		baseURL:    "https://api.openai.com/v1",
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
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.apiKey)

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
	return parseResponse(data)
}

// APIError is a non-200 response from the OpenAI API.
type APIError struct {
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("openai: API returned status %d: %s", e.StatusCode, e.Body)
}

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
}

type chatMessage struct {
	Role       string     `json:"role"`
	Content    any        `json:"content,omitempty"` // string or []contentPart
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
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
	req := &chatRequest{
		Model:       m.name,
		MaxTokens:   params.Settings.MaxTokens,
		Temperature: params.Settings.Temperature,
		TopP:        params.Settings.TopP,
		Seed:        params.Settings.Seed,
		Stop:        params.Settings.StopSequences,
	}
	if params.Instructions != "" {
		req.Messages = append(req.Messages, chatMessage{Role: "system", Content: params.Instructions})
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
			req.ToolChoice = "required"
		}
	}
	if len(req.Tools) > 0 {
		req.ParallelToolCalls = params.Settings.ParallelToolCalls
	}
	if params.OutputSchema != nil {
		req.ResponseFormat = &responseFormat{
			Type:       "json_schema",
			JSONSchema: jsonSchemaFormat{Name: "final_result", Schema: params.OutputSchema, Strict: true},
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
	Strict bool           `json:"strict"`
}

func convertMessage(msg ai.ModelMessage) ([]chatMessage, error) {
	switch m := msg.(type) {
	case ai.ModelRequest:
		return convertRequest(m)
	case ai.ModelResponse:
		return convertResponse(m), nil
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
		case ai.RetryPromptPart:
			if p.ToolCallID != "" {
				out = append(out, chatMessage{Role: "tool", Content: p.Content, ToolCallID: p.ToolCallID})
			} else {
				out = append(out, chatMessage{Role: "user", Content: p.Content})
			}
		default:
			return nil, fmt.Errorf("openai: unknown request part type %T", part)
		}
	}
	return out, nil
}

func convertResponse(m ai.ModelResponse) []chatMessage {
	msg := chatMessage{Role: "assistant"}
	for _, part := range m.Parts {
		switch p := part.(type) {
		case ai.TextPart:
			msg.Content = p.Content
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

func convertTool(def ai.ToolDefinition) chatTool {
	return chatTool{
		Type:     "function",
		Function: chatFunction{Name: def.Name, Description: def.Description, Parameters: def.Schema, Strict: def.Strict},
	}
}

type chatResponse struct {
	Model   string `json:"model"`
	Created int64  `json:"created"`
	Choices []struct {
		Message struct {
			Content   string     `json:"content"`
			ToolCalls []toolCall `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

func parseResponse(data []byte) (*ai.ModelResponse, error) {
	var cr chatResponse
	if err := json.Unmarshal(data, &cr); err != nil {
		return nil, fmt.Errorf("openai: parse response: %w", err)
	}
	if len(cr.Choices) == 0 {
		return nil, fmt.Errorf("openai: response has no choices")
	}
	resp := &ai.ModelResponse{
		ModelName: cr.Model,
		Timestamp: time.Unix(cr.Created, 0).UTC(),
		Usage: ai.Usage{
			Requests:     1,
			InputTokens:  cr.Usage.PromptTokens,
			OutputTokens: cr.Usage.CompletionTokens,
		},
	}
	msg := cr.Choices[0].Message
	if msg.Content != "" {
		resp.Parts = append(resp.Parts, ai.TextPart{Content: msg.Content})
	}
	for _, call := range msg.ToolCalls {
		resp.Parts = append(resp.Parts, ai.ToolCallPart{
			ToolName:   call.Function.Name,
			Args:       json.RawMessage(call.Function.Arguments),
			ToolCallID: call.ID,
		})
	}
	return resp, nil
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
