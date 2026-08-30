package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	ai "github.com/Kludex/pydantic-ai-go"
)

// ResponsesModel calls the OpenAI Responses API, the successor to Chat
// Completions. Create one with NewResponsesModel.
type ResponsesModel struct {
	name       string
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// NewResponsesModel creates a ResponsesModel for the named OpenAI model.
// It accepts the same options as NewModel.
func NewResponsesModel(name string, opts ...Option) *ResponsesModel {
	m := NewModel(name, opts...)
	return &ResponsesModel{name: m.name, apiKey: m.apiKey, baseURL: m.baseURL, httpClient: m.httpClient}
}

// Name returns the model name.
func (m *ResponsesModel) Name() string { return m.name }

// Request implements ai.Model.
func (m *ResponsesModel) Request(ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
	payload, err := m.buildResponsesPayload(msgs, params)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/responses", bytes.NewReader(body))
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
	return parseResponsesResponse(data)
}

type responsesRequest struct {
	Model             string           `json:"model"`
	Instructions      string           `json:"instructions,omitempty"`
	Input             []responsesInput `json:"input"`
	Tools             []responsesTool  `json:"tools,omitempty"`
	ToolChoice        any              `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool            `json:"parallel_tool_calls,omitempty"`
	MaxTokens         int              `json:"max_output_tokens,omitempty"`
	Temperature       *float64         `json:"temperature,omitempty"`
	TopP              *float64         `json:"top_p,omitempty"`
	Stream            bool             `json:"stream,omitempty"`
}

type responsesInput struct {
	// message
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
	// function_call and function_call_output items
	Type      string `json:"type,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Output    string `json:"output,omitempty"`
}

type responsesTool struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
	Strict      *bool          `json:"strict,omitempty"`
}

func (m *ResponsesModel) buildResponsesPayload(msgs []ai.ModelMessage, params ai.ModelRequestParams) (*responsesRequest, error) {
	req := &responsesRequest{
		Model:        m.name,
		Instructions: params.Instructions,
		MaxTokens:    params.Settings.MaxTokens,
		Temperature:  params.Settings.Temperature,
		TopP:         params.Settings.TopP,
	}
	for _, msg := range msgs {
		items, err := convertResponsesMessage(msg)
		if err != nil {
			return nil, err
		}
		req.Input = append(req.Input, items...)
	}
	for _, tool := range params.Tools {
		req.Tools = append(req.Tools, responsesTool{
			Type: "function", Name: tool.Name, Description: tool.Description, Parameters: tool.Schema, Strict: tool.Strict,
		})
	}
	if params.OutputTool != nil {
		req.Tools = append(req.Tools, responsesTool{
			Type: "function", Name: params.OutputTool.Name, Description: params.OutputTool.Description,
			Parameters: params.OutputTool.Schema, Strict: params.OutputTool.Strict,
		})
		if !params.AllowText {
			req.ToolChoice = "required"
		}
	}
	if len(req.Tools) > 0 {
		req.ParallelToolCalls = params.Settings.ParallelToolCalls
	}
	if params.OutputSchema != nil {
		return nil, fmt.Errorf("openai: the Responses model does not support native JSON output mode yet; use OutputModeTool")
	}
	return req, nil
}

func convertResponsesMessage(msg ai.ModelMessage) ([]responsesInput, error) {
	switch m := msg.(type) {
	case ai.ModelRequest:
		return convertResponsesRequest(m)
	case ai.ModelResponse:
		return convertResponsesResponse(m), nil
	default:
		return nil, fmt.Errorf("openai: unknown message type %T", msg)
	}
}

func convertResponsesRequest(m ai.ModelRequest) ([]responsesInput, error) {
	var out []responsesInput
	for _, p := range m.Parts {
		switch part := p.(type) {
		case ai.SystemPromptPart:
			out = append(out, responsesInput{Role: "system", Content: part.Content})
		case ai.UserPromptPart:
			out = append(out, responsesInput{Role: "user", Content: part.Content})
		case ai.ToolReturnPart:
			content, err := contentString(part.Content)
			if err != nil {
				return nil, err
			}
			out = append(out, responsesInput{Type: "function_call_output", CallID: part.ToolCallID, Output: content})
		case ai.RetryPromptPart:
			if part.ToolCallID != "" {
				out = append(out, responsesInput{Type: "function_call_output", CallID: part.ToolCallID, Output: part.Content})
			} else {
				out = append(out, responsesInput{Role: "user", Content: part.Content})
			}
		default:
			return nil, fmt.Errorf("openai: unknown request part type %T", p)
		}
	}
	return out, nil
}

func convertResponsesResponse(m ai.ModelResponse) []responsesInput {
	var out []responsesInput
	for _, p := range m.Parts {
		switch part := p.(type) {
		case ai.TextPart:
			out = append(out, responsesInput{Role: "assistant", Content: part.Content})
		case ai.ToolCallPart:
			out = append(out, responsesInput{Type: "function_call", CallID: part.ToolCallID, Name: part.ToolName, Arguments: string(part.Args)})
		}
	}
	return out
}

type responsesResponse struct {
	Model  string `json:"model"`
	Output []struct {
		Type    string `json:"type"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		CallID    string `json:"call_id"`
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
		Summary   []struct {
			Text string `json:"text"`
		} `json:"summary"`
	} `json:"output"`
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

func parseResponsesResponse(data []byte) (*ai.ModelResponse, error) {
	var rr responsesResponse
	if err := json.Unmarshal(data, &rr); err != nil {
		return nil, fmt.Errorf("openai: parse response: %w", err)
	}
	resp := &ai.ModelResponse{
		ModelName: rr.Model,
		Usage: ai.Usage{
			Requests:     1,
			InputTokens:  rr.Usage.InputTokens,
			OutputTokens: rr.Usage.OutputTokens,
		},
	}
	for _, item := range rr.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				if c.Type == "output_text" {
					resp.Parts = append(resp.Parts, ai.TextPart{Content: c.Text})
				}
			}
		case "function_call":
			resp.Parts = append(resp.Parts, ai.ToolCallPart{
				ToolName:   item.Name,
				Args:       json.RawMessage(item.Arguments),
				ToolCallID: item.CallID,
			})
		case "reasoning":
			for _, s := range item.Summary {
				resp.Parts = append(resp.Parts, ai.ThinkingPart{Content: s.Text})
			}
		}
	}
	return resp, nil
}

var _ ai.Model = (*ResponsesModel)(nil)
