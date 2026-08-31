package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
)

// ResponsesModel calls the OpenAI Responses API, the successor to Chat
// Completions. Create one with NewResponsesModel.
type ResponsesModel struct {
	name              string
	apiKey            string
	baseURL           string
	httpClient        *http.Client
	strictToolSupport bool
	defaultSettings   ai.ModelSettings
}

// NewResponsesModel creates a ResponsesModel for the named OpenAI model.
// It accepts the same options as NewModel.
func NewResponsesModel(name string, opts ...Option) *ResponsesModel {
	m := NewModel(name, opts...)
	return &ResponsesModel{
		name: m.name, apiKey: m.apiKey, baseURL: m.baseURL, httpClient: m.httpClient,
		strictToolSupport: m.strictToolSupport, defaultSettings: m.defaultSettings,
	}
}

// Name returns the model name.
func (m *ResponsesModel) Name() string { return m.name }

// DefaultModelSettings returns this model's request defaults.
func (m *ResponsesModel) DefaultModelSettings() ai.ModelSettings { return m.defaultSettings.Clone() }

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
	response, err := parseResponsesResponse(data)
	if response != nil {
		response.ProviderName = "openai"
		response.ProviderURL = m.baseURL
	}
	return response, err
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
	Type             string `json:"type,omitempty"`
	ID               string `json:"id,omitempty"`
	CallID           string `json:"call_id,omitempty"`
	Name             string `json:"name,omitempty"`
	Arguments        string `json:"arguments,omitempty"`
	Namespace        string `json:"namespace,omitempty"`
	Output           string `json:"output,omitempty"`
	EncryptedContent string `json:"encrypted_content,omitempty"`
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
		schema, strict, err := prepareOpenAITool(tool, m.strictToolSupport)
		if err != nil {
			return nil, err
		}
		req.Tools = append(req.Tools, responsesTool{
			Type: "function", Name: tool.Name, Description: tool.Description, Parameters: schema, Strict: strict,
		})
	}
	if params.OutputTool != nil {
		schema, strict, err := prepareOpenAITool(*params.OutputTool, m.strictToolSupport)
		if err != nil {
			return nil, err
		}
		req.Tools = append(req.Tools, responsesTool{
			Type: "function", Name: params.OutputTool.Name, Description: params.OutputTool.Description,
			Parameters: schema, Strict: strict,
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
		case ai.ToolAvailabilityDeltaPart:
		case ai.RetryPromptPart:
			content := part.ModelResponse()
			if part.ToolCallID != "" {
				out = append(out, responsesInput{Type: "function_call_output", CallID: part.ToolCallID, Output: content})
			} else {
				out = append(out, responsesInput{Role: "user", Content: content})
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
			id := ""
			if part.ProviderName == "" || part.ProviderName == "openai" {
				id = part.ID
			}
			out = append(out, responsesInput{Role: "assistant", Content: part.Content, ID: id})
		case ai.ThinkingPart:
			if (part.ProviderName == "" || part.ProviderName == "openai") &&
				(part.ID != "" || part.Signature != "") {
				out = append(out, responsesInput{
					Type: "reasoning", ID: part.ID, EncryptedContent: part.Signature,
				})
			}
		case ai.ToolCallPart:
			id := ""
			if part.ProviderName == "" || part.ProviderName == "openai" {
				id = part.ID
			}
			namespace := ""
			if part.ProviderName == "" || part.ProviderName == "openai" {
				namespace, _ = part.ProviderDetails["namespace"].(string)
			}
			out = append(out, responsesInput{
				Type: "function_call", ID: id, CallID: part.ToolCallID,
				Name: part.ToolName, Arguments: string(part.Args), Namespace: namespace,
			})
		}
	}
	return out
}

type incompleteDetails struct {
	Reason string `json:"reason"`
}

type responsesResponse struct {
	ID                string             `json:"id"`
	Model             string             `json:"model"`
	CreatedAt         float64            `json:"created_at"`
	Status            string             `json:"status"`
	Background        bool               `json:"background"`
	IncompleteDetails *incompleteDetails `json:"incomplete_details"`
	Output            []struct {
		ID      string `json:"id"`
		Type    string `json:"type"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		CallID           string `json:"call_id"`
		Name             string `json:"name"`
		Arguments        string `json:"arguments"`
		Namespace        string `json:"namespace"`
		EncryptedContent string `json:"encrypted_content"`
		Summary          []struct {
			Text string `json:"text"`
		} `json:"summary"`
	} `json:"output"`
	Usage responsesUsage `json:"usage"`
}

type responsesUsage struct {
	InputTokens        int `json:"input_tokens"`
	OutputTokens       int `json:"output_tokens"`
	InputTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"input_tokens_details"`
	OutputTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
}

func (u responsesUsage) usage() ai.Usage {
	return ai.Usage{
		Requests: 1, InputTokens: u.InputTokens, OutputTokens: u.OutputTokens,
		CacheReadTokens: u.InputTokensDetails.CachedTokens,
		ReasoningTokens: u.OutputTokensDetails.ReasoningTokens,
		Details:         map[string]int{"reasoning_tokens": u.OutputTokensDetails.ReasoningTokens},
	}
}

func openAIResponsesFinishReason(reason string) ai.FinishReason {
	return map[string]ai.FinishReason{
		"completed": ai.FinishReasonStop, "max_output_tokens": ai.FinishReasonLength,
		"content_filter": ai.FinishReasonContentFilter, "cancelled": ai.FinishReasonError,
		"failed": ai.FinishReasonError,
	}[reason]
}

func openAIResponsesState(status string, background bool) ai.ModelResponseState {
	switch status {
	case "queued", "in_progress":
		if background {
			return ai.ModelResponseStateSuspended
		}
		return ai.ModelResponseStateIncomplete
	default:
		return ai.ModelResponseStateComplete
	}
}

func responsesMetadata(
	status string, details *incompleteDetails, createdAt float64, background bool,
) (string, map[string]any, time.Time, ai.ModelResponseState) {
	rawFinishReason := status
	if details != nil {
		rawFinishReason = details.Reason
	}
	providerDetails := map[string]any{}
	if rawFinishReason != "" {
		providerDetails["finish_reason"] = rawFinishReason
	}
	var timestamp time.Time
	if createdAt != 0 {
		seconds, fraction := math.Modf(createdAt)
		timestamp = time.Unix(int64(seconds), int64(fraction*float64(time.Second))).UTC()
		providerDetails["timestamp"] = timestamp
	}
	if background {
		providerDetails["background"] = true
	}
	if len(providerDetails) == 0 {
		providerDetails = nil
	}
	return rawFinishReason, providerDetails, timestamp, openAIResponsesState(status, background)
}

func parseResponsesResponse(data []byte) (*ai.ModelResponse, error) {
	var rr responsesResponse
	if err := json.Unmarshal(data, &rr); err != nil {
		return nil, fmt.Errorf("openai: parse response: %w", err)
	}
	rawFinishReason, providerDetails, _, state := responsesMetadata(
		rr.Status, rr.IncompleteDetails, rr.CreatedAt, rr.Background,
	)
	resp := &ai.ModelResponse{
		ModelName: rr.Model, Usage: rr.Usage.usage(), ProviderDetails: providerDetails,
		ProviderResponseID: rr.ID, FinishReason: openAIResponsesFinishReason(rawFinishReason), State: state,
	}
	for _, item := range rr.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				if c.Type == "output_text" {
					resp.Parts = append(resp.Parts, ai.TextPart{
						Content: c.Text, ID: item.ID, ProviderName: "openai",
					})
				}
			}
		case "function_call":
			var providerDetails map[string]any
			if item.Namespace != "" {
				providerDetails = map[string]any{"namespace": item.Namespace}
			}
			resp.Parts = append(resp.Parts, ai.ToolCallPart{
				ToolName: item.Name, Args: json.RawMessage(item.Arguments), ToolCallID: item.CallID,
				ID: item.ID, ProviderName: "openai", ProviderDetails: providerDetails,
			})
		case "reasoning":
			if len(item.Summary) == 0 && item.EncryptedContent != "" {
				resp.Parts = append(resp.Parts, ai.ThinkingPart{
					ID: item.ID, Signature: item.EncryptedContent, ProviderName: "openai",
				})
			}
			for index, s := range item.Summary {
				signature := ""
				if index == 0 {
					signature = item.EncryptedContent
				}
				resp.Parts = append(resp.Parts, ai.ThinkingPart{
					Content: s.Text, ID: item.ID, Signature: signature, ProviderName: "openai",
				})
			}
		}
	}
	return resp, nil
}

var _ ai.Model = (*ResponsesModel)(nil)
