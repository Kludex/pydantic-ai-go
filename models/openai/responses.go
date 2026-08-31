package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
)

// ResponsesModel calls the OpenAI Responses API, the successor to Chat
// Completions. Create one with NewResponsesModel.
type ResponsesModel struct {
	name                   string
	apiKey                 string
	baseURL                string
	httpClient             *http.Client
	strictToolSupport      bool
	deferredToolSupport    bool
	defaultSettings        ai.ModelSettings
	background             *bool
	backgroundPollInterval time.Duration
}

// NewResponsesModel creates a ResponsesModel for the named OpenAI model.
// It accepts the same options as NewModel.
func NewResponsesModel(name string, opts ...Option) *ResponsesModel {
	m := NewModel(name, opts...)
	return &ResponsesModel{
		name: m.name, apiKey: m.apiKey, baseURL: m.baseURL, httpClient: m.httpClient,
		strictToolSupport: m.strictToolSupport, deferredToolSupport: m.deferredToolSupport,
		defaultSettings: m.defaultSettings, background: m.background,
		backgroundPollInterval: m.backgroundPollInterval,
	}
}

// Name returns the model name.
func (m *ResponsesModel) Name() string { return m.name }

// DefaultModelSettings returns this model's request defaults.
func (m *ResponsesModel) DefaultModelSettings() ai.ModelSettings { return m.defaultSettings.Clone() }

// Request implements ai.Model.
func (m *ResponsesModel) Request(ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
	if responseID, ok := suspendedResponsesID(msgs); ok {
		return m.retrieveResponse(ctx, responseID, params.Settings.ExtraHeaders)
	}
	payload, err := m.buildResponsesPayload(msgs, params, true)
	if err != nil {
		return nil, err
	}
	body, err := marshalRequest(payload, params.Settings.ExtraBody)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/responses", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.apiKey)
	setExtraHeaders(req, params.Settings.ExtraHeaders)

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

// ContinuationDelay implements ai.ModelContinuationDelayer.
func (m *ResponsesModel) ContinuationDelay(response ai.ModelResponse) time.Duration {
	if response.State == ai.ModelResponseStateSuspended && providerBool(response.ProviderDetails, "background") {
		return m.backgroundPollInterval
	}
	return 0
}

// CancelSuspendedResponse implements ai.SuspendedResponseCanceler.
func (m *ResponsesModel) CancelSuspendedResponse(ctx context.Context, response ai.ModelResponse) error {
	if response.ProviderName != "openai" || response.ProviderResponseID == "" ||
		!providerBool(response.ProviderDetails, "background") {
		return nil
	}
	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost,
		m.baseURL+"/responses/"+url.PathEscape(response.ProviderResponseID)+"/cancel", nil,
	)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+m.apiKey)
	resp, err := m.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("openai: cancel background response: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("openai: read cancel response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return &APIError{StatusCode: resp.StatusCode, Body: string(data)}
	}
	return nil
}

func (m *ResponsesModel) retrieveResponse(
	ctx context.Context, responseID string, headers map[string]string,
) (*ai.ModelResponse, error) {
	req, err := http.NewRequestWithContext(
		ctx, http.MethodGet, m.baseURL+"/responses/"+url.PathEscape(responseID), nil,
	)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+m.apiKey)
	setExtraHeaders(req, headers)
	resp, err := m.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai: retrieve background response: %w", err)
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

func suspendedResponsesID(messages []ai.ModelMessage) (string, bool) {
	if len(messages) == 0 {
		return "", false
	}
	response, ok := messages[len(messages)-1].(ai.ModelResponse)
	if !ok || response.State != ai.ModelResponseStateSuspended || response.ProviderName != "openai" ||
		response.ProviderResponseID == "" || !providerBool(response.ProviderDetails, "background") {
		return "", false
	}
	return response.ProviderResponseID, true
}

func providerBool(details map[string]any, key string) bool {
	value, _ := details[key].(bool)
	return value
}

type responsesRequest struct {
	Model             string              `json:"model"`
	Instructions      string              `json:"instructions,omitempty"`
	Input             []responsesInput    `json:"input"`
	Tools             []responsesTool     `json:"tools,omitempty"`
	ToolChoice        any                 `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool               `json:"parallel_tool_calls,omitempty"`
	MaxTokens         int                 `json:"max_output_tokens,omitempty"`
	Temperature       *float64            `json:"temperature,omitempty"`
	TopP              *float64            `json:"top_p,omitempty"`
	Stream            bool                `json:"stream,omitempty"`
	Background        *bool               `json:"background,omitempty"`
	Reasoning         *responsesReasoning `json:"reasoning,omitempty"`
	TopLogprobs       *int                `json:"top_logprobs,omitempty"`
	Include           []string            `json:"include,omitempty"`
	ServiceTier       ai.ServiceTier      `json:"service_tier,omitempty"`
}

type responsesReasoning struct {
	Effort string `json:"effort,omitempty"`
}

type responsesInput struct {
	// message
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
	// function_call and function_call_output items
	Type             string          `json:"type,omitempty"`
	ID               string          `json:"id,omitempty"`
	CallID           string          `json:"call_id,omitempty"`
	Name             string          `json:"name,omitempty"`
	Arguments        any             `json:"arguments,omitempty"`
	Namespace        string          `json:"namespace,omitempty"`
	Output           string          `json:"output,omitempty"`
	Execution        string          `json:"execution,omitempty"`
	Status           string          `json:"status,omitempty"`
	Tools            []responsesTool `json:"tools,omitempty"`
	EncryptedContent string          `json:"encrypted_content,omitempty"`
}

type responsesTool struct {
	Type         string         `json:"type"`
	Name         string         `json:"name,omitempty"`
	Description  string         `json:"description,omitempty"`
	Parameters   map[string]any `json:"parameters,omitempty"`
	Strict       *bool          `json:"strict,omitempty"`
	DeferLoading bool           `json:"defer_loading,omitempty"`
	Execution    string         `json:"execution,omitempty"`
}

func (m *ResponsesModel) buildResponsesPayload(
	msgs []ai.ModelMessage, params ai.ModelRequestParams, nativeDeferred bool,
) (*responsesRequest, error) {
	reasoningEffort, err := openAIThinkingEffort(params.Settings.Thinking)
	if err != nil {
		return nil, err
	}
	serviceTier, err := openAIServiceTier(params.Settings.ServiceTier)
	if err != nil {
		return nil, err
	}
	req := &responsesRequest{
		Model:        m.name,
		Instructions: params.Instructions,
		MaxTokens:    params.Settings.MaxTokens,
		Temperature:  params.Settings.Temperature,
		TopP:         params.Settings.TopP,
		Background:   m.background,
		TopLogprobs:  params.Settings.TopLogprobs,
		ServiceTier:  serviceTier,
	}
	if params.Settings.Logprobs != nil && *params.Settings.Logprobs {
		req.Include = append(req.Include, "message.output_text.logprobs")
	}
	if reasoningEffort != "" {
		req.Reasoning = &responsesReasoning{Effort: reasoningEffort}
	}
	if openAIReasoningActive(reasoningEffort) {
		req.Temperature = nil
		req.TopP = nil
	}
	var searchTool *ai.ToolDefinition
	if nativeDeferred && m.deferredToolSupport && len(params.DeferredTools) > 0 {
		for _, tool := range params.Tools {
			if tool.ToolKind == ai.ToolPartKindToolSearch && tool.Name == ai.ToolSearchName {
				definition := tool
				searchTool = &definition
				break
			}
		}
	}
	deferred := make(map[string]ai.ToolDefinition, len(params.DeferredTools))
	if searchTool != nil {
		for _, tool := range params.DeferredTools {
			deferred[tool.Name] = tool
		}
	}
	converter := responsesMessageConverter{
		clientToolSearch: searchTool != nil,
		deferred:         deferred,
		rendered:         make(map[string]struct{}),
		strictSupport:    m.strictToolSupport,
	}
	for _, msg := range msgs {
		items, err := converter.convert(msg)
		if err != nil {
			return nil, err
		}
		req.Input = append(req.Input, items...)
	}
	for _, tool := range params.Tools {
		if searchTool != nil && (tool.Name == ai.ToolSearchName || tool.DeferLoading) {
			continue
		}
		converted, err := prepareResponsesFunctionTool(tool, m.strictToolSupport)
		if err != nil {
			return nil, err
		}
		req.Tools = append(req.Tools, converted)
	}
	if searchTool != nil {
		for _, tool := range params.DeferredTools {
			converted, err := prepareResponsesFunctionTool(tool, m.strictToolSupport)
			if err != nil {
				return nil, err
			}
			converted.DeferLoading = true
			req.Tools = append(req.Tools, converted)
		}
	}
	if params.OutputTool != nil {
		converted, err := prepareResponsesFunctionTool(*params.OutputTool, m.strictToolSupport)
		if err != nil {
			return nil, err
		}
		req.Tools = append(req.Tools, converted)
		if !params.AllowText {
			req.ToolChoice = "required"
		}
	}
	if searchTool != nil {
		schema, _, err := prepareOpenAITool(*searchTool, m.strictToolSupport)
		if err != nil {
			return nil, err
		}
		req.Tools = append(req.Tools, responsesTool{
			Type: "tool_search", Description: searchTool.Description, Parameters: schema, Execution: "client",
		})
	}
	if len(req.Tools) > 0 {
		req.ParallelToolCalls = params.Settings.ParallelToolCalls
	}
	if params.OutputSchema != nil && params.OutputMode != ai.OutputModePrompted {
		return nil, fmt.Errorf("openai: the Responses model does not support native JSON output mode yet; use OutputModeTool")
	}
	return req, nil
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
	ServiceTier       string             `json:"service_tier"`
	IncompleteDetails *incompleteDetails `json:"incomplete_details"`
	Error             *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
	Output []struct {
		ID      string `json:"id"`
		Type    string `json:"type"`
		Content []struct {
			Type     string           `json:"type"`
			Text     string           `json:"text"`
			Logprobs []map[string]any `json:"logprobs"`
		} `json:"content"`
		CallID           string          `json:"call_id"`
		Name             string          `json:"name"`
		Arguments        json.RawMessage `json:"arguments"`
		Namespace        string          `json:"namespace"`
		Execution        string          `json:"execution"`
		Status           string          `json:"status"`
		EncryptedContent string          `json:"encrypted_content"`
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
	return modelResponseFromResponses(rr)
}

func modelResponseFromResponses(rr responsesResponse) (*ai.ModelResponse, error) {
	rawFinishReason, providerDetails, timestamp, state := responsesMetadata(
		rr.Status, rr.IncompleteDetails, rr.CreatedAt, rr.Background,
	)
	if rr.ServiceTier != "" {
		if providerDetails == nil {
			providerDetails = map[string]any{}
		}
		providerDetails["service_tier"] = rr.ServiceTier
	}
	resp := &ai.ModelResponse{
		ModelName: rr.Model, Usage: rr.Usage.usage(), Timestamp: timestamp, ProviderDetails: providerDetails,
		ProviderResponseID: rr.ID, FinishReason: openAIResponsesFinishReason(rawFinishReason), State: state,
	}
	for _, item := range rr.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				if c.Type == "output_text" {
					var details map[string]any
					if c.Logprobs != nil {
						details = map[string]any{"logprobs": c.Logprobs}
					}
					resp.Parts = append(resp.Parts, ai.TextPart{
						Content: c.Text, ID: item.ID, ProviderName: "openai", ProviderDetails: details,
					})
				}
			}
		case "function_call":
			arguments, err := normalizeResponsesArguments(item.Arguments)
			if err != nil {
				return nil, err
			}
			var providerDetails map[string]any
			if item.Namespace != "" {
				providerDetails = map[string]any{"namespace": item.Namespace}
			}
			resp.Parts = append(resp.Parts, ai.ToolCallPart{
				ToolName: item.Name, Args: arguments, ToolCallID: item.CallID,
				ID: item.ID, ProviderName: "openai", ProviderDetails: providerDetails,
			})
		case "tool_search_call":
			if item.Execution != "client" {
				return nil, fmt.Errorf("openai: server-executed tool search is not supported yet")
			}
			arguments, err := normalizeResponsesArguments(item.Arguments)
			if err != nil {
				return nil, err
			}
			callID := item.CallID
			if callID == "" {
				callID = item.ID
			}
			resp.Parts = append(resp.Parts, ai.ToolCallPart{
				ToolName: ai.ToolSearchName, Args: arguments, ToolCallID: callID,
				ToolKind: ai.ToolPartKindToolSearch, ID: item.ID, ProviderName: "openai",
				ProviderDetails: map[string]any{"execution": item.Execution, "status": item.Status},
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

func normalizeResponsesArguments(raw json.RawMessage) (json.RawMessage, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return json.RawMessage(`{}`), nil
	}
	if raw[0] != '"' {
		return append(json.RawMessage(nil), raw...), nil
	}
	var arguments string
	_ = json.Unmarshal(raw, &arguments)
	if !json.Valid([]byte(arguments)) {
		return nil, fmt.Errorf("openai: invalid response arguments %q", arguments)
	}
	return json.RawMessage(arguments), nil
}

var (
	_ ai.Model                     = (*ResponsesModel)(nil)
	_ ai.ModelContinuationDelayer  = (*ResponsesModel)(nil)
	_ ai.SuspendedResponseCanceler = (*ResponsesModel)(nil)
)
