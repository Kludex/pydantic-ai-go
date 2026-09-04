// Package cohere implements generation and shared provider configuration for Cohere.
package cohere

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

const defaultBaseURL = "https://api.cohere.com"

// Settings combines portable settings with Cohere-specific generation settings.
type Settings struct {
	// Common contains portable model settings.
	Common ai.ModelSettings
	// TopK limits token sampling to the K most likely candidates.
	TopK *int
}

// Build returns detached portable settings accepted by agents and direct requests.
func (settings Settings) Build() (ai.ModelSettings, error) {
	common := settings.Common.Clone()
	if settings.TopK == nil {
		return common, nil
	}
	if *settings.TopK < 0 {
		return ai.ModelSettings{}, fmt.Errorf("cohere: top k must be non-negative")
	}
	extra := maps.Clone(common.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	if _, exists := extra["k"]; exists {
		return ai.ModelSettings{}, fmt.Errorf("cohere: extra body field %q conflicts with typed settings", "k")
	}
	extra["k"] = *settings.TopK
	common.ExtraBody = extra
	return common, nil
}

// Model calls Cohere's v2 Chat API.
type Model struct {
	name            string
	providerName    string
	baseURL         string
	apiKey          string
	httpClient      *http.Client
	headers         http.Header
	prepareRequest  RequestPreparationFunc
	defaultSettings ai.ModelSettings
}

// Option configures a Cohere model.
type Option func(*Model)

// WithAPIKey sets the API key. The default is CO_API_KEY.
func WithAPIKey(key string) Option { return func(model *Model) { model.apiKey = key } }

// WithBaseURL points the model at a Cohere-compatible v2 endpoint.
func WithBaseURL(baseURL string) Option {
	return func(model *Model) { model.baseURL = strings.TrimRight(baseURL, "/") }
}

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(model *Model) { model.httpClient = client }
}

// WithProvider configures a Cohere-compatible endpoint.
func WithProvider(provider ProviderConfig) Option {
	if provider.Name == "" {
		panic("cohere: provider name must not be empty")
	}
	if provider.BaseURL == "" {
		panic("cohere: provider base URL must not be empty")
	}
	provider = provider.Clone()
	return func(model *Model) {
		model.providerName = provider.Name
		model.baseURL = strings.TrimRight(provider.BaseURL, "/")
		model.apiKey = provider.APIKey
		if provider.HTTPClient != nil {
			model.httpClient = provider.HTTPClient
		}
		model.headers = provider.Headers.Clone()
		model.prepareRequest = provider.PrepareRequest
	}
}

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(model *Model) { model.defaultSettings = settings.Clone() }
}

// NewProviderConfig returns reusable Cohere endpoint and environment configuration.
func NewProviderConfig() ProviderConfig {
	baseURL := os.Getenv("CO_BASE_URL")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return ProviderConfig{Name: "cohere", BaseURL: strings.TrimRight(baseURL, "/"), APIKey: os.Getenv("CO_API_KEY")}
}

// NewModel creates a Cohere model, such as command-r7b-12-2024.
func NewModel(name string, options ...Option) *Model {
	provider := NewProviderConfig()
	model := &Model{
		name: name, providerName: provider.Name, baseURL: provider.BaseURL, apiKey: provider.APIKey,
		httpClient: http.DefaultClient,
	}
	for _, option := range options {
		option(model)
	}
	return model
}

// Name returns the configured model name.
func (model *Model) Name() string { return model.name }

// ProviderName returns the durable provider identity.
func (model *Model) ProviderName() string { return model.providerName }

// ProviderURL returns the configured Cohere API URL.
func (model *Model) ProviderURL() string { return model.baseURL }

// DefaultModelSettings returns a detached settings snapshot.
func (model *Model) DefaultModelSettings() ai.ModelSettings { return model.defaultSettings.Clone() }

// Request implements ai.Model.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	payload, err := model.buildRequest(messages, params)
	if err != nil {
		return nil, err
	}
	body, err := marshalRequest(payload, params.Settings.ExtraBody)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, model.baseURL+"/v2/chat", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header = model.headers.Clone()
	if request.Header == nil {
		request.Header = make(http.Header)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")
	if model.apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+model.apiKey)
	}
	if model.prepareRequest != nil {
		if err := model.prepareRequest(request); err != nil {
			return nil, err
		}
	}
	for key, value := range params.Settings.ExtraHeaders {
		request.Header.Set(key, value)
	}
	response, err := model.httpClient.Do(request)
	if err != nil {
		return nil, ai.NewModelTransportError(ctx, model, "request", err)
	}
	defer func() { _ = response.Body.Close() }()
	data, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, ai.NewModelTransportError(ctx, model, "read response", err)
	}
	if response.StatusCode != http.StatusOK {
		return nil, &APIError{
			StatusCode: response.StatusCode, Body: string(data), Headers: response.Header.Clone(),
			ProviderName: model.providerName,
		}
	}
	return model.parseResponse(data)
}

func (model *Model) parseResponse(data []byte) (*ai.ModelResponse, error) {
	var response chatResponse
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, fmt.Errorf("cohere: decode response: %w", err)
	}
	parts := make([]ai.ResponsePart, 0, len(response.Message.Content)+len(response.Message.ToolCalls))
	for _, content := range response.Message.Content {
		switch content.Type {
		case "text":
			parts = append(parts, ai.TextPart{Content: content.Text, ProviderName: model.providerName})
		case "thinking":
			parts = append(parts, ai.ThinkingPart{Content: content.Thinking, ProviderName: model.providerName})
		}
	}
	for _, call := range response.Message.ToolCalls {
		if call.Function == nil || call.Function.Name == "" {
			continue
		}
		callID := call.ID
		if callID == "" {
			digest := sha256.Sum256([]byte(call.Function.Name + "\x00" + call.Function.Arguments))
			callID = "cohere-" + hex.EncodeToString(digest[:8])
		}
		parts = append(parts, ai.ToolCallPart{
			ToolName: call.Function.Name, Args: json.RawMessage(call.Function.Arguments),
			ToolCallID: callID, ProviderName: model.providerName,
		})
	}
	providerDetails := map[string]any{"finish_reason": response.FinishReason}
	return &ai.ModelResponse{
		Parts: parts, Usage: response.Usage.normalized(), ModelName: model.name, Timestamp: time.Now().UTC(),
		ProviderName: model.providerName, ProviderURL: model.baseURL, ProviderResponseID: response.ID,
		ProviderDetails: providerDetails, FinishReason: cohereFinishReason(response.FinishReason),
	}, nil
}

func marshalRequest(payload chatRequest, extra map[string]any) ([]byte, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("cohere: marshal request: %w", err)
	}
	if len(extra) == 0 {
		return data, nil
	}
	var object map[string]any
	_ = json.Unmarshal(data, &object)
	for key, value := range extra {
		if _, exists := object[key]; exists {
			return nil, fmt.Errorf("cohere: extra body field %q conflicts with a typed field", key)
		}
		object[key] = value
	}
	return json.Marshal(object)
}

func cohereFinishReason(reason string) ai.FinishReason {
	return map[string]ai.FinishReason{
		"COMPLETE": ai.FinishReasonStop, "STOP_SEQUENCE": ai.FinishReasonStop,
		"MAX_TOKENS": ai.FinishReasonLength, "TOOL_CALL": ai.FinishReasonToolCall,
		"ERROR": ai.FinishReasonError,
	}[reason]
}

type chatResponse struct {
	ID           string `json:"id"`
	FinishReason string `json:"finish_reason"`
	Message      struct {
		Content   []responseContent `json:"content"`
		ToolCalls []cohereToolCall  `json:"tool_calls"`
	} `json:"message"`
	Usage cohereUsage `json:"usage"`
}

type responseContent struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	Thinking string `json:"thinking"`
}

type cohereUsage struct {
	BilledUnits struct {
		InputTokens     int `json:"input_tokens"`
		OutputTokens    int `json:"output_tokens"`
		SearchUnits     int `json:"search_units"`
		Classifications int `json:"classifications"`
	} `json:"billed_units"`
	Tokens struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"tokens"`
	CachedTokens int `json:"cached_tokens"`
}

func (usage cohereUsage) normalized() ai.Usage {
	details := map[string]int{}
	values := []struct {
		name  string
		value int
	}{
		{name: "billed_input_tokens", value: usage.BilledUnits.InputTokens},
		{name: "billed_output_tokens", value: usage.BilledUnits.OutputTokens},
		{name: "search_units", value: usage.BilledUnits.SearchUnits},
		{name: "classifications", value: usage.BilledUnits.Classifications},
	}
	for _, value := range values {
		if value.value != 0 {
			details[value.name] = value.value
		}
	}
	if len(details) == 0 {
		details = nil
	}
	return ai.Usage{
		Requests: 1, InputTokens: usage.Tokens.InputTokens, OutputTokens: usage.Tokens.OutputTokens,
		CacheReadTokens: usage.CachedTokens, Details: details,
	}
}
