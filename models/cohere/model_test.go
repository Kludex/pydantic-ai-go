package cohere_test

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/cohere"
)

func TestSettings(t *testing.T) {
	settings, err := (cohere.Settings{}).Build()
	if err != nil || settings.ExtraBody != nil {
		t.Fatalf("unexpected zero settings: %#v %v", settings, err)
	}
	topK := 10
	settings, err = (cohere.Settings{
		Common: ai.ModelSettings{ExtraBody: map[string]any{"value": "kept"}}, TopK: &topK,
	}).Build()
	if err != nil || settings.ExtraBody["k"] != 10 || settings.ExtraBody["value"] != "kept" {
		t.Fatalf("unexpected settings: %#v %v", settings, err)
	}
	topK = -1
	if _, err := (cohere.Settings{TopK: &topK}).Build(); err == nil || !strings.Contains(err.Error(), "non-negative") {
		t.Fatalf("unexpected top-k error: %v", err)
	}
	topK = 1
	if _, err := (cohere.Settings{
		Common: ai.ModelSettings{ExtraBody: map[string]any{"k": 2}}, TopK: &topK,
	}).Build(); err == nil || !strings.Contains(err.Error(), "conflicts") {
		t.Fatalf("unexpected conflict: %v", err)
	}
	settings, err = (cohere.Settings{TopK: &topK}).Build()
	if err != nil || settings.ExtraBody["k"] != 1 {
		t.Fatalf("unexpected top-k-only settings: %#v %v", settings, err)
	}
}

func TestModelRequest(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v2/chat" || request.Header.Get("Authorization") != "Bearer key" ||
			request.Header.Get("X-Provider") != "provider" || request.Header.Get("X-Dynamic") != "dynamic" ||
			request.Header.Get("X-Request") != "request" {
			t.Errorf("unexpected request: %s %#v", request.URL, request.Header)
		}
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		response.Header().Set("X-Response", "value")
		_, _ = io.WriteString(response, `{"id":"response-id","finish_reason":"TOOL_CALL","message":{"content":[`+
			`{"type":"thinking","thinking":"reason"},{"type":"text","text":"answer"},{"type":"unknown"}],`+
			`"tool_calls":[{"id":"","type":"function","function":{"name":"lookup","arguments":"{}"}},`+
			`{"id":"ignored","type":"function"}]},"usage":{"billed_units":{"input_tokens":3,"output_tokens":4,`+
			`"search_units":5,"classifications":6},"tokens":{"input_tokens":10,"output_tokens":20},"cached_tokens":7}}`)
	}))
	defer server.Close()
	temperature := 0.2
	topP := 0.8
	seed := 42
	presence := 0.1
	frequency := 0.3
	model := cohere.NewModel("command-r7b-12-2024", cohere.WithProvider(cohere.ProviderConfig{
		Name: "cohere-gateway", BaseURL: server.URL, APIKey: "key", HTTPClient: server.Client(),
		Headers: http.Header{"X-Provider": {"provider"}},
		PrepareRequest: func(request *http.Request) error {
			request.Header.Set("X-Dynamic", "dynamic")
			return nil
		},
	}), cohere.WithDefaultSettings(ai.ModelSettings{MaxTokens: 99}))
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "history system"},
			ai.UserPromptPart{Contents: []ai.UserContent{
				ai.TextContent{Text: "question"}, ai.CachePoint{},
			}},
		}},
		ai.ModelResponse{},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ThinkingPart{Content: "earlier reason"}, ai.TextPart{Content: "earlier answer"},
			ai.ToolCallPart{ToolName: "lookup", ToolCallID: "call", Args: json.RawMessage(`{"x":1}`)},
			ai.NativeToolCallPart{}, ai.NativeToolReturnPart{}, ai.FilePart{}, ai.CompactionPart{},
		}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "lookup", ToolCallID: "call", Content: ai.ToolReturn{ReturnValue: map[string]any{"ok": true}}},
			ai.RetryPromptPart{Content: "retry output"},
			ai.RetryPromptPart{Content: "retry tool", ToolName: "lookup", ToolCallID: "call"},
		}},
	}
	strict := true
	params := ai.ModelRequestParams{
		Instructions:     "fallback instructions",
		InstructionParts: []ai.InstructionPart{{Content: "static"}, {Content: "dynamic", Dynamic: true}},
		Tools:            []ai.ToolDefinition{{Name: "lookup", Description: "Look up", Schema: map[string]any{"type": "object"}, Strict: &strict}},
		OutputTool:       &ai.ToolDefinition{Name: "final", Schema: map[string]any{"type": "object"}},
		Settings: ai.ModelSettings{
			MaxTokens: 100, Temperature: &temperature, TopP: &topP, Seed: &seed,
			PresencePenalty: &presence, FrequencyPenalty: &frequency, StopSequences: []string{"stop"},
			ExtraHeaders: map[string]string{"X-Request": "request"}, ExtraBody: map[string]any{"k": 10},
		},
	}
	result, err := model.Request(t.Context(), messages, params)
	if err != nil {
		t.Fatal(err)
	}
	if result.ProviderName != "cohere-gateway" || result.ProviderURL != server.URL ||
		result.ProviderResponseID != "response-id" || result.FinishReason != ai.FinishReasonToolCall ||
		result.Parts[0].(ai.ThinkingPart).Content != "reason" || result.Parts[1].(ai.TextPart).Content != "answer" ||
		!strings.HasPrefix(result.Parts[2].(ai.ToolCallPart).ToolCallID, "cohere-") || result.Usage.InputTokens != 10 ||
		result.Usage.OutputTokens != 20 || result.Usage.CacheReadTokens != 7 ||
		result.Usage.Details["classifications"] != 6 {
		t.Fatalf("unexpected result: %#v", result)
	}
	if body["model"] != "command-r7b-12-2024" || body["stream"] != false || body["max_tokens"] != float64(100) ||
		body["p"] != 0.8 || body["k"] != float64(10) || body["tool_choice"] != "REQUIRED" {
		t.Fatalf("unexpected body: %#v", body)
	}
	wireMessages := body["messages"].([]any)
	if len(wireMessages) != 8 || wireMessages[1].(map[string]any)["content"] != "static" ||
		wireMessages[2].(map[string]any)["content"] != "dynamic" {
		t.Fatalf("unexpected messages: %#v", wireMessages)
	}
	assistant := wireMessages[4].(map[string]any)
	if len(assistant["content"].([]any)) != 2 || len(assistant["tool_calls"].([]any)) != 1 {
		t.Fatalf("unexpected assistant history: %#v", assistant)
	}
	if len(body["tools"].([]any)) != 2 {
		t.Fatalf("unexpected tools: %#v", body["tools"])
	}
	if model.Name() != "command-r7b-12-2024" || model.ProviderName() != "cohere-gateway" ||
		model.ProviderURL() != server.URL || model.DefaultModelSettings().MaxTokens != 99 {
		t.Fatalf("unexpected model identity/defaults: %#v", model)
	}
}

func TestProviderOptions(t *testing.T) {
	t.Setenv("CO_API_KEY", "environment-key")
	t.Setenv("CO_BASE_URL", "")
	provider := cohere.NewProviderConfig()
	if provider.Name != "cohere" || provider.BaseURL != "https://api.cohere.com" || provider.APIKey != "environment-key" {
		t.Fatalf("unexpected provider: %#v", provider)
	}
	t.Setenv("CO_BASE_URL", "https://cohere.example/")
	if provider := cohere.NewProviderConfig(); provider.BaseURL != "https://cohere.example" {
		t.Fatalf("unexpected custom base URL: %#v", provider)
	}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		_, _ = io.WriteString(response, `{"finish_reason":"COMPLETE","message":{"content":[{"type":"text","text":"ok"}]}}`)
	}))
	defer server.Close()
	model := cohere.NewModel("model", cohere.WithBaseURL(server.URL+"/"), cohere.WithAPIKey(""),
		cohere.WithHTTPClient(server.Client()))
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Content: "question"},
		ai.ToolReturnPart{ToolName: "one", ToolCallID: "one", Content: "text"},
		ai.ToolReturnPart{ToolName: "two", ToolCallID: "two", Content: &ai.ToolReturn{ReturnValue: "pointer"}},
	}}}
	result, err := model.Request(t.Context(), messages, ai.ModelRequestParams{Instructions: "instruction"})
	if err != nil || result.Text() != "ok" || result.FinishReason != ai.FinishReasonStop {
		t.Fatalf("unexpected result: %#v %v", result, err)
	}
}

func TestMappingFailures(t *testing.T) {
	model := cohere.NewModel("model")
	tests := []struct {
		name     string
		messages []ai.ModelMessage
		params   ai.ModelRequestParams
		match    string
	}{
		{name: "native tool", params: ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.WebSearchTool{}}}, match: "native tools"},
		{name: "native output", params: ai.ModelRequestParams{
			OutputMode: ai.OutputModeNative, OutputSchema: map[string]any{"type": "object"},
		}, match: "native structured"},
		{name: "multimodal", messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.UserPromptPart{Contents: []ai.UserContent{ai.ImageURL{URL: "https://example.com/image.png"}}},
		}}}, match: "multimodal"},
		{name: "tool return marshal", messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "tool", ToolCallID: "call", Content: make(chan int)},
		}}}, match: "marshal tool result"},
		{name: "tool return ID", messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "tool", Content: "result"},
		}}}, match: "requires a tool call ID"},
		{name: "retry ID", messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.RetryPromptPart{ToolName: "tool", Content: "retry"},
		}}}, match: "requires a tool call ID"},
		{name: "availability", messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"tool"}},
		}}}, match: "must be synthesized"},
		{name: "request speech", messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SpeechPart{Speaker: ai.SpeechSpeakerUser},
		}}}, match: ai.ErrUnpreparedSpeech.Error()},
		{name: "tool call ID", messages: []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "tool", Args: json.RawMessage(`{}`)},
		}}}, match: "requires an ID"},
		{name: "response speech", messages: []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.SpeechPart{Speaker: ai.SpeechSpeakerAssistant},
		}}}, match: ai.ErrUnpreparedSpeech.Error()},
	}
	for _, test := range tests {
		_, err := model.Request(t.Context(), test.messages, test.params)
		if err == nil || !strings.Contains(err.Error(), test.match) {
			t.Fatalf("name=%s: unexpected error: %v", test.name, err)
		}
	}
}

func TestInvalidProviderPanics(t *testing.T) {
	for _, provider := range []cohere.ProviderConfig{
		{BaseURL: "https://example.com/v2"},
		{Name: "cohere"},
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			_ = cohere.WithProvider(provider)
		}()
	}
}

func TestRequestFailures(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		switch request.Header.Get("X-Case") {
		case "api":
			response.Header().Set("X-Error", "value")
			response.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(response, "limited")
		case "json":
			_, _ = io.WriteString(response, "not-json")
		}
	}))
	defer server.Close()
	model := cohere.NewModel("model", cohere.WithBaseURL(server.URL), cohere.WithHTTPClient(server.Client()))
	_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{ExtraHeaders: map[string]string{"X-Case": "api"}}})
	var apiError *cohere.APIError
	if !errors.As(err, &apiError) || apiError.StatusCode != http.StatusTooManyRequests || apiError.Headers.Get("X-Error") != "value" {
		t.Fatalf("unexpected API error: %T %v", err, err)
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{ExtraHeaders: map[string]string{"X-Case": "json"}}})
	if err == nil || !strings.Contains(err.Error(), "decode response") {
		t.Fatalf("unexpected JSON error: %v", err)
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{ExtraBody: map[string]any{"model": "other"}}})
	if err == nil || !strings.Contains(err.Error(), "conflicts") {
		t.Fatalf("unexpected body conflict: %v", err)
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{ExtraBody: map[string]any{"bad": make(chan int)}}})
	if err == nil || !strings.Contains(err.Error(), "unsupported type") {
		t.Fatalf("unexpected marshal error: %v", err)
	}

	prepareFailure := errors.New("prepare failed")
	model = cohere.NewModel("model", cohere.WithProvider(cohere.ProviderConfig{
		Name: "cohere", BaseURL: server.URL, HTTPClient: server.Client(),
		PrepareRequest: func(*http.Request) error { return prepareFailure },
	}))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, prepareFailure) {
		t.Fatalf("unexpected prepare error: %v", err)
	}
	model = cohere.NewModel("model", cohere.WithBaseURL("://invalid"))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected URL error")
	}
	model = cohere.NewModel("model", cohere.WithHTTPClient(&http.Client{Transport: roundTripFunc(
		func(*http.Request) (*http.Response, error) { return nil, errors.New("transport failed") },
	)}))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil ||
		!strings.Contains(err.Error(), "transport failed") {
		t.Fatalf("unexpected transport error: %v", err)
	}
	model = cohere.NewModel("model", cohere.WithHTTPClient(&http.Client{Transport: roundTripFunc(
		func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: errorBody{}}, nil
		},
	)}))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil ||
		!strings.Contains(err.Error(), "read failed") {
		t.Fatalf("unexpected read error: %v", err)
	}
	_, err = cohere.NewModel("model").Request(t.Context(), nil, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{Name: "tool", Schema: map[string]any{"bad": make(chan int)}}},
	})
	if err == nil || !strings.Contains(err.Error(), "marshal request") {
		t.Fatalf("unexpected payload marshal error: %v", err)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

type errorBody struct{}

func (errorBody) Read([]byte) (int, error) { return 0, errors.New("read failed") }
func (errorBody) Close() error             { return nil }
