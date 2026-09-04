package mistral_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/mistral"
)

func TestRequestValidation(t *testing.T) {
	model, requests := noRequestModel(t)
	settingsPrompt, err := (mistral.Settings{
		PromptCacheKey: "cache", ToolChoice: mistral.ToolChoiceNone,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name     string
		messages []ai.ModelMessage
		params   ai.ModelRequestParams
	}{
		{name: "native tool", params: ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.WebSearchTool{}}}},
		{name: "native output", params: ai.ModelRequestParams{
			OutputMode: ai.OutputModeNative, OutputSchema: map[string]any{"type": "object"},
		}},
		{name: "portable unsupported", params: ai.ModelRequestParams{Settings: ai.ModelSettings{
			LogitBias: map[string]int{"1": 1},
		}}},
		{name: "thinking token budget", params: ai.ModelRequestParams{Settings: ai.ModelSettings{
			Thinking: &ai.ThinkingSettings{TokenBudget: intPointer(10)},
		}}},
		{name: "unknown message", messages: []ai.ModelMessage{nil}},
		{name: "tool result ID", messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "tool", Content: "result"},
		}}}},
		{name: "tool retry ID", messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.RetryPromptPart{ToolName: "tool", Content: "again"},
		}}}},
		{name: "availability", messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"tool"}},
		}}}},
		{name: "request speech", messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SpeechPart{Speaker: ai.SpeechSpeakerUser},
		}}}},
		{name: "response speech", messages: []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.SpeechPart{Speaker: ai.SpeechSpeakerAssistant},
		}}}},
		{name: "tool call ID", messages: []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "tool", Args: []byte("{}")},
		}}}},
		{name: "tool marshal", messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "tool", ToolCallID: "call", Content: make(chan int)},
		}}}},
		{name: "reserved setting type", params: ai.ModelRequestParams{Settings: ai.ModelSettings{
			ExtraBody: map[string]any{"mistral_prompt_cache_key": 1},
		}}},
		{name: "extra marshal", params: ai.ModelRequestParams{Settings: ai.ModelSettings{
			ExtraBody: map[string]any{"channel": make(chan int)},
		}}},
		{name: "extra conflict", params: ai.ModelRequestParams{Settings: ai.ModelSettings{
			ExtraBody: map[string]any{"model": "other"},
		}}},
		{name: "structured tool none override", params: ai.ModelRequestParams{
			Instructions: "return a result",
			OutputTool:   &ai.ToolDefinition{Name: "result", Schema: map[string]any{"type": "object"}},
			Settings:     settingsPrompt,
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := model.Request(t.Context(), test.messages, test.params)
			if test.name == "structured tool none override" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil {
				t.Fatal("expected error")
			}
		})
	}
	filtered, err := (mistral.Settings{
		ToolChoice: mistral.ToolChoiceNone, AllowedTools: []string{"missing"},
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	_, err = model.Request(t.Context(), []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.RetryPromptPart{Content: "retry"},
			ai.RetryPromptPart{ToolName: "tool", ToolCallID: "call", Content: "retry"},
			ai.ToolReturnPart{ToolName: "tool", ToolCallID: "call", Content: "done"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.FilePart{}, ai.CompactionPart{}, ai.NativeToolCallPart{}, ai.NativeToolReturnPart{},
		}},
	}, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{Name: "tool", Schema: map[string]any{"type": "object"}}}, Settings: filtered,
	})
	if err != nil {
		t.Fatal(err)
	}
	if *requests != 2 {
		t.Fatalf("unexpected request count: %d", *requests)
	}
}

func TestSettingsValidation(t *testing.T) {
	empty, err := (mistral.Settings{}).Build()
	if err != nil || empty.ExtraBody != nil {
		t.Fatalf("unexpected empty settings: %#v %v", empty, err)
	}
	tests := []mistral.Settings{
		{ToolChoice: mistral.ToolChoice("bad")},
		{AllowedTools: []string{}},
		{AllowedTools: []string{""}},
		{AllowedTools: []string{"same", "same"}},
		{Common: ai.ModelSettings{ExtraBody: map[string]any{"mistral_tool_choice": "auto"}}, ToolChoice: mistral.ToolChoiceAuto},
	}
	for _, settings := range tests {
		if _, err := settings.Build(); err == nil {
			t.Fatalf("expected settings error for %#v", settings)
		}
	}
	for _, extra := range []map[string]any{
		{"mistral_tool_choice": "auto"},
		{"mistral_tool_choice": mistral.ToolChoice("bad")},
		{"mistral_allowed_tools": []int{1}},
		{"mistral_allowed_tools": []string{}},
	} {
		model, _ := noRequestModel(t)
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
			Settings: ai.ModelSettings{ExtraBody: extra},
		}); err == nil {
			t.Fatalf("expected runtime settings error for %#v", extra)
		}
	}
}

func TestResponseFailures(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{name: "malformed", body: `{`},
		{name: "choices", body: `{"choices":[]}`},
		{name: "content", body: `{"choices":[{"message":{"content":{}},"finish_reason":"stop"}]}`},
		{name: "tool type", body: `{"choices":[{"message":{"content":null,"tool_calls":[{"type":"custom","function":{"name":"x","arguments":{}}}]}}]}`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := responseServer(t, test.body, nil)
			defer server.Close()
			model := mistral.NewModel("model", mistral.WithBaseURL(server.URL), mistral.WithHTTPClient(server.Client()))
			if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
				t.Fatal("expected error")
			}
		})
	}
	server := responseServer(t, `{"choices":[{"message":{"content":null,"tool_calls":[
		{"type":"function","function":{"name":"x","arguments":null}}]},"finish_reason":"error"}]}`, nil)
	defer server.Close()
	model := mistral.NewModel("fallback", mistral.WithBaseURL(server.URL), mistral.WithHTTPClient(server.Client()))
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil || response.ModelName != "fallback" || response.FinishReason != ai.FinishReasonError ||
		string(response.Parts[0].(ai.ToolCallPart).Args) != "{}" {
		t.Fatalf("unexpected fallback response: %#v %v", response, err)
	}
}

func TestTransportFailures(t *testing.T) {
	for _, constructor := range []func(){
		func() { mistral.NewModel("model", mistral.WithBaseURL("")) },
		func() { mistral.NewModel("model", mistral.WithProvider(mistral.ProviderConfig{BaseURL: "url"})) },
		func() { mistral.NewModel("model", mistral.WithProvider(mistral.ProviderConfig{Name: "name"})) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("expected configuration panic")
				}
			}()
			constructor()
		}()
	}
	_ = mistral.NewModel("model", mistral.WithProvider(mistral.ProviderConfig{
		Name: "gateway", BaseURL: "https://example.com/v1",
	}))
	model := mistral.NewModel("model", mistral.WithBaseURL("://bad"))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected URL error")
	}
	model = mistral.NewModel("model", mistral.WithProvider(mistral.ProviderConfig{
		Name: "mistral", BaseURL: "https://example.com/v1", PrepareRequest: func(*http.Request) error {
			return errors.New("prepare")
		},
	}))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected preparation error")
	}
	model = mistral.NewModel("model", mistral.WithProvider(mistral.ProviderConfig{
		Name: "mistral", BaseURL: "https://example.com/v1", HTTPClient: &http.Client{Transport: roundTripFunc(
			func(*http.Request) (*http.Response, error) { return nil, errors.New("network") },
		)},
	}))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected transport error")
	}
	model = mistral.NewModel("model", mistral.WithProvider(mistral.ProviderConfig{
		Name: "mistral", BaseURL: "https://example.com/v1", HTTPClient: &http.Client{Transport: roundTripFunc(
			func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: failingBody{}}, nil
			},
		)},
	}))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected read error")
	}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("X-Test", "value")
		http.Error(response, "denied", http.StatusUnauthorized)
	}))
	defer server.Close()
	model = mistral.NewModel("model", mistral.WithBaseURL(server.URL), mistral.WithHTTPClient(server.Client()))
	_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	var apiError *mistral.APIError
	if !errors.As(err, &apiError) || apiError.StatusCode != http.StatusUnauthorized ||
		apiError.Headers.Get("X-Test") != "value" || !strings.Contains(apiError.Error(), "denied") ||
		!apiError.IsModelAPIError() {
		t.Fatalf("unexpected API error: %#v %v", apiError, err)
	}
}

func noRequestModel(t *testing.T) (*mistral.Model, *int) {
	t.Helper()
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		requests++
		_, _ = io.WriteString(response, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(server.Close)
	return mistral.NewModel("model", mistral.WithBaseURL(server.URL), mistral.WithHTTPClient(server.Client())), &requests
}

func responseServer(t *testing.T, body string, captured *map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if captured != nil {
			if err := jsonNewDecoder(request.Body, captured); err != nil {
				t.Error(err)
			}
		}
		_, _ = io.WriteString(response, body)
	}))
}

func jsonNewDecoder(reader io.Reader, destination any) error {
	decoder := json.NewDecoder(reader)
	return decoder.Decode(destination)
}

type failingBody struct{}

func (failingBody) Read([]byte) (int, error) { return 0, fmt.Errorf("read") }
func (failingBody) Close() error             { return nil }
