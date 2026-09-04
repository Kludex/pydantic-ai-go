package crusoe_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/crusoe"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestModels(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		requests++
		if request.Header.Get("Authorization") != "Bearer key" || request.URL.Path != "/chat/completions" {
			t.Errorf("unexpected request: %s %#v", request.URL, request.Header)
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["response_format"] == nil {
			t.Errorf("native output schema was omitted: %#v", body)
		}
		if request.Header.Get("Accept") == "text/event-stream" {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(response, "data: "+`{"model":"model","choices":[{"delta":{"reasoning":"stream reason","content":"ok"},"finish_reason":"stop"}]}`+"\n\n"+
				"data: [DONE]\n\n")
			return
		}
		if body["model"] == "deepseek-ai/DeepSeek-V3-0324" {
			_, _ = io.WriteString(response, `{"model":"model","choices":[{"message":{"reasoning_content":"deep reason","content":"ok"},"finish_reason":"stop"}]}`)
			return
		}
		_, _ = io.WriteString(response, `{"model":"model","choices":[{"message":{"reasoning":"reason","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()
	params := ai.ModelRequestParams{OutputMode: ai.OutputModeNative, OutputSchema: map[string]any{"type": "object"}}
	model := crusoe.NewModel(
		"openai/gpt-oss-120b",
		crusoe.WithBaseURL(server.URL),
		crusoe.WithAPIKey("key"),
		crusoe.WithHTTPClient(server.Client()),
		crusoe.WithDefaultSettings(ai.ModelSettings{MaxTokens: 20}),
	)
	result, err := model.Request(t.Context(), nil, params)
	if err != nil || result.ProviderName != "crusoe" || result.Parts[0].(ai.ThinkingPart).Content != "reason" {
		t.Fatalf("unexpected result: %#v %v", result, err)
	}
	stream, err := model.StreamRequest(t.Context(), nil, params)
	if err != nil {
		t.Fatal(err)
	}
	var thinking string
	for event, eventErr := range stream {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		if event, ok := event.(ai.ThinkingDeltaEvent); ok {
			thinking += event.Delta
		}
	}
	if thinking != "stream reason" {
		t.Fatalf("unexpected stream reasoning: %q", thinking)
	}
	deepseek := crusoe.NewModel("deepseek-ai/DeepSeek-V3-0324", crusoe.WithProvider(openai.ProviderConfig{
		Name: "crusoe-gateway", BaseURL: server.URL, APIKey: "key", HTTPClient: server.Client(),
	}))
	result, err = deepseek.Request(t.Context(), nil, params)
	if err != nil || result.ProviderName != "crusoe-gateway" || result.Parts[0].(ai.ThinkingPart).Content != "deep reason" {
		t.Fatalf("unexpected DeepSeek result: %#v %v", result, err)
	}
	if requests != 3 {
		t.Fatalf("unexpected requests: %d", requests)
	}
}

func TestProviderConfiguration(t *testing.T) {
	t.Setenv("CRUSOE_API_KEY", "environment-key")
	provider := crusoe.NewProviderConfig()
	if provider.Name != "crusoe" || provider.BaseURL != "https://api.inference.crusoecloud.com/v1" ||
		provider.APIKey != "environment-key" {
		t.Fatalf("unexpected provider: %#v", provider)
	}
}
