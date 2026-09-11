package together_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
	"github.com/Kludex/pydantic-ai-go/ai/models/together"
)

func TestTogetherRequestAndForcedToolChoice(t *testing.T) {
	for name, test := range map[string]struct {
		model      string
		wantChoice string
	}{
		"DeepSeek V4": {model: "deepseek-ai/DeepSeek-V4-Pro", wantChoice: "auto"},
		"Qwen":        {model: "Qwen/Qwen3-32B", wantChoice: "required"},
	} {
		t.Run(name, func(t *testing.T) {
			var body map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
				if request.URL.Path != "/v1/chat/completions" || request.Header.Get("Authorization") != "Bearer secret" {
					t.Errorf("unexpected request: %s headers=%v", request.URL.Path, request.Header)
				}
				if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				_, _ = io.WriteString(response, `{"model":"`+test.model+`","choices":[{"message":{"reasoning_content":"think","content":"done"},"finish_reason":"stop"}],"usage":{}}`)
			}))
			defer server.Close()
			client := server.Client()
			client.Timeout = time.Second
			model := together.NewModel(test.model, together.WithAPIKey("secret"), together.WithBaseURL(server.URL+"/v1"),
				together.WithHTTPClient(client), together.WithDefaultSettings(ai.ModelSettings{MaxTokens: 12}))
			response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
				OutputTool: &ai.ToolDefinition{Name: "final", Schema: map[string]any{"type": "object"}},
			})
			if err != nil {
				t.Fatal(err)
			}
			if body["tool_choice"] != test.wantChoice || len(response.Parts) != 2 ||
				response.Parts[0].(ai.ThinkingPart).Content != "think" || client.Timeout != time.Second ||
				model.DefaultModelSettings().MaxTokens != 12 {
				t.Fatalf("unexpected request or response: body=%#v response=%#v", body, response)
			}
		})
	}
}

func TestTogetherProviderAndStreaming(t *testing.T) {
	t.Setenv("TOGETHER_API_KEY", "environment-key")
	provider := together.NewProviderConfig()
	if provider.Name != "together" || provider.BaseURL != "https://api.together.xyz/v1" ||
		provider.APIKey != "environment-key" {
		t.Fatalf("unexpected provider config: %#v", provider)
	}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("X-Test") != "value" {
			t.Errorf("provider header missing: %v", request.Header)
		}
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(response, "data: {\"choices\":[{\"delta\":{\"reasoning\":\"think\",\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	model := together.NewModel("meta-llama/Llama-4", together.WithProvider(openai.ProviderConfig{
		Name: "gateway", BaseURL: server.URL, APIKey: "token", HTTPClient: server.Client(),
		Headers: http.Header{"X-Test": {"value"}},
	}))
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var sawThinking bool
	for event, streamErr := range stream {
		if streamErr != nil {
			t.Fatal(streamErr)
		}
		if delta, ok := event.(ai.ThinkingDeltaEvent); ok && delta.Delta == "think" {
			sawThinking = true
		}
	}
	if !sawThinking || model.ProviderName() != "together" {
		t.Fatalf("unexpected streamed model: thinking=%v provider=%q", sawThinking, model.ProviderName())
	}
}
