package vllm_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/infer"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
	"github.com/Kludex/pydantic-ai-go/ai/models/vllm"
)

func TestVLLMRequest(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/chat/completions" || request.Header.Get("Authorization") != "Bearer secret" {
			t.Errorf("unexpected request: %s authorization=%q", request.URL.Path, request.Header.Get("Authorization"))
		}
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		_, _ = io.WriteString(response, `{"model":"Qwen/Qwen3-32B","choices":[{"message":{"reasoning":"think","reasoning_content":"think","content":"done"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	model, err := vllm.NewModel("Qwen/Qwen3-32B", vllm.WithBaseURL(server.URL+"/v1"),
		vllm.WithAPIKey("secret"), vllm.WithHTTPClient(server.Client()))
	if err != nil {
		t.Fatal(err)
	}
	response, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.SystemPromptPart{Content: "history"}, ai.UserPromptPart{Content: "hello"},
	}}}, ai.ModelRequestParams{
		Instructions: "current", Settings: ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelHigh}},
	})
	if err != nil {
		t.Fatal(err)
	}
	messages := body["messages"].([]any)
	if len(messages) != 2 || messages[0].(map[string]any)["content"] != "current\n\nhistory" ||
		body["reasoning_effort"] != "high" {
		t.Fatalf("unexpected request body: %#v", body)
	}
	if len(response.Parts) != 2 || response.Parts[0].(ai.ThinkingPart).Content != "think" ||
		response.Parts[0].(ai.ThinkingPart).ID != "reasoning" {
		t.Fatalf("unexpected response: %#v", response)
	}
}

func TestVLLMConfigurationAndInference(t *testing.T) {
	t.Setenv("VLLM_BASE_URL", "")
	if _, err := vllm.NewProviderConfig(); err == nil {
		t.Fatal("expected missing provider base URL error")
	}
	if _, err := vllm.NewModel("model"); err == nil {
		t.Fatal("expected missing base URL error")
	}
	t.Setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
	t.Setenv("VLLM_API_KEY", "key")
	model, err := infer.Model("vllm:model")
	if err != nil {
		t.Fatal(err)
	}
	identity := model.(ai.ModelProviderIdentity)
	if identity.ProviderName() != "vllm" || identity.ProviderURL() != "http://localhost:8000/v1" {
		t.Fatalf("unexpected inferred model: %q %q", identity.ProviderName(), identity.ProviderURL())
	}
	profile := model.(ai.ModelProfiler).ModelProfile()
	if !profile.NativeOutputRequiresPrompt {
		t.Fatal("vLLM native output must include schema instructions")
	}
}

func TestVLLMOptionsAndStreaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("X-Test") != "value" {
			t.Errorf("provider header missing: %v", request.Header)
		}
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(response, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\",\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	model, err := vllm.NewModel("deepseek-ai/DeepSeek-V4-Pro",
		vllm.WithAPIKey("ignored"), vllm.WithBaseURL("https://ignored.example"),
		vllm.WithHTTPClient(http.DefaultClient), vllm.WithProvider(openai.ProviderConfig{
			Name: "gateway", BaseURL: server.URL, HTTPClient: server.Client(), Headers: http.Header{"X-Test": {"value"}},
		}), vllm.WithDefaultSettings(ai.ModelSettings{MaxTokens: 10}))
	if err != nil {
		t.Fatal(err)
	}
	stream, err := model.StreamRequest(t.Context(), []ai.ModelMessage{ai.ModelResponse{}}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var thinking bool
	for event, streamErr := range stream {
		if streamErr != nil {
			t.Fatal(streamErr)
		}
		if delta, ok := event.(ai.ThinkingDeltaEvent); ok {
			thinking = delta.ID == "reasoning_content" && delta.Delta == "think"
		}
	}
	if !thinking || model.DefaultModelSettings().MaxTokens != 10 {
		t.Fatal("vLLM options or streamed reasoning were not applied")
	}
}

func TestVLLMProviderConfig(t *testing.T) {
	t.Setenv("VLLM_BASE_URL", "https://vllm.example/v1")
	t.Setenv("VLLM_API_KEY", "key")
	provider, err := vllm.NewProviderConfig()
	if err != nil || provider.BaseURL != "https://vllm.example/v1" || provider.APIKey != "key" {
		t.Fatalf("unexpected provider config: %#v %v", provider, err)
	}
}

func TestVLLMForcedToolChoiceByModel(t *testing.T) {
	for name, test := range map[string]struct {
		model    string
		thinking *ai.ThinkingSettings
		required bool
	}{
		"Qwen coder":                {model: "Qwen/Qwen3-Coder-480B-A35B-Instruct", required: true},
		"GPT OSS":                   {model: "openai/gpt-oss-20b"},
		"DeepSeek default thinking": {model: "deepseek-ai/DeepSeek-V4-Pro"},
		"DeepSeek thinking disabled": {
			model: "deepseek-ai/DeepSeek-V4-Pro", thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled},
			required: true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			var body map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
				_ = json.NewDecoder(request.Body).Decode(&body)
				_, _ = io.WriteString(response, `{"choices":[{"message":{"tool_calls":[{"id":"call","type":"function","function":{"name":"final","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`)
			}))
			defer server.Close()
			model, err := vllm.NewModel(test.model, vllm.WithBaseURL(server.URL), vllm.WithHTTPClient(server.Client()))
			if err != nil {
				t.Fatal(err)
			}
			_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{
				OutputTool: &ai.ToolDefinition{Name: "final", Schema: map[string]any{"type": "object"}, Strict: boolPointer(true)},
				Settings:   ai.ModelSettings{Thinking: test.thinking},
			})
			if err != nil {
				t.Fatal(err)
			}
			_, required := body["tool_choice"]
			if required != test.required {
				t.Fatalf("unexpected tool choice: %#v", body)
			}
		})
	}
}

func boolPointer(value bool) *bool { return &value }

func TestVLLMUnsupportedThinkingIsOmitted(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(request.Body).Decode(&body)
		if _, exists := body["reasoning_effort"]; exists {
			t.Errorf("unsupported reasoning effort was sent: %#v", body)
		}
		_, _ = io.WriteString(response, `{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()
	model, err := vllm.NewModel("Qwen/Qwen3.8-27B", vllm.WithBaseURL(server.URL), vllm.WithHTTPClient(server.Client()))
	if err != nil {
		t.Fatal(err)
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{
		Settings: ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelHigh}},
	})
	if err != nil {
		t.Fatal(err)
	}
}
