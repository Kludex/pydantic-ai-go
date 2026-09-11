package githubcopilot_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/githubcopilot"
	"github.com/Kludex/pydantic-ai-go/ai/models/infer"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestGitHubCopilotRequestAndReasoningRoundTrip(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		calls++
		if request.URL.Path != "/chat/completions" || request.Header.Get("Authorization") != "Bearer token" ||
			request.Header.Get("Copilot-Integration-Id") != "vscode-chat" {
			t.Errorf("unexpected request: %s headers=%v", request.URL.Path, request.Header)
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body["model"] != "claude-sonnet-5" {
			t.Errorf("model ID changed: %#v", body["model"])
		}
		if _, exists := body["temperature"]; exists {
			t.Errorf("unsupported Claude temperature was sent: %#v", body)
		}
		if calls == 2 {
			messages := body["messages"].([]any)
			assistant := messages[0].(map[string]any)
			if assistant["reasoning_text"] != "consider" {
				t.Errorf("reasoning was not replayed: %#v", assistant)
			}
		}
		_, _ = io.WriteString(response, `{"id":"id","model":"claude-sonnet-5","choices":[{"message":{"reasoning_text":"consider","content":"done"},"finish_reason":"stop"}],"usage":{}}`)
	}))
	defer server.Close()

	temperature := 0.5
	model, err := githubcopilot.NewModel("claude-sonnet-5", githubcopilot.WithAPIKey("token"),
		githubcopilot.WithBaseURL(server.URL), githubcopilot.WithHTTPClient(server.Client()))
	if err != nil {
		t.Fatal(err)
	}
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		Settings: ai.ModelSettings{Temperature: &temperature},
	})
	if err != nil {
		t.Fatal(err)
	}
	thinking := response.Parts[0].(ai.ThinkingPart)
	if thinking.Content != "consider" || thinking.ID != "reasoning_text" {
		t.Fatalf("unexpected reasoning: %#v", thinking)
	}
	_, err = model.Request(t.Context(), []ai.ModelMessage{*response}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestGitHubCopilotOptionsAndStreaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("X-Test") != "value" {
			t.Errorf("provider header missing: %v", request.Header)
		}
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(response, "data: {\"choices\":[{\"delta\":{\"reasoning_text\":\"think\",\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	model, err := githubcopilot.NewModel("gemini-3.8-flash",
		githubcopilot.WithAPIKey("ignored"), githubcopilot.WithBaseURL("https://ignored.example"),
		githubcopilot.WithHTTPClient(http.DefaultClient), githubcopilot.WithProvider(openai.ProviderConfig{
			Name: "gateway", BaseURL: server.URL, APIKey: "token", HTTPClient: server.Client(),
			Headers: http.Header{"X-Test": {"value"}},
		}), githubcopilot.WithDefaultSettings(ai.ModelSettings{MaxTokens: 10}))
	if err != nil {
		t.Fatal(err)
	}
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var thinking bool
	for event, streamErr := range stream {
		if streamErr != nil {
			t.Fatal(streamErr)
		}
		if delta, ok := event.(ai.ThinkingDeltaEvent); ok {
			thinking = delta.ID == "reasoning_text" && delta.Delta == "think"
		}
	}
	if !thinking || model.DefaultModelSettings().MaxTokens != 10 {
		t.Fatal("Copilot options or streamed reasoning were not applied")
	}
}

func TestGitHubCopilotProviderConfig(t *testing.T) {
	t.Setenv("GITHUB_COPILOT_API_KEY", "key")
	t.Setenv("GITHUB_COPILOT_BASE_URL", "")
	t.Setenv("COPILOT_API_URL", "")
	t.Setenv("GITHUB_COPILOT_API_BASE", "")
	model, err := githubcopilot.NewModel("gpt-5.4")
	if err != nil || model.ProviderURL() != "https://api.githubcopilot.com" {
		t.Fatalf("unexpected default model: %#v %v", model, err)
	}
	defaultProvider, err := githubcopilot.NewProviderConfig()
	if err != nil || defaultProvider.BaseURL != "https://api.githubcopilot.com" {
		t.Fatalf("unexpected default provider: %#v %v", defaultProvider, err)
	}
	t.Setenv("GITHUB_COPILOT_BASE_URL", "https://copilot.example")
	provider, err := githubcopilot.NewProviderConfig()
	if err != nil || provider.BaseURL != "https://copilot.example" || provider.APIKey != "key" ||
		provider.Headers.Get("Copilot-Integration-Id") != "vscode-chat" {
		t.Fatalf("unexpected provider config: %#v %v", provider, err)
	}
}

func TestGitHubCopilotEnvironmentAndInference(t *testing.T) {
	t.Setenv("GITHUB_COPILOT_API_KEY", "")
	t.Setenv("GITHUB_COPILOT_API_TOKEN", "")
	t.Setenv("COPILOT_GITHUB_TOKEN", "")
	if _, err := githubcopilot.NewProviderConfig(); err == nil {
		t.Fatal("expected missing provider credential error")
	}
	if _, err := githubcopilot.NewModel("gpt-5.4"); err == nil {
		t.Fatal("expected missing credential error")
	}
	t.Setenv("GITHUB_COPILOT_API_TOKEN", "fallback")
	t.Setenv("GITHUB_COPILOT_BASE_URL", "https://copilot.example")
	model, err := infer.Model("github-copilot:gpt-5.4")
	if err != nil {
		t.Fatal(err)
	}
	identity := model.(ai.ModelProviderIdentity)
	if model.Name() != "gpt-5.4" || identity.ProviderName() != "github-copilot" ||
		identity.ProviderURL() != "https://copilot.example" {
		t.Fatalf("unexpected inferred model: %q %q %q", model.Name(), identity.ProviderName(), identity.ProviderURL())
	}
}
