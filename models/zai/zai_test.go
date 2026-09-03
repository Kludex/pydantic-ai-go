package zai_test

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
	"github.com/Kludex/pydantic-ai-go/models/zai"
)

func TestProviderConfig(t *testing.T) {
	t.Setenv("ZAI_API_KEY", "key")
	provider := zai.NewProviderConfig()
	if provider.Name != "zai" || provider.BaseURL != "https://api.z.ai/api/paas/v4" || provider.APIKey != "key" {
		t.Fatalf("unexpected provider: %#v", provider)
	}
}

type responseServer struct {
	*httptest.Server
	mu        sync.Mutex
	bodies    []map[string]any
	headers   []http.Header
	responses []string
}

func newResponseServer(t *testing.T, responses ...string) *responseServer {
	t.Helper()
	captured := &responseServer{responses: responses}
	captured.Server = httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		captured.mu.Lock()
		defer captured.mu.Unlock()
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		captured.bodies = append(captured.bodies, body)
		captured.headers = append(captured.headers, request.Header.Clone())
		index := len(captured.bodies) - 1
		if index >= len(captured.responses) {
			t.Errorf("unexpected request %d", index)
			response.WriteHeader(http.StatusInternalServerError)
			return
		}
		_, _ = io.WriteString(response, captured.responses[index])
	}))
	return captured
}

func chatResponse(content, reasoning, finishReason string) string {
	return fmt.Sprintf(`{
		"id":"response-id","model":"glm-response","created":123,
		"choices":[{"message":{"content":%q,"reasoning_content":%q},"finish_reason":%q}],
		"usage":{"prompt_tokens":2,"completion_tokens":3}
	}`, content, reasoning, finishReason)
}

func TestZAIThinkingRoundTripAndSettings(t *testing.T) {
	server := newResponseServer(t,
		chatResponse("answer", "private reasoning", "stop"),
		chatResponse("", "", "sensitive"),
	)
	defer server.Close()
	model := zai.NewModel(
		"glm-5.3-flash",
		zai.WithBaseURL(server.URL),
		zai.WithAPIKey("secret"),
		zai.WithHTTPClient(server.Client()),
	)
	params := ai.ModelRequestParams{Settings: ai.ModelSettings{
		Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelMedium},
	}}
	first, err := model.Request(t.Context(), []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}}},
	}, params)
	if err != nil {
		t.Fatal(err)
	}
	if first.ProviderName != "zai" || first.FinishReason != ai.FinishReasonStop || len(first.Parts) != 2 {
		t.Fatalf("unexpected first response: %+v", first)
	}
	thinking, ok := first.Parts[0].(ai.ThinkingPart)
	if !ok || thinking.Content != "private reasoning" || thinking.ProviderName != "zai" {
		t.Fatalf("unexpected thinking part: %+v", first.Parts[0])
	}
	text, ok := first.Parts[1].(ai.TextPart)
	if !ok || text.Content != "answer" || text.ProviderName != "zai" {
		t.Fatalf("unexpected text part: %+v", first.Parts[1])
	}

	second, err := model.Request(t.Context(), []ai.ModelMessage{*first}, params)
	if err != nil {
		t.Fatal(err)
	}
	if second.FinishReason != ai.FinishReasonContentFilter {
		t.Fatalf("unexpected sensitive finish reason: %+v", second)
	}
	if server.headers[0].Get("Authorization") != "Bearer secret" {
		t.Fatalf("unexpected authorization %q", server.headers[0].Get("Authorization"))
	}
	for index, body := range server.bodies {
		thinkingBody := body["thinking"].(map[string]any)
		if thinkingBody["type"] != "enabled" || thinkingBody["clear_thinking"] != false ||
			body["reasoning_effort"] != "high" {
			t.Fatalf("unexpected request %d thinking fields: %v", index, body)
		}
	}
	messages := server.bodies[1]["messages"].([]any)
	assistant := messages[0].(map[string]any)
	if assistant["reasoning_content"] != "private reasoning" || assistant["content"] != "answer" {
		t.Fatalf("thinking history was not preserved: %v", assistant)
	}
}

func TestZAIThinkingProfiles(t *testing.T) {
	t.Setenv("ZAI_API_KEY", "environment-key")
	for _, test := range []struct {
		name           string
		model          string
		level          ai.ThinkingLevel
		wantType       string
		wantEffort     string
		wantThinking   bool
		clearThinking  *bool
		wantClearValue bool
	}{
		{name: "5.2 forwards minimal", model: "glm-5.2", level: ai.ThinkingLevelMinimal,
			wantType: "enabled", wantEffort: "minimal", wantThinking: true},
		{name: "5.3 maps minimal", model: "glm-5.3", level: ai.ThinkingLevelMinimal,
			wantType: "enabled", wantEffort: "low", wantThinking: true},
		{name: "5.3 maps medium", model: "glm-5.3-flash", level: ai.ThinkingLevelMedium,
			wantType: "enabled", wantEffort: "high", wantThinking: true},
		{name: "5.3 maps xhigh", model: "glm-5.3", level: ai.ThinkingLevelXHigh,
			wantType: "enabled", wantEffort: "max", wantThinking: true},
		{name: "5.3 forwards low", model: "glm-5.3", level: ai.ThinkingLevelLow,
			wantType: "enabled", wantEffort: "low", wantThinking: true},
		{name: "5.3 forwards high", model: "glm-5.3", level: ai.ThinkingLevelHigh,
			wantType: "enabled", wantEffort: "high", wantThinking: true},
		{name: "5.1 collapses effort", model: "glm-5.1", level: ai.ThinkingLevelHigh,
			wantType: "enabled", wantThinking: true},
		{name: "disabled", model: "glm-4.7", level: ai.ThinkingLevelDisabled,
			wantType: "disabled", wantThinking: true},
		{name: "enabled without effort", model: "glm-4.6", level: ai.ThinkingLevelEnabled,
			wantType: "enabled", wantThinking: true},
		{name: "vision thinking", model: "glm-4.5v", wantThinking: true},
		{name: "non-thinking model", model: "glm-4-32b", level: ""},
		{name: "explicit clear", model: "glm-5", wantThinking: true,
			clearThinking: boolPointer(true), wantClearValue: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := newResponseServer(t, chatResponse("done", "", "stop"))
			defer server.Close()
			settings := ai.ModelSettings{}
			if test.level != "" {
				settings.Thinking = &ai.ThinkingSettings{Level: test.level}
			}
			if test.clearThinking != nil {
				var err error
				settings, err = (zai.Settings{Common: settings, ClearThinking: test.clearThinking}).Build()
				if err != nil {
					t.Fatal(err)
				}
			}
			model := zai.NewModel(test.model, zai.WithBaseURL(server.URL))
			if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: settings}); err != nil {
				t.Fatal(err)
			}
			body := server.bodies[0]
			if server.headers[0].Get("Authorization") != "Bearer environment-key" {
				t.Fatalf("unexpected environment authorization: %q", server.headers[0].Get("Authorization"))
			}
			thinkingValue, exists := body["thinking"]
			if exists != test.wantThinking {
				t.Fatalf("unexpected thinking presence: %v", body)
			}
			if exists {
				thinking := thinkingValue.(map[string]any)
				if test.wantType != "" && thinking["type"] != test.wantType {
					t.Fatalf("unexpected thinking type: %v", thinking)
				}
				if thinking["clear_thinking"] != test.wantClearValue {
					t.Fatalf("unexpected clear_thinking: %v", thinking)
				}
			}
			effort, hasEffort := body["reasoning_effort"]
			if test.wantEffort == "" {
				if hasEffort {
					t.Fatalf("unexpected reasoning effort: %v", effort)
				}
			} else if effort != test.wantEffort {
				t.Fatalf("unexpected reasoning effort: %v", effort)
			}
		})
	}
}

func TestZAIStreamingReasoningAndFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(response, "data: "+`{"id":"stream-id","model":"glm-5.3-flash",`+
			`"choices":[{"delta":{"reasoning_content":"think "},"finish_reason":""}]}`+"\n\n")
		_, _ = io.WriteString(response, "data: "+`{"choices":[{"delta":{"reasoning_content":"more",`+
			`"content":"done"},"finish_reason":"network_error"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":2}}`+"\n\n")
		_, _ = io.WriteString(response, "data: [DONE]\n\n")
	}))
	defer server.Close()
	model := zai.NewModel("glm-5.3-flash", zai.WithBaseURL(server.URL))
	stream := ai.StreamModel(t.Context(), model, nil, ai.ModelRequestParams{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	response := stream.Response()
	if response == nil || response.FinishReason != ai.FinishReasonError || response.ProviderName != "zai" ||
		response.Text() != "done" {
		t.Fatalf("unexpected streamed response: %+v", response)
	}
	thinking, ok := response.Parts[0].(ai.ThinkingPart)
	if !ok || thinking.Content != "think more" || thinking.ProviderName != "zai" {
		t.Fatalf("unexpected streamed thinking: %+v", response.Parts)
	}
}

func TestZAIFinishReasonMappings(t *testing.T) {
	for reason, want := range map[string]ai.FinishReason{
		"model_context_window_exceeded": ai.FinishReasonLength,
		"network_error":                 ai.FinishReasonError,
		"tool_calls":                    ai.FinishReasonToolCall,
	} {
		t.Run(reason, func(t *testing.T) {
			server := newResponseServer(t, chatResponse("done", "", reason))
			defer server.Close()
			model := zai.NewModel("glm-5.3", zai.WithBaseURL(server.URL))
			response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			if response.FinishReason != want || response.ProviderDetails["finish_reason"] != reason {
				t.Fatalf("unexpected finish mapping: %+v", response)
			}
		})
	}
}

func TestZAISettingsValidationAndProvider(t *testing.T) {
	common := ai.ModelSettings{ExtraBody: map[string]any{"custom": map[string]any{"value": true}}}
	built, err := (zai.Settings{Common: common}).Build()
	if err != nil {
		t.Fatal(err)
	}
	built.ExtraBody["custom"].(map[string]any)["value"] = false
	if !reflect.DeepEqual(common.ExtraBody, map[string]any{"custom": map[string]any{"value": true}}) {
		t.Fatal("settings were not detached")
	}
	if _, err := (zai.Settings{
		Common:        ai.ModelSettings{ExtraBody: map[string]any{"thinking": "bad"}},
		ClearThinking: boolPointer(true),
	}).Build(); err == nil || !strings.Contains(err.Error(), "must be an object") {
		t.Fatalf("unexpected malformed settings error: %v", err)
	}
	if _, err := (zai.Settings{
		Common: ai.ModelSettings{ExtraBody: map[string]any{
			"thinking": map[string]any{"clear_thinking": false},
		}},
		ClearThinking: boolPointer(true),
	}).Build(); err == nil || !strings.Contains(err.Error(), "conflicts with typed settings") {
		t.Fatalf("unexpected setting conflict error: %v", err)
	}

	for name, settings := range map[string]ai.ModelSettings{
		"malformed thinking": {ExtraBody: map[string]any{"thinking": "bad"}},
		"thinking type conflict": {
			Thinking:  aiThinking(ai.ThinkingLevelHigh),
			ExtraBody: map[string]any{"thinking": map[string]any{"type": "enabled"}},
		},
		"effort conflict": {
			Thinking:  aiThinking(ai.ThinkingLevelHigh),
			ExtraBody: map[string]any{"reasoning_effort": "low"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			model := zai.NewModel("glm-5.3")
			if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: settings}); err == nil {
				t.Fatal("expected request preparation error")
			}
		})
	}
	streamErrorModel := zai.NewModel("glm-5.3")
	if _, err := streamErrorModel.StreamRequest(t.Context(), nil, ai.ModelRequestParams{
		Settings: ai.ModelSettings{ExtraBody: map[string]any{"thinking": "bad"}},
	}); err == nil || !strings.Contains(err.Error(), "must be an object") {
		t.Fatalf("unexpected stream preparation error: %v", err)
	}

	server := newResponseServer(t, chatResponse("custom", "reason", "stop"))
	defer server.Close()
	provider := openai.ProviderConfig{
		Name: "zai-gateway", BaseURL: server.URL, APIKey: "gateway-key", HTTPClient: server.Client(),
	}
	defaults := ai.ModelSettings{MaxTokens: 99}
	model := zai.NewModel("glm-5.3", zai.WithProvider(provider), zai.WithDefaultSettings(defaults))
	defaults.MaxTokens = 1
	if model.DefaultModelSettings().MaxTokens != 99 {
		t.Fatal("default settings were not detached")
	}
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if response.ProviderName != "zai-gateway" || response.Parts[0].(ai.ThinkingPart).ProviderName != "zai-gateway" {
		t.Fatalf("custom provider identity was not retained: %+v", response)
	}
}

func boolPointer(value bool) *bool { return &value }

func aiThinking(level ai.ThinkingLevel) *ai.ThinkingSettings {
	return &ai.ThinkingSettings{Level: level}
}
