package cerebras_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/cerebras"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func TestSettings(t *testing.T) {
	settings, err := (cerebras.Settings{}).Build()
	if err != nil || settings.Thinking != nil || settings.ExtraBody != nil {
		t.Fatalf("unexpected zero settings: %#v %v", settings, err)
	}
	disable := true
	clear := false
	settings, err = (cerebras.Settings{
		Common:           ai.ModelSettings{ExtraBody: map[string]any{"value": "kept"}},
		DisableReasoning: &disable,
		ClearThinking:    &clear,
	}).Build()
	if err != nil || settings.Thinking.Level != ai.ThinkingLevelDisabled ||
		settings.ExtraBody["clear_thinking"] != false || settings.ExtraBody["value"] != "kept" {
		t.Fatalf("unexpected settings: %#v %v", settings, err)
	}
	settings.ExtraBody["value"] = "changed"
	settings, err = (cerebras.Settings{ClearThinking: &clear}).Build()
	if err != nil || settings.ExtraBody["clear_thinking"] != false {
		t.Fatalf("unexpected clear-only settings: %#v %v", settings, err)
	}
	common := ai.ModelSettings{ExtraBody: map[string]any{"clear_thinking": true}}
	if _, err := (cerebras.Settings{Common: common, ClearThinking: &clear}).Build(); err == nil ||
		!strings.Contains(err.Error(), "conflicts") {
		t.Fatalf("unexpected conflict: %v", err)
	}
}

func TestZAIReasoningRequestAndStream(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		requests++
		if request.URL.Path != "/chat/completions" || request.Header.Get("Authorization") != "Bearer secret" ||
			request.Header.Get("X-Cerebras-3rd-Party-Integration") != "pydantic-ai" {
			t.Errorf("unexpected request: %s %#v", request.URL, request.Header)
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["clear_thinking"] != false || body["reasoning_effort"] != "none" || body["logit_bias"] != nil {
			t.Errorf("unexpected reasoning settings: %#v", body)
		}
		wireMessages := body["messages"].([]any)
		userContent := wireMessages[0].(map[string]any)["content"].([]any)
		if userContent[1].(map[string]any)["type"] != "image_url" || body["response_format"] == nil {
			t.Errorf("unexpected image or output schema: %#v", body)
		}
		assistant := wireMessages[1].(map[string]any)
		content := assistant["content"].(string)
		if !strings.Contains(content, "answer") || !strings.Contains(content, "<think>\nreason\n</think>") ||
			strings.Contains(content, "foreign") {
			t.Errorf("unexpected replay content: %q", content)
		}
		if request.Header.Get("Accept") == "text/event-stream" {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(response, "data: "+`{"id":"id","model":"zai-glm-4.7","choices":[{"delta":{"content":"<think>why</think>ok","tool_calls":[{"index":0,"id":"result-call","function":{"name":"result","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}`+"\n\n"+
				"data: [DONE]\n\n")
			return
		}
		_, _ = io.WriteString(response, `{"id":"id","model":"zai-glm-4.7","choices":[`+
			`{"message":{"content":"<think>why</think>ok","tool_calls":[{"id":"result-call","type":"function","function":{"name":"result","arguments":"{}"}}]},"finish_reason":"tool_calls"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":2}}`)
	}))
	defer server.Close()
	temperature := 0.5
	model := cerebras.NewModel(
		"zai-glm-4.7",
		cerebras.WithBaseURL(server.URL),
		cerebras.WithAPIKey("secret"),
		cerebras.WithHTTPClient(server.Client()),
		cerebras.WithDefaultSettings(ai.ModelSettings{Temperature: &temperature}),
	)
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.TextContent{Text: "question"}, ai.ImageURL{URL: "https://example.com/image.png"},
		}}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.TextPart{Content: "answer"},
			ai.ThinkingPart{Content: "reason", ProviderName: "cerebras"},
			ai.ThinkingPart{Content: "foreign", ProviderName: "other"},
			ai.ToolCallPart{ToolName: "lookup", Args: json.RawMessage(`{}`), ToolCallID: "call"},
		}},
	}
	params := ai.ModelRequestParams{
		OutputMode: ai.OutputModeNative, OutputSchema: map[string]any{"type": "object"},
		Settings: ai.ModelSettings{
			Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled}, LogitBias: map[string]int{"1": 100},
		},
	}
	result, err := model.Request(t.Context(), messages, params)
	if err != nil || result.Text() != "ok" || len(result.Parts) != 3 ||
		result.Parts[0].(ai.ThinkingPart).Content != "why" ||
		result.Parts[2].(ai.ToolCallPart).ToolName != "result" || result.ProviderName != "cerebras" {
		t.Fatalf("unexpected response: %#v %v", result, err)
	}
	stream, err := model.StreamRequest(t.Context(), messages, params)
	if err != nil {
		t.Fatal(err)
	}
	var thinking, text string
	for event, eventErr := range stream {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		switch event := event.(type) {
		case ai.ThinkingDeltaEvent:
			thinking += event.Delta
		case ai.TextDeltaEvent:
			text += event.Delta
		}
	}
	if thinking != "why" || text != "ok" || requests != 2 {
		t.Fatalf("unexpected stream: %q %q %d", thinking, text, requests)
	}
}

func TestModelProfilesAndGateway(t *testing.T) {
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		_, _ = io.WriteString(response, `{"model":"model","choices":[{"message":{"reasoning":"server reason","content":"ok"}}]}`)
	}))
	defer server.Close()
	provider := openai.ProviderConfig{Name: "cerebras-gateway", BaseURL: server.URL, HTTPClient: server.Client()}
	model := cerebras.NewModel("gpt-oss-120b", cerebras.WithProvider(provider))
	messages := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.ThinkingPart{Content: "reason", ProviderName: "cerebras-gateway"},
	}}}
	params := ai.ModelRequestParams{Settings: ai.ModelSettings{
		Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled},
	}}
	response, err := model.Request(t.Context(), messages, params)
	if err != nil || response.ProviderName != "cerebras-gateway" ||
		response.Parts[0].(ai.ThinkingPart).Content != "server reason" || bodies[0]["reasoning_effort"] != nil ||
		bodies[0]["messages"].([]any)[0].(map[string]any)["reasoning"] != "reason" {
		t.Fatalf("unexpected GPT-OSS request: %#v %#v %v", bodies, response, err)
	}

	conflict := ai.ModelRequestParams{Settings: ai.ModelSettings{
		Thinking:  &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled},
		ExtraBody: map[string]any{"reasoning_effort": "high"},
	}}
	zai := cerebras.NewModel("zai-glm-4.7", cerebras.WithProvider(provider))
	if _, err := zai.Request(t.Context(), nil, conflict); err == nil || !strings.Contains(err.Error(), "conflicts") {
		t.Fatalf("unexpected conflict: %v", err)
	}
	if _, err := zai.StreamRequest(t.Context(), nil, conflict); err == nil || !strings.Contains(err.Error(), "conflicts") {
		t.Fatalf("unexpected stream conflict: %v", err)
	}
}

func TestProviderConfiguration(t *testing.T) {
	t.Setenv("CEREBRAS_API_KEY", "environment-key")
	provider := cerebras.NewProviderConfig()
	if provider.Name != "cerebras" || provider.BaseURL != "https://api.cerebras.ai/v1" ||
		provider.APIKey != "environment-key" || provider.Headers.Get("X-Cerebras-3rd-Party-Integration") != "pydantic-ai" {
		t.Fatalf("unexpected provider: %#v", provider)
	}
}
