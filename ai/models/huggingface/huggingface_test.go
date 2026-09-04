package huggingface_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/huggingface"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestModelRequestAndStream(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		requests++
		if request.URL.Path != "/chat/completions" || request.Header.Get("Authorization") != "Bearer token" {
			t.Errorf("unexpected request: %s %#v", request.URL, request.Header)
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["model"] != "Qwen/Qwen3-32B" || body["max_tokens"] != float64(20) ||
			body["max_completion_tokens"] != nil {
			t.Errorf("unexpected body: %#v", body)
		}
		messages := body["messages"].([]any)
		userContent := messages[0].(map[string]any)["content"].([]any)
		if userContent[1].(map[string]any)["type"] != "image_url" {
			t.Errorf("unexpected image: %#v", userContent)
		}
		assistant := messages[1].(map[string]any)
		if content := assistant["content"].(string); !strings.Contains(content, "<think>\nreason\n</think>") ||
			strings.Contains(content, "foreign") {
			t.Errorf("unexpected thinking replay: %q", content)
		}
		function := body["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
		if _, exists := function["strict"]; exists {
			t.Errorf("unexpected strict tool definition: %#v", function)
		}
		if request.Header.Get("Accept") == "text/event-stream" {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(response, "data: "+`{"id":"id","model":"Qwen/Qwen3-32B","created":10,"choices":[{"delta":{"content":"before<think>why</think>after","tool_calls":[{"index":0,"id":"call","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"stop_sequence"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}`+"\n\n"+
				"data: [DONE]\n\n")
			return
		}
		_, _ = io.WriteString(response, `{"id":"id","model":"Qwen/Qwen3-32B","created":10,"choices":[`+
			`{"message":{"content":"before<think>why</think>after","tool_calls":[{"id":"call","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"eos_token"}],`+
			`"usage":{"prompt_tokens":2,"completion_tokens":3}}`)
	}))
	defer server.Close()
	temperature := 0.2
	model := huggingface.NewModel(
		"Qwen/Qwen3-32B",
		huggingface.WithBaseURL(server.URL),
		huggingface.WithToken("token"),
		huggingface.WithHTTPClient(server.Client()),
		huggingface.WithDefaultSettings(ai.ModelSettings{Temperature: &temperature}),
	)
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.TextContent{Text: "question"}, ai.ImageURL{URL: "https://example.com/image.png"},
		}}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.TextPart{Content: "answer"},
			ai.ThinkingPart{Content: "reason", ProviderName: "huggingface"},
			ai.ThinkingPart{Content: "foreign", ProviderName: "other"},
			ai.ToolCallPart{ToolName: "earlier", Args: json.RawMessage(`{}`), ToolCallID: "earlier-call"},
		}},
	}
	strict := true
	params := ai.ModelRequestParams{
		Settings: ai.ModelSettings{MaxTokens: 20},
		Tools:    []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}, Strict: &strict}},
	}
	result, err := model.Request(t.Context(), messages, params)
	if err != nil || result.ProviderName != "huggingface" || result.FinishReason != ai.FinishReasonStop ||
		result.Text() != "before\n\nafter" || result.Parts[1].(ai.ThinkingPart).Content != "why" ||
		result.Parts[3].(ai.ToolCallPart).ToolName != "lookup" || result.Usage.InputTokens != 2 {
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
	if thinking != "why" || text != "beforeafter" || requests != 2 {
		t.Fatalf("unexpected stream: %q %q %d", thinking, text, requests)
	}
}

func TestConfiguration(t *testing.T) {
	t.Setenv("HF_TOKEN", "environment-token")
	provider := huggingface.NewProviderConfig()
	if provider.Name != "huggingface" || provider.BaseURL != "https://router.huggingface.co/v1" ||
		provider.APIKey != "environment-token" {
		t.Fatalf("unexpected provider config: %#v", provider)
	}
	model := huggingface.NewModel("model", huggingface.WithInferenceProvider("together"))
	if model.ProviderURL() != "https://router.huggingface.co/together/v1" {
		t.Fatalf("unexpected inference provider URL: %q", model.ProviderURL())
	}

	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("X-Gateway") != "yes" {
			t.Errorf("missing gateway header: %#v", request.Header)
		}
		_, _ = io.WriteString(response, `{"model":"model","choices":[{"message":{"content":"ok"}}]}`)
	}))
	defer server.Close()
	model = huggingface.NewModel("model", huggingface.WithProvider(openai.ProviderConfig{
		Name: "hf-gateway", BaseURL: server.URL, HTTPClient: server.Client(),
		Headers: http.Header{"X-Gateway": {"yes"}},
	}))
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil || response.ProviderName != "hf-gateway" || response.Text() != "ok" {
		t.Fatalf("unexpected gateway response: %#v %v", response, err)
	}
}

func TestUnsupportedRequests(t *testing.T) {
	model := huggingface.NewModel("model")
	params := ai.ModelRequestParams{OutputMode: ai.OutputModeNative, OutputSchema: map[string]any{"type": "object"}}
	if _, err := model.Request(t.Context(), nil, params); err == nil || !strings.Contains(err.Error(), "native structured") {
		t.Fatalf("unexpected native output error: %v", err)
	}
	if _, err := model.StreamRequest(t.Context(), nil, params); err == nil || !strings.Contains(err.Error(), "native structured") {
		t.Fatalf("unexpected streamed native output error: %v", err)
	}

	tests := []ai.UserContent{
		ai.AudioURL{URL: "https://example.com/audio.mp3"},
		ai.DocumentURL{URL: "https://example.com/file.pdf"},
		ai.VideoURL{URL: "https://example.com/video.mp4"},
		ai.UploadedFile{FileID: "file"},
		ai.BinaryContent{MediaType: "audio/wav", Data: []byte("audio")},
	}
	for _, content := range tests {
		messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.UserPromptPart{Contents: []ai.UserContent{content}},
		}}}
		if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err == nil ||
			!strings.Contains(err.Error(), "not supported") {
			t.Fatalf("content=%T: unexpected error: %v", content, err)
		}
	}
	for _, content := range []any{
		ai.ToolReturn{Content: []ai.UserContent{ai.VideoURL{URL: "https://example.com/video.mp4"}}},
		&ai.ToolReturn{Content: []ai.UserContent{ai.UploadedFile{FileID: "file"}}},
	} {
		messages := []ai.ModelMessage{
			ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "earlier"}}},
			ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{Content: content}}},
		}
		if _, err := model.StreamRequest(t.Context(), messages, ai.ModelRequestParams{}); err == nil {
			t.Fatalf("unexpected rich return success: %#v", content)
		}
	}
}

func TestEmptyInferenceProviderPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	_ = huggingface.WithInferenceProvider("")
}
