package ollama_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/ollama"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestModelRequestAndStream(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		requests++
		if request.URL.Path != "/v1/chat/completions" || request.Header.Get("Authorization") != "Bearer secret" {
			t.Errorf("unexpected request: %s %#v", request.URL, request.Header)
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["model"] != "qwen3" || body["max_tokens"] != float64(20) || body["max_completion_tokens"] != nil {
			t.Errorf("unexpected body: %#v", body)
		}
		content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
		if content[0].(map[string]any)["type"] != "image_url" {
			t.Errorf("unexpected image content: %#v", content)
		}
		function := body["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
		if _, exists := function["strict"]; exists {
			t.Errorf("unexpected strict tool definition: %#v", function)
		}
		if request.Header.Get("Accept") == "text/event-stream" {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(response, "data: "+`{"id":"response","model":"qwen3","created":10,"choices":[{"delta":{"reasoning":"why"}}]}`+"\n\n"+
				"data: "+`{"id":"response","model":"qwen3","created":10,"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}`+"\n\n"+
				"data: [DONE]\n\n")
			return
		}
		_, _ = io.WriteString(response, `{"id":"response","model":"qwen3","created":10,"choices":[`+
			`{"message":{"reasoning":"why","content":"hello"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":2,"completion_tokens":3}}`)
	}))
	defer server.Close()
	model := ollama.NewModel(
		"qwen3",
		ollama.WithBaseURL(server.URL),
		ollama.WithAPIKey("secret"),
		ollama.WithHTTPClient(server.Client()),
		ollama.WithDefaultSettings(ai.ModelSettings{MaxTokens: 20}),
	)
	strict := true
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
		ai.ImageURL{URL: "https://example.com/image.png"},
	}}}}}
	params := ai.ModelRequestParams{
		Settings: ai.ModelSettings{MaxTokens: 20},
		Tools:    []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}, Strict: &strict}},
	}
	response, err := model.Request(t.Context(), messages, params)
	if err != nil {
		t.Fatal(err)
	}
	if response.ProviderName != "ollama" || response.ProviderURL != server.URL+"/v1" || response.Text() != "hello" ||
		response.Parts[0].(ai.ThinkingPart).Content != "why" || response.Usage.InputTokens != 2 {
		t.Fatalf("unexpected response: %#v", response)
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
	if thinking != "why" || text != "hello" || requests != 2 {
		t.Fatalf("unexpected stream: thinking=%q text=%q requests=%d", thinking, text, requests)
	}
	_, err = model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{ai.DocumentURL{URL: "https://example.com/file.pdf"}}},
	}}}, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), "document input") {
		t.Fatalf("unexpected document error: %v", err)
	}
	if requests != 2 {
		t.Fatalf("validation reached transport: %d", requests)
	}
}

func TestProviderConfiguration(t *testing.T) {
	t.Setenv("OLLAMA_BASE_URL", "")
	t.Setenv("OLLAMA_HOST", "")
	t.Setenv("OLLAMA_API_KEY", "")
	model := ollama.NewModel("llama3.2")
	if model.ProviderURL() != "http://localhost:11434/v1" || model.ProviderName() != "ollama" {
		t.Fatalf("unexpected defaults: %s %s", model.ProviderURL(), model.ProviderName())
	}

	t.Setenv("OLLAMA_HOST", "http://host.example")
	if config := ollama.NewProviderConfig(); config.BaseURL != "http://host.example/v1" {
		t.Fatalf("unexpected host config: %#v", config)
	}
	t.Setenv("OLLAMA_BASE_URL", "http://base.example/v1/")
	t.Setenv("OLLAMA_API_KEY", "environment-key")
	if config := ollama.NewProviderConfig(); config.BaseURL != "http://base.example/v1" || config.APIKey != "environment-key" {
		t.Fatalf("unexpected base config: %#v", config)
	}

	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("X-Provider") != "value" || request.Header.Get("X-Dynamic") != "yes" {
			t.Errorf("missing provider headers: %#v", request.Header)
		}
		_, _ = io.WriteString(response, `{"model":"llama","choices":[{"message":{"content":"ok"}}]}`)
	}))
	defer server.Close()
	provider := openai.ProviderConfig{
		Name: "ollama-gateway", BaseURL: server.URL, HTTPClient: server.Client(),
		Headers: http.Header{"X-Provider": {"value"}},
		PrepareRequest: func(request *http.Request) error {
			request.Header.Set("X-Dynamic", "yes")
			return nil
		},
	}
	model = ollama.NewModel("llama", ollama.WithProvider(provider))
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil || response.ProviderName != "ollama-gateway" || response.Text() != "ok" {
		t.Fatalf("unexpected gateway response: %#v %v", response, err)
	}
}

func TestUnsupportedContent(t *testing.T) {
	model := ollama.NewModel("qwen3")
	tests := []struct {
		name    string
		content ai.UserContent
	}{
		{name: "audio", content: ai.AudioURL{URL: "https://example.com/audio.mp3"}},
		{name: "document", content: ai.DocumentURL{URL: "https://example.com/file.pdf"}},
		{name: "video", content: ai.VideoURL{URL: "https://example.com/video.mp4"}},
		{name: "uploaded", content: ai.UploadedFile{FileID: "file"}},
		{name: "binary", content: ai.BinaryContent{MediaType: "audio/wav", Data: []byte("audio")}},
	}
	for _, test := range tests {
		messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.UserPromptPart{Contents: []ai.UserContent{test.content}},
		}}}
		if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err == nil ||
			!strings.Contains(err.Error(), "not supported") {
			t.Fatalf("name=%s: unexpected error: %v", test.name, err)
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

func TestInvalidConfigurationPanics(t *testing.T) {
	for _, call := range []func(){
		func() { _ = ollama.WithBaseURL("") },
		func() { _ = ollama.WithProvider(openai.ProviderConfig{Name: "ollama"}) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			call()
		}()
	}
}

func TestOllamaCloudRejectsNativeOutput(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		requests++
	}))
	defer server.Close()
	model := ollama.NewModel("qwen3-cloud", ollama.WithBaseURL(server.URL), ollama.WithHTTPClient(server.Client()))
	params := ai.ModelRequestParams{OutputMode: ai.OutputModeNative, OutputSchema: map[string]any{"type": "object"}}
	if _, err := model.Request(t.Context(), nil, params); err == nil || !strings.Contains(err.Error(), "not enforced") {
		t.Fatalf("unexpected request error: %v", err)
	}
	if _, err := model.StreamRequest(t.Context(), nil, params); err == nil || !strings.Contains(err.Error(), "not enforced") {
		t.Fatalf("unexpected stream error: %v", err)
	}
	if requests != 0 {
		t.Fatalf("unexpected transport requests: %d", requests)
	}

	cloud := ollama.NewModel("qwen3", ollama.WithProvider(openai.ProviderConfig{
		Name: "ollama", BaseURL: "https://api.ollama.com", APIKey: "key",
	}))
	if _, err := cloud.Request(t.Context(), nil, params); err == nil {
		t.Fatal("expected cloud host rejection")
	}
}
