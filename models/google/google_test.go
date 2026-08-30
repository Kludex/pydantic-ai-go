package google_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/google"
)

func newServer(t *testing.T, handler http.HandlerFunc) *google.Model {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return google.NewModel("gemini-2.5-flash",
		google.WithAPIKey("test-key"),
		google.WithBaseURL(server.URL),
		google.WithHTTPClient(server.Client()),
	)
}

func TestRequestTextResponse(t *testing.T) {
	var gotBody map[string]any
	var gotKey, gotPath string
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("x-goog-api-key")
		gotPath = r.URL.Path
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{
			"modelVersion": "gemini-2.5-flash",
			"candidates": [{"content": {"parts": [{"text": "Hello!"}]}}],
			"usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3}
		}`))
	})
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hi"}}}}
	temp := 0.5
	resp, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{
		Instructions: "be brief",
		AllowText:    true,
		Settings:     ai.ModelSettings{MaxTokens: 100, Temperature: &temp},
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotKey != "test-key" {
		t.Fatalf("unexpected api key header %q", gotKey)
	}
	if !strings.HasSuffix(gotPath, "/models/gemini-2.5-flash:generateContent") {
		t.Fatalf("unexpected path %q", gotPath)
	}
	if gotBody["systemInstruction"] == nil {
		t.Fatal("system instruction not sent")
	}
	gen := gotBody["generationConfig"].(map[string]any)
	if gen["maxOutputTokens"].(float64) != 100 || gen["temperature"].(float64) != 0.5 {
		t.Fatalf("generation config not sent: %v", gen)
	}
	if resp.Text() != "Hello!" {
		t.Fatalf("unexpected text %q", resp.Text())
	}
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 3 || resp.Usage.Requests != 1 {
		t.Fatalf("unexpected usage %+v", resp.Usage)
	}
	if resp.ModelName != "gemini-2.5-flash" {
		t.Fatalf("unexpected model name %q", resp.ModelName)
	}
}

func TestRequestFunctionCallRoundTrip(t *testing.T) {
	var gotBody map[string]any
	first := true
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		if first {
			first = false
			_, _ = w.Write([]byte(`{
				"modelVersion": "gemini-2.5-flash",
				"candidates": [{"content": {"parts": [
					{"thought": true, "text": "checking"},
					{"functionCall": {"name": "get_weather", "args": {"city": "SF"}}}
				]}}],
				"usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 8}
			}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"modelVersion": "gemini-2.5-flash",
			"candidates": [{"content": {"parts": [{"text": "Sunny."}]}}],
			"usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 4}
		}`))
	})

	params := ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{Name: "get_weather", Schema: map[string]any{
			"type":                 "object",
			"additionalProperties": false,
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
		}}},
		AllowText: true,
	}
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "weather?"}}}}
	resp, err := model.Request(t.Context(), msgs, params)
	if err != nil {
		t.Fatal(err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ToolName != "get_weather" || string(calls[0].Args) != `{"city":"SF"}` {
		t.Fatalf("unexpected calls %+v", calls)
	}
	if _, ok := resp.Parts[0].(ai.ThinkingPart); !ok {
		t.Fatalf("thought part lost: %+v", resp.Parts)
	}

	msgs = append(msgs, *resp, ai.ModelRequest{Parts: []ai.RequestPart{
		ai.ToolReturnPart{ToolName: "get_weather", Content: "sunny"},
	}})
	if _, err := model.Request(t.Context(), msgs, params); err != nil {
		t.Fatal(err)
	}
	contents := gotBody["contents"].([]any)
	modelTurn := contents[1].(map[string]any)
	if modelTurn["role"] != "model" {
		t.Fatalf("unexpected roles %v", contents)
	}
	toolTurn := contents[2].(map[string]any)["parts"].([]any)[0].(map[string]any)
	if toolTurn["functionResponse"] == nil {
		t.Fatalf("function response not sent: %v", toolTurn)
	}
	declared := gotBody["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any)[0].(map[string]any)
	schema := declared["parameters"].(map[string]any)
	if _, ok := schema["additionalProperties"]; ok {
		t.Fatal("additionalProperties should be sanitized for Gemini")
	}
}

func TestOutputToolForcesFunctionCalling(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"final_result","args":{}}}]}}],"usageMetadata":{}}`))
	})
	params := ai.ModelRequestParams{
		OutputTool: &ai.ToolDefinition{Name: "final_result", Schema: map[string]any{"type": "object"}},
	}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	mode := gotBody["toolConfig"].(map[string]any)["functionCallingConfig"].(map[string]any)["mode"]
	if mode != "ANY" {
		t.Fatalf("expected ANY mode, got %v", mode)
	}
}

func TestRetryAndSystemParts(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]}}],"usageMetadata":{}}`))
	})
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.SystemPromptPart{Content: "sys"},
		ai.RetryPromptPart{Content: "bad args", ToolName: "t"},
		ai.RetryPromptPart{Content: "plain retry"},
	}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	parts := gotBody["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	if parts[1].(map[string]any)["functionResponse"] == nil {
		t.Fatalf("tool retry should be a function response: %v", parts[1])
	}
	if parts[2].(map[string]any)["text"] != "plain retry" {
		t.Fatalf("plain retry should be text: %v", parts[2])
	}
}

func TestErrors(t *testing.T) {
	t.Run("api error", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte("rate limited"))
		})
		var apiErr *google.APIError
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
		if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusTooManyRequests {
			t.Fatalf("expected APIError 429, got %v", err)
		}
		if apiErr.Error() != "google: API returned status 429: rate limited" {
			t.Fatalf("unexpected message %q", apiErr.Error())
		}
	})
	t.Run("invalid json", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("not json")) })
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected parse error")
		}
	})
	t.Run("no candidates", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{"candidates":[]}`)) })
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("unknown message type", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		if _, err := model.Request(t.Context(), []ai.ModelMessage{nil}, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("unknown request part", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{nil}}}
		if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("bad assistant tool args", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		msgs := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "t", Args: json.RawMessage(`not json`)},
		}}}
		if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("unserializable tool return", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "t", Content: make(chan int)},
		}}}
		if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("transport error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		server.Close()
		model := google.NewModel("m", google.WithBaseURL(server.URL))
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected transport error")
		}
	})
	t.Run("invalid URL", func(t *testing.T) {
		model := google.NewModel("m", google.WithBaseURL("http://[::1"))
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected URL error")
		}
	})
	t.Run("truncated body", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "1000")
			_, _ = w.Write([]byte(`{"model`))
		})
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected read error")
		}
	})
}

func TestModelName(t *testing.T) {
	if google.NewModel("gemini-2.5-flash").Name() != "gemini-2.5-flash" {
		t.Fatal("unexpected name")
	}
}

func TestEndToEndAgentRun(t *testing.T) {
	first := true
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		if first {
			first = false
			_, _ = w.Write([]byte(`{
				"candidates": [{"content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"city": "SF"}}}]}}],
				"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}
			}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"candidates": [{"content": {"parts": [{"text": "It is sunny in SF."}]}}],
			"usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 6}
		}`))
	})
	agent := ai.NewAgent[struct{}, string](model)
	ai.AddSimpleTool(agent, "get_weather", func(_ context.Context, args struct {
		City string `json:"city"`
	}) (string, error) {
		return "sunny in " + args.City, nil
	})
	result, err := agent.Run(t.Context(), "weather in SF?", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "It is sunny in SF." || result.Usage().Requests != 2 {
		t.Fatalf("unexpected result %+v", result)
	}
}

func TestAssistantHistoryWithText(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]}}],"usageMetadata":{}}`))
	})
	msgs := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.TextPart{Content: "previous answer"},
		ai.ToolCallPart{ToolName: "t"},
	}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	parts := gotBody["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	if parts[0].(map[string]any)["text"] != "previous answer" {
		t.Fatalf("text part lost: %v", parts)
	}
	if parts[1].(map[string]any)["functionCall"] == nil {
		t.Fatalf("empty-args tool call lost: %v", parts)
	}
}

func TestMultimodalUserPrompt(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"a cat"}]}}],"usageMetadata":{}}`))
	})
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
		ai.TextContent{Text: "what is this?"},
		ai.BinaryContent{Data: []byte("hi"), MediaType: "image/png"},
		ai.ImageURL{URL: "https://example.com/cat.png"},
	}}}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	parts := gotBody["contents"].([]any)[0].(map[string]any)["parts"].([]any)
	inline := parts[1].(map[string]any)["inlineData"].(map[string]any)
	if inline["mimeType"] != "image/png" || inline["data"] != "aGk=" {
		t.Fatalf("unexpected inline data %v", inline)
	}
	file := parts[2].(map[string]any)["fileData"].(map[string]any)
	if file["fileUri"] != "https://example.com/cat.png" {
		t.Fatalf("unexpected file data %v", file)
	}
}

func TestMultimodalUnknownContent(t *testing.T) {
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{nil}}}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected error")
	}
}
