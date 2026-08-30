package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func newServer(t *testing.T, handler http.HandlerFunc) *openai.Model {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return openai.NewModel("gpt-5",
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(server.URL),
		openai.WithHTTPClient(server.Client()),
	)
}

func TestRequestTextResponse(t *testing.T) {
	var gotBody map[string]any
	var gotAuth string
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{
			"model": "gpt-5", "created": 1735689600,
			"choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
			"usage": {"prompt_tokens": 12, "completion_tokens": 3}
		}`))
	})

	msgs := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hi"}}},
	}
	temp := 0.5
	resp, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{
		Instructions: "be brief",
		AllowText:    true,
		Settings:     ai.ModelSettings{MaxTokens: 100, Temperature: &temp},
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer test-key" {
		t.Fatalf("unexpected auth header %q", gotAuth)
	}
	messages := gotBody["messages"].([]any)
	if len(messages) != 2 {
		t.Fatalf("expected system + user messages, got %v", messages)
	}
	if gotBody["max_completion_tokens"].(float64) != 100 || gotBody["temperature"].(float64) != 0.5 {
		t.Fatalf("settings not sent: %v", gotBody)
	}
	if resp.Text() != "Hello!" {
		t.Fatalf("unexpected text %q", resp.Text())
	}
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 3 || resp.Usage.Requests != 1 {
		t.Fatalf("unexpected usage %+v", resp.Usage)
	}
	if resp.ModelName != "gpt-5" {
		t.Fatalf("unexpected model name %q", resp.ModelName)
	}
}

func TestRequestToolCallRoundTrip(t *testing.T) {
	var gotBody map[string]any
	first := true
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		if first {
			first = false
			_, _ = w.Write([]byte(`{
				"model": "gpt-5", "created": 1735689600,
				"choices": [{"message": {"role": "assistant", "tool_calls": [
					{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}}
				]}}],
				"usage": {"prompt_tokens": 20, "completion_tokens": 8}
			}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"model": "gpt-5", "created": 1735689600,
			"choices": [{"message": {"role": "assistant", "content": "Sunny."}}],
			"usage": {"prompt_tokens": 30, "completion_tokens": 4}
		}`))
	})

	tools := []ai.ToolDefinition{{Name: "get_weather", Description: "d", Schema: map[string]any{"type": "object"}}}
	params := ai.ModelRequestParams{Tools: tools, AllowText: true}
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "weather?"}}}}

	resp, err := model.Request(t.Context(), msgs, params)
	if err != nil {
		t.Fatal(err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ToolName != "get_weather" || calls[0].ToolCallID != "c1" {
		t.Fatalf("unexpected calls %+v", calls)
	}

	msgs = append(msgs, *resp, ai.ModelRequest{Parts: []ai.RequestPart{
		ai.ToolReturnPart{ToolName: "get_weather", Content: "sunny", ToolCallID: "c1"},
	}})
	resp, err = model.Request(t.Context(), msgs, params)
	if err != nil {
		t.Fatal(err)
	}
	sent := gotBody["messages"].([]any)
	assistant := sent[1].(map[string]any)
	if assistant["tool_calls"] == nil {
		t.Fatalf("assistant tool calls not echoed back: %v", assistant)
	}
	toolMsg := sent[2].(map[string]any)
	if toolMsg["role"] != "tool" || toolMsg["tool_call_id"] != "c1" || toolMsg["content"] != "sunny" {
		t.Fatalf("unexpected tool message %v", toolMsg)
	}
	if resp.Text() != "Sunny." {
		t.Fatalf("unexpected text %q", resp.Text())
	}
}

func TestOutputToolForcesToolChoice(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{
			"model": "gpt-5", "created": 0,
			"choices": [{"message": {"role": "assistant", "tool_calls": [
				{"id": "c1", "type": "function", "function": {"name": "final_result", "arguments": "{}"}}
			]}}],
			"usage": {"prompt_tokens": 1, "completion_tokens": 1}
		}`))
	})
	params := ai.ModelRequestParams{
		OutputTool: &ai.ToolDefinition{Name: "final_result", Schema: map[string]any{"type": "object"}},
	}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	if gotBody["tool_choice"] != "required" {
		t.Fatalf("expected required tool choice, got %v", gotBody["tool_choice"])
	}
	if len(gotBody["tools"].([]any)) != 1 {
		t.Fatalf("output tool not sent: %v", gotBody["tools"])
	}
}

func TestRetryPromptMessages(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","created":0,"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`))
	})
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.SystemPromptPart{Content: "sys"},
		ai.RetryPromptPart{Content: "bad args", ToolName: "t", ToolCallID: "c1"},
		ai.RetryPromptPart{Content: "plain retry"},
	}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	sent := gotBody["messages"].([]any)
	if sent[1].(map[string]any)["role"] != "tool" {
		t.Fatalf("tool retry should be a tool message: %v", sent[1])
	}
	if sent[2].(map[string]any)["role"] != "user" {
		t.Fatalf("plain retry should be a user message: %v", sent[2])
	}
}

func TestAPIError(t *testing.T) {
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error": {"message": "rate limited"}}`))
	})
	_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{AllowText: true})
	var apiErr *openai.APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("expected APIError 429, got %v", err)
	}
}

func TestEndToEndAgentRun(t *testing.T) {
	first := true
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		if first {
			first = false
			_, _ = w.Write([]byte(`{
				"model": "gpt-5", "created": 0,
				"choices": [{"message": {"role": "assistant", "tool_calls": [
					{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}}
				]}}],
				"usage": {"prompt_tokens": 10, "completion_tokens": 5}
			}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"model": "gpt-5", "created": 0,
			"choices": [{"message": {"role": "assistant", "content": "It is sunny in SF."}}],
			"usage": {"prompt_tokens": 20, "completion_tokens": 6}
		}`))
	})

	agent := ai.NewAgent[struct{}, string](model, ai.WithInstructions("be helpful"))
	ai.AddSimpleTool(agent, "get_weather", func(_ context.Context, args struct {
		City string `json:"city"`
	}) (string, error) {
		return "sunny in " + args.City, nil
	})

	result, err := agent.Run(t.Context(), "weather in SF?", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "It is sunny in SF." {
		t.Fatalf("unexpected output %q", result.Output)
	}
	if result.Usage().Requests != 2 || result.Usage().TotalTokens() != 41 {
		t.Fatalf("unexpected usage %+v", result.Usage())
	}
}

func TestAPIErrorMessage(t *testing.T) {
	err := &openai.APIError{StatusCode: 500, Body: "oops"}
	if err.Error() != "openai: API returned status 500: oops" {
		t.Fatalf("unexpected message %q", err.Error())
	}
}

func TestStructuredToolReturnContent(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","created":0,"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`))
	})
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.ToolReturnPart{ToolName: "t", Content: map[string]any{"temp": 20}, ToolCallID: "c1"},
	}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	content := gotBody["messages"].([]any)[0].(map[string]any)["content"]
	if content != `{"temp":20}` {
		t.Fatalf("unexpected content %v", content)
	}
}

func TestUnserializableToolReturn(t *testing.T) {
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{}`))
	})
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.ToolReturnPart{ToolName: "t", Content: make(chan int)},
	}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err == nil {
		t.Fatal("expected error")
	}
}

func TestParseErrors(t *testing.T) {
	for name, body := range map[string]string{
		"invalid json": `not json`,
		"no choices":   `{"model":"gpt-5","choices":[],"usage":{}}`,
	} {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(body))
		})
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{AllowText: true}); err == nil {
			t.Fatalf("%s: expected error", name)
		}
	}
}

func TestRequestTransportError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	server.Close()
	model := openai.NewModel("gpt-5", openai.WithAPIKey("k"), openai.WithBaseURL(server.URL))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{AllowText: true}); err == nil {
		t.Fatal("expected transport error")
	}
}

func TestModelName(t *testing.T) {
	if openai.NewModel("gpt-5").Name() != "gpt-5" {
		t.Fatal("unexpected name")
	}
}

func TestUnserializableToolSchema(t *testing.T) {
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{}`))
	})
	params := ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{Name: "t", Schema: map[string]any{
			"type": "object", "properties": map[string]any{}, "required": []string{}, "bad": make(chan int),
		}}},
	}
	if _, err := model.Request(t.Context(), nil, params); err == nil {
		t.Fatal("expected marshal error")
	}
}

func TestTruncatedResponseBody(t *testing.T) {
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Length", "1000")
		_, _ = w.Write([]byte(`{"model"`))
	})
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{AllowText: true}); err == nil {
		t.Fatal("expected read error")
	}
}

func TestConvertUnknownMessageAndPartTypes(t *testing.T) {
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{}`))
	})
	if _, err := model.Request(t.Context(), []ai.ModelMessage{nil}, ai.ModelRequestParams{AllowText: true}); err == nil {
		t.Fatal("expected error for unknown message type")
	}
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{nil}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err == nil {
		t.Fatal("expected error for unknown part type")
	}
}

func TestAssistantMessageWithTextAndToolCalls(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","created":0,"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`))
	})
	msgs := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.TextPart{Content: "let me check"},
		ai.ThinkingPart{Content: "hidden"},
		ai.ToolCallPart{ToolName: "t", Args: json.RawMessage(`{}`), ToolCallID: "c1"},
	}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	assistant := gotBody["messages"].([]any)[0].(map[string]any)
	if assistant["content"] != "let me check" || assistant["tool_calls"] == nil {
		t.Fatalf("unexpected assistant message %v", assistant)
	}
}

func TestSystemPromptPartInHistory(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","created":0,"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`))
	})
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.SystemPromptPart{Content: "sys"}}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	if gotBody["messages"].([]any)[0].(map[string]any)["role"] != "system" {
		t.Fatalf("unexpected messages %v", gotBody["messages"])
	}
}

func TestInvalidBaseURL(t *testing.T) {
	model := openai.NewModel("gpt-5", openai.WithBaseURL("http://[::1"))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{AllowText: true}); err == nil {
		t.Fatal("expected URL error")
	}
}

func TestMultimodalUserPrompt(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","created":0,"choices":[{"message":{"role":"assistant","content":"a cat"}}],"usage":{}}`))
	})
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
		ai.TextContent{Text: "what is this?"},
		ai.ImageURL{URL: "https://example.com/cat.png"},
		ai.BinaryContent{Data: []byte("hi"), MediaType: "image/png"},
	}}}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	parts := gotBody["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(parts) != 3 {
		t.Fatalf("unexpected parts %v", parts)
	}
	if parts[0].(map[string]any)["type"] != "text" {
		t.Fatalf("unexpected first part %v", parts[0])
	}
	if parts[1].(map[string]any)["image_url"].(map[string]any)["url"] != "https://example.com/cat.png" {
		t.Fatalf("unexpected image part %v", parts[1])
	}
	dataURL := parts[2].(map[string]any)["image_url"].(map[string]any)["url"].(string)
	if dataURL != "data:image/png;base64,aGk=" {
		t.Fatalf("unexpected data URL %q", dataURL)
	}
}

func TestMultimodalUnknownContent(t *testing.T) {
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{nil}}}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected error for unknown content type")
	}
}

func TestNativeJSONOutputMode(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","created":0,"choices":[{"message":{"role":"assistant","content":"{}"}}],"usage":{}}`))
	})
	params := ai.ModelRequestParams{
		AllowText:    true,
		OutputSchema: map[string]any{"type": "object"},
	}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	format := gotBody["response_format"].(map[string]any)
	if format["type"] != "json_schema" {
		t.Fatalf("unexpected response format %v", format)
	}
	js := format["json_schema"].(map[string]any)
	if js["name"] != "final_result" || js["strict"] != true {
		t.Fatalf("unexpected json_schema %v", js)
	}
}

func TestStrictToolDefinition(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","created":0,"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`))
	})
	strict := true
	params := ai.ModelRequestParams{AllowText: true, Tools: []ai.ToolDefinition{{
		Name: "search", Schema: map[string]any{"type": "object"}, Strict: &strict,
	}}}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	function := gotBody["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	if function["strict"] != true {
		t.Fatalf("strict flag not sent: %v", function)
	}
}

func TestParallelToolCallsSetting(t *testing.T) {
	var bodies []map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		_, _ = w.Write([]byte(`{"model":"gpt-5","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`))
	})
	disabled := false
	params := ai.ModelRequestParams{
		AllowText: true,
		Tools:     []ai.ToolDefinition{{Name: "work", Schema: map[string]any{"type": "object"}}},
		Settings:  ai.ModelSettings{ParallelToolCalls: &disabled},
	}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	enabled := true
	params.Tools = nil
	params.Settings.ParallelToolCalls = &enabled
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	if bodies[0]["parallel_tool_calls"] != false {
		t.Fatalf("parallel setting not forwarded: %v", bodies[0])
	}
	if _, exists := bodies[1]["parallel_tool_calls"]; exists {
		t.Fatalf("parallel setting sent without tools: %v", bodies[1])
	}
}
