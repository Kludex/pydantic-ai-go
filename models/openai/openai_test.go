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
