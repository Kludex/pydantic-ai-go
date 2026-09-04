package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
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

func TestDefaultSettingsAreDetached(t *testing.T) {
	stop := []string{"stop"}
	settings := ai.ModelSettings{MaxTokens: 42, StopSequences: stop}
	models := []ai.Model{
		openai.NewModel("gpt-test", openai.WithDefaultSettings(settings)),
		openai.NewResponsesModel("gpt-test", openai.WithDefaultSettings(settings)),
	}
	stop[0] = "changed"
	for _, model := range models {
		defaults := model.(ai.ModelDefaultSettings).DefaultModelSettings()
		if defaults.MaxTokens != 42 || defaults.StopSequences[0] != "stop" {
			t.Fatalf("unexpected defaults for %T: %+v", model, defaults)
		}
		defaults.StopSequences[0] = "mutated"
		if model.(ai.ModelDefaultSettings).DefaultModelSettings().StopSequences[0] != "stop" {
			t.Fatalf("defaults were mutable for %T", model)
		}
	}
}

func TestChatThinkingSettings(t *testing.T) {
	for name, test := range map[string]struct {
		level ai.ThinkingLevel
		want  string
	}{
		"disabled": {level: ai.ThinkingLevelDisabled, want: "none"},
		"enabled":  {level: ai.ThinkingLevelEnabled, want: "medium"},
		"effort":   {level: ai.ThinkingLevelHigh, want: "high"},
	} {
		t.Run(name, func(t *testing.T) {
			var body map[string]any
			model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"done"}}],"usage":{}}`))
			})
			temperature := 0.5
			_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
				Thinking: &ai.ThinkingSettings{Level: test.level}, Temperature: &temperature,
			}})
			if err != nil || body["reasoning_effort"] != test.want {
				t.Fatalf("unexpected thinking payload body=%v err=%v", body, err)
			}
			if test.level == ai.ThinkingLevelDisabled && body["temperature"] != temperature {
				t.Fatalf("disabled reasoning removed sampling settings: %v", body)
			}
			if test.level != ai.ThinkingLevelDisabled && body["temperature"] != nil {
				t.Fatalf("active reasoning retained incompatible sampling settings: %v", body)
			}
		})
	}

	model := openai.NewModel("gpt-5")
	_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		Thinking: &ai.ThinkingSettings{Level: "extreme"},
	}})
	if err == nil || err.Error() != `openai: invalid thinking level "extreme"` {
		t.Fatalf("unexpected invalid thinking error: %v", err)
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		ServiceTier: "expedited",
	}})
	if err == nil || err.Error() != `openai: invalid service tier "expedited"` {
		t.Fatalf("unexpected invalid service tier error: %v", err)
	}
}

func TestExtraBodyRejectsConflictsAndInvalidValues(t *testing.T) {
	model := openai.NewModel("gpt-5")
	for name, body := range map[string]map[string]any{
		"typed field conflict": {"model": "other"},
		"invalid value":        {"custom": func() {}},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
				ExtraBody: body,
			}})
			if err == nil || !strings.Contains(err.Error(), "openai: marshal request") {
				t.Fatalf("unexpected extra body error: %v", err)
			}
		})
	}
}

func TestChatRefusalResponse(t *testing.T) {
	model := newServer(t, func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(`{
			"id":"response","model":"gpt-5","created":1735689600,
			"choices":[{"message":{"refusal":"I cannot help with that."},"finish_reason":"stop"}]
		}`))
	})
	_, err := ai.NewAgent[struct{}, string](model).Run(t.Context(), "blocked", struct{}{})
	var filtered *ai.ContentFilterError
	if !errors.As(err, &filtered) || filtered.Response().FinishReason != ai.FinishReasonContentFilter ||
		filtered.Response().ProviderDetails["refusal"] != "I cannot help with that." ||
		filtered.Response().ProviderDetails["finish_reason"] != nil {
		t.Fatalf("unexpected refusal error: %v response=%+v", err, filtered)
	}
}

func TestRequestTextResponse(t *testing.T) {
	var gotBody map[string]any
	var gotAuth, gotCustom string
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotCustom = r.Header.Get("x-custom")
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{
			"id": "chat-1", "model": "gpt-5", "created": 1735689600,
			"service_tier": "default", "system_fingerprint": "fp-1",
			"choices": [{
				"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop",
				"logprobs": {"content": [{"token": "Hello", "logprob": -0.1, "top_logprobs": []}]}
			}],
			"usage": {
				"prompt_tokens": 12, "completion_tokens": 9,
				"prompt_tokens_details": {"cached_tokens": 4, "audio_tokens": 2},
				"completion_tokens_details": {
					"reasoning_tokens": 3, "audio_tokens": 1,
					"accepted_prediction_tokens": 2, "rejected_prediction_tokens": 1
				}
			}
		}`))
	})

	msgs := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hi"}}},
	}
	temp := 0.5
	presencePenalty := 0.2
	frequencyPenalty := 0.3
	logprobs := true
	topLogprobs := 3
	resp, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{
		Instructions: "be brief",
		AllowText:    true,
		Settings: ai.ModelSettings{
			MaxTokens: 100, Temperature: &temp,
			PresencePenalty: &presencePenalty, FrequencyPenalty: &frequencyPenalty,
			LogitBias: map[string]int{"42": 10}, Logprobs: &logprobs, TopLogprobs: &topLogprobs,
			ServiceTier:  ai.ServiceTierPriority,
			ExtraHeaders: map[string]string{"x-custom": "value"},
			ExtraBody:    map[string]any{"store": true},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer test-key" || gotCustom != "value" {
		t.Fatalf("unexpected headers auth=%q custom=%q", gotAuth, gotCustom)
	}
	messages := gotBody["messages"].([]any)
	if len(messages) != 2 {
		t.Fatalf("expected system + user messages, got %v", messages)
	}
	if gotBody["max_completion_tokens"].(float64) != 100 || gotBody["temperature"].(float64) != 0.5 ||
		gotBody["presence_penalty"].(float64) != presencePenalty ||
		gotBody["frequency_penalty"].(float64) != frequencyPenalty ||
		gotBody["logit_bias"].(map[string]any)["42"].(float64) != 10 || gotBody["logprobs"] != true ||
		gotBody["top_logprobs"].(float64) != float64(topLogprobs) || gotBody["service_tier"] != "priority" ||
		gotBody["store"] != true {
		t.Fatalf("settings not sent: %v", gotBody)
	}
	if resp.Text() != "Hello!" {
		t.Fatalf("unexpected text %q", resp.Text())
	}
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 9 || resp.Usage.Requests != 1 ||
		resp.Usage.CacheReadTokens != 4 || resp.Usage.InputAudioTokens != 2 ||
		resp.Usage.OutputAudioTokens != 1 || resp.Usage.ReasoningTokens != 3 ||
		resp.Usage.AcceptedPredictionTokens != 2 || resp.Usage.RejectedPredictionTokens != 1 ||
		resp.Usage.Details["reasoning_tokens"] != 3 || resp.Usage.Details["audio_tokens"] != 1 ||
		resp.Usage.Details["accepted_prediction_tokens"] != 2 ||
		resp.Usage.Details["rejected_prediction_tokens"] != 1 {
		t.Fatalf("unexpected usage %+v", resp.Usage)
	}
	if resp.ModelName != "gpt-5" || resp.ProviderName != "openai" ||
		resp.ProviderURL == "" || resp.ProviderResponseID != "chat-1" ||
		resp.FinishReason != ai.FinishReasonStop || resp.ProviderDetails["finish_reason"] != "stop" ||
		resp.ProviderDetails["service_tier"] != "default" || resp.ProviderDetails["system_fingerprint"] != "fp-1" ||
		len(resp.ProviderDetails["logprobs"].([]map[string]any)) != 1 {
		t.Fatalf("unexpected response metadata %+v", resp)
	}
}

func TestChatNativeToolCompatibility(t *testing.T) {
	model := newServer(t, openAIRequestRecorder(t, nil))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
	}); err == nil || !strings.Contains(err.Error(), `Chat Completions does not support native tool "web_search"`) {
		t.Fatalf("unexpected required native-tool error: %v", err)
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.WebSearchTool{Optional: true}},
	}); err != nil {
		t.Fatalf("optional native tool should be omitted: %v", err)
	}
	for _, nativeTool := range []ai.NativeTool{nil, (*ai.WebSearchTool)(nil)} {
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{nativeTool},
		}); err == nil || !strings.Contains(err.Error(), "native tool must not be nil") {
			t.Fatalf("unexpected nil native-tool error: %v", err)
		}
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.WebSearchTool{Optional: true}, ai.WebSearchTool{Optional: true}},
	}); err == nil || !strings.Contains(err.Error(), "duplicate native tool ID") {
		t.Fatalf("unexpected duplicate native-tool error: %v", err)
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.CodeExecutionTool{}},
	}); err == nil || !strings.Contains(err.Error(), `does not support native tool "code_execution"`) {
		t.Fatalf("unexpected unsupported native-tool error: %v", err)
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
		ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"archive"}, ToolCallID: "c1"},
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
	if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusTooManyRequests || !apiErr.IsModelAPIError() {
		t.Fatalf("expected fallback-eligible APIError 429, got %v", err)
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
	_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{AllowText: true})
	var transportError *ai.ModelTransportError
	if !errors.As(err, &transportError) || transportError.ModelName != "gpt-5" ||
		transportError.ProviderName != "openai" || transportError.Operation != "request" {
		t.Fatalf("unexpected transport error: %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err = model.Request(ctx, nil, ai.ModelRequestParams{AllowText: true})
	var apiError ai.ModelAPIError
	if !errors.Is(err, context.Canceled) || errors.As(err, &apiError) {
		t.Fatalf("caller cancellation was misclassified: %v", err)
	}
}

func TestModelName(t *testing.T) {
	model := openai.NewModel("gpt-5")
	if model.Name() != "gpt-5" || model.ProviderName() != "openai" ||
		model.ProviderURL() != "https://api.openai.com/v1" {
		t.Fatalf("unexpected model identity: %q %q %q", model.Name(), model.ProviderName(), model.ProviderURL())
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
	_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{AllowText: true})
	var transportError *ai.ModelTransportError
	if !errors.As(err, &transportError) || transportError.Operation != "read response" {
		t.Fatalf("unexpected read error: %v", err)
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

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"o1-mini","choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer server.Close()
	o1 := openai.NewModel("o1-mini", openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client()))
	if _, err := o1.Request(t.Context(), msgs, ai.ModelRequestParams{Instructions: "instructions"}); err != nil {
		t.Fatal(err)
	}
	messages := gotBody["messages"].([]any)
	if messages[0].(map[string]any)["role"] != "user" || messages[1].(map[string]any)["role"] != "user" {
		t.Fatalf("unexpected o1-mini instruction roles: %#v", messages)
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
		ai.ImageURL{URL: "https://example.com/cat.png", VendorMetadata: map[string]any{"detail": "low"}},
		ai.BinaryContent{
			Data: []byte("hi"), MediaType: "image/png", VendorMetadata: map[string]any{"detail": "high"},
		},
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
	if parts[1].(map[string]any)["image_url"].(map[string]any)["url"] != "https://example.com/cat.png" ||
		parts[1].(map[string]any)["image_url"].(map[string]any)["detail"] != "low" {
		t.Fatalf("unexpected image part %v", parts[1])
	}
	dataURL := parts[2].(map[string]any)["image_url"].(map[string]any)["url"].(string)
	if dataURL != "data:image/png;base64,aGk=" ||
		parts[2].(map[string]any)["image_url"].(map[string]any)["detail"] != "high" {
		t.Fatalf("unexpected data URL %q", dataURL)
	}
}

func TestChatFileContent(t *testing.T) {
	fileServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/audio":
			response.Header().Set("Content-Type", "audio/mpeg")
		case "/document.pdf":
			response.Header().Set("Content-Type", "application/pdf")
		case "/text.txt":
			response.Header().Set("Content-Type", "text/plain")
		case "/image":
			response.Header().Set("Content-Type", "image/png")
		default:
			response.Header().Set("Content-Type", "application/octet-stream")
		}
		_, _ = response.Write([]byte("file"))
	}))
	defer fileServer.Close()
	var body map[string]any
	model := newServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"choices":[{"message":{"content":"done"}}]}`))
	})
	textURL := fileServer.URL + "/text.txt"
	textIdentifier := (ai.DocumentURL{URL: textURL}).ResolvedIdentifier()
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
		ai.AudioURL{URL: fileServer.URL + "/audio", ForceDownload: ai.FileDownloadAllowLocal},
		ai.DocumentURL{URL: fileServer.URL + "/document.pdf", ForceDownload: ai.FileDownloadAllowLocal},
		ai.DocumentURL{URL: textURL, ForceDownload: ai.FileDownloadAllowLocal},
		ai.ImageURL{URL: fileServer.URL + "/image", ForceDownload: ai.FileDownloadAllowLocal},
		ai.BinaryContent{Data: []byte("audio"), MediaType: "audio/wav"},
		ai.BinaryContent{Data: []byte("document"), MediaType: "application/pdf"},
		ai.BinaryContent{Data: []byte("config"), MediaType: "text/plain", Identifier: "config"},
		ai.UploadedFile{FileID: "file-report", ProviderName: "openai", MediaType: "application/pdf"},
	}}}}}
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	parts := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if parts[0].(map[string]any)["input_audio"].(map[string]any)["data"] != "ZmlsZQ==" ||
		parts[0].(map[string]any)["input_audio"].(map[string]any)["format"] != "mp3" ||
		parts[1].(map[string]any)["file"].(map[string]any)["file_data"] !=
			"data:application/pdf;base64,ZmlsZQ==" ||
		parts[1].(map[string]any)["file"].(map[string]any)["filename"] != "filename.pdf" ||
		parts[2].(map[string]any)["text"] !=
			"-----BEGIN FILE id=\""+textIdentifier+"\" type=\"text/plain\"-----\nfile\n-----END FILE id=\""+
				textIdentifier+"\"-----" ||
		parts[3].(map[string]any)["image_url"].(map[string]any)["url"] != "data:image/png;base64,ZmlsZQ==" ||
		parts[4].(map[string]any)["input_audio"].(map[string]any)["format"] != "wav" ||
		parts[5].(map[string]any)["file"].(map[string]any)["filename"] != "filename.pdf" ||
		parts[6].(map[string]any)["text"] !=
			"-----BEGIN FILE id=\"config\" type=\"text/plain\"-----\nconfig\n-----END FILE id=\"config\"-----" ||
		parts[7].(map[string]any)["file"].(map[string]any)["file_id"] != "file-report" {
		t.Fatalf("unexpected Chat file content: %#v", parts)
	}
}

func TestChatFileContentCompatibility(t *testing.T) {
	contentServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/audio.ogg":
			response.Header().Set("Content-Type", "audio/ogg")
		case "/unknown.bin":
			response.Header().Set("Content-Type", "application/unknown")
		default:
			response.Header().Set("Content-Type", "application/octet-stream")
		}
		_, _ = response.Write([]byte("content"))
	}))
	defer contentServer.Close()
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		_, _ = response.Write([]byte(`{"choices":[{"message":{"content":"done"}}]}`))
	}))
	defer server.Close()
	model := openai.NewModel(
		"compatible",
		openai.WithBaseURL(server.URL),
		openai.WithHTTPClient(server.Client()),
		openai.WithChatCompatibility(openai.ChatCompatibility{
			FileURLInput: true, AudioInputDataURI: true,
		}),
	)
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
		ai.DocumentURL{URL: "https://example.com/report.pdf"},
		ai.BinaryContent{Data: []byte("audio"), MediaType: "audio/mpeg"},
	}}}}}
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	parts := bodies[0]["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if parts[0].(map[string]any)["file"].(map[string]any)["file_data"] != "https://example.com/report.pdf" ||
		parts[1].(map[string]any)["input_audio"].(map[string]any)["data"] !=
			"data:audio/mpeg;base64,YXVkaW8=" {
		t.Fatalf("unexpected compatible file content: %#v", parts)
	}

	disabled := openai.NewModel(
		"compatible", openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client()),
		openai.WithChatDocumentInput(false),
	)
	enabled := openai.NewModel(
		"compatible", openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client()),
	)
	for _, test := range []struct {
		name    string
		model   *openai.Model
		content ai.UserContent
	}{
		{name: "document disabled", model: disabled, content: ai.BinaryContent{
			Data: []byte("pdf"), MediaType: "application/pdf",
		}},
		{name: "unsupported audio", model: enabled, content: ai.BinaryContent{
			Data: []byte("audio"), MediaType: "audio/ogg",
		}},
		{name: "unsupported file", model: enabled, content: ai.BinaryContent{
			Data: []byte("data"), MediaType: "application/unknown",
		}},
		{name: "foreign uploaded file", model: enabled, content: ai.UploadedFile{
			FileID: "file", ProviderName: "anthropic", MediaType: "application/pdf",
		}},
		{name: "uploaded image", model: enabled, content: ai.UploadedFile{
			FileID: "file", ProviderName: "openai", MediaType: "image/png",
		}},
		{name: "invalid image mode", model: enabled, content: ai.ImageURL{
			URL: "https://example.com/image.png", ForceDownload: "invalid",
		}},
		{name: "blocked image", model: enabled, content: ai.ImageURL{
			URL: "http://127.0.0.1/image.png", ForceDownload: ai.FileDownloadSafe,
		}},
		{name: "blocked audio", model: enabled, content: ai.AudioURL{
			URL: "http://127.0.0.1/audio.mp3",
		}},
		{name: "invalid audio mode", model: enabled, content: ai.AudioURL{
			URL: "https://example.com/audio.mp3", ForceDownload: "invalid",
		}},
		{name: "unsupported downloaded audio", model: enabled, content: ai.AudioURL{
			URL: contentServer.URL + "/audio.ogg", ForceDownload: ai.FileDownloadAllowLocal,
		}},
		{name: "unknown document media", model: enabled, content: ai.DocumentURL{
			URL: "https://example.com/report",
		}},
		{name: "invalid document mode", model: enabled, content: ai.DocumentURL{
			URL: "https://example.com/report.pdf", ForceDownload: "invalid",
		}},
		{name: "blocked document", model: enabled, content: ai.DocumentURL{
			URL: "http://127.0.0.1/report.pdf", ForceDownload: ai.FileDownloadSafe,
		}},
		{name: "unsupported downloaded document", model: enabled, content: ai.DocumentURL{
			URL: contentServer.URL + "/unknown.bin", MediaType: "application/pdf",
			ForceDownload: ai.FileDownloadAllowLocal,
		}},
		{name: "unsupported direct document", model: model, content: ai.DocumentURL{
			URL: "https://example.com/report.json",
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := test.model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Contents: []ai.UserContent{test.content}},
			}}}, ai.ModelRequestParams{})
			if err == nil {
				t.Fatal("unsupported Chat file content succeeded")
			}
		})
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
	params.OutputMode = ai.OutputModePrompted
	gotBody = nil
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	if gotBody["response_format"] != nil {
		t.Fatalf("prompted output enabled native response format: %+v", gotBody)
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
