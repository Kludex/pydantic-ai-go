package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func newResponsesServer(t *testing.T, handler http.HandlerFunc) *openai.ResponsesModel {
	t.Helper()
	return newResponsesServerWithOptions(t, handler)
}

func newResponsesServerWithOptions(
	t *testing.T, handler http.HandlerFunc, opts ...openai.Option,
) *openai.ResponsesModel {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	options := []openai.Option{
		openai.WithAPIKey("test-key"),
		openai.WithBaseURL(server.URL),
		openai.WithHTTPClient(server.Client()),
	}
	return openai.NewResponsesModel("gpt-5", append(options, opts...)...)
}

func TestResponsesCountTokens(t *testing.T) {
	var body map[string]any
	model := newResponsesServer(t, func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/responses/input_tokens" || request.Header.Get("X-Custom") != "value" {
			t.Errorf("unexpected token count request: %s headers=%v", request.URL.Path, request.Header)
		}
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"input_tokens":17}`))
	})
	if model.ProviderName() != "openai" || model.ProviderURL() == "" {
		t.Fatalf("unexpected provider identity: %q %q", model.ProviderName(), model.ProviderURL())
	}
	parallel := true
	usage, err := model.CountTokens(t.Context(), []ai.ModelMessage{ai.ModelRequest{
		Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}},
	}}, ai.ModelRequestParams{
		Instructions: "Be brief.",
		Tools:        []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}}},
		Settings: ai.ModelSettings{
			MaxTokens: 100, ParallelToolCalls: &parallel,
			ExtraHeaders: map[string]string{"X-Custom": "value"}, ExtraBody: map[string]any{"store": true},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if usage.InputTokens != 17 || usage.Requests != 0 {
		t.Fatalf("unexpected token usage: %+v", usage)
	}
	if body["model"] != "gpt-5" || body["instructions"] != "Be brief." || body["store"] != true ||
		body["max_output_tokens"] != nil || body["background"] != nil || body["include"] != nil {
		t.Fatalf("unexpected token count body: %v", body)
	}
	if _, err := model.CountTokens(t.Context(), nil, ai.ModelRequestParams{}); err == nil ||
		!strings.Contains(err.Error(), "cannot count tokens without messages") {
		t.Fatalf("unexpected empty token count error: %v", err)
	}
}

func TestResponsesCountTokensErrors(t *testing.T) {
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}}}}
	t.Run("payload", func(t *testing.T) {
		model := openai.NewResponsesModel("gpt-5")
		_, err := model.CountTokens(t.Context(), messages, ai.ModelRequestParams{Settings: ai.ModelSettings{
			Thinking: &ai.ThinkingSettings{Level: "extreme"},
		}})
		if err == nil || !strings.Contains(err.Error(), "invalid thinking level") {
			t.Fatalf("unexpected payload error: %v", err)
		}
	})
	t.Run("marshal", func(t *testing.T) {
		model := openai.NewResponsesModel("gpt-5")
		_, err := model.CountTokens(t.Context(), messages, ai.ModelRequestParams{Settings: ai.ModelSettings{
			ExtraBody: map[string]any{"bad": make(chan struct{})},
		}})
		if err == nil || !strings.Contains(err.Error(), "marshal token count request") {
			t.Fatalf("unexpected marshal error: %v", err)
		}
	})
	t.Run("request", func(t *testing.T) {
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL(":"))
		if _, err := model.CountTokens(t.Context(), messages, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected request construction error")
		}
	})
	t.Run("prepare", func(t *testing.T) {
		prepareErr := errors.New("prepare failed")
		model := openai.NewResponsesModel("gpt-5", openai.WithProvider(openai.ProviderConfig{
			Name: "compatible", BaseURL: "http://example.test",
			PrepareRequest: func(*http.Request) error { return prepareErr },
		}))
		if _, err := model.CountTokens(
			t.Context(), messages, ai.ModelRequestParams{},
		); !errors.Is(err, prepareErr) {
			t.Fatalf("unexpected prepare error: %v", err)
		}
	})
	for _, test := range []struct {
		name       string
		response   *http.Response
		requestErr error
		contains   string
	}{
		{name: "request failure", requestErr: errors.New("request failed"), contains: "token count request"},
		{name: "read failure", response: &http.Response{
			StatusCode: http.StatusOK, Body: compactionErrorBody{}, Header: make(http.Header),
		}, contains: "read token count response"},
		{name: "status", response: &http.Response{
			StatusCode: http.StatusBadRequest, Body: io.NopCloser(strings.NewReader("bad")), Header: make(http.Header),
		}, contains: "status 400"},
		{name: "decode", response: &http.Response{
			StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("{")), Header: make(http.Header),
		}, contains: "decode token count response"},
	} {
		t.Run(test.name, func(t *testing.T) {
			client := &http.Client{Transport: compactionRoundTripFunc(func(*http.Request) (*http.Response, error) {
				return test.response, test.requestErr
			})}
			model := openai.NewResponsesModel(
				"gpt-5", openai.WithBaseURL("http://example.test"), openai.WithHTTPClient(client),
			)
			if _, err := model.CountTokens(
				t.Context(), messages, ai.ModelRequestParams{},
			); err == nil || !strings.Contains(err.Error(), test.contains) {
				t.Fatalf("unexpected token count error: %v", err)
			}
		})
	}
}

func TestResponsesTextResponse(t *testing.T) {
	var gotBody map[string]any
	var gotPath, gotCustom string
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotCustom = r.Header.Get("x-custom")
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{
			"id": "response-1", "model": "gpt-5", "created_at": 1735689600.25,
			"status": "completed", "service_tier": "default",
			"output": [
				{"id": "reasoning-1", "type": "reasoning", "encrypted_content": "signature", "summary": [{"text": "thinking"}]},
				{"id": "message-1", "type": "message", "content": [{
					"type": "output_text", "text": "Hello!", "logprobs": [{"token": "Hello", "logprob": -0.1}]
				}]}
			],
			"usage": {
				"input_tokens": 12, "output_tokens": 5,
				"input_tokens_details": {"cached_tokens": 4},
				"output_tokens_details": {"reasoning_tokens": 2}
			}
		}`))
	})
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hi"}}}}
	temperature := 0.5
	logprobs := true
	topLogprobs := 3
	resp, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{
		Instructions: "be brief", AllowText: true,
		Settings: ai.ModelSettings{
			Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelHigh}, Temperature: &temperature,
			Logprobs: &logprobs, TopLogprobs: &topLogprobs, ServiceTier: ai.ServiceTierPriority,
			ExtraHeaders: map[string]string{"x-custom": "value"}, ExtraBody: map[string]any{"store": true},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != "/responses" || gotCustom != "value" {
		t.Fatalf("unexpected path %q or custom header %q", gotPath, gotCustom)
	}
	if gotBody["instructions"] != "be brief" || gotBody["temperature"] != nil ||
		gotBody["reasoning"].(map[string]any)["effort"] != "high" ||
		gotBody["top_logprobs"].(float64) != float64(topLogprobs) ||
		gotBody["include"].([]any)[0] != "message.output_text.logprobs" || gotBody["service_tier"] != "priority" ||
		gotBody["store"] != true {
		t.Fatalf("instructions or reasoning not sent: %v", gotBody)
	}
	if resp.Text() != "Hello!" {
		t.Fatalf("unexpected text %q", resp.Text())
	}
	thinking, ok := resp.Parts[0].(ai.ThinkingPart)
	if !ok || thinking.ID != "reasoning-1" || thinking.Signature != "signature" ||
		thinking.ProviderName != "openai" {
		t.Fatalf("reasoning metadata lost: %+v", resp.Parts)
	}
	text := resp.Parts[1].(ai.TextPart)
	if text.ID != "message-1" || text.ProviderName != "openai" ||
		len(text.ProviderDetails["logprobs"].([]map[string]any)) != 1 {
		t.Fatalf("text metadata lost: %+v", text)
	}
	if resp.ProviderName != "openai" || resp.ProviderURL == "" || resp.ProviderResponseID != "response-1" ||
		resp.FinishReason != ai.FinishReasonStop || resp.State != ai.ModelResponseStateComplete ||
		resp.ProviderDetails["finish_reason"] != "completed" || resp.ProviderDetails["timestamp"] == nil ||
		resp.ProviderDetails["service_tier"] != "default" {
		t.Fatalf("unexpected response metadata %+v", resp)
	}
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 5 ||
		resp.Usage.CacheReadTokens != 4 || resp.Usage.ReasoningTokens != 2 ||
		resp.Usage.Details["reasoning_tokens"] != 2 {
		t.Fatalf("unexpected usage %+v", resp.Usage)
	}
}

func TestResponsesRejectInvalidThinkingLevel(t *testing.T) {
	model := openai.NewResponsesModel("gpt-5")
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

func TestResponsesPreservesEncryptedReasoningWithoutSummary(t *testing.T) {
	model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"model":"gpt-5","service_tier":"priority",
			"output":[{"id":"reasoning-1","type":"reasoning","encrypted_content":"signature"}]
		}`))
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	thinking := response.Parts[0].(ai.ThinkingPart)
	if thinking.Content != "" || thinking.ID != "reasoning-1" || thinking.Signature != "signature" ||
		thinking.ProviderName != "openai" || response.ProviderDetails["service_tier"] != "priority" {
		t.Fatalf("encrypted reasoning was not retained: %+v", thinking)
	}
}

func TestResponsesPendingStateMetadata(t *testing.T) {
	for name, test := range map[string]struct {
		status     string
		reason     string
		background bool
		want       ai.ModelResponseState
		wantFinish ai.FinishReason
	}{
		"foreground": {status: "queued", want: ai.ModelResponseStateIncomplete},
		"background": {status: "queued", background: true, want: ai.ModelResponseStateSuspended},
		"incomplete reason": {
			status: "incomplete", reason: "max_output_tokens",
			want: ai.ModelResponseStateComplete, wantFinish: ai.FinishReasonLength,
		},
	} {
		t.Run(name, func(t *testing.T) {
			model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
				details := ""
				if test.reason != "" {
					details = fmt.Sprintf(`,"incomplete_details":{"reason":%q}`, test.reason)
				}
				_, _ = fmt.Fprintf(w,
					`{"id":"pending","model":"gpt-5","status":%q,"background":%t%s}`,
					test.status, test.background, details,
				)
			})
			response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			wantRawReason := test.status
			if test.reason != "" {
				wantRawReason = test.reason
			}
			if response.State != test.want || response.FinishReason != test.wantFinish ||
				response.ProviderDetails["finish_reason"] != wantRawReason ||
				response.ProviderDetails["background"] != test.background && test.background {
				t.Fatalf("unexpected pending metadata: %+v", response)
			}
		})
	}
}

func TestResponsesAgentContinuesBackgroundResponse(t *testing.T) {
	var methods, paths []string
	var initialBody map[string]any
	model := newResponsesServerWithOptions(t, func(w http.ResponseWriter, r *http.Request) {
		methods = append(methods, r.Method)
		paths = append(paths, r.URL.Path)
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("missing authorization header: %q", r.Header.Get("Authorization"))
		}
		if r.Method == http.MethodPost {
			if err := json.NewDecoder(r.Body).Decode(&initialBody); err != nil {
				t.Error(err)
			}
			_, _ = w.Write([]byte(`{
				"id":"job","model":"gpt-5","created_at":1735689600,
				"status":"queued","background":true,"usage":{"input_tokens":2}
			}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"id":"job","model":"gpt-5","created_at":1735689601,
			"status":"completed","background":true,
			"output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]}],
			"usage":{"input_tokens":2,"output_tokens":1}
		}`))
	}, openai.WithBackgroundMode(true), openai.WithBackgroundPollInterval(0))
	result, err := ai.NewAgent[struct{}, string](model).Run(t.Context(), "go", struct{}{})
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected background result=%+v err=%v", result, err)
	}
	if !slices.Equal(methods, []string{http.MethodPost, http.MethodGet}) ||
		!slices.Equal(paths, []string{"/responses", "/responses/job"}) || initialBody["background"] != true {
		t.Fatalf("unexpected background requests methods=%v paths=%v body=%v", methods, paths, initialBody)
	}
	if usage := result.Usage(); usage.Requests != 1 || usage.InputTokens != 2 || usage.OutputTokens != 1 {
		t.Fatalf("background usage was double counted: %+v", usage)
	}
	response := result.NewMessages()[1].(ai.ModelResponse)
	if response.Timestamp.IsZero() {
		t.Fatalf("retrieved response timestamp was lost: %+v", response)
	}
}

func TestResponsesBackgroundLifecycle(t *testing.T) {
	t.Run("delay", func(t *testing.T) {
		model := openai.NewResponsesModel("gpt-5", openai.WithBackgroundPollInterval(time.Millisecond))
		if delay := model.ContinuationDelay(ai.ModelResponse{
			State: ai.ModelResponseStateSuspended, ProviderDetails: map[string]any{"background": true},
		}); delay != time.Millisecond {
			t.Fatalf("unexpected background delay: %s", delay)
		}
		if delay := model.ContinuationDelay(ai.ModelResponse{State: ai.ModelResponseStateComplete}); delay != 0 {
			t.Fatalf("completed response had delay: %s", delay)
		}
	})

	t.Run("cancel", func(t *testing.T) {
		calls := 0
		model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
			calls++
			if r.Method != http.MethodPost || r.URL.Path != "/responses/job/cancel" {
				t.Fatalf("unexpected cancellation request %s %s", r.Method, r.URL.Path)
			}
			_, _ = w.Write([]byte(`{"id":"job","status":"cancelled"}`))
		})
		if err := model.CancelSuspendedResponse(t.Context(), ai.ModelResponse{
			ProviderName: "openai", ProviderResponseID: "job", ProviderDetails: map[string]any{"background": true},
		}); err != nil {
			t.Fatal(err)
		}
		if err := model.CancelSuspendedResponse(t.Context(), ai.ModelResponse{ProviderName: "other"}); err != nil {
			t.Fatal(err)
		}
		if calls != 1 {
			t.Fatalf("unexpected cancellation calls: %d", calls)
		}
	})

	t.Run("cancel error", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "cannot cancel", http.StatusConflict)
		})
		err := model.CancelSuspendedResponse(t.Context(), ai.ModelResponse{
			ProviderName: "openai", ProviderResponseID: "job", ProviderDetails: map[string]any{"background": true},
		})
		var apiErr *openai.APIError
		if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusConflict {
			t.Fatalf("unexpected cancel error: %v", err)
		}
	})
}

func TestResponsesBackgroundRequestFailures(t *testing.T) {
	messages := []ai.ModelMessage{ai.ModelResponse{
		ProviderName: "openai", ProviderResponseID: "job", State: ai.ModelResponseStateSuspended,
		ProviderDetails: map[string]any{"background": true},
	}}
	t.Run("invalid retrieve URL", func(t *testing.T) {
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL("http://[::1"))
		if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected retrieve URL error")
		}
	})
	t.Run("retrieve transport", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		server.Close()
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL(server.URL))
		if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected retrieve transport error")
		}
	})
	t.Run("retrieve read", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "100")
			_, _ = w.Write([]byte("short"))
		})
		if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected retrieve read error")
		}
	})
	t.Run("retrieve HTTP", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "missing", http.StatusNotFound)
		})
		if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected retrieve API error")
		}
	})

	response := ai.ModelResponse{
		ProviderName: "openai", ProviderResponseID: "job", ProviderDetails: map[string]any{"background": true},
	}
	t.Run("invalid cancel URL", func(t *testing.T) {
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL("http://[::1"))
		if err := model.CancelSuspendedResponse(t.Context(), response); err == nil {
			t.Fatal("expected cancel URL error")
		}
	})
	t.Run("cancel transport", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		server.Close()
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL(server.URL))
		if err := model.CancelSuspendedResponse(t.Context(), response); err == nil {
			t.Fatal("expected cancel transport error")
		}
	})
	t.Run("cancel read", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "100")
			_, _ = w.Write([]byte("short"))
		})
		if err := model.CancelSuspendedResponse(t.Context(), response); err == nil {
			t.Fatal("expected cancel read error")
		}
	})
}

func TestBackgroundPollIntervalRejectsNegativeValues(t *testing.T) {
	defer func() {
		if recovered := recover(); recovered != "openai: background poll interval must not be negative" {
			t.Fatalf("unexpected panic: %v", recovered)
		}
	}()
	_ = openai.WithBackgroundPollInterval(-time.Second)
}

func TestResponsesToolCallRoundTrip(t *testing.T) {
	var gotBody map[string]any
	first := true
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		if first {
			first = false
			_, _ = w.Write([]byte(`{
				"model": "gpt-5",
				"output": [{"id": "function-1", "type": "function_call", "call_id": "c1", "name": "get_weather", "namespace": "weather", "arguments": "{\"city\":\"SF\"}"}],
				"usage": {"input_tokens": 20, "output_tokens": 8}
			}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"model": "gpt-5",
			"output": [{"type": "message", "content": [{"type": "output_text", "text": "Sunny."}]}],
			"usage": {"input_tokens": 30, "output_tokens": 4}
		}`))
	})
	params := ai.ModelRequestParams{
		Tools:     []ai.ToolDefinition{{Name: "get_weather", Description: "d", Schema: map[string]any{"type": "object"}}},
		AllowText: true,
	}
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "weather?"}}}}
	resp, err := model.Request(t.Context(), msgs, params)
	if err != nil {
		t.Fatal(err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ToolCallID != "c1" || calls[0].ID != "function-1" ||
		calls[0].ProviderName != "openai" || calls[0].ProviderDetails["namespace"] != "weather" {
		t.Fatalf("unexpected calls %+v", calls)
	}
	msgs = append(msgs, *resp, ai.ModelRequest{Parts: []ai.RequestPart{
		ai.ToolReturnPart{ToolName: "get_weather", Content: "sunny", ToolCallID: "c1"},
		ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"archive"}, ToolCallID: "c1"},
	}})
	if _, err := model.Request(t.Context(), msgs, params); err != nil {
		t.Fatal(err)
	}
	input := gotBody["input"].([]any)
	callItem := input[1].(map[string]any)
	if callItem["type"] != "function_call" || callItem["call_id"] != "c1" || callItem["id"] != "function-1" ||
		callItem["namespace"] != "weather" {
		t.Fatalf("function call not echoed: %v", callItem)
	}
	outputItem := input[2].(map[string]any)
	if outputItem["type"] != "function_call_output" || outputItem["output"] != "sunny" {
		t.Fatalf("function output not sent: %v", outputItem)
	}
	if len(gotBody["tools"].([]any)) != 1 {
		t.Fatalf("tools not sent: %v", gotBody["tools"])
	}
}

func TestResponsesNativeDeferredToolSearch(t *testing.T) {
	var gotBody map[string]any
	request := 0
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		request++
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		if request == 1 {
			_, _ = w.Write([]byte(`{
				"id":"response-search","model":"gpt-5","status":"completed",
				"output":[{
					"id":"search-item","type":"tool_search_call","call_id":"search-call",
					"execution":"client","status":"completed","arguments":{"queries":["first"]}
				}],"usage":{}
			}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"id":"response-final","model":"gpt-5","status":"completed",
			"output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}],"usage":{}
		}`))
	})
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	search := ai.ToolDefinition{
		Name: ai.ToolSearchName, Description: "Find tools.", Schema: map[string]any{
			"type": "object", "properties": map[string]any{
				"queries": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
			},
		},
		ToolKind: ai.ToolPartKindToolSearch, ToolSearchStrategy: ai.ToolSearchStrategyCustom,
	}
	first := ai.ToolDefinition{Name: "first", Schema: schema, DeferLoading: true}
	second := ai.ToolDefinition{Name: "second", Schema: schema, DeferLoading: true}
	params := ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{search}, DeferredTools: []ai.ToolDefinition{first, second}, AllowText: true,
	}
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "find"}}}}
	response, err := model.Request(t.Context(), messages, params)
	if err != nil {
		t.Fatal(err)
	}
	call := response.Parts[0].(ai.ToolCallPart)
	if call.ToolName != ai.ToolSearchName || call.ToolKind != ai.ToolPartKindToolSearch ||
		call.ToolCallID != "search-call" || call.ID != "search-item" || string(call.Args) != `{"queries":["first"]}` ||
		call.ProviderDetails["execution"] != "client" || call.ProviderDetails["status"] != "completed" {
		t.Fatalf("unexpected client tool-search call: %+v", call)
	}
	wireTools := gotBody["tools"].([]any)
	if len(wireTools) != 3 || wireTools[0].(map[string]any)["name"] != "first" ||
		wireTools[0].(map[string]any)["defer_loading"] != true ||
		wireTools[1].(map[string]any)["name"] != "second" ||
		wireTools[2].(map[string]any)["type"] != "tool_search" ||
		wireTools[2].(map[string]any)["execution"] != "client" || wireTools[2].(map[string]any)["name"] != nil {
		t.Fatalf("unexpected native deferred tools: %+v", wireTools)
	}

	messages = append(messages, *response,
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{
				ToolName: ai.ToolSearchName, ToolCallID: "search-call", ToolKind: ai.ToolPartKindToolSearch,
				Content: ai.ToolSearchResult{DiscoveredTools: []ai.ToolSearchMatch{{Name: "first"}}},
			},
			ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"first"}, ToolCallID: "search-call"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "loader", ToolCallID: "loader", Args: []byte(`{}`),
		}}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "loader", ToolCallID: "loader", Content: "loaded"},
			ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"unknown", "second"}, ToolCallID: "loader"},
		}},
	)
	if _, err := model.Request(t.Context(), messages, params); err != nil {
		t.Fatal(err)
	}
	input := gotBody["input"].([]any)
	if len(input) != 6 {
		t.Fatalf("unexpected native search replay items: %+v", input)
	}
	searchCall := input[1].(map[string]any)
	searchOutput := input[2].(map[string]any)
	if searchCall["type"] != "tool_search_call" || searchCall["execution"] != "client" ||
		searchCall["arguments"].(map[string]any)["queries"].([]any)[0] != "first" ||
		searchOutput["type"] != "tool_search_output" || searchOutput["execution"] != "client" ||
		len(searchOutput["tools"].([]any)) != 1 || searchOutput["tools"].([]any)[0].(map[string]any)["name"] != "first" ||
		searchOutput["tools"].([]any)[0].(map[string]any)["defer_loading"] != nil {
		t.Fatalf("unexpected native search replay: call=%+v output=%+v", searchCall, searchOutput)
	}
	additional := input[5].(map[string]any)
	if additional["type"] != "additional_tools" || additional["role"] != "developer" ||
		len(additional["tools"].([]any)) != 1 || additional["tools"].([]any)[0].(map[string]any)["name"] != "second" {
		t.Fatalf("unexpected additional tools item: %+v", additional)
	}
}

func TestResponsesDeferredToolSupportCanBeDisabled(t *testing.T) {
	var gotBody map[string]any
	model := newResponsesServerWithOptions(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","status":"completed","output":[],"usage":{}}`))
	}, openai.WithDeferredToolSupport(false))
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{
			Name: ai.ToolSearchName, Schema: schema, ToolKind: ai.ToolPartKindToolSearch,
		}},
		DeferredTools: []ai.ToolDefinition{{Name: "hidden", Schema: schema, DeferLoading: true}},
	}); err != nil {
		t.Fatal(err)
	}
	tools := gotBody["tools"].([]any)
	if len(tools) != 1 || tools[0].(map[string]any)["type"] != "function" ||
		tools[0].(map[string]any)["name"] != ai.ToolSearchName {
		t.Fatalf("deferred support override was ignored: %+v", tools)
	}
}

func TestResponsesNativeDeferredErrors(t *testing.T) {
	model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model":"gpt-5","status":"completed","output":[],"usage":{}}`))
	})
	validSchema := map[string]any{"type": "object", "properties": map[string]any{}}
	invalidSchema := map[string]any{"type": "string"}
	search := ai.ToolDefinition{
		Name: ai.ToolSearchName, Schema: validSchema, ToolKind: ai.ToolPartKindToolSearch,
		ToolSearchStrategy: ai.ToolSearchStrategyCustom,
	}
	deferred := ai.ToolDefinition{Name: "hidden", Schema: validSchema, DeferLoading: true}
	searchReturn := func(content any) []ai.ModelMessage {
		return []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{
			ToolName: ai.ToolSearchName, ToolCallID: "search", ToolKind: ai.ToolPartKindToolSearch,
			Content: content,
		}}}}
	}
	invalidDeferred := deferred
	invalidDeferred.Schema = invalidSchema
	invalidSearch := search
	invalidSearch.Schema = invalidSchema
	for name, test := range map[string]struct {
		messages []ai.ModelMessage
		search   ai.ToolDefinition
		deferred ai.ToolDefinition
	}{
		"deferred definition": {search: search, deferred: invalidDeferred},
		"search definition":   {search: invalidSearch, deferred: deferred},
		"search result marshal": {
			messages: searchReturn(make(chan int)), search: search, deferred: deferred,
		},
		"search result parse": {
			messages: searchReturn("bad"), search: search, deferred: deferred,
		},
		"search result definition": {
			messages: searchReturn(ai.ToolSearchResult{DiscoveredTools: []ai.ToolSearchMatch{{Name: "hidden"}}}),
			search:   search, deferred: invalidDeferred,
		},
		"additional definition": {
			messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"hidden"}},
			}}},
			search: search, deferred: invalidDeferred,
		},
		"historical search arguments": {
			messages: []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: ai.ToolSearchName, ToolKind: ai.ToolPartKindToolSearch, Args: []byte(`{`),
			}}}},
			search: search, deferred: deferred,
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := model.Request(t.Context(), test.messages, ai.ModelRequestParams{
				Tools: []ai.ToolDefinition{test.search}, DeferredTools: []ai.ToolDefinition{test.deferred},
			})
			if err == nil {
				t.Fatal("expected native deferred conversion error")
			}
		})
	}
}

func TestResponsesOutputToolAndRetries(t *testing.T) {
	var gotBody map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","output":[{"type":"function_call","call_id":"c1","name":"final_result","arguments":"{}"}],"usage":{}}`))
	})
	params := ai.ModelRequestParams{
		OutputTool: &ai.ToolDefinition{Name: "final_result", Schema: map[string]any{"type": "object"}},
	}
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.SystemPromptPart{Content: "sys"},
		ai.RetryPromptPart{Content: "bad", ToolCallID: "c0"},
		ai.RetryPromptPart{Content: "plain"},
	}}}
	if _, err := model.Request(t.Context(), msgs, params); err != nil {
		t.Fatal(err)
	}
	if gotBody["tool_choice"] != "required" {
		t.Fatalf("expected required tool choice, got %v", gotBody["tool_choice"])
	}
	input := gotBody["input"].([]any)
	if input[0].(map[string]any)["role"] != "system" {
		t.Fatalf("system part lost: %v", input[0])
	}
	if input[1].(map[string]any)["type"] != "function_call_output" {
		t.Fatalf("tool retry should be function output: %v", input[1])
	}
	if input[2].(map[string]any)["role"] != "user" {
		t.Fatalf("plain retry should be user: %v", input[2])
	}
}

func TestResponsesToolSearchResponseErrorsAndFallbacks(t *testing.T) {
	for name, item := range map[string]string{
		"invalid function arguments":      `{"type":"function_call","call_id":"call","name":"work","arguments":"{"}`,
		"invalid client search arguments": `{"type":"tool_search_call","call_id":"search","execution":"client","arguments":"{"}`,
	} {
		t.Run(name, func(t *testing.T) {
			model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
				_, _ = fmt.Fprintf(w, `{"model":"gpt-5","status":"completed","output":[%s]}`, item)
			})
			if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
				t.Fatal("expected response conversion error")
			}
		})
	}

	model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"model":"gpt-5","status":"completed","output":[
				{"id":"fallback","type":"tool_search_call","execution":"client","arguments":null},
				{"type":"function_call","call_id":"empty","name":"work"}
			]
		}`))
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	search := response.Parts[0].(ai.ToolCallPart)
	function := response.Parts[1].(ai.ToolCallPart)
	if search.ToolCallID != "fallback" || string(search.Args) != `{}` || string(function.Args) != `{}` {
		t.Fatalf("missing argument fallbacks were not normalized: %+v", response.Parts)
	}
}

func TestResponsesErrors(t *testing.T) {
	t.Run("native output unsupported", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		params := ai.ModelRequestParams{OutputSchema: map[string]any{"type": "object"}}
		_, err := model.Request(t.Context(), nil, params)
		if err == nil || !strings.Contains(err.Error(), "OutputModeTool") {
			t.Fatalf("expected unsupported error, got %v", err)
		}
		params.OutputMode = ai.OutputModePrompted
		if _, err := model.Request(t.Context(), nil, params); err != nil {
			t.Fatalf("prompted output should not request native mode: %v", err)
		}
	})
	t.Run("api error", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte("bad"))
		})
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected API error")
		}
	})
	t.Run("invalid json", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("not json")) })
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected parse error")
		}
	})
	t.Run("unknown message type", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		if _, err := model.Request(t.Context(), []ai.ModelMessage{nil}, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("unknown request part", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{nil}}}
		if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("unserializable tool return", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{ToolName: "t", Content: make(chan int)}}}}
		if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("unserializable tool schema", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		params := ai.ModelRequestParams{Tools: []ai.ToolDefinition{{Name: "t", Schema: map[string]any{
			"type": "object", "properties": map[string]any{}, "required": []string{}, "bad": make(chan int),
		}}}}
		if _, err := model.Request(t.Context(), nil, params); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("transport error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		server.Close()
		model := openai.NewResponsesModel("m", openai.WithBaseURL(server.URL))
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected transport error")
		}
	})
	t.Run("invalid URL", func(t *testing.T) {
		model := openai.NewResponsesModel("m", openai.WithBaseURL("http://[::1"))
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected URL error")
		}
	})
	t.Run("truncated body", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "1000")
			_, _ = w.Write([]byte(`{"model`))
		})
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected read error")
		}
	})
}

func TestResponsesModelName(t *testing.T) {
	if openai.NewResponsesModel("gpt-5").Name() != "gpt-5" {
		t.Fatal("unexpected name")
	}
}

func vcrResponsesModel(t *testing.T, name string) *openai.ResponsesModel {
	t.Helper()
	mode := recorder.ModeReplayOnly
	if _, err := os.Stat("testdata/" + name + ".yaml"); os.IsNotExist(err) && os.Getenv("OPENAI_API_KEY") != "" {
		mode = recorder.ModeRecordOnce
	}
	r, err := recorder.New("testdata/"+name,
		recorder.WithMode(mode),
		recorder.WithHook(func(i *cassette.Interaction) error {
			delete(i.Request.Headers, "Authorization")
			return nil
		}, recorder.AfterCaptureHook),
		recorder.WithMatcher(cassette.MatcherFunc(func(r *http.Request, i cassette.Request) bool {
			return r.Method == i.Method && r.URL.String() == i.URL
		})),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Stop(); err != nil {
			t.Error(err)
		}
	})
	return openai.NewResponsesModel("gpt-4o-mini", openai.WithHTTPClient(r.GetDefaultClient()))
}

func TestRecordedResponsesRun(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](vcrResponsesModel(t, "responses_run"),
		ai.WithInstructions("Answer with a single word."),
	)
	result, err := agent.Run(t.Context(), "What is the capital of Spain?", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output == "" || result.Usage().TotalTokens() == 0 {
		t.Fatalf("unexpected result %+v", result)
	}
}

func TestRecordedResponsesToolRun(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](vcrResponsesModel(t, "responses_tool_run"),
		ai.WithInstructions("Use the get_weather tool, then answer briefly."),
	)
	called := false
	ai.AddSimpleTool(agent, "get_weather", func(_ context.Context, args struct {
		City string `json:"city" jsonschema:"description=City name"`
	}) (string, error) {
		called = true
		return "sunny, 21C in " + args.City, nil
	}, ai.WithDescription("Get current weather for a city"))
	result, err := agent.Run(t.Context(), "What is the weather in Berlin?", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if !called || result.Output == "" {
		t.Fatalf("called=%v output=%q", called, result.Output)
	}
}

func TestResponsesAssistantHistoryWithThinking(t *testing.T) {
	var gotBody map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`))
	})
	msgs := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.ThinkingPart{Content: "hidden"},
		ai.ThinkingPart{ID: "reasoning-1", Signature: "signature", ProviderName: "openai"},
		ai.TextPart{Content: "previous"},
	}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	input := gotBody["input"].([]any)
	if len(input) != 2 || input[0].(map[string]any)["type"] != "reasoning" ||
		input[0].(map[string]any)["id"] != "reasoning-1" ||
		input[0].(map[string]any)["encrypted_content"] != "signature" ||
		input[1].(map[string]any)["role"] != "assistant" {
		t.Fatalf("provider reasoning metadata was not round-tripped: %v", input)
	}
}

func TestResponsesStrictToolDefinition(t *testing.T) {
	var gotBody map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`))
	})
	strict := true
	params := ai.ModelRequestParams{AllowText: true, Tools: []ai.ToolDefinition{{
		Name: "search", Schema: map[string]any{"type": "object"}, Strict: &strict,
	}}}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	tool := gotBody["tools"].([]any)[0].(map[string]any)
	if tool["strict"] != true {
		t.Fatalf("strict flag not sent: %v", tool)
	}
}

func TestResponsesParallelToolCallsSetting(t *testing.T) {
	var gotBody map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`))
	})
	enabled := true
	params := ai.ModelRequestParams{
		AllowText: true,
		Tools:     []ai.ToolDefinition{{Name: "work", Schema: map[string]any{"type": "object"}}},
		Settings:  ai.ModelSettings{ParallelToolCalls: &enabled},
	}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	if gotBody["parallel_tool_calls"] != true {
		t.Fatalf("parallel setting not forwarded: %v", gotBody)
	}
}

func TestResponsesCompactionRoundTripTrimsHistory(t *testing.T) {
	var input []map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Input []map[string]any `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		input = body.Input
		_, _ = w.Write([]byte(`{
			"id":"response", "model":"gpt-5",
			"output":[
				{"id":"cmp-new","type":"compaction","encrypted_content":"new-encrypted"},
				{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]}
			],
			"usage":{"input_tokens":1,"output_tokens":1}
		}`))
	})
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "standing"}, ai.UserPromptPart{Content: "drop"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "drop older response"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.TextPart{Content: "drop before boundary"},
			ai.CompactionPart{
				Content: "summary", ID: "cmp-old", ProviderName: "openai",
				ProviderDetails: map[string]any{"encrypted_content": "old-encrypted"},
			},
			ai.TextPart{Content: "keep after boundary"},
		}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "keep tail"}}},
	}
	response, err := model.Request(t.Context(), messages, ai.ModelRequestParams{AllowText: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(input) != 4 || input[0]["role"] != "system" || input[0]["content"] != "standing" ||
		input[1]["type"] != "compaction" || input[1]["id"] != "cmp-old" ||
		input[1]["encrypted_content"] != "old-encrypted" || input[2]["content"] != "keep after boundary" ||
		input[3]["content"] != "keep tail" {
		t.Fatalf("unexpected compacted Responses input: %+v", input)
	}
	compaction, ok := response.Parts[0].(ai.CompactionPart)
	if !ok || compaction.ID != "cmp-new" || compaction.ProviderName != "openai" ||
		compaction.ProviderDetails["encrypted_content"] != "new-encrypted" ||
		response.ProviderDetails["compaction"] != true || response.Text() != "done" {
		t.Fatalf("unexpected Responses compaction: %+v", response.Parts)
	}
}

func TestResponsesIgnoresInvalidCompactionBoundaries(t *testing.T) {
	for name, compaction := range map[string]ai.CompactionPart{
		"foreign": {
			Content: "summary", ProviderName: "anthropic",
			ProviderDetails: map[string]any{"encrypted_content": "foreign"},
		},
		"missing encrypted content": {Content: "summary", ProviderName: "openai"},
	} {
		t.Run(name, func(t *testing.T) {
			var input []map[string]any
			model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
				var body struct {
					Input []map[string]any `json:"input"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				input = body.Input
				_, _ = w.Write([]byte(`{
					"id":"response", "model":"gpt-5", "status":"completed",
					"output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]}],
					"usage":{"input_tokens":1,"output_tokens":1}
				}`))
			})
			messages := []ai.ModelMessage{
				ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "keep"}}},
				ai.ModelResponse{Parts: []ai.ResponsePart{compaction, ai.TextPart{Content: "assistant"}}},
			}
			if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{AllowText: true}); err != nil {
				t.Fatal(err)
			}
			if len(input) != 2 || input[0]["content"] != "keep" || input[1]["content"] != "assistant" {
				t.Fatalf("invalid compaction changed input: %+v", input)
			}
		})
	}
}

func TestResponsesCompactionDoesNotDuplicatePlantedPrompt(t *testing.T) {
	var input []map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Input []map[string]any `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		input = body.Input
		_, _ = w.Write([]byte(`{
			"id":"response", "model":"gpt-5", "status":"completed",
			"output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]}],
			"usage":{}
		}`))
	})
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.SystemPromptPart{Content: "standing"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.CompactionPart{
			ProviderName: "openai", ProviderDetails: map[string]any{
				"encrypted_content": "opaque", ai.StandingPromptPlantedKey: true,
			},
		}}},
	}
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	if len(input) != 1 || input[0]["type"] != "compaction" {
		t.Fatalf("planted prompt was duplicated: %+v", input)
	}
}

func TestResponsesServerManagedToolSearch(t *testing.T) {
	request := 0
	var bodies []map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		request++
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		if request == 1 {
			_, _ = w.Write([]byte(`{
				"id":"response-search","model":"gpt-5.4","created_at":1735689600,"status":"completed",
				"output":[
					{"id":"tso-1","type":"tool_search_output","call_id":null,"execution":"server","status":"completed","tools":[{"type":"function","name":"weather","description":"","parameters":{"type":"object"}}]},
					{"id":"ts-1","type":"tool_search_call","call_id":null,"execution":"server","status":"completed","arguments":{"paths":["weather"]}},
					{"id":"fc-1","type":"function_call","call_id":"weather-1","name":"weather","namespace":"weather","arguments":{"city":"Paris"}}
				],"usage":{}
			}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"id":"response-final","model":"gpt-5.4","status":"completed",
			"output":[{"type":"message","content":[{"type":"output_text","text":"sunny"}]}],"usage":{}
		}`))
	})
	weather := ai.NewTool[struct{}, struct {
		City string `json:"city"`
	}, string]("weather", func(_ context.Context, _ *ai.RunContext[struct{}], args struct {
		City string `json:"city"`
	}) (string, error) {
		return args.City + ": sunny", nil
	}, ai.WithDeferredLoading())
	agent := ai.NewAgent[struct{}, string](model)
	agent.AddToolset(ai.WithToolSearch(ai.NewFunctionToolset(weather), ai.ToolSearchConfig[struct{}]{}))
	result, err := agent.Run(t.Context(), "weather", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "sunny" || request != 2 {
		t.Fatalf("unexpected server-search run: output=%q requests=%d", result.Output, request)
	}
	messages := result.Messages()
	searchResponse := messages[1].(ai.ModelResponse)
	call := searchResponse.Parts[0].(ai.NativeToolCallPart)
	returned := searchResponse.Parts[1].(ai.NativeToolReturnPart)
	function := searchResponse.Parts[2].(ai.ToolCallPart)
	if call.ToolCallID != "ts-1" || string(call.Args) != `{"queries":["weather"]}` ||
		returned.ToolCallID != "ts-1" || returned.Timestamp.IsZero() ||
		returned.Content.(ai.ToolSearchResult).DiscoveredTools[0].Name != "weather" ||
		function.ToolName != "weather" {
		t.Fatalf("unexpected normalized server search: %+v", searchResponse.Parts)
	}
	tools := bodies[0]["tools"].([]any)
	if len(tools) != 2 || tools[0].(map[string]any)["name"] != "weather" ||
		tools[0].(map[string]any)["defer_loading"] != true ||
		tools[1].(map[string]any)["type"] != "tool_search" ||
		tools[1].(map[string]any)["execution"] != nil {
		t.Fatalf("unexpected server search tools: %+v", tools)
	}
	input := bodies[1]["input"].([]any)
	var replayCall, replayOutput map[string]any
	for _, raw := range input {
		item := raw.(map[string]any)
		switch item["type"] {
		case "tool_search_call":
			replayCall = item
		case "tool_search_output":
			replayOutput = item
		}
	}
	if replayCall == nil || replayOutput == nil || replayCall["call_id"] != nil || replayOutput["call_id"] != nil ||
		replayCall["execution"] != "server" || replayOutput["execution"] != "server" ||
		replayCall["id"] != "ts-1" || replayOutput["id"] != "tso-1" ||
		len(replayOutput["tools"].([]any)) != 1 {
		t.Fatalf("unexpected server search replay: call=%+v output=%+v", replayCall, replayOutput)
	}
}

func TestResponsesRejectsNamedToolSearchStrategies(t *testing.T) {
	model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model":"gpt-5.4","status":"completed","output":[],"usage":{}}`))
	})
	if !model.SupportsToolSearchStrategy(ai.ToolSearchStrategyAuto) ||
		model.SupportsToolSearchStrategy(ai.ToolSearchStrategyBM25) ||
		model.SupportsToolSearchStrategy(ai.ToolSearchStrategyRegex) {
		t.Fatal("unexpected OpenAI tool-search strategy support")
	}
	for _, strategy := range []ai.ToolSearchStrategy{ai.ToolSearchStrategyBM25, ai.ToolSearchStrategyRegex} {
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
			Tools: []ai.ToolDefinition{{
				Name: ai.ToolSearchName, ToolKind: ai.ToolPartKindToolSearch, ToolSearchStrategy: strategy,
			}},
			DeferredTools: []ai.ToolDefinition{{Name: "hidden", DeferLoading: true}},
		})
		if err == nil || !strings.Contains(err.Error(), `tool search strategy "`+string(strategy)+`" is not supported`) {
			t.Fatalf("unexpected %s strategy error: %v", strategy, err)
		}
	}
}

func TestResponsesServerManagedToolSearchPairingEdges(t *testing.T) {
	t.Run("ambiguous null IDs", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{
				"model":"gpt-5.4","status":"completed","output":[
					{"id":"ts-a","type":"tool_search_call","call_id":null,"execution":"server","status":"completed","arguments":{"paths":["a"]}},
					{"id":"tso-a","type":"tool_search_output","call_id":null,"execution":"server","status":"completed","tools":[{"type":"function","name":"a"}]},
					{"id":"ts-b","type":"tool_search_call","call_id":null,"execution":"server","status":"completed","arguments":{"paths":["b"]}},
					{"id":"tso-b","type":"tool_search_output","call_id":null,"execution":"server","status":"completed","tools":[{"type":"function","name":"b"}]}
				],"usage":{}
			}`))
		})
		response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		if len(response.Parts) != 4 ||
			response.Parts[0].(ai.NativeToolCallPart).ToolCallID != "ts-a" ||
			response.Parts[1].(ai.NativeToolReturnPart).ToolCallID != "tso-a" ||
			response.Parts[2].(ai.NativeToolCallPart).ToolCallID != "ts-b" ||
			response.Parts[3].(ai.NativeToolReturnPart).ToolCallID != "tso-b" {
			t.Fatalf("ambiguous null IDs were guessed: %+v", response.Parts)
		}
	})
	t.Run("explicit output first", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{
				"model":"gpt-5.4","status":"completed","output":[
					{"id":"tso","type":"tool_search_output","call_id":"call","execution":"server","status":"in_progress","tools":[{"type":"function","name":"real"},{"type":"file_search"}]},
					{"id":"ts","type":"tool_search_call","call_id":"call","execution":"server","status":"incomplete","arguments":{}}
				],"usage":{}
			}`))
		})
		response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		call := response.Parts[0].(ai.NativeToolCallPart)
		returned := response.Parts[1].(ai.NativeToolReturnPart)
		matches := returned.Content.(ai.ToolSearchResult).DiscoveredTools
		if call.ToolCallID != "call" || returned.ToolCallID != "call" || len(matches) != 1 ||
			matches[0].Name != "real" || call.ProviderDetails["status"] != "incomplete" ||
			returned.ProviderDetails["status"] != "in_progress" {
			t.Fatalf("unexpected explicit pairing: %+v", response.Parts)
		}
	})
	t.Run("unknown execution and non-object arguments", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{
				"model":"gpt-5.4","status":"completed","output":[
					{"id":"server","type":"tool_search_call","execution":"server","arguments":[]},
					{"id":"future","type":"tool_search_call","execution":"future","arguments":{}}
				],"usage":{}
			}`))
		})
		response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		if len(response.Parts) != 1 || string(response.Parts[0].(ai.NativeToolCallPart).Args) != `[]` {
			t.Fatalf("unexpected search execution normalization: %+v", response.Parts)
		}
	})
	t.Run("unmatched and client output", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{
				"model":"gpt-5.4","status":"completed","output":[
					{"id":"client","type":"tool_search_output","execution":"client","status":"completed","tools":[]},
					{"id":"server","type":"tool_search_output","execution":"server","status":"completed","tools":[]}
				],"usage":{}
			}`))
		})
		response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		if len(response.Parts) != 1 || response.Parts[0].(ai.NativeToolReturnPart).ToolCallID != "server" {
			t.Fatalf("unexpected unmatched outputs: %+v", response.Parts)
		}
	})
}

func TestResponsesServerToolSearchReplayEdges(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	params := ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{
			Name: ai.ToolSearchName, Schema: schema, ToolKind: ai.ToolPartKindToolSearch,
		}},
		DeferredTools: []ai.ToolDefinition{{Name: "hidden", Schema: schema, DeferLoading: true}},
	}
	model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model":"gpt-5.4","status":"completed","output":[],"usage":{}}`))
	})
	for name, part := range map[string]ai.ResponsePart{
		"malformed call": ai.NativeToolCallPart{
			ToolName: ai.ToolSearchName, ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "openai", Args: []byte(`{`),
		},
		"malformed return": ai.NativeToolReturnPart{
			ToolName: ai.ToolSearchName, ToolKind: ai.ToolPartKindToolSearch, ProviderName: "openai",
			Content: make(chan int), ProviderDetails: map[string]any{"id": "output"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{part}}}, params)
			if err == nil {
				t.Fatal("expected native replay error")
			}
		})
	}

	var body map[string]any
	model = newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"gpt-5.4","status":"completed","output":[],"usage":{}}`))
	})
	history := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.NativeToolCallPart{
			ToolName: ai.ToolSearchName, ToolCallID: "fallback", ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "openai", ProviderDetails: map[string]any{"call_id": 42, "status": "future"},
		},
		ai.NativeToolReturnPart{
			ToolName: ai.ToolSearchName, ToolCallID: "ignored", ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "openai", Content: ai.ToolSearchResult{},
		},
		ai.NativeToolCallPart{
			ToolName: ai.ToolSearchName, ToolKind: ai.ToolPartKindToolSearch, ProviderName: "anthropic",
		},
		ai.NativeToolReturnPart{
			ToolName: "capability", ToolKind: ai.ToolPartKindCapabilityLoad, ProviderName: "openai",
		},
		ai.NativeToolReturnPart{
			ToolName: ai.ToolSearchName, ToolCallID: "explicit", ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "openai", Content: ai.ToolSearchResult{},
			ProviderDetails: map[string]any{
				"id": "output", "call_id": "explicit", "status": "in_progress",
			},
		},
	}}}
	if _, err := model.Request(t.Context(), history, params); err != nil {
		t.Fatal(err)
	}
	input := body["input"].([]any)
	if len(input) != 2 {
		t.Fatalf("unexpected replay items: %+v", input)
	}
	call := input[0].(map[string]any)
	output := input[1].(map[string]any)
	if call["call_id"] != "fallback" || call["status"] != "completed" ||
		output["call_id"] != "explicit" || output["status"] != "in_progress" {
		t.Fatalf("unexpected replay defaults: call=%+v output=%+v", call, output)
	}
}

func TestResponsesReplaysForeignNativeSearchThroughClientProtocol(t *testing.T) {
	var body map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{
			"model":"gpt-5.4","status":"completed",
			"output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}],"usage":{}
		}`))
	})
	hidden := ai.NewSimpleTool[struct{}](
		"hidden", func(context.Context, struct{}) (string, error) { return "hidden", nil },
		ai.WithDeferredLoading(),
	)
	agent := ai.NewAgent[struct{}, string](model)
	agent.AddToolset(ai.WithToolSearch(ai.NewFunctionToolset(hidden), ai.ToolSearchConfig[struct{}]{}))
	history := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.NativeToolCallPart{
			ToolName: ai.ToolSearchName, ToolCallID: "search", ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "anthropic", Args: []byte(`{"queries":["hidden"]}`),
		},
		ai.NativeToolReturnPart{
			ToolName: ai.ToolSearchName, ToolCallID: "search", ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "anthropic",
			Content:      ai.ToolSearchResult{DiscoveredTools: []ai.ToolSearchMatch{{Name: "hidden"}}},
		},
	}}}
	if _, err := agent.Run(t.Context(), "continue", struct{}{}, ai.WithMessageHistory(history)); err != nil {
		t.Fatal(err)
	}
	input := body["input"].([]any)
	if input[0].(map[string]any)["type"] != "tool_search_call" ||
		input[0].(map[string]any)["execution"] != "client" ||
		input[1].(map[string]any)["type"] != "tool_search_output" ||
		input[1].(map[string]any)["execution"] != "client" ||
		len(input[1].(map[string]any)["tools"].([]any)) != 1 {
		t.Fatalf("unexpected foreign native search replay: %+v", input)
	}
	tools := body["tools"].([]any)
	if tools[len(tools)-1].(map[string]any)["type"] != "tool_search" ||
		tools[len(tools)-1].(map[string]any)["execution"] != nil {
		t.Fatalf("current automatic search did not remain server-managed: %+v", tools)
	}
}
