package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/anthropic"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

type unsupportedNativeTool struct{ optional bool }

func (tool unsupportedNativeTool) Kind() string                   { return "unsupported" }
func (tool unsupportedNativeTool) UniqueID() string               { return "unsupported" }
func (tool unsupportedNativeTool) IsOptional() bool               { return tool.optional }
func (tool unsupportedNativeTool) CloneNativeTool() ai.NativeTool { return tool }

func newServer(t *testing.T, handler http.HandlerFunc) *anthropic.Model {
	t.Helper()
	return newServerWithOptions(t, handler)
}

func newNamedServer(t *testing.T, name string, handler http.HandlerFunc) *anthropic.Model {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return anthropic.NewModel(
		name, anthropic.WithAPIKey("test-key"), anthropic.WithBaseURL(server.URL),
		anthropic.WithHTTPClient(server.Client()),
	)
}

func newServerWithOptions(
	t *testing.T, handler http.HandlerFunc, opts ...anthropic.Option,
) *anthropic.Model {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	options := []anthropic.Option{
		anthropic.WithAPIKey("test-key"),
		anthropic.WithBaseURL(server.URL),
		anthropic.WithHTTPClient(server.Client()),
	}
	return anthropic.NewModel("claude-sonnet-4-5", append(options, opts...)...)
}

func TestAnthropicCountTokens(t *testing.T) {
	var body map[string]any
	model := newServer(t, func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/messages/count_tokens" || request.URL.Query().Get("beta") != "true" ||
			request.Header.Get("X-Test") != "value" || request.Header.Get("x-api-key") != "test-key" {
			t.Errorf("unexpected token count request: %s headers=%v", request.URL.Path, request.Header)
		}
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"input_tokens":19}`))
	})
	parallel := false
	temperature := 0.5
	usage, err := model.CountTokens(t.Context(), []ai.ModelMessage{ai.ModelRequest{
		Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}},
	}}, ai.ModelRequestParams{
		Instructions: "Be brief.",
		Tools:        []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}}},
		NativeTools:  []ai.NativeTool{ai.WebSearchTool{}},
		Settings: ai.ModelSettings{
			MaxTokens: 100, Temperature: &temperature, ParallelToolCalls: &parallel,
			ExtraHeaders: map[string]string{"X-Test": "value"}, ExtraBody: map[string]any{"custom": true},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if usage.InputTokens != 19 || usage.Requests != 0 {
		t.Fatalf("unexpected token usage: %+v", usage)
	}
	if body["model"] != "claude-sonnet-4-5" || body["system"] != "Be brief." || body["custom"] != true ||
		body["max_tokens"] != nil || body["temperature"] != nil {
		t.Fatalf("unexpected token count body: %v", body)
	}
	tools := body["tools"].([]any)
	if len(tools) != 1 || tools[0].(map[string]any)["name"] != "lookup" ||
		tools[0].(map[string]any)["type"] != nil {
		t.Fatalf("token count did not isolate function tools: %#v", tools)
	}
}

type anthropicRoundTripFunc func(*http.Request) (*http.Response, error)

func (function anthropicRoundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

type anthropicErrorBody struct{}

func (anthropicErrorBody) Read([]byte) (int, error) { return 0, errors.New("read failed") }
func (anthropicErrorBody) Close() error             { return nil }

func TestAnthropicCountTokensErrors(t *testing.T) {
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}}}}
	t.Run("payload", func(t *testing.T) {
		model := anthropic.NewModel("claude")
		_, err := model.CountTokens(t.Context(), messages, ai.ModelRequestParams{Settings: ai.ModelSettings{
			Thinking: &ai.ThinkingSettings{Level: "extreme"},
		}})
		if err == nil || !strings.Contains(err.Error(), "invalid thinking level") {
			t.Fatalf("unexpected payload error: %v", err)
		}
	})
	t.Run("marshal", func(t *testing.T) {
		model := anthropic.NewModel("claude")
		_, err := model.CountTokens(t.Context(), messages, ai.ModelRequestParams{Settings: ai.ModelSettings{
			ExtraBody: map[string]any{"bad": make(chan struct{})},
		}})
		if err == nil || !strings.Contains(err.Error(), "marshal token count request") {
			t.Fatalf("unexpected marshal error: %v", err)
		}
	})
	t.Run("request", func(t *testing.T) {
		model := anthropic.NewModel("claude", anthropic.WithBaseURL(":"))
		if _, err := model.CountTokens(t.Context(), messages, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected request construction error")
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
			StatusCode: http.StatusOK, Body: anthropicErrorBody{}, Header: make(http.Header),
		}, contains: "read token count response"},
		{name: "status", response: &http.Response{
			StatusCode: http.StatusBadRequest, Body: io.NopCloser(strings.NewReader("bad")), Header: make(http.Header),
		}, contains: "status 400"},
		{name: "decode", response: &http.Response{
			StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("{")), Header: make(http.Header),
		}, contains: "decode token count response"},
		{name: "missing", response: &http.Response{
			StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("{}")), Header: make(http.Header),
		}, contains: "omitted input_tokens"},
	} {
		t.Run(test.name, func(t *testing.T) {
			client := &http.Client{Transport: anthropicRoundTripFunc(func(*http.Request) (*http.Response, error) {
				return test.response, test.requestErr
			})}
			model := anthropic.NewModel(
				"claude", anthropic.WithBaseURL("http://example.test"), anthropic.WithHTTPClient(client),
			)
			if _, err := model.CountTokens(
				t.Context(), messages, ai.ModelRequestParams{},
			); err == nil || !strings.Contains(err.Error(), test.contains) {
				t.Fatalf("unexpected token count error: %v", err)
			}
		})
	}
}

func TestDefaultSettingsAreDetached(t *testing.T) {
	stop := []string{"stop"}
	model := anthropic.NewModel("claude-test", anthropic.WithDefaultSettings(ai.ModelSettings{
		MaxTokens: 42, StopSequences: stop,
	}))
	stop[0] = "changed"
	defaults := model.DefaultModelSettings()
	if defaults.MaxTokens != 42 || defaults.StopSequences[0] != "stop" {
		t.Fatalf("unexpected defaults: %+v", defaults)
	}
	defaults.StopSequences[0] = "mutated"
	if model.DefaultModelSettings().StopSequences[0] != "stop" {
		t.Fatal("model defaults were mutable")
	}
}

func TestThinkingSettings(t *testing.T) {
	for name, test := range map[string]struct {
		level  ai.ThinkingLevel
		budget *int
		want   int
	}{
		"enabled": {level: ai.ThinkingLevelEnabled, want: 10000},
		"minimal": {level: ai.ThinkingLevelMinimal, want: 1024},
		"low":     {level: ai.ThinkingLevelLow, want: 2048},
		"medium":  {level: ai.ThinkingLevelMedium, want: 10000},
		"high":    {level: ai.ThinkingLevelHigh, want: 16384},
		"xhigh":   {level: ai.ThinkingLevelXHigh, want: 32768},
		"budget":  {budget: func() *int { value := 777; return &value }(), want: 777},
	} {
		t.Run(name, func(t *testing.T) {
			var body map[string]any
			model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				_, _ = w.Write([]byte(`{"model":"claude","stop_reason":"end_turn","content":[{"type":"text","text":"done"}],"usage":{}}`))
			})
			_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
				Thinking: &ai.ThinkingSettings{Level: test.level, TokenBudget: test.budget},
			}})
			thinking := body["thinking"].(map[string]any)
			if err != nil || thinking["type"] != "enabled" || int(thinking["budget_tokens"].(float64)) != test.want {
				t.Fatalf("unexpected thinking payload body=%v err=%v", body, err)
			}
		})
	}

	t.Run("disabled and empty", func(t *testing.T) {
		requests := 0
		model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
			requests++
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			if body["thinking"] != nil {
				t.Fatalf("disabled thinking was sent: %v", body)
			}
			_, _ = w.Write([]byte(`{"model":"claude","content":[],"usage":{}}`))
		})
		include := false
		for _, thinking := range []*ai.ThinkingSettings{
			{Level: ai.ThinkingLevelDisabled}, {IncludeThoughts: &include},
		} {
			if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
				Settings: ai.ModelSettings{Thinking: thinking},
			}); err != nil {
				t.Fatal(err)
			}
		}
		if requests != 2 {
			t.Fatalf("unexpected request count: %d", requests)
		}
	})

	for name, thinking := range map[string]*ai.ThinkingSettings{
		"invalid level": {Level: "extreme"},
		"zero budget":   {TokenBudget: func() *int { value := 0; return &value }()},
	} {
		t.Run(name, func(t *testing.T) {
			model := anthropic.NewModel("claude")
			if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
				Settings: ai.ModelSettings{Thinking: thinking},
			}); err == nil {
				t.Fatal("expected thinking error")
			}
		})
	}

	t.Run("forced output", func(t *testing.T) {
		model := anthropic.NewModel("claude")
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
			Settings:   ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelEnabled}},
			OutputTool: &ai.ToolDefinition{Name: "final", Schema: map[string]any{"type": "object"}},
		})
		if err == nil || !strings.Contains(err.Error(), "extended thinking and forced output tools") {
			t.Fatalf("unexpected forced output error: %v", err)
		}
	})
}

func TestServiceTierMapping(t *testing.T) {
	for name, test := range map[string]struct {
		tier ai.ServiceTier
		want any
	}{
		"auto":         {tier: ai.ServiceTierAuto, want: "auto"},
		"flex omitted": {tier: ai.ServiceTierFlex},
	} {
		t.Run(name, func(t *testing.T) {
			var body map[string]any
			model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				_, _ = w.Write([]byte(`{"model":"claude","content":[],"usage":{}}`))
			})
			if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
				ServiceTier: test.tier,
			}}); err != nil {
				t.Fatal(err)
			}
			if body["service_tier"] != test.want {
				t.Fatalf("unexpected service tier payload: %v", body)
			}
		})
	}

	_, err := anthropic.NewModel("claude").Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		ServiceTier: "expedited",
	}})
	if err == nil || err.Error() != `anthropic: invalid service tier "expedited"` {
		t.Fatalf("unexpected service tier error: %v", err)
	}
}

func TestExtraBodyRejectsConflictsAndInvalidValues(t *testing.T) {
	model := anthropic.NewModel("claude")
	for name, body := range map[string]map[string]any{
		"typed field conflict": {"model": "other"},
		"invalid value":        {"custom": func() {}},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
				ExtraBody: body,
			}})
			if err == nil || !strings.Contains(err.Error(), "anthropic: marshal request") {
				t.Fatalf("unexpected extra body error: %v", err)
			}
		})
	}
}

func TestRequestTextResponse(t *testing.T) {
	var gotBody map[string]any
	var gotKey, gotVersion, gotCustom string
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("x-api-key")
		gotVersion = r.Header.Get("anthropic-version")
		gotCustom = r.Header.Get("x-custom")
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{
			"id": "message-1", "model": "claude-sonnet-4-5", "stop_reason": "end_turn",
			"service_tier": "standard",
			"content": [{"type": "text", "text": "Hello!"}],
			"usage": {
				"input_tokens": 12, "output_tokens": 3,
				"cache_creation_input_tokens": 3, "cache_read_input_tokens": 4
			}
		}`))
	})
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hi"}}}}
	resp, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{
		Instructions: "be brief",
		AllowText:    true,
		Settings: ai.ModelSettings{
			ServiceTier:  ai.ServiceTierDefault,
			ExtraHeaders: map[string]string{"x-custom": "value"},
			ExtraBody:    map[string]any{"container": "test-container"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotKey != "test-key" || gotVersion != "2023-06-01" || gotCustom != "value" {
		t.Fatalf("unexpected headers key=%q version=%q custom=%q", gotKey, gotVersion, gotCustom)
	}
	if gotBody["system"] != "be brief" {
		t.Fatalf("system prompt not sent: %v", gotBody)
	}
	if gotBody["max_tokens"].(float64) != 4096 || gotBody["service_tier"] != "standard_only" ||
		gotBody["container"] != "test-container" {
		t.Fatalf("default settings not applied: %v", gotBody)
	}
	if resp.Text() != "Hello!" {
		t.Fatalf("unexpected text %q", resp.Text())
	}
	if resp.ProviderName != "anthropic" || resp.ProviderURL == "" || resp.ProviderResponseID != "message-1" ||
		resp.FinishReason != ai.FinishReasonStop || resp.ProviderDetails["finish_reason"] != "end_turn" ||
		resp.ProviderDetails["service_tier"] != "standard" || resp.State != ai.ModelResponseStateComplete {
		t.Fatalf("unexpected response metadata %+v", resp)
	}
	if resp.Usage.InputTokens != 19 || resp.Usage.OutputTokens != 3 ||
		resp.Usage.CacheWriteTokens != 3 || resp.Usage.CacheReadTokens != 4 ||
		resp.Usage.Details["input_tokens"] != 12 || resp.Usage.Details["output_tokens"] != 3 ||
		resp.Usage.Details["cache_creation_input_tokens"] != 3 ||
		resp.Usage.Details["cache_read_input_tokens"] != 4 {
		t.Fatalf("unexpected usage %+v", resp.Usage)
	}
}

func TestPauseTurnResponseIsSuspended(t *testing.T) {
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"paused","model":"claude","stop_reason":"pause_turn","content":[],
			"usage":{"input_tokens":1,"output_tokens":1}
		}`))
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if response.State != ai.ModelResponseStateSuspended || response.FinishReason != "" ||
		response.ProviderDetails["finish_reason"] != "pause_turn" {
		t.Fatalf("unexpected paused response: %+v", response)
	}
}

func TestAgentAutomaticallyContinuesPauseTurn(t *testing.T) {
	var bodies []map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		if len(bodies) == 1 {
			_, _ = w.Write([]byte(`{
				"id":"paused","model":"claude-sonnet-4-5","stop_reason":"pause_turn",
				"content":[{"type":"text","text":"partial "}],
				"usage":{"input_tokens":2,"output_tokens":1}
			}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"id":"complete","model":"claude-sonnet-4-5","stop_reason":"end_turn",
			"content":[{"type":"text","text":"done"}],
			"usage":{"input_tokens":3,"output_tokens":1}
		}`))
	})
	result, err := ai.NewAgent[struct{}, string](model).Run(t.Context(), "go", struct{}{})
	if err != nil || result.Output != "partial done" || result.Usage().Requests != 2 {
		t.Fatalf("unexpected pause continuation result=%+v err=%v", result, err)
	}
	if len(bodies) != 2 {
		t.Fatalf("expected two Anthropic requests, got %d", len(bodies))
	}
	messages := bodies[1]["messages"].([]any)
	if len(messages) != 2 || messages[1].(map[string]any)["role"] != "assistant" {
		t.Fatalf("paused response was not echoed to Anthropic: %+v", messages)
	}
	content := messages[1].(map[string]any)["content"].([]any)
	if content[0].(map[string]any)["text"] != "partial " {
		t.Fatalf("paused content was not preserved: %+v", content)
	}
}

func TestRequestToolUseRoundTrip(t *testing.T) {
	var gotBody map[string]any
	first := true
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		if first {
			first = false
			_, _ = w.Write([]byte(`{
				"model": "claude-sonnet-4-5",
				"content": [
					{"type": "thinking", "thinking": "checking", "signature": "signature"},
					{"type": "tool_use", "id": "tu1", "name": "get_weather", "input": {"city": "SF"}}
				],
				"usage": {"input_tokens": 20, "output_tokens": 8}
			}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"model": "claude-sonnet-4-5",
			"content": [{"type": "text", "text": "Sunny."}],
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
	if len(calls) != 1 || calls[0].ToolName != "get_weather" || calls[0].ToolCallID != "tu1" {
		t.Fatalf("unexpected calls %+v", calls)
	}
	thinking, ok := resp.Parts[0].(ai.ThinkingPart)
	if !ok || thinking.Signature != "signature" || thinking.ProviderName != "anthropic" {
		t.Fatalf("thinking metadata lost: %+v", resp.Parts)
	}

	msgs = append(msgs, *resp, ai.ModelRequest{Parts: []ai.RequestPart{
		ai.ToolReturnPart{ToolName: "get_weather", Content: "sunny", ToolCallID: "tu1"},
		ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"archive"}, ToolCallID: "tu1"},
	}})
	if _, err := model.Request(t.Context(), msgs, params); err != nil {
		t.Fatal(err)
	}
	sent := gotBody["messages"].([]any)
	assistant := sent[1].(map[string]any)
	if assistant["role"] != "assistant" {
		t.Fatalf("unexpected roles %v", sent)
	}
	thinkingBlock := assistant["content"].([]any)[0].(map[string]any)
	if thinkingBlock["type"] != "thinking" || thinkingBlock["signature"] != "signature" {
		t.Fatalf("thinking signature was not round-tripped: %v", thinkingBlock)
	}
	toolResult := sent[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	if toolResult["type"] != "tool_result" || toolResult["tool_use_id"] != "tu1" || toolResult["content"] != "sunny" {
		t.Fatalf("unexpected tool result %v", toolResult)
	}
	if len(gotBody["tools"].([]any)) != 1 {
		t.Fatalf("tools not sent: %v", gotBody["tools"])
	}
}

func TestNativeDeferredToolRendering(t *testing.T) {
	var gotBody map[string]any
	var gotBeta string
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotBeta = r.Header.Get("anthropic-beta")
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{
			"model":"claude-sonnet-4-5","stop_reason":"end_turn",
			"content":[{"type":"text","text":"done"}],"usage":{}
		}`))
	})
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	search := ai.ToolDefinition{
		Name: ai.ToolSearchName, Schema: schema, ToolKind: ai.ToolPartKindToolSearch,
		ToolSearchStrategy: ai.ToolSearchStrategyCustom,
	}
	first := ai.ToolDefinition{Name: "first", Schema: schema, DeferLoading: true}
	second := ai.ToolDefinition{Name: "second", Schema: schema, DeferLoading: true}
	messages := []ai.ModelMessage{
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.CompactionPart{Content: "Summary.", ProviderName: "anthropic"},
			ai.ToolCallPart{
				ToolName: ai.ToolSearchName, ToolCallID: "search", ToolKind: ai.ToolPartKindToolSearch,
				Args: []byte(`{"queries":["first"]}`),
			},
		}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{
				ToolName: ai.ToolSearchName, ToolCallID: "search", ToolKind: ai.ToolPartKindToolSearch,
				Content: ai.ToolSearchResult{DiscoveredTools: []ai.ToolSearchMatch{{Name: "first"}}},
			},
			ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"first"}, ToolCallID: "search"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "loader", ToolCallID: "loader", Args: []byte(`{}`),
		}}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "loader", ToolCallID: "loader", Content: "loaded"},
			ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"unknown", "second"}, ToolCallID: "loader"},
		}},
	}
	_, err := model.Request(t.Context(), messages, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{search, first}, DeferredTools: []ai.ToolDefinition{first, second}, AllowText: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	wireTools := gotBody["tools"].([]any)
	if gotBeta != "mid-conversation-tool-changes-2026-07-01,compact-2026-01-12" {
		t.Fatalf("tool-addition and compaction beta headers missing: %q", gotBeta)
	}
	if len(wireTools) != 3 || wireTools[0].(map[string]any)["name"] != ai.ToolSearchName ||
		wireTools[0].(map[string]any)["defer_loading"] != nil ||
		wireTools[1].(map[string]any)["defer_loading"] != true ||
		wireTools[2].(map[string]any)["defer_loading"] != true {
		t.Fatalf("unexpected deferred tool definitions: %+v", wireTools)
	}
	wireMessages := gotBody["messages"].([]any)
	searchResult := wireMessages[1].(map[string]any)["content"].([]any)
	if len(searchResult) != 1 {
		t.Fatalf("search availability delta was rendered twice: %+v", searchResult)
	}
	reference := searchResult[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	if reference["type"] != "tool_reference" || reference["tool_name"] != "first" {
		t.Fatalf("unexpected search tool reference: %+v", reference)
	}
	loaderResult := wireMessages[3].(map[string]any)["content"].([]any)
	addition := loaderResult[1].(map[string]any)
	toolReference := addition["tool"].(map[string]any)
	if addition["type"] != "tool_addition" || toolReference["type"] != "tool_reference" ||
		toolReference["name"] != "second" {
		t.Fatalf("unexpected tool addition: %+v", addition)
	}
}

func TestNativeDeferredToolRenderingEdgeCases(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{
			"model":"claude-sonnet-4-5","stop_reason":"end_turn",
			"content":[{"type":"text","text":"done"}],"usage":{}
		}`))
	})
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	deferred := ai.ToolDefinition{Name: "hidden", Schema: schema, DeferLoading: true}
	output := ai.ToolDefinition{Name: "final_result", Schema: schema}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		OutputTool: &output, DeferredTools: []ai.ToolDefinition{deferred},
	}); err != nil {
		t.Fatal(err)
	}
	if tools := gotBody["tools"].([]any); len(tools) != 2 ||
		tools[0].(map[string]any)["name"] != "hidden" || tools[0].(map[string]any)["defer_loading"] != true {
		t.Fatalf("output tool did not stabilize deferred definitions: %+v", tools)
	}

	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{deferred}, DeferredTools: []ai.ToolDefinition{deferred},
	}); err != nil {
		t.Fatal(err)
	}
	if tool := gotBody["tools"].([]any)[0].(map[string]any); tool["defer_loading"] != nil {
		t.Fatalf("all-deferred request was not downgraded safely: %+v", tool)
	}

	search := ai.ToolDefinition{Name: ai.ToolSearchName, Schema: schema}
	searchCall := ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
		ToolName: ai.ToolSearchName, ToolCallID: "search", ToolKind: ai.ToolPartKindToolSearch,
	}}}
	for name, content := range map[string]any{
		"empty": ai.ToolSearchResult{Message: "nothing"},
		"filtered": ai.ToolSearchResult{DiscoveredTools: []ai.ToolSearchMatch{
			{Name: "unknown"}, {Name: "hidden"}, {Name: "hidden"},
		}},
	} {
		t.Run(name, func(t *testing.T) {
			messages := []ai.ModelMessage{searchCall, ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{
				ToolName: ai.ToolSearchName, ToolCallID: "search", ToolKind: ai.ToolPartKindToolSearch,
				Content: content,
			}}}}
			if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{
				Tools: []ai.ToolDefinition{search}, DeferredTools: []ai.ToolDefinition{deferred},
			}); err != nil {
				t.Fatal(err)
			}
		})
	}

	for name, content := range map[string]any{"marshal": make(chan int), "parse": "bad"} {
		t.Run(name, func(t *testing.T) {
			messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{
				ToolName: ai.ToolSearchName, ToolCallID: "search", ToolKind: ai.ToolPartKindToolSearch,
				Content: content,
			}}}}
			if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{
				Tools: []ai.ToolDefinition{search}, DeferredTools: []ai.ToolDefinition{deferred},
			}); err == nil {
				t.Fatal("expected invalid tool-search history error")
			}
		})
	}

	var disabledBody map[string]any
	disabled := newServerWithOptions(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&disabledBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"m","content":[{"type":"text","text":"done"}],"usage":{}}`))
	}, anthropic.WithDeferredToolSupport(false))
	if _, err := disabled.Request(t.Context(), nil, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{search}, DeferredTools: []ai.ToolDefinition{deferred},
	}); err != nil {
		t.Fatal(err)
	}
	if tools := disabledBody["tools"].([]any); len(tools) != 1 ||
		tools[0].(map[string]any)["name"] != ai.ToolSearchName {
		t.Fatalf("deferred support override was ignored: %+v", tools)
	}

	strict := true
	invalid := deferred
	invalid.Strict = &strict
	invalid.Schema = map[string]any{"$defs": map[string]any{"bad": "not a schema"}}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{search}, DeferredTools: []ai.ToolDefinition{invalid},
	}); err == nil {
		t.Fatal("expected invalid deferred schema error")
	}
}

func TestOutputToolForcesToolChoice(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"m","content":[{"type":"tool_use","id":"tu1","name":"final_result","input":{}}],"usage":{}}`))
	})
	params := ai.ModelRequestParams{
		OutputTool: &ai.ToolDefinition{Name: "final_result", Schema: map[string]any{"type": "object"}},
	}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	if gotBody["tool_choice"].(map[string]any)["type"] != "any" {
		t.Fatalf("expected tool_choice any, got %v", gotBody["tool_choice"])
	}
}

func TestRetryAndSystemParts(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"m","content":[{"type":"text","text":"ok"}],"usage":{}}`))
	})
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.SystemPromptPart{Content: "sys"},
		ai.RetryPromptPart{Content: "bad args", ToolCallID: "tu1"},
		ai.RetryPromptPart{Content: "plain retry"},
		ai.ToolReturnPart{ToolName: "t", Content: map[string]any{"x": 1}, ToolCallID: "tu2"},
		ai.ToolReturnPart{ToolName: "t", Content: "failed", ToolCallID: "tu3", Outcome: ai.ToolReturnOutcomeFailed},
		ai.ToolReturnPart{
			ToolName: "t", Content: "interrupted", ToolCallID: "tu4", Outcome: ai.ToolReturnOutcomeInterrupted,
		},
	}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	blocks := gotBody["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if blocks[1].(map[string]any)["is_error"] != true {
		t.Fatalf("tool retry should be an error tool result: %v", blocks[1])
	}
	if blocks[2].(map[string]any)["type"] != "text" {
		t.Fatalf("plain retry should be text: %v", blocks[2])
	}
	if blocks[3].(map[string]any)["is_error"] != nil || blocks[4].(map[string]any)["is_error"] != true ||
		blocks[5].(map[string]any)["is_error"] != true {
		t.Fatalf("tool outcomes were not mapped to Anthropic errors: %v", blocks)
	}
	if blocks[3].(map[string]any)["content"] != `{"x":1}` {
		t.Fatalf("structured tool return not serialized: %v", blocks[3])
	}
}

func TestErrors(t *testing.T) {
	t.Run("native tools", func(t *testing.T) {
		var body map[string]any
		model := newServer(t, func(w http.ResponseWriter, request *http.Request) {
			if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"done"}]}`))
		})
		messages := []ai.ModelMessage{
			ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "earlier"}}},
			ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "use the files"}}},
		}
		if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{ai.WebSearchTool{
				UserLocation: &ai.WebSearchUserLocation{
					City: "Paris", Country: "FR", Region: "IDF", Timezone: "Europe/Paris",
				},
				AllowedDomains: []string{"go.dev"}, BlockedDomains: []string{"example.com"}, MaxUses: 3,
			}, ai.WebFetchTool{
				AllowedDomains: []string{"go.dev"}, BlockedDomains: []string{"example.com"}, MaxUses: 2,
				EnableCitations: true, MaxContentTokens: 4096,
			}, ai.CodeExecutionTool{Files: []ai.UploadedFile{
				{FileID: "file-anthropic", ProviderName: "anthropic"},
				{FileID: "file-openai", ProviderName: "openai"},
			}}},
		}); err != nil {
			t.Fatal(err)
		}
		tool := body["tools"].([]any)[0].(map[string]any)
		location := tool["user_location"].(map[string]any)
		if tool["type"] != "web_search_20250305" || tool["name"] != "web_search" ||
			tool["max_uses"] != float64(3) || tool["allowed_domains"].([]any)[0] != "go.dev" ||
			tool["blocked_domains"].([]any)[0] != "example.com" || location["type"] != "approximate" ||
			location["city"] != "Paris" {
			t.Fatalf("unexpected Anthropic web-search tool: %#v", tool)
		}
		fetch := body["tools"].([]any)[1].(map[string]any)
		if fetch["type"] != "web_fetch_20250910" || fetch["name"] != "web_fetch" ||
			fetch["max_uses"] != float64(2) || fetch["max_content_tokens"] != float64(4096) ||
			fetch["citations"].(map[string]any)["enabled"] != true {
			t.Fatalf("unexpected Anthropic web-fetch tool: %#v", fetch)
		}
		code := body["tools"].([]any)[2].(map[string]any)
		var userContent []any
		for _, rawMessage := range body["messages"].([]any) {
			message := rawMessage.(map[string]any)
			if message["role"] == "user" {
				userContent = message["content"].([]any)
				break
			}
		}
		upload := userContent[len(userContent)-1].(map[string]any)
		if code["type"] != "code_execution_20260120" || code["name"] != "code_execution" ||
			upload["type"] != "container_upload" || upload["file_id"] != "file-anthropic" {
			t.Fatalf("unexpected Anthropic code execution request: code=%#v content=%#v", code, userContent)
		}
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{unsupportedNativeTool{}},
		}); err == nil || !strings.Contains(err.Error(), `native tool "unsupported" is not implemented`) {
			t.Fatalf("unexpected native-tool error: %v", err)
		}
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{unsupportedNativeTool{optional: true}},
		}); err != nil {
			t.Fatalf("optional native tool should be omitted: %v", err)
		}
		var nilTool *ai.WebSearchTool
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
			NativeTools: []ai.NativeTool{nilTool},
		}); err == nil || !strings.Contains(err.Error(), "native tool must not be nil") {
			t.Fatalf("unexpected nil native-tool error: %v", err)
		}
	})
	t.Run("api error", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte("rate limited"))
		})
		var apiErr *anthropic.APIError
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{AllowText: true})
		if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusTooManyRequests || !apiErr.IsModelAPIError() {
			t.Fatalf("expected fallback-eligible APIError 429, got %v", err)
		}
		if apiErr.Error() != "anthropic: API returned status 429: rate limited" {
			t.Fatalf("unexpected message %q", apiErr.Error())
		}
	})
	t.Run("invalid json", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte("not json"))
		})
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{AllowText: true}); err == nil {
			t.Fatal("expected parse error")
		}
	})
	t.Run("unknown content block", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"model":"m","content":[{"type":"mystery"}],"usage":{}}`))
		})
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{AllowText: true}); err == nil {
			t.Fatal("expected error for unknown block")
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
	t.Run("unserializable tool return", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "t", Content: make(chan int)},
		}}}
		if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("unserializable tool schema", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
		params := ai.ModelRequestParams{Tools: []ai.ToolDefinition{{Name: "t", Schema: map[string]any{"bad": make(chan int)}}}}
		if _, err := model.Request(t.Context(), nil, params); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("transport error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		server.Close()
		model := anthropic.NewModel("m", anthropic.WithBaseURL(server.URL))
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected transport error")
		}
	})
	t.Run("invalid URL", func(t *testing.T) {
		model := anthropic.NewModel("m", anthropic.WithBaseURL("http://[::1"))
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected URL error")
		}
	})
	t.Run("truncated body", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "1000")
			_, _ = w.Write([]byte(`{"model"`))
		})
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected read error")
		}
	})
}

func TestModelName(t *testing.T) {
	model := anthropic.NewModel("claude-sonnet-4-5")
	if model.Name() != "claude-sonnet-4-5" || model.ProviderName() != "anthropic" ||
		model.ProviderURL() != "https://api.anthropic.com/v1" {
		t.Fatalf("unexpected model identity: %q %q %q", model.Name(), model.ProviderName(), model.ProviderURL())
	}
}

func vcrModel(t *testing.T, name string) *anthropic.Model {
	t.Helper()
	mode := recorder.ModeReplayOnly
	if _, err := os.Stat("testdata/" + name + ".yaml"); os.IsNotExist(err) && os.Getenv("ANTHROPIC_API_KEY") != "" {
		mode = recorder.ModeRecordOnce
	}
	r, err := recorder.New("testdata/"+name,
		recorder.WithMode(mode),
		recorder.WithHook(func(i *cassette.Interaction) error {
			delete(i.Request.Headers, "X-Api-Key")
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
	return anthropic.NewModel("claude-haiku-4-5", anthropic.WithHTTPClient(r.GetDefaultClient()))
}

func TestRecordedSimpleRun(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](vcrModel(t, "simple_run"),
		ai.WithInstructions("Answer with a single word."),
	)
	result, err := agent.Run(t.Context(), "What is the capital of France?", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output == "" || result.Usage().TotalTokens() == 0 {
		t.Fatalf("unexpected result %+v", result)
	}
}

func TestRecordedToolRun(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](vcrModel(t, "tool_run"),
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

func TestAssistantHistoryWithMixedParts(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"m","content":[{"type":"text","text":"ok"}],"usage":{}}`))
	})
	msgs := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.TextPart{Content: "let me check"},
		ai.ThinkingPart{Content: "hidden"},
		ai.ToolCallPart{ToolName: "t", Args: json.RawMessage(`{}`), ToolCallID: "tu1"},
	}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	blocks := gotBody["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(blocks) != 2 {
		t.Fatalf("expected text + tool_use (thinking dropped), got %v", blocks)
	}
	if blocks[1].(map[string]any)["type"] != "tool_use" {
		t.Fatalf("unexpected blocks %v", blocks)
	}
}

func TestMultimodalUserPrompt(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"m","content":[{"type":"text","text":"a cat"}],"usage":{}}`))
	})
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
		ai.TextContent{Text: "what is this?"},
		ai.BinaryContent{Data: []byte("hi"), MediaType: "image/png"},
		ai.ImageURL{URL: "https://example.com/cat.png"},
		ai.UploadedFile{FileID: "file-image", ProviderName: "anthropic", MediaType: "image/png"},
		ai.UploadedFile{FileID: "file-document", ProviderName: "anthropic", MediaType: "application/pdf"},
	}}}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	blocks := gotBody["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(blocks) != 5 {
		t.Fatalf("unexpected blocks %v", blocks)
	}
	source := blocks[1].(map[string]any)["source"].(map[string]any)
	if source["type"] != "base64" || source["data"] != "aGk=" || source["media_type"] != "image/png" {
		t.Fatalf("unexpected image source %v", source)
	}
	urlSource := blocks[2].(map[string]any)["source"].(map[string]any)
	if urlSource["type"] != "url" || urlSource["url"] != "https://example.com/cat.png" {
		t.Fatalf("unexpected url source %v", urlSource)
	}
	imageFile := blocks[3].(map[string]any)
	documentFile := blocks[4].(map[string]any)
	if imageFile["type"] != "image" || imageFile["source"].(map[string]any)["file_id"] != "file-image" ||
		documentFile["type"] != "document" ||
		documentFile["source"].(map[string]any)["file_id"] != "file-document" {
		t.Fatalf("unexpected uploaded files: image=%#v document=%#v", imageFile, documentFile)
	}
}

func TestAnthropicCachePoints(t *testing.T) {
	var body map[string]any
	model := newServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"model":"m","content":[{"type":"text","text":"ok"}],"usage":{}}`))
	})
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.TextContent{Text: "old"}, ai.CachePoint{TTL: ai.CachePointTTL1Hour},
			ai.TextContent{Text: "middle"}, ai.CachePoint{},
		}}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.TextContent{Text: "new one"}, ai.CachePoint{},
			ai.TextContent{Text: "new two"}, ai.CachePoint{},
			ai.TextContent{Text: "new three"}, ai.CachePoint{TTL: ai.CachePointTTL1Hour},
		}}}},
	}
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	wireMessages := body["messages"].([]any)
	old := wireMessages[0].(map[string]any)["content"].([]any)
	newer := wireMessages[1].(map[string]any)["content"].([]any)
	if old[0].(map[string]any)["cache_control"] != nil ||
		old[1].(map[string]any)["cache_control"].(map[string]any)["ttl"] != "5m" {
		t.Fatalf("old Anthropic cache points were not limited: %#v", old)
	}
	for index, block := range newer {
		cache := block.(map[string]any)["cache_control"].(map[string]any)
		want := "5m"
		if index == 2 {
			want = "1h"
		}
		if cache["type"] != "ephemeral" || cache["ttl"] != want {
			t.Fatalf("unexpected Anthropic cache point %d: %#v", index, block)
		}
	}

	messages = []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "prior"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.CachePoint{TTL: ai.CachePointTTL1Hour},
		}}}},
	}
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	wireMessages = body["messages"].([]any)
	if len(wireMessages) != 1 ||
		wireMessages[0].(map[string]any)["content"].([]any)[0].(map[string]any)["cache_control"].(map[string]any)["ttl"] != "1h" {
		t.Fatalf("leading cache point did not attach to prior content: %#v", wireMessages)
	}
}

func TestAnthropicCachePointErrors(t *testing.T) {
	model := newServer(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("invalid cache point reached transport")
	})
	for name, contents := range map[string][]ai.UserContent{
		"first":   {ai.CachePoint{}, ai.TextContent{Text: "later"}},
		"invalid": {ai.TextContent{Text: "first"}, ai.CachePoint{TTL: "1d"}},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Contents: contents},
			}}}, ai.ModelRequestParams{})
			if err == nil || !strings.Contains(err.Error(), "cache point") {
				t.Fatalf("unexpected cache-point error: %v", err)
			}
		})
	}
	_, err := model.Request(t.Context(), []ai.ModelMessage{
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.ThinkingPart{
			Content: "thought", Signature: "signed", ProviderName: "anthropic",
		}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{ai.CachePoint{}}}}},
	}, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), `cache control not supported for content type "thinking"`) {
		t.Fatalf("unexpected unsupported cache target error: %v", err)
	}
}

func TestMultimodalUnknownContent(t *testing.T) {
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
	for _, test := range []struct {
		name    string
		content ai.UserContent
		want    string
	}{
		{name: "unknown", content: nil, want: "unsupported user content"},
		{name: "foreign file", content: ai.UploadedFile{
			FileID: "file", ProviderName: "openai", MediaType: "image/png",
		}, want: "belongs to provider"},
		{name: "unsupported file", content: ai.UploadedFile{
			FileID: "file", ProviderName: "anthropic", MediaType: "audio/wav",
		}, want: "unsupported uploaded file media type"},
	} {
		t.Run(test.name, func(t *testing.T) {
			msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Contents: []ai.UserContent{test.content}},
			}}}
			if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{}); err == nil ||
				!strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected user content error: %v", err)
			}
		})
	}
}

func TestNativeJSONOutputModeUnsupported(t *testing.T) {
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
	params := ai.ModelRequestParams{OutputSchema: map[string]any{"type": "object"}}
	_, err := model.Request(t.Context(), nil, params)
	if err == nil || !strings.Contains(err.Error(), "not supported") {
		t.Fatalf("expected unsupported error, got %v", err)
	}
	params.OutputMode = ai.OutputModePrompted
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatalf("prompted output should not request native mode: %v", err)
	}
}

func TestStrictToolDefinition(t *testing.T) {
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"m","content":[{"type":"text","text":"ok"}],"usage":{}}`))
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

func TestParallelToolCallsSetting(t *testing.T) {
	for _, test := range []struct {
		name        string
		parallel    bool
		outputTool  bool
		wantType    string
		wantDisable bool
	}{
		{name: "disable automatic tools", parallel: false, wantType: "auto", wantDisable: true},
		{name: "enable required output", parallel: true, outputTool: true, wantType: "any", wantDisable: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			var gotBody map[string]any
			model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
					t.Error(err)
				}
				_, _ = w.Write([]byte(`{"model":"m","content":[{"type":"text","text":"ok"}],"usage":{}}`))
			})
			params := ai.ModelRequestParams{
				AllowText: true,
				Tools:     []ai.ToolDefinition{{Name: "work", Schema: map[string]any{"type": "object"}}},
				Settings:  ai.ModelSettings{ParallelToolCalls: &test.parallel},
			}
			if test.outputTool {
				params.AllowText = false
				params.OutputTool = &ai.ToolDefinition{Name: "final_result", Schema: map[string]any{"type": "object"}}
			}
			if _, err := model.Request(t.Context(), nil, params); err != nil {
				t.Fatal(err)
			}
			choice := gotBody["tool_choice"].(map[string]any)
			if choice["type"] != test.wantType || choice["disable_parallel_tool_use"] != test.wantDisable {
				t.Fatalf("unexpected tool choice %v", choice)
			}
		})
	}
}

func TestAnthropicCompactionRoundTripTrimsHistory(t *testing.T) {
	var body map[string]any
	var beta string
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		beta = r.Header.Get("anthropic-beta")
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{
			"id":"response", "model":"claude", "stop_reason":"end_turn",
			"content":[
				{"type":"compaction","content":"New summary.","encrypted_content":"new-encrypted"},
				{"type":"text","text":"done"}
			],
			"usage":{"input_tokens":1,"output_tokens":1}
		}`))
	})
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "drop"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.TextPart{Content: "drop before boundary"},
			ai.CompactionPart{
				Content: "Old summary.", ProviderName: "anthropic",
				ProviderDetails: map[string]any{"encrypted_content": "old-encrypted"},
			},
			ai.TextPart{Content: "keep after boundary"},
		}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "keep tail"}}},
	}
	response, err := model.Request(t.Context(), messages, ai.ModelRequestParams{
		AllowText: true, Instructions: "standing",
	})
	if err != nil {
		t.Fatal(err)
	}
	wireMessages := body["messages"].([]any)
	contextManagement := body["context_management"].(map[string]any)
	edits := contextManagement["edits"].([]any)
	if len(wireMessages) != 2 || body["system"] != "standing" || len(edits) != 1 ||
		edits[0].(map[string]any)["type"] != "compact_20260112" ||
		!strings.Contains(beta, "compact-2026-01-12") {
		t.Fatalf("unexpected compacted Anthropic request messages=%+v beta=%q", wireMessages, beta)
	}
	assistant := wireMessages[0].(map[string]any)["content"].([]any)
	if len(assistant) != 2 || assistant[0].(map[string]any)["type"] != "compaction" ||
		assistant[0].(map[string]any)["content"] != "Old summary." ||
		assistant[0].(map[string]any)["encrypted_content"] != "old-encrypted" ||
		assistant[1].(map[string]any)["text"] != "keep after boundary" {
		t.Fatalf("unexpected Anthropic compaction blocks: %+v", assistant)
	}
	compaction, ok := response.Parts[0].(ai.CompactionPart)
	if !ok || compaction.Content != "New summary." || compaction.ProviderName != "anthropic" ||
		compaction.ProviderDetails["encrypted_content"] != "new-encrypted" || response.Text() != "done" {
		t.Fatalf("unexpected Anthropic compaction response: %+v", response.Parts)
	}
}

func TestAnthropicIgnoresInvalidCompactionBoundaries(t *testing.T) {
	for name, compaction := range map[string]ai.CompactionPart{
		"foreign": {Content: "Summary.", ProviderName: "openai"},
		"contentless": {
			ProviderName: "anthropic", ProviderDetails: map[string]any{"encrypted_content": "opaque"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			var body map[string]any
			var beta string
			model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
				beta = r.Header.Get("anthropic-beta")
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				_, _ = w.Write([]byte(`{
					"id":"response", "model":"claude", "stop_reason":"end_turn",
					"content":[{"type":"text","text":"done"}], "usage":{}
				}`))
			})
			messages := []ai.ModelMessage{
				ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "keep"}}},
				ai.ModelResponse{Parts: []ai.ResponsePart{compaction, ai.TextPart{Content: "assistant"}}},
			}
			if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{AllowText: true}); err != nil {
				t.Fatal(err)
			}
			if len(body["messages"].([]any)) != 2 || strings.Contains(beta, "compact-2026-01-12") {
				t.Fatalf("invalid compaction changed Anthropic input=%+v beta=%q", body["messages"], beta)
			}
		})
	}
}

func TestAnthropicCompactionContextManagementOverride(t *testing.T) {
	var body map[string]any
	var beta string
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		beta = r.Header.Get("anthropic-beta")
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{
			"id":"response", "model":"claude", "stop_reason":"end_turn",
			"content":[{"type":"text","text":"done"}], "usage":{}
		}`))
	})
	messages := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.CompactionPart{
		Content: "Summary.", ProviderName: "anthropic",
	}}}}
	settings := ai.ModelSettings{
		ExtraHeaders: map[string]string{"anthropic-beta": "custom-beta, compact-2026-01-12"},
		ExtraBody: map[string]any{"context_management": map[string]any{
			"edits": []any{map[string]any{"type": "custom"}},
		}},
	}
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{
		AllowText: true, Settings: settings,
	}); err != nil {
		t.Fatal(err)
	}
	edits := body["context_management"].(map[string]any)["edits"].([]any)
	if len(edits) != 1 || edits[0].(map[string]any)["type"] != "custom" ||
		beta != "custom-beta,compact-2026-01-12" {
		t.Fatalf("compaction overrides were replaced: body=%+v beta=%q", body, beta)
	}
}

func TestAnthropicDynamicWebSearchVersion(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"content":[{"type":"text","text":"done"}]}`))
	}))
	defer server.Close()
	model := anthropic.NewModel(
		"claude-sonnet-4-6", anthropic.WithAPIKey("key"), anthropic.WithBaseURL(server.URL),
		anthropic.WithHTTPClient(server.Client()),
	)
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{&ai.WebSearchTool{}, &ai.WebFetchTool{}, &ai.CodeExecutionTool{}},
	}); err != nil {
		t.Fatal(err)
	}
	tools := body["tools"].([]any)
	if tool := tools[0].(map[string]any); tool["type"] != "web_search_20260209" ||
		tool["max_uses"] != nil || tool["user_location"] != nil {
		t.Fatalf("unexpected dynamic web search tool: %#v", tool)
	}
	if tool := tools[1].(map[string]any); tool["type"] != "web_fetch_20260209" || tool["citations"] != nil {
		t.Fatalf("unexpected dynamic web fetch tool: %#v", tool)
	}
	if tool := tools[2].(map[string]any); tool["type"] != "code_execution_20260120" {
		t.Fatalf("unexpected latest code execution tool: %#v", tool)
	}
}

func TestAnthropicWebSearchResponseAndReplay(t *testing.T) {
	var body map[string]any
	model := newServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{
			"id":"response","model":"claude-sonnet-4-5","content":[
				{"type":"server_tool_use","id":"web-1","name":"web_search","input":{"query":"Go news"},"caller":{"type":"code_execution_20250825"}},
				{"type":"web_search_tool_result","tool_use_id":"web-1","content":[{"type":"web_search_result","url":"https://go.dev","title":"Go"}],"caller":{"type":"code_execution_20250825"}},
				{"type":"text","text":"done"}
			]
		}`))
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.WebSearchTool{}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Parts) != 3 {
		t.Fatalf("unexpected web search response: %#v", response.Parts)
	}
	call := response.Parts[0].(ai.NativeToolCallPart)
	returned := response.Parts[1].(ai.NativeToolReturnPart)
	results := returned.Content.([]any)
	if call.ToolKind != ai.ToolPartKindWebSearch || call.ToolCallID != "web-1" ||
		string(call.Args) != `{"query":"Go news"}` || call.ProviderDetails["anthropic_caller"] == nil ||
		returned.ToolKind != ai.ToolPartKindWebSearch || returned.ToolCallID != "web-1" || len(results) != 1 ||
		returned.ProviderDetails["anthropic_caller"] == nil {
		t.Fatalf("unexpected normalized web search: call=%+v return=%+v", call, returned)
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{*response}, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(content) != 3 || content[0].(map[string]any)["type"] != "server_tool_use" ||
		content[0].(map[string]any)["name"] != "web_search" || content[0].(map[string]any)["caller"] == nil ||
		content[1].(map[string]any)["type"] != "web_search_tool_result" ||
		content[1].(map[string]any)["caller"] == nil {
		t.Fatalf("unexpected web search replay: %#v", content)
	}

	history := ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.NativeToolCallPart{ToolName: "web_search", ToolCallID: "empty", ToolKind: ai.ToolPartKindWebSearch,
			ProviderName: "anthropic"},
		ai.NativeToolReturnPart{ToolName: "web_search", ToolCallID: "empty", ToolKind: ai.ToolPartKindWebSearch,
			ProviderName: "anthropic", Content: nil},
		ai.NativeToolCallPart{ToolName: "web_search", ToolKind: ai.ToolPartKindWebSearch, ProviderName: "other"},
		ai.NativeToolReturnPart{ToolName: "web_search", ToolKind: ai.ToolPartKindWebSearch, ProviderName: "other"},
	}}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{history}, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	content = body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(content) != 2 || content[0].(map[string]any)["input"].(map[string]any) == nil {
		t.Fatalf("unexpected empty or foreign web search replay: %#v", content)
	}
}

func TestAnthropicLegacyCodeExecutionRequest(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"content":[{"type":"text","text":"done"}]}`))
	}))
	t.Cleanup(server.Close)
	model := anthropic.NewModel(
		"claude-haiku-4-5-20251001", anthropic.WithAPIKey("test"), anthropic.WithBaseURL(server.URL),
		anthropic.WithHTTPClient(server.Client()),
	)
	history := []ai.ModelMessage{
		ai.ModelResponse{ProviderName: "anthropic", Parts: []ai.ResponsePart{ai.NativeToolCallPart{
			ToolName: "code_execution", ToolCallID: "code", ToolKind: ai.ToolPartKindCodeExecution,
			ProviderName: "anthropic",
		}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "continue"}}},
	}
	tool := &ai.CodeExecutionTool{Files: []ai.UploadedFile{{FileID: "file-1", ProviderName: "anthropic"}}}
	if _, err := model.Request(t.Context(), history, ai.ModelRequestParams{NativeTools: []ai.NativeTool{tool}}); err != nil {
		t.Fatal(err)
	}
	if got := body["tools"].([]any)[0].(map[string]any)["type"]; got != "code_execution_20250825" {
		t.Fatalf("unexpected legacy code tool version: %v", got)
	}
	messages := body["messages"].([]any)
	call := messages[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	upload := messages[1].(map[string]any)["content"].([]any)[1].(map[string]any)
	if input := call["input"].(map[string]any); len(input) != 0 || upload["file_id"] != "file-1" {
		t.Fatalf("unexpected legacy code replay or upload: call=%#v upload=%#v", call, upload)
	}
}

func TestAnthropicCodeExecutionResponseAndReplay(t *testing.T) {
	var body map[string]any
	model := newServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"id":"response","container":{"id":"container-1"},"content":[
			{"type":"server_tool_use","id":"bash-1","name":"bash_code_execution","input":{"command":"python x.py"},"caller":{"type":"code_execution_20260120","tool_id":"parent"}},
			{"type":"bash_code_execution_tool_result","tool_use_id":"bash-1","content":{"type":"bash_code_execution_result","stdout":"ok"}},
			{"type":"server_tool_use","id":"python-1","name":"code_execution","input":null,"caller":{"type":"code_execution_20250825","tool_id":"parent"}},
			{"type":"code_execution_tool_result","tool_use_id":"python-1","content":{"type":"code_execution_result","stdout":"1"}},
			{"type":"server_tool_use","id":"edit-1","name":"text_editor_code_execution","input":{"command":"view","path":"/tmp/x"}},
			{"type":"text_editor_code_execution_tool_result","tool_use_id":"edit-1","content":{"type":"text_editor_code_execution_result","content":"file"}}
		]}`))
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.CodeExecutionTool{}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if response.ProviderDetails["container_id"] != "container-1" || len(response.Parts) != 6 {
		t.Fatalf("unexpected code execution response: %+v", response)
	}
	bashCall := response.Parts[0].(ai.NativeToolCallPart)
	bashReturn := response.Parts[1].(ai.NativeToolReturnPart)
	pythonCall := response.Parts[2].(ai.NativeToolCallPart)
	editReturn := response.Parts[5].(ai.NativeToolReturnPart)
	if bashCall.ToolKind != ai.ToolPartKindCodeExecution || bashCall.ToolName != "code_execution" ||
		bashCall.ProviderDetails["anthropic_tool_name"] != "bash_code_execution" ||
		bashCall.ProviderDetails["anthropic_caller"] == nil ||
		bashReturn.ProviderDetails["anthropic_tool_name"] != "bash_code_execution" ||
		string(pythonCall.Args) != `{}` || pythonCall.ProviderDetails["anthropic_tool_name"] != nil ||
		pythonCall.ProviderDetails["anthropic_caller"] == nil || editReturn.ProviderDetails["anthropic_tool_name"] !=
		"text_editor_code_execution" {
		t.Fatalf("unexpected normalized code execution parts: %#v", response.Parts)
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{*response}, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.CodeExecutionTool{}},
	}); err != nil {
		t.Fatal(err)
	}
	if body["container"] != "container-1" {
		t.Fatalf("container was not reused: %#v", body["container"])
	}
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	wantTypes := []string{
		"server_tool_use", "bash_code_execution_tool_result", "server_tool_use",
		"code_execution_tool_result", "server_tool_use", "text_editor_code_execution_tool_result",
	}
	for index, want := range wantTypes {
		if content[index].(map[string]any)["type"] != want {
			t.Fatalf("unexpected replay block %d: %#v", index, content[index])
		}
	}
	if content[0].(map[string]any)["name"] != "bash_code_execution" ||
		content[2].(map[string]any)["name"] != "code_execution" ||
		content[4].(map[string]any)["name"] != "text_editor_code_execution" {
		t.Fatalf("unexpected replay tool names: %#v", content)
	}
}

func TestAnthropicWebFetchResponseAndReplay(t *testing.T) {
	var body map[string]any
	model := newServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"content":[
			{"type":"server_tool_use","id":"fetch-1","name":"web_fetch","input":{"url":"https://go.dev"}},
			{"type":"web_fetch_tool_result","tool_use_id":"fetch-1","content":{"type":"web_fetch_result","url":"https://go.dev","content":"Go"}},
			{"type":"text","text":"done"}
		]}`))
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.WebFetchTool{}},
	})
	if err != nil {
		t.Fatal(err)
	}
	call := response.Parts[0].(ai.NativeToolCallPart)
	returned := response.Parts[1].(ai.NativeToolReturnPart)
	if call.ToolName != "web_fetch" || call.ToolKind != ai.ToolPartKindWebFetch ||
		string(call.Args) != `{"url":"https://go.dev"}` || returned.ToolKind != ai.ToolPartKindWebFetch ||
		returned.Content.(map[string]any)["content"] != "Go" {
		t.Fatalf("unexpected web fetch parts: call=%+v return=%+v", call, returned)
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{*response}, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if content[0].(map[string]any)["name"] != "web_fetch" ||
		content[1].(map[string]any)["type"] != "web_fetch_tool_result" {
		t.Fatalf("unexpected web fetch replay: %#v", content)
	}
}

func TestAnthropicWebSearchEmptyPayloads(t *testing.T) {
	model := newServer(t, func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(`{"content":[
			{"type":"server_tool_use","id":"web","name":"web_search","input":null,"caller":{"type":"direct"}},
			{"type":"web_search_tool_result","tool_use_id":"web","caller":{"type":"direct"}}
		]}`))
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	call := response.Parts[0].(ai.NativeToolCallPart)
	returned := response.Parts[1].(ai.NativeToolReturnPart)
	if string(call.Args) != `{}` || call.ProviderDetails != nil || returned.Content != nil ||
		returned.ProviderDetails != nil {
		t.Fatalf("unexpected empty web-search parts: call=%+v return=%+v", call, returned)
	}
}

func TestAnthropicServerManagedToolSearch(t *testing.T) {
	request := 0
	var bodies []map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		request++
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		if request == 1 {
			_, _ = w.Write([]byte(`{
				"id":"message-search","model":"claude-sonnet-4-6","stop_reason":"tool_use","content":[
					{"type":"server_tool_use","id":"search-1","name":"tool_search_tool_bm25","input":{"query":"weather"},"caller":{"type":"code_execution_20250825"}},
					{"type":"tool_search_tool_result","tool_use_id":"search-1","content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"weather"}]}},
					{"type":"tool_use","id":"weather-1","name":"weather","input":{"city":"Paris"}}
				],"usage":{}
			}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"id":"message-final","model":"claude-sonnet-4-6","stop_reason":"end_turn",
			"content":[{"type":"text","text":"sunny"}],"usage":{}
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
		t.Fatalf("unexpected native Anthropic run: output=%q requests=%d", result.Output, request)
	}
	messages := result.Messages()
	response := messages[1].(ai.ModelResponse)
	call := response.Parts[0].(ai.NativeToolCallPart)
	returned := response.Parts[1].(ai.NativeToolReturnPart)
	if call.ToolCallID != "search-1" || string(call.Args) != `{"queries":["weather"]}` ||
		call.ProviderDetails["strategy"] != "bm25" || call.ProviderDetails["anthropic_caller"] == nil ||
		returned.ToolCallID != "search-1" ||
		returned.Content.(ai.ToolSearchResult).DiscoveredTools[0].Name != "weather" {
		t.Fatalf("unexpected normalized Anthropic search: %+v", response.Parts)
	}
	tools := bodies[0]["tools"].([]any)
	if len(tools) != 2 || tools[0].(map[string]any)["name"] != "weather" ||
		tools[0].(map[string]any)["defer_loading"] != true ||
		tools[1].(map[string]any)["name"] != "tool_search_tool_bm25" ||
		tools[1].(map[string]any)["type"] != "tool_search_tool_bm25_20251119" ||
		tools[1].(map[string]any)["input_schema"] != nil {
		t.Fatalf("unexpected Anthropic native tools: %+v", tools)
	}
	wireHistory := bodies[1]["messages"].([]any)
	assistant := wireHistory[1].(map[string]any)["content"].([]any)
	searchCall := assistant[0].(map[string]any)
	searchResult := assistant[1].(map[string]any)
	if searchCall["name"] != "tool_search_tool_bm25" ||
		searchCall["input"].(map[string]any)["query"] != "weather" || searchCall["caller"] == nil ||
		searchResult["type"] != "tool_search_tool_result" ||
		searchResult["content"].(map[string]any)["type"] != "tool_search_tool_search_result" {
		t.Fatalf("unexpected Anthropic native replay: %+v", assistant)
	}
}

func TestAnthropicAdvisorTool(t *testing.T) {
	var bodies []map[string]any
	var betas []string
	model := newNamedServer(t, "claude-opus-4-8", func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		betas = append(betas, request.Header.Get("anthropic-beta"))
		_, _ = response.Write([]byte(`{
			"id":"message","model":"claude-opus-4-8","stop_reason":"end_turn","content":[
				{"type":"server_tool_use","id":"advisor-1","name":"advisor","input":{},"caller":{"type":"code_execution_20260120"}},
				{"type":"advisor_tool_result","tool_use_id":"advisor-1","content":{"type":"advisor_result","text":"Use four.","stop_reason":"max_tokens"}},
				{"type":"advisor_tool_result","tool_use_id":"advisor-2","content":{"type":"advisor_redacted_result","encrypted_content":"secret","stop_reason":"end_turn"}},
				{"type":"advisor_tool_result","tool_use_id":"advisor-3","content":{"type":"advisor_tool_result_error","error_code":"max_uses_exceeded"}}
			],"usage":{"input_tokens":100,"output_tokens":50,"iterations":[
				{"type":"message","input_tokens":100,"output_tokens":50},
				{"type":"advisor_message","input_tokens":500,"output_tokens":200,"cache_creation_input_tokens":10},
				{"type":"compaction","input_tokens":20,"output_tokens":5}
			]}
		}`))
	})
	maxUses := 2
	maxTokens := 2048
	advisor := &ai.AdvisorTool{
		Model: "claude-opus-4-8", MaxUses: &maxUses, MaxTokens: &maxTokens, Caching: ai.AdvisorCaching1Hour,
	}
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{advisor}})
	if err != nil {
		t.Fatal(err)
	}
	tool := bodies[0]["tools"].([]any)[0].(map[string]any)
	if tool["type"] != "advisor_20260301" || tool["name"] != "advisor" ||
		tool["model"] != "claude-opus-4-8" || tool["max_uses"] != float64(2) ||
		tool["max_tokens"] != float64(2048) || tool["caching"].(map[string]any)["type"] != "ephemeral" ||
		tool["caching"].(map[string]any)["ttl"] != "1h" ||
		!strings.Contains(betas[0], "advisor-tool-2026-03-01") {
		t.Fatalf("unexpected advisor request: tool=%#v beta=%q", tool, betas[0])
	}
	call := response.Parts[0].(ai.NativeToolCallPart)
	result := response.Parts[1].(ai.NativeToolReturnPart)
	redacted := response.Parts[2].(ai.NativeToolReturnPart)
	failed := response.Parts[3].(ai.NativeToolReturnPart)
	if call.ToolName != "advisor" || call.ToolKind != ai.ToolPartKindAdvisor || len(call.Args) != 0 ||
		call.ProviderDetails["anthropic_caller"] == nil ||
		result.Content.(map[string]any)["text"] != "Use four." ||
		redacted.Content.(map[string]any)["encrypted_content"] != "secret" ||
		failed.Content.(map[string]any)["error_code"] != "max_uses_exceeded" {
		t.Fatalf("unexpected advisor parts: %#v", response.Parts)
	}
	usage := response.Usage
	if usage.InputTokens != 100 || usage.OutputTokens != 50 || usage.Details["message_iterations"] != 1 ||
		usage.Details["advisor_iterations"] != 1 || usage.Details["advisor_input_tokens"] != 500 ||
		usage.Details["advisor_output_tokens"] != 200 || usage.Details["advisor_cache_creation_input_tokens"] != 10 ||
		usage.Details["compaction_iterations"] != 1 || usage.Details["compaction_input_tokens"] != 20 {
		t.Fatalf("unexpected advisor usage: %+v", usage)
	}
	response.Parts = append(response.Parts, ai.NativeToolReturnPart{
		ToolName: "advisor", ToolCallID: "invalid", ToolKind: ai.ToolPartKindAdvisor,
		Content: "invalid", ProviderName: "anthropic",
	})
	if _, err := model.Request(t.Context(), []ai.ModelMessage{*response}, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.AdvisorTool{Model: "claude-opus-4-8"}},
	}); err != nil {
		t.Fatal(err)
	}
	replayed := bodies[1]["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if replayed[0].(map[string]any)["name"] != "advisor" ||
		replayed[0].(map[string]any)["caller"].(map[string]any)["type"] != "code_execution_20260120" ||
		replayed[1].(map[string]any)["type"] != "advisor_tool_result" ||
		replayed[1].(map[string]any)["content"].(map[string]any)["text"] != "Use four." {
		t.Fatalf("unexpected advisor replay: %#v", replayed)
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{*response}, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	inactiveBlocks, _ := bodies[2]["messages"].([]any)[0].(map[string]any)["content"].([]any)
	for _, block := range inactiveBlocks {
		block := block.(map[string]any)
		if block["name"] == "advisor" || block["type"] == "advisor_tool_result" {
			t.Fatalf("inactive advisor history was replayed: %#v", block)
		}
	}
}

func TestAnthropicAdvisorToolSupportAndCounting(t *testing.T) {
	model := newServer(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("unsupported advisor reached transport")
	})
	for _, advisor := range []ai.NativeTool{
		ai.AdvisorTool{Model: "claude-opus-4-8"},
		&ai.AdvisorTool{Model: "claude-opus-4-8"},
	} {
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{advisor}})
		if err == nil || !strings.Contains(err.Error(), "does not support the advisor tool") {
			t.Fatalf("unexpected advisor support error: %v", err)
		}
	}

	var body map[string]any
	var beta string
	model = newServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		beta = request.Header.Get("anthropic-beta")
		_, _ = response.Write([]byte(`{"model":"claude","content":[],"usage":{}}`))
	})
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		&ai.AdvisorTool{Model: "claude-opus-4-8", Optional: true},
	}}); err != nil {
		t.Fatal(err)
	}
	if body["tools"] != nil || strings.Contains(beta, "advisor-tool-2026-03-01") {
		t.Fatalf("optional advisor was not omitted: body=%#v beta=%q", body, beta)
	}

	var countBody map[string]any
	model = newNamedServer(t, "claude-opus-4-8", func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&countBody); err != nil {
			t.Error(err)
		}
		beta = request.Header.Get("anthropic-beta")
		_, _ = response.Write([]byte(`{"input_tokens":7}`))
	})
	history := []ai.ModelMessage{ai.ModelResponse{ProviderName: "anthropic", Parts: []ai.ResponsePart{
		ai.NativeToolCallPart{ToolName: "advisor", ToolCallID: "advisor", ToolKind: ai.ToolPartKindAdvisor, ProviderName: "anthropic"},
		ai.NativeToolReturnPart{
			ToolName: "advisor", ToolCallID: "advisor", ToolKind: ai.ToolPartKindAdvisor,
			Content: map[string]any{"type": "advisor_result", "text": "four"}, ProviderName: "anthropic",
		},
	}}}
	if _, err := model.CountTokens(t.Context(), history, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		ai.AdvisorTool{Model: "claude-opus-4-8"},
	}}); err != nil {
		t.Fatal(err)
	}
	if countBody["tools"] != nil || strings.Contains(beta, "advisor-tool-2026-03-01") {
		t.Fatalf("advisor leaked into token count: body=%#v beta=%q", countBody, beta)
	}
	for _, block := range countBody["messages"].([]any)[0].(map[string]any)["content"].([]any) {
		block := block.(map[string]any)
		if block["name"] == "advisor" || block["type"] == "advisor_tool_result" {
			t.Fatalf("advisor history leaked into token count: %#v", block)
		}
	}
}

func TestAnthropicMCPServerTool(t *testing.T) {
	var bodies []map[string]any
	var betas []string
	model := newServer(t, func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		betas = append(betas, request.Header.Get("anthropic-beta"))
		_, _ = response.Write([]byte(`{
			"id":"message","model":"claude-sonnet-4-6","stop_reason":"end_turn","content":[
				{"type":"mcp_tool_use","id":"call","server_name":"docs","name":"search","input":{"query":"Go"}},
				{"type":"mcp_tool_result","tool_use_id":"call","content":[{"type":"text","text":"result"}],"is_error":false},
				{"type":"mcp_tool_result","tool_use_id":"orphan","content":"missing call","is_error":true}
			],"usage":{}
		}`))
	})
	params := ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		&ai.MCPServerTool{
			ID: "docs", URL: "https://example.com/mcp", AuthorizationToken: "secret",
			Description: "ignored", AllowedTools: []string{}, Headers: map[string]string{"ignored": "value"},
		},
		ai.MCPServerTool{ID: "calendar", URL: "https://calendar.example.com/mcp", AllowedTools: []string{"events"}},
	}}
	response, err := model.Request(t.Context(), nil, params)
	if err != nil {
		t.Fatal(err)
	}
	servers := bodies[0]["mcp_servers"].([]any)
	first := servers[0].(map[string]any)
	firstConfig := first["tool_configuration"].(map[string]any)
	secondConfig := servers[1].(map[string]any)["tool_configuration"].(map[string]any)
	if len(servers) != 2 || first["type"] != "url" || first["name"] != "docs" ||
		first["url"] != "https://example.com/mcp" || first["authorization_token"] != "secret" ||
		first["description"] != nil || first["headers"] != nil || firstConfig["enabled"] != false ||
		len(firstConfig["allowed_tools"].([]any)) != 0 || secondConfig["enabled"] != true ||
		secondConfig["allowed_tools"].([]any)[0] != "events" ||
		!strings.Contains(betas[0], "mcp-client-2025-04-04") {
		t.Fatalf("unexpected Anthropic MCP request: servers=%#v beta=%q", servers, betas[0])
	}
	call := response.Parts[0].(ai.NativeToolCallPart)
	returned := response.Parts[1].(ai.NativeToolReturnPart)
	orphan := response.Parts[2].(ai.NativeToolReturnPart)
	content := returned.Content.(map[string]any)
	if call.ToolName != "mcp_server:docs" || call.ToolCallID != "call" ||
		call.ToolKind != ai.ToolPartKindMCPServer ||
		string(call.Args) != `{"action":"call_tool","tool_args":{"query":"Go"},"tool_name":"search"}` ||
		returned.ToolName != "mcp_server:docs" || content["is_error"] != false ||
		content["content"].([]any)[0].(map[string]any)["text"] != "result" ||
		orphan.ToolName != "mcp_server" || orphan.Content.(map[string]any)["is_error"] != true {
		t.Fatalf("unexpected Anthropic MCP parts: %#v", response.Parts)
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{*response}, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	replayed := bodies[1]["messages"].([]any)[0].(map[string]any)["content"].([]any)
	use := replayed[0].(map[string]any)
	result := replayed[1].(map[string]any)
	if use["type"] != "mcp_tool_use" || use["server_name"] != "docs" || use["name"] != "search" ||
		use["input"].(map[string]any)["query"] != "Go" || result["type"] != "mcp_tool_result" ||
		result["tool_use_id"] != "call" || result["content"].([]any)[0].(map[string]any)["text"] != "result" {
		t.Fatalf("unexpected Anthropic MCP replay: %#v", replayed)
	}
}

func TestAnthropicMCPServerCountTokensOmission(t *testing.T) {
	var body map[string]any
	var beta string
	model := newServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		beta = request.Header.Get("anthropic-beta")
		_, _ = response.Write([]byte(`{"input_tokens":12}`))
	})
	usage, err := model.CountTokens(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Content: "hello"},
	}}}, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{Name: "memory", Schema: map[string]any{"type": "object"}}},
		NativeTools: []ai.NativeTool{
			ai.MCPServerTool{ID: "docs", URL: "https://example.com/mcp"}, ai.MemoryTool{},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	tools := body["tools"].([]any)
	if usage.InputTokens != 12 || body["mcp_servers"] != nil || len(tools) != 1 ||
		tools[0].(map[string]any)["type"] != "memory_20250818" ||
		strings.Contains(beta, "mcp-client-2025-04-04") ||
		!strings.Contains(beta, "context-management-2025-06-27") {
		t.Fatalf("token count native isolation failed: usage=%+v body=%#v beta=%q", usage, body, beta)
	}
}

func TestAnthropicMCPServerErrors(t *testing.T) {
	model := newServer(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("unsupported MCP connector reached transport")
	})
	_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		ai.MCPServerTool{ID: "calendar", URL: "x-openai-connector:calendar"},
	}})
	if err == nil || !strings.Contains(err.Error(), "OpenAI MCP connector URLs are not supported") {
		t.Fatalf("unexpected connector error: %v", err)
	}
	var optionalBody map[string]any
	var optionalBeta string
	model = newServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&optionalBody); err != nil {
			t.Error(err)
		}
		optionalBeta = request.Header.Get("anthropic-beta")
		_, _ = response.Write([]byte(`{"model":"claude","content":[{"type":"text","text":"ok"}],"usage":{}}`))
	})
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		ai.MCPServerTool{ID: "calendar", URL: "x-openai-connector:calendar", Optional: true},
	}}); err != nil {
		t.Fatal(err)
	}
	if optionalBody["mcp_servers"] != nil || strings.Contains(optionalBeta, "mcp-client-2025-04-04") {
		t.Fatalf("optional connector was not omitted: body=%#v beta=%q", optionalBody, optionalBeta)
	}

	tests := []struct {
		name string
		part ai.ResponsePart
		want string
	}{
		{name: "server name", part: ai.NativeToolCallPart{
			ToolName: "mcp_server", ToolCallID: "call", ToolKind: ai.ToolPartKindMCPServer,
			Args: json.RawMessage(`{"action":"call_tool","tool_name":"search","tool_args":{}}`), ProviderName: "anthropic",
		}, want: "invalid MCP server tool name"},
		{name: "arguments", part: ai.NativeToolCallPart{
			ToolName: "mcp_server:docs", ToolCallID: "call", ToolKind: ai.ToolPartKindMCPServer,
			Args: json.RawMessage(`{`), ProviderName: "anthropic",
		}, want: "parse MCP server arguments"},
		{name: "action", part: ai.NativeToolCallPart{
			ToolName: "mcp_server:docs", ToolCallID: "call", ToolKind: ai.ToolPartKindMCPServer,
			Args: json.RawMessage(`{"action":"list_tools"}`), ProviderName: "anthropic",
		}, want: "invalid MCP server action"},
		{name: "tool name", part: ai.NativeToolCallPart{
			ToolName: "mcp_server:docs", ToolCallID: "call", ToolKind: ai.ToolPartKindMCPServer,
			Args: json.RawMessage(`{"action":"call_tool"}`), ProviderName: "anthropic",
		}, want: "call tool name must not be empty"},
		{name: "result marshal", part: ai.NativeToolReturnPart{
			ToolName: "mcp_server:docs", ToolCallID: "call", ToolKind: ai.ToolPartKindMCPServer,
			Content: make(chan int), ProviderName: "anthropic",
		}, want: "marshal MCP server result"},
		{name: "result shape", part: ai.NativeToolReturnPart{
			ToolName: "mcp_server:docs", ToolCallID: "call", ToolKind: ai.ToolPartKindMCPServer,
			Content: "invalid", ProviderName: "anthropic",
		}, want: "parse MCP server result"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := newServer(t, func(http.ResponseWriter, *http.Request) {
				t.Fatal("invalid MCP history reached transport")
			})
			_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelResponse{
				ProviderName: "anthropic", Parts: []ai.ResponsePart{test.part},
			}}, ai.ModelRequestParams{})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected MCP history error: %v", err)
			}
		})
	}

	model = newServer(t, func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte(`{"model":"claude","content":[{
			"type":"mcp_tool_use","id":"call","server_name":"docs","name":"search","input":
		}]}`))
	})
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), "parse response") {
		t.Fatalf("unexpected malformed MCP response error: %v", err)
	}

	var body map[string]any
	model = newServer(t, func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = response.Write([]byte(`{"model":"claude","content":[],"usage":{}}`))
	})
	_, err = model.Request(t.Context(), []ai.ModelMessage{ai.ModelResponse{
		ProviderName: "anthropic", Parts: []ai.ResponsePart{
			ai.NativeToolCallPart{
				ToolName: "mcp_server:docs", ToolCallID: "call", ToolKind: ai.ToolPartKindMCPServer,
				Args: json.RawMessage(`{"action":"call_tool","tool_name":"search"}`), ProviderName: "anthropic",
			},
			ai.NativeToolReturnPart{
				ToolName: "mcp_server:docs", ToolCallID: "call", ToolKind: ai.ToolPartKindMCPServer,
				Content: map[string]any{"is_error": false}, ProviderName: "anthropic",
			},
		},
	}}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	blocks := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	input := blocks[0].(map[string]any)["input"]
	resultContent, hasContent := blocks[1].(map[string]any)["content"]
	if len(input.(map[string]any)) != 0 || !hasContent || resultContent != nil {
		t.Fatalf("empty MCP values were not replayed: %#v", body)
	}
}

func TestAnthropicMemoryTool(t *testing.T) {
	requestCount := 0
	var bodies []map[string]any
	var betas []string
	model := newServer(t, func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		betas = append(betas, request.Header.Get("anthropic-beta"))
		requestCount++
		if requestCount == 1 {
			_, _ = response.Write([]byte(`{
				"id":"message-memory","model":"claude-sonnet-4-6","stop_reason":"tool_use",
				"content":[{"type":"tool_use","id":"memory-1","name":"memory","input":{"command":"view","path":"/memories"}}],
				"usage":{}
			}`))
			return
		}
		_, _ = response.Write([]byte(`{
			"id":"message-final","model":"claude-sonnet-4-6","stop_reason":"end_turn",
			"content":[{"type":"text","text":"Mexico City"}],"usage":{}
		}`))
	})
	type memoryCommand struct {
		Command string `json:"command"`
		Path    string `json:"path"`
	}
	called := false
	agent := ai.NewAgent[struct{}, string](model, ai.WithNativeTools(ai.MemoryTool{}))
	ai.AddSimpleTool(agent, "memory", func(_ context.Context, command memoryCommand) (string, error) {
		called = true
		if command.Command != "view" || command.Path != "/memories" {
			t.Fatalf("unexpected memory command: %+v", command)
		}
		return "The user lives in Mexico City.", nil
	})
	result, err := agent.Run(t.Context(), "Where do I live?", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if !called || result.Output != "Mexico City" || requestCount != 2 {
		t.Fatalf("unexpected memory run: called=%v output=%q requests=%d", called, result.Output, requestCount)
	}
	for index, body := range bodies {
		tools := body["tools"].([]any)
		if len(tools) != 1 || tools[0].(map[string]any)["type"] != "memory_20250818" ||
			tools[0].(map[string]any)["name"] != "memory" || tools[0].(map[string]any)["input_schema"] != nil ||
			!strings.Contains(betas[index], "context-management-2025-06-27") {
			t.Fatalf("unexpected native memory definition: tools=%#v beta=%q", tools, betas[index])
		}
	}
	secondMessages := bodies[1]["messages"].([]any)
	toolResult := secondMessages[len(secondMessages)-1].(map[string]any)["content"].([]any)[0].(map[string]any)
	if toolResult["type"] != "tool_result" || toolResult["tool_use_id"] != "memory-1" {
		t.Fatalf("unexpected memory result: %#v", toolResult)
	}
}

func TestAnthropicNativeToolBetaHeaders(t *testing.T) {
	var beta string
	model := newServer(t, func(response http.ResponseWriter, request *http.Request) {
		beta = request.Header.Get("anthropic-beta")
		_, _ = response.Write([]byte(`{"model":"claude-sonnet-4-5","content":[{"type":"text","text":"ok"}],"usage":{}}`))
	})
	_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		ai.WebFetchTool{},
		ai.CodeExecutionTool{Files: []ai.UploadedFile{{
			ProviderName: "anthropic", FileID: "file-1",
		}}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(beta, "web-fetch-2025-09-10") ||
		!strings.Contains(beta, "files-api-2025-04-14") {
		t.Fatalf("missing native tool beta headers: %q", beta)
	}
}

func TestAnthropicMemoryToolRequiresFunction(t *testing.T) {
	model := newServer(t, func(http.ResponseWriter, *http.Request) {
		t.Fatal("memory request was sent without a function implementation")
	})
	for _, native := range []ai.NativeTool{ai.MemoryTool{}, &ai.MemoryTool{}} {
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{native}})
		if err == nil || !strings.Contains(err.Error(), `requires a function tool named "memory"`) {
			t.Fatalf("unexpected memory validation error: %v", err)
		}
	}
}

func TestAnthropicNativeToolSearchStrategies(t *testing.T) {
	var body map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"claude-sonnet-4-6","stop_reason":"end_turn","content":[],"usage":{}}`))
	})
	if !model.SupportsToolSearchStrategy(ai.ToolSearchStrategyBM25) ||
		!model.SupportsToolSearchStrategy(ai.ToolSearchStrategyRegex) ||
		model.SupportsToolSearchStrategy(ai.ToolSearchStrategyKeywords) {
		t.Fatal("unexpected Anthropic tool-search strategy support")
	}
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{
			Name: ai.ToolSearchName, Schema: schema, ToolKind: ai.ToolPartKindToolSearch,
			ToolSearchStrategy: ai.ToolSearchStrategyRegex,
		}},
		DeferredTools: []ai.ToolDefinition{{Name: "hidden", Schema: schema, DeferLoading: true}},
	}); err != nil {
		t.Fatal(err)
	}
	tools := body["tools"].([]any)
	if len(tools) != 2 || tools[1].(map[string]any)["type"] != "tool_search_tool_regex_20251119" {
		t.Fatalf("regex search was not selected: %+v", tools)
	}

	disabled := newServerWithOptions(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model":"m","content":[],"usage":{}}`))
	}, anthropic.WithDeferredToolSupport(false))
	if disabled.SupportsToolSearchStrategy(ai.ToolSearchStrategyBM25) {
		t.Fatal("disabled model advertised native search")
	}
	_, err := disabled.Request(t.Context(), nil, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{
			Name: ai.ToolSearchName, Schema: schema, ToolKind: ai.ToolPartKindToolSearch,
			ToolSearchStrategy: ai.ToolSearchStrategyBM25,
		}},
		DeferredTools: []ai.ToolDefinition{{Name: "hidden", Schema: schema, DeferLoading: true}},
	})
	if err == nil || !strings.Contains(err.Error(), `tool search strategy "bm25" requires deferred-tool support`) {
		t.Fatalf("unexpected disabled native search error: %v", err)
	}
}

func TestAnthropicNativeToolSearchErrorReplay(t *testing.T) {
	request := 0
	var replay map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		request++
		if request == 1 {
			_, _ = w.Write([]byte(`{
				"model":"claude-sonnet-4-6","stop_reason":"end_turn","content":[
					{"type":"server_tool_use","id":"search-1","name":"tool_search_tool_regex","input":{"pattern":"weather"}},
					{"type":"tool_search_tool_result","tool_use_id":"search-1","content":{"type":"tool_search_tool_result_error","error_code":"unavailable","error_message":"temporary"}}
				],"usage":{}
			}`))
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&replay); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"claude-sonnet-4-6","stop_reason":"end_turn","content":[],"usage":{}}`))
	})
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	params := ai.ModelRequestParams{
		Tools:         []ai.ToolDefinition{{Name: ai.ToolSearchName, Schema: schema, ToolKind: ai.ToolPartKindToolSearch}},
		DeferredTools: []ai.ToolDefinition{{Name: "weather", Schema: schema, DeferLoading: true}},
	}
	response, err := model.Request(t.Context(), nil, params)
	if err != nil {
		t.Fatal(err)
	}
	returned := response.Parts[1].(ai.NativeToolReturnPart)
	if returned.ProviderDetails["error_code"] != "unavailable" ||
		returned.ProviderDetails["error_message"] != "temporary" {
		t.Fatalf("native search error details were lost: %+v", returned)
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{*response}, params); err != nil {
		t.Fatal(err)
	}
	content := replay["messages"].([]any)[0].(map[string]any)["content"].([]any)
	errorContent := content[1].(map[string]any)["content"].(map[string]any)
	if errorContent["error_code"] != "unavailable" || errorContent["error_message"] != nil {
		t.Fatalf("unexpected native search error replay: %+v", errorContent)
	}
}

func TestAnthropicNativeToolSearchReplayEdges(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	params := ai.ModelRequestParams{
		Tools:         []ai.ToolDefinition{{Name: ai.ToolSearchName, Schema: schema, ToolKind: ai.ToolPartKindToolSearch}},
		DeferredTools: []ai.ToolDefinition{{Name: "hidden", Schema: schema, DeferLoading: true}},
	}
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model":"claude-sonnet-4-6","stop_reason":"end_turn","content":[],"usage":{}}`))
	})
	for name, part := range map[string]ai.ResponsePart{
		"malformed call": ai.NativeToolCallPart{
			ToolName: ai.ToolSearchName, ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "anthropic", Args: []byte(`{`),
		},
		"unencodable return": ai.NativeToolReturnPart{
			ToolName: ai.ToolSearchName, ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "anthropic", Content: make(chan int),
		},
		"invalid return shape": ai.NativeToolReturnPart{
			ToolName: ai.ToolSearchName, ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "anthropic", Content: "bad",
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{part}}}, params)
			if err == nil {
				t.Fatal("expected native Anthropic replay error")
			}
		})
	}

	var body map[string]any
	model = newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"claude-sonnet-4-6","stop_reason":"end_turn","content":[],"usage":{}}`))
	})
	history := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.NativeToolCallPart{
			ToolName: ai.ToolSearchName, ToolCallID: "empty", ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "anthropic",
		},
		ai.NativeToolCallPart{
			ToolName: ai.ToolSearchName, ToolCallID: "raw", ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "anthropic", Args: []byte(`{"pattern":"already"}`),
			ProviderDetails: map[string]any{"strategy": "regex"},
		},
		ai.NativeToolCallPart{
			ToolName: ai.ToolSearchName, ToolKind: ai.ToolPartKindToolSearch, ProviderName: "openai",
		},
		ai.NativeToolReturnPart{
			ToolName: "capability", ToolKind: ai.ToolPartKindCapabilityLoad, ProviderName: "anthropic",
		},
	}}}
	if _, err := model.Request(t.Context(), history, params); err != nil {
		t.Fatal(err)
	}
	blocks := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(blocks) != 2 || blocks[0].(map[string]any)["input"] == nil ||
		blocks[1].(map[string]any)["name"] != "tool_search_tool_regex" ||
		blocks[1].(map[string]any)["input"].(map[string]any)["pattern"] != "already" {
		t.Fatalf("unexpected replay edge blocks: %+v", blocks)
	}

	if _, err := model.Request(t.Context(), history, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
}

func TestAnthropicNativeToolSearchResponseErrors(t *testing.T) {
	for name, block := range map[string]string{
		"unsupported server tool": `{"type":"server_tool_use","id":"call","name":"future_tool","input":{}}`,
		"malformed server input":  `{"type":"server_tool_use","id":"call","name":"tool_search_tool_bm25","input":"bad"}`,
		"malformed result":        `{"type":"tool_search_tool_result","tool_use_id":"call","content":"bad"}`,
		"unknown result":          `{"type":"tool_search_tool_result","tool_use_id":"call","content":{"type":"future"}}`,
	} {
		t.Run(name, func(t *testing.T) {
			model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
				_, _ = fmt.Fprintf(w, `{"model":"claude-sonnet-4-6","content":[%s],"usage":{}}`, block)
			})
			if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
				t.Fatal("expected native Anthropic response error")
			}
		})
	}

	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"model":"claude-sonnet-4-6","content":[
				{"type":"server_tool_use","id":"empty","name":"tool_search_tool_bm25","input":null,"caller":{"type":"direct"}}
			],"usage":{}
		}`))
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	call := response.Parts[0].(ai.NativeToolCallPart)
	if string(call.Args) != `{"queries":[]}` || call.ProviderDetails["anthropic_caller"] != nil {
		t.Fatalf("unexpected empty native search call: %+v", call)
	}
}

func TestAnthropicReplaysForeignNativeSearchLocally(t *testing.T) {
	var body map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{
			"model":"claude-sonnet-4-6","stop_reason":"end_turn",
			"content":[{"type":"text","text":"done"}],"usage":{}
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
			ProviderName: "openai", Args: []byte(`{"queries":["hidden"]}`),
		},
		ai.NativeToolReturnPart{
			ToolName: ai.ToolSearchName, ToolCallID: "search", ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "openai",
			Content:      ai.ToolSearchResult{DiscoveredTools: []ai.ToolSearchMatch{{Name: "hidden"}}},
		},
	}}}
	if _, err := agent.Run(t.Context(), "continue", struct{}{}, ai.WithMessageHistory(history)); err != nil {
		t.Fatal(err)
	}
	messages := body["messages"].([]any)
	call := messages[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	result := messages[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	if call["type"] != "tool_use" || call["name"] != ai.ToolSearchName ||
		result["type"] != "tool_result" ||
		result["content"].([]any)[0].(map[string]any)["tool_name"] != "hidden" {
		t.Fatalf("unexpected foreign search replay: call=%+v result=%+v", call, result)
	}
	tools := body["tools"].([]any)
	if tools[len(tools)-1].(map[string]any)["name"] != "tool_search_tool_bm25" {
		t.Fatalf("current search did not remain hosted: %+v", tools)
	}
}
