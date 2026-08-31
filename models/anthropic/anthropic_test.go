package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
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

func newServer(t *testing.T, handler http.HandlerFunc) *anthropic.Model {
	t.Helper()
	return newServerWithOptions(t, handler)
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

func TestRequestTextResponse(t *testing.T) {
	var gotBody map[string]any
	var gotKey, gotVersion string
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("x-api-key")
		gotVersion = r.Header.Get("anthropic-version")
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{
			"id": "message-1", "model": "claude-sonnet-4-5", "stop_reason": "end_turn",
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
	})
	if err != nil {
		t.Fatal(err)
	}
	if gotKey != "test-key" || gotVersion != "2023-06-01" {
		t.Fatalf("unexpected headers key=%q version=%q", gotKey, gotVersion)
	}
	if gotBody["system"] != "be brief" {
		t.Fatalf("system prompt not sent: %v", gotBody)
	}
	if gotBody["max_tokens"].(float64) != 4096 {
		t.Fatalf("default max_tokens not applied: %v", gotBody["max_tokens"])
	}
	if resp.Text() != "Hello!" {
		t.Fatalf("unexpected text %q", resp.Text())
	}
	if resp.ProviderName != "anthropic" || resp.ProviderURL == "" || resp.ProviderResponseID != "message-1" ||
		resp.FinishReason != ai.FinishReasonStop || resp.ProviderDetails["finish_reason"] != "end_turn" ||
		resp.State != ai.ModelResponseStateComplete {
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
	search := ai.ToolDefinition{Name: ai.ToolSearchName, Schema: schema, ToolKind: ai.ToolPartKindToolSearch}
	first := ai.ToolDefinition{Name: "first", Schema: schema, DeferLoading: true}
	second := ai.ToolDefinition{Name: "second", Schema: schema, DeferLoading: true}
	messages := []ai.ModelMessage{
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: ai.ToolSearchName, ToolCallID: "search", ToolKind: ai.ToolPartKindToolSearch,
			Args: []byte(`{"queries":["first"]}`),
		}}},
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
	if gotBeta != "mid-conversation-tool-changes-2026-07-01" {
		t.Fatalf("tool-addition beta header missing: %q", gotBeta)
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
	t.Run("api error", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte("rate limited"))
		})
		var apiErr *anthropic.APIError
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{AllowText: true})
		if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusTooManyRequests {
			t.Fatalf("expected APIError 429, got %v", err)
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
	if anthropic.NewModel("claude-sonnet-4-5").Name() != "claude-sonnet-4-5" {
		t.Fatal("unexpected name")
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
	}}}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{AllowText: true}); err != nil {
		t.Fatal(err)
	}
	blocks := gotBody["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(blocks) != 3 {
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
}

func TestMultimodalUnknownContent(t *testing.T) {
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) })
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{nil}}}}}
	if _, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected error")
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
