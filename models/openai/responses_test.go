package openai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

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

func TestResponsesTextResponse(t *testing.T) {
	var gotBody map[string]any
	var gotPath string
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{
			"id": "response-1", "model": "gpt-5", "created_at": 1735689600.25,
			"status": "completed",
			"output": [
				{"id": "reasoning-1", "type": "reasoning", "encrypted_content": "signature", "summary": [{"text": "thinking"}]},
				{"id": "message-1", "type": "message", "content": [{"type": "output_text", "text": "Hello!"}]}
			],
			"usage": {
				"input_tokens": 12, "output_tokens": 5,
				"input_tokens_details": {"cached_tokens": 4},
				"output_tokens_details": {"reasoning_tokens": 2}
			}
		}`))
	})
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hi"}}}}
	resp, err := model.Request(t.Context(), msgs, ai.ModelRequestParams{Instructions: "be brief", AllowText: true})
	if err != nil {
		t.Fatal(err)
	}
	if gotPath != "/responses" {
		t.Fatalf("unexpected path %q", gotPath)
	}
	if gotBody["instructions"] != "be brief" {
		t.Fatalf("instructions not sent: %v", gotBody)
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
	if text.ID != "message-1" || text.ProviderName != "openai" {
		t.Fatalf("text metadata lost: %+v", text)
	}
	if resp.ProviderName != "openai" || resp.ProviderURL == "" || resp.ProviderResponseID != "response-1" ||
		resp.FinishReason != ai.FinishReasonStop || resp.State != ai.ModelResponseStateComplete ||
		resp.ProviderDetails["finish_reason"] != "completed" || resp.ProviderDetails["timestamp"] == nil {
		t.Fatalf("unexpected response metadata %+v", resp)
	}
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 5 ||
		resp.Usage.CacheReadTokens != 4 || resp.Usage.ReasoningTokens != 2 ||
		resp.Usage.Details["reasoning_tokens"] != 2 {
		t.Fatalf("unexpected usage %+v", resp.Usage)
	}
}

func TestResponsesPreservesEncryptedReasoningWithoutSummary(t *testing.T) {
	model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"model":"gpt-5","status":"completed",
			"output":[{"id":"reasoning-1","type":"reasoning","encrypted_content":"signature"}]
		}`))
	})
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	thinking := response.Parts[0].(ai.ThinkingPart)
	if thinking.Content != "" || thinking.ID != "reasoning-1" || thinking.Signature != "signature" ||
		thinking.ProviderName != "openai" {
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
		ToolKind: ai.ToolPartKindToolSearch,
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
		"server search":                   `{"type":"tool_search_call","call_id":"search","execution":"server","arguments":{}}`,
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
