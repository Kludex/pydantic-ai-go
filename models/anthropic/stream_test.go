package anthropic_test

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/anthropic"
)

func anthropicSSE(t *testing.T, events []string) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range events {
			_, _ = fmt.Fprintf(w, "event: ignored\ndata: %s\n\n", event)
		}
	}
}

func collectAnthropicStream(
	t *testing.T, model ai.StreamingModel, params ai.ModelRequestParams,
) ([]ai.ModelStreamEvent, error) {
	t.Helper()
	stream, err := model.StreamRequest(t.Context(), nil, params)
	if err != nil {
		return nil, err
	}
	var events []ai.ModelStreamEvent
	for event, err := range stream {
		if err != nil {
			return events, err
		}
		events = append(events, event)
	}
	return events, nil
}

func normalizedAnthropicText(event ai.StreamEvent) string {
	switch event := event.(type) {
	case ai.PartStartEvent:
		if text, ok := event.Part.(ai.TextPart); ok {
			return text.Content
		}
	case ai.PartDeltaEvent:
		if text, ok := event.Delta.(ai.TextPartDelta); ok {
			return text.ContentDelta
		}
	}
	return ""
}

func TestStreamEvents(t *testing.T) {
	var gotStream bool
	var gotAccept, gotCustom string
	var gotBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotAccept = r.Header.Get("Accept")
		gotCustom = r.Header.Get("x-custom")
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		gotStream = gotBody["stream"] == true
		anthropicSSE(t, []string{
			`{"type":"message_start","message":{"id":"message-stream","model":"claude-stream","usage":{"input_tokens":5,"output_tokens":1,"cache_creation_input_tokens":3,"cache_read_input_tokens":4}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"H"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"i"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"content_block_start","index":1,"content_block":{"type":"thinking","thinking":"A"}}`,
			`{"type":"content_block_delta","index":1,"delta":{"type":"thinking_delta","thinking":"B"}}`,
			`{"type":"content_block_delta","index":1,"delta":{"type":"signature_delta","signature":"ignored"}}`,
			`{"type":"content_block_delta","index":1,"delta":{"type":"citations_delta"}}`,
			`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"c1","name":"work","input":{}}}`,
			`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"x\":"}}`,
			`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"1}"}}`,
			`{"type":"content_block_start","index":3,"content_block":{"type":"compaction","content":"Summary.","encrypted_content":"opaque"}}`,
			`{"type":"ping"}`,
			`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":8}}`,
			`{"type":"message_stop"}`,
		})(w, r)
	})
	events, err := collectAnthropicStream(t, model, ai.ModelRequestParams{Settings: ai.ModelSettings{
		ExtraHeaders: map[string]string{"x-custom": "stream"}, ExtraBody: map[string]any{"container": "stream"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if !gotStream || gotAccept != "text/event-stream" || gotCustom != "stream" || gotBody["container"] != "stream" {
		t.Fatalf("stream request not configured: stream=%v accept=%q custom=%q body=%v", gotStream, gotAccept, gotCustom, gotBody)
	}
	var text, thinking, signature, args string
	var textPartID, thinkingPartID, argsPartID string
	var start ai.ToolCallStartEvent
	var compaction ai.CompactionEvent
	var finish ai.FinishEvent
	for _, event := range events {
		switch event := event.(type) {
		case ai.TextDeltaEvent:
			text += event.Delta
			textPartID = event.PartID
		case ai.ThinkingDeltaEvent:
			thinking += event.Delta
			thinkingPartID = event.PartID
			if event.SignatureDelta != "" {
				signature = event.SignatureDelta
			}
		case ai.CompactionEvent:
			compaction = event
		case ai.ToolCallStartEvent:
			start = event
		case ai.ToolCallDeltaEvent:
			args += event.ArgsDelta
			argsPartID = event.PartID
		case ai.FinishEvent:
			finish = event
		}
	}
	if text != "Hi" || thinking != "AB" || signature != "ignored" || start.ToolName != "work" ||
		start.ToolCallID != "c1" || args != `{"x":1}` || compaction.PartID != "3" ||
		compaction.Content != "Summary." || compaction.ProviderName != "anthropic" ||
		compaction.ProviderDetails["encrypted_content"] != "opaque" {
		t.Fatalf(
			"unexpected events: text=%q thinking=%q compaction=%+v start=%+v args=%q",
			text, thinking, compaction, start, args,
		)
	}
	if textPartID != "0" || thinkingPartID != "1" || start.PartID != "2" || argsPartID != "2" {
		t.Fatalf("unstable Anthropic part IDs: text=%q thinking=%q start=%q args=%q", textPartID, thinkingPartID, start.PartID, argsPartID)
	}
	if finish.ModelName != "claude-stream" || finish.Usage.Requests != 1 || finish.Usage.InputTokens != 12 ||
		finish.Usage.OutputTokens != 8 || finish.Usage.CacheWriteTokens != 3 || finish.Usage.CacheReadTokens != 4 ||
		finish.ProviderName != "anthropic" || finish.ProviderURL == "" ||
		finish.ProviderResponseID != "message-stream" || finish.FinishReason != ai.FinishReasonToolCall ||
		finish.ProviderDetails["finish_reason"] != "tool_use" || finish.State != ai.ModelResponseStateComplete {
		t.Fatalf("unexpected finish %+v", finish)
	}
}

func TestPauseTurnStreamIsSuspended(t *testing.T) {
	model := newServer(t, anthropicSSE(t, []string{
		`{"type":"message_start","message":{"id":"paused","model":"claude","usage":{"input_tokens":1}}}`,
		`{"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":1}}`,
		`{"type":"message_stop"}`,
	}))
	events, err := collectAnthropicStream(t, model, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	finish := events[len(events)-1].(ai.FinishEvent)
	if finish.State != ai.ModelResponseStateSuspended || finish.FinishReason != "" ||
		finish.ProviderDetails["finish_reason"] != "pause_turn" {
		t.Fatalf("unexpected paused stream: %+v", finish)
	}
}

func TestAgentStreamAutomaticallyContinuesPauseTurn(t *testing.T) {
	requests := 0
	var secondBody map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		requests++
		if requests == 1 {
			anthropicSSE(t, []string{
				`{"type":"message_start","message":{"id":"paused","model":"claude","usage":{"input_tokens":1}}}`,
				`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial "}}`,
				`{"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":1}}`,
				`{"type":"message_stop"}`,
			})(w, r)
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&secondBody); err != nil {
			t.Error(err)
		}
		anthropicSSE(t, []string{
			`{"type":"message_start","message":{"id":"complete","model":"claude","usage":{"input_tokens":2}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"done"}}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
			`{"type":"message_stop"}`,
		})(w, r)
	})
	stream := ai.NewAgent[struct{}, string](model).RunStream(t.Context(), "go", struct{}{})
	var text string
	var starts []int
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		text += normalizedAnthropicText(event)
		if event, ok := event.(ai.PartStartEvent); ok {
			starts = append(starts, event.Index)
		}
	}
	result := stream.Result()
	if result == nil || result.Output != "partial done" || result.Usage().Requests != 2 ||
		text != "partial done" || len(starts) != 2 || starts[0] != 0 || starts[1] != 1 {
		t.Fatalf("unexpected pause stream result=%+v text=%q starts=%v", result, text, starts)
	}
	messages := secondBody["messages"].([]any)
	if len(messages) != 2 || messages[1].(map[string]any)["role"] != "assistant" {
		t.Fatalf("paused stream response was not echoed: %+v", messages)
	}
}

func TestStreamEndToEnd(t *testing.T) {
	model := newServer(t, anthropicSSE(t, []string{
		`{"type":"message_start","message":{"model":"claude","usage":{"input_tokens":2,"output_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}`,
		`{"type":"message_delta","usage":{"output_tokens":1}}`,
		`{"type":"message_stop"}`,
	}))
	agent := ai.NewAgent[struct{}, string](model)
	stream := agent.RunStream(t.Context(), "go", struct{}{})
	var text string
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		text += normalizedAnthropicText(event)
	}
	if text != "hello" || stream.Result().Output != "hello" {
		t.Fatalf("unexpected stream text %q and result %+v", text, stream.Result())
	}
}

func TestStreamProtocolErrors(t *testing.T) {
	tests := []struct {
		name   string
		events []string
		want   string
	}{
		{name: "malformed", events: []string{`not json`}, want: "parse stream event"},
		{name: "api error", events: []string{`{"type":"error","error":{"type":"overloaded_error","message":"busy"}}`}, want: "overloaded_error"},
		{name: "unknown event", events: []string{`{"type":"mystery"}`}, want: "unknown stream event"},
		{name: "unknown block", events: []string{`{"type":"content_block_start","content_block":{"type":"audio"}}`}, want: "unsupported content block"},
		{name: "unknown delta", events: []string{`{"type":"content_block_delta","delta":{"type":"audio_delta"}}`}, want: "unsupported content block delta"},
		{name: "missing stop", events: []string{`{"type":"ping"}`}, want: "without message_stop"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := newServer(t, anthropicSSE(t, test.events))
			_, err := collectAnthropicStream(t, model, ai.ModelRequestParams{})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("expected %q error, got %v", test.want, err)
			}
		})
	}
}

func TestStreamRequestErrors(t *testing.T) {
	t.Run("bad payload", func(t *testing.T) {
		model := newServer(t, anthropicSSE(t, nil))
		params := ai.ModelRequestParams{Tools: []ai.ToolDefinition{{Name: "bad", Schema: map[string]any{"x": make(chan int)}}}}
		if _, err := collectAnthropicStream(t, model, params); err == nil {
			t.Fatal("expected marshal error")
		}
	})
	t.Run("bad message", func(t *testing.T) {
		model := newServer(t, anthropicSSE(t, nil))
		if _, err := model.StreamRequest(t.Context(), []ai.ModelMessage{nil}, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected message error")
		}
	})
	t.Run("invalid URL", func(t *testing.T) {
		model := anthropic.NewModel("claude", anthropic.WithBaseURL("http://[::1"))
		if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected URL error")
		}
	})
	t.Run("transport", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		server.Close()
		model := anthropic.NewModel("claude", anthropic.WithBaseURL(server.URL))
		if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected transport error")
		}
	})
	t.Run("http error", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte("bad"))
		})
		if _, err := collectAnthropicStream(t, model, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected API error")
		}
	})
	t.Run("truncated error", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "100")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte("short"))
		})
		if _, err := collectAnthropicStream(t, model, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected read error")
		}
	})
}

func TestStreamScannerError(t *testing.T) {
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("data: "))
		_, _ = w.Write(make([]byte, 2*1024*1024))
	})
	if _, err := collectAnthropicStream(t, model, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected scanner error")
	}
}

func TestStreamEarlyBreak(t *testing.T) {
	events := []string{
		`{"type":"message_start","message":{"usage":{}}}`,
		`{"type":"content_block_start","content_block":{"type":"text","text":"a"}}`,
		`{"type":"content_block_start","content_block":{"type":"thinking","thinking":"b"}}`,
		`{"type":"content_block_start","content_block":{"type":"tool_use","id":"c","name":"work","input":{"x":1}}}`,
		`{"type":"content_block_delta","delta":{"type":"text_delta","text":"c"}}`,
		`{"type":"message_stop"}`,
	}
	for breakAt := 1; breakAt <= 5; breakAt++ {
		model := newServer(t, anthropicSSE(t, events))
		stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		count := 0
		for range stream {
			count++
			if count == breakAt {
				break
			}
		}
	}
}

func TestAnthropicStreamWebSearch(t *testing.T) {
	model := newServer(t, anthropicSSE(t, []string{
		`{"type":"message_start","message":{"id":"response","model":"claude-sonnet-4-5","usage":{}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"web-1","name":"web_search","input":{},"caller":{"type":"code_execution_20250825"}}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"Go news\"}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"web-1","content":[{"type":"web_search_result","url":"https://go.dev"}],"caller":{"type":"code_execution_20250825"}}}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`,
		`{"type":"message_stop"}`,
	}))
	events, err := collectAnthropicStream(t, model, ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.WebSearchTool{}}})
	if err != nil {
		t.Fatal(err)
	}
	var start ai.ToolCallStartEvent
	var delta ai.ToolCallDeltaEvent
	var returned ai.NativeToolReturnEvent
	for _, event := range events {
		switch event := event.(type) {
		case ai.ToolCallStartEvent:
			start = event
		case ai.ToolCallDeltaEvent:
			delta = event
		case ai.NativeToolReturnEvent:
			returned = event
		}
	}
	if !start.Native || start.ToolName != "web_search" || start.ToolKind != ai.ToolPartKindWebSearch ||
		start.ProviderDetails["anthropic_caller"] == nil || delta.ArgsDelta != `{"query":"Go news"}` ||
		returned.Part.ToolKind != ai.ToolPartKindWebSearch || returned.Part.ToolCallID != "web-1" ||
		len(returned.Part.Content.([]any)) != 1 || returned.Part.ProviderDetails["anthropic_caller"] == nil {
		t.Fatalf("unexpected streamed web search events: %#v", events)
	}
}

func TestAnthropicStreamWebFetch(t *testing.T) {
	model := newServer(t, anthropicSSE(t, []string{
		`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"fetch","name":"web_fetch","input":{}}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"url\":\"https://go.dev\"}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"web_fetch_tool_result","tool_use_id":"fetch","content":{"type":"web_fetch_result","url":"https://go.dev"}}}`,
		`{"type":"message_stop"}`,
	}))
	events, err := collectAnthropicStream(t, model, ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.WebFetchTool{}}})
	if err != nil {
		t.Fatal(err)
	}
	start := events[0].(ai.ToolCallStartEvent)
	delta := events[1].(ai.ToolCallDeltaEvent)
	returned := events[2].(ai.NativeToolReturnEvent)
	if start.ToolName != "web_fetch" || start.ToolKind != ai.ToolPartKindWebFetch || !start.Native ||
		delta.ArgsDelta != `{"url":"https://go.dev"}` || returned.Part.ToolKind != ai.ToolPartKindWebFetch ||
		returned.Part.Content.(map[string]any)["url"] != "https://go.dev" {
		t.Fatalf("unexpected streamed web fetch events: %#v", events)
	}
}

func TestAnthropicStreamWebSearchEdges(t *testing.T) {
	t.Run("empty arguments", func(t *testing.T) {
		model := newServer(t, anthropicSSE(t, []string{
			`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"web","name":"web_search","input":null}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_stop"}`,
		}))
		events, err := collectAnthropicStream(t, model, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		if delta := events[1].(ai.ToolCallDeltaEvent); delta.ArgsDelta != `{}` {
			t.Fatalf("unexpected empty web-search delta: %+v", delta)
		}
	})
	t.Run("consumer break on return", func(t *testing.T) {
		model := newServer(t, anthropicSSE(t, []string{
			`{"type":"content_block_start","index":0,"content_block":{"type":"web_search_tool_result","tool_use_id":"web","content":[]}}`,
		}))
		stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		seen := 0
		for _, err := range stream {
			if err != nil {
				t.Fatal(err)
			}
			seen++
			break
		}
		if seen != 1 {
			t.Fatalf("stream yielded %d events before break", seen)
		}
	})
}

func TestAnthropicStreamServerManagedToolSearch(t *testing.T) {
	model := newServer(t, anthropicSSE(t, []string{
		`{"type":"message_start","message":{"id":"message-1","model":"claude-sonnet-4-6","usage":{}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"search-1","name":"tool_search_tool_bm25","input":{},"caller":{"type":"direct"}}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"weather\"}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_search_tool_result","tool_use_id":"search-1","content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"weather"}]}}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`,
		`{"type":"message_stop"}`,
	}))
	events, err := collectAnthropicStream(t, model, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 4 {
		t.Fatalf("unexpected native search events: %+v", events)
	}
	start := events[0].(ai.ToolCallStartEvent)
	delta := events[1].(ai.ToolCallDeltaEvent)
	returned := events[2].(ai.NativeToolReturnEvent)
	if !start.Native || start.ToolCallID != "search-1" || start.ProviderName != "anthropic" ||
		start.ProviderDetails["strategy"] != "bm25" || start.ProviderDetails["anthropic_caller"] != nil ||
		delta.ArgsDelta != `{"queries":["weather"]}` ||
		returned.Part.ToolCallID != "search-1" ||
		returned.Part.Content.(ai.ToolSearchResult).DiscoveredTools[0].Name != "weather" {
		t.Fatalf("unexpected native search lifecycle: %+v", events)
	}
}

func TestAnthropicStreamNativeToolSearchEdges(t *testing.T) {
	t.Run("regex without deltas", func(t *testing.T) {
		model := newServer(t, anthropicSSE(t, []string{
			`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"search","name":"tool_search_tool_regex","input":null,"caller":{"type":"code_execution_20250825"}}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_stop"}`,
		}))
		events, err := collectAnthropicStream(t, model, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		start := events[0].(ai.ToolCallStartEvent)
		delta := events[1].(ai.ToolCallDeltaEvent)
		if start.ProviderDetails["strategy"] != "regex" || start.ProviderDetails["anthropic_caller"] == nil ||
			delta.ArgsDelta != `{"queries":[]}` {
			t.Fatalf("unexpected regex stream: %+v", events)
		}
	})

	for name, events := range map[string][]string{
		"malformed call": {
			`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"search","name":"tool_search_tool_bm25","input":"bad"}}`,
			`{"type":"content_block_stop","index":0}`,
		},
		"malformed result": {
			`{"type":"content_block_start","index":0,"content_block":{"type":"tool_search_tool_result","tool_use_id":"search","content":"bad"}}`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			model := newServer(t, anthropicSSE(t, events))
			if _, err := collectAnthropicStream(t, model, ai.ModelRequestParams{}); err == nil {
				t.Fatal("expected malformed native stream error")
			}
		})
	}
}

func TestAnthropicStreamNativeToolSearchCanStop(t *testing.T) {
	callEvents := []string{
		`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"search","name":"tool_search_tool_bm25","input":{"query":"x"}}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"future"}`,
	}
	returnEvents := []string{
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_search_tool_result","tool_use_id":"search","content":{"type":"tool_search_tool_search_result","tool_references":[]}}}`,
		`{"type":"future"}`,
	}
	for name, test := range map[string]struct {
		events []string
		stop   func(ai.ModelStreamEvent) bool
	}{
		"call": {events: callEvents, stop: func(event ai.ModelStreamEvent) bool {
			_, ok := event.(ai.ToolCallStartEvent)
			return ok
		}},
		"delta": {events: callEvents, stop: func(event ai.ModelStreamEvent) bool {
			_, ok := event.(ai.ToolCallDeltaEvent)
			return ok
		}},
		"return": {events: returnEvents, stop: func(event ai.ModelStreamEvent) bool {
			_, ok := event.(ai.NativeToolReturnEvent)
			return ok
		}},
	} {
		t.Run(name, func(t *testing.T) {
			model := newServer(t, anthropicSSE(t, test.events))
			stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			for event, err := range stream {
				if err != nil {
					t.Fatal(err)
				}
				if test.stop(event) {
					break
				}
			}
		})
	}
}
