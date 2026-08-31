package openai_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func TestResponsesStreamEvents(t *testing.T) {
	var streamed bool
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Stream bool `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		streamed = body.Stream
		sseHandler(t, []string{
			`{"type":"response.created","response":{"model":"gpt-5"}}`,
			`{"type":"response.output_item.added","item":{"id":"msg","type":"message"}}`,
			`{"type":"response.output_text.delta","item_id":"msg","delta":"Hi"}`,
			`{"type":"response.output_item.added","item":{"id":"reason","type":"reasoning","encrypted_content":"signature"}}`,
			`{"type":"response.reasoning_summary_part.added","item_id":"reason","part":{"text":"A"}}`,
			`{"type":"response.reasoning_summary_text.delta","item_id":"reason","delta":"B"}`,
			`{"type":"response.reasoning_text.delta","item_id":"reason","delta":"C"}`,
			`{"type":"response.output_item.added","item":{"id":"fc","type":"function_call","call_id":"c1","name":"work","namespace":"tools","arguments":""}}`,
			`{"type":"response.function_call_arguments.delta","item_id":"fc","delta":"{\"x\":"}`,
			`{"type":"response.function_call_arguments.delta","item_id":"fc","delta":"1}"}`,
			`{"type":"response.output_text.done"}`,
			`{"type":"response.completed","response":{"id":"response-stream","model":"gpt-5","created_at":1735689600.25,"status":"completed","usage":{"input_tokens":5,"output_tokens":3}}}`,
			`[DONE]`,
		})(w, r)
	})
	events, err := collect(t, model, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if !streamed {
		t.Fatal("Responses request did not enable streaming")
	}
	var text, thinking, args string
	var textPartID, textID, thinkingPartID, thinkingID, thinkingSignature, argsPartID string
	var start ai.ToolCallStartEvent
	var finish ai.FinishEvent
	for _, event := range events {
		switch event := event.(type) {
		case ai.TextDeltaEvent:
			text += event.Delta
			textPartID = event.PartID
			textID = event.ID
		case ai.ThinkingDeltaEvent:
			thinking += event.Delta
			thinkingPartID = event.PartID
			if event.ID != "" {
				thinkingID = event.ID
			}
			if event.SignatureDelta != "" {
				thinkingSignature = event.SignatureDelta
			}
		case ai.ToolCallStartEvent:
			start = event
		case ai.ToolCallDeltaEvent:
			args += event.ArgsDelta
			argsPartID = event.PartID
		case ai.FinishEvent:
			finish = event
		}
	}
	if text != "Hi" || thinking != "ABC" || start.ToolName != "work" || start.ToolCallID != "c1" || args != `{"x":1}` {
		t.Fatalf("unexpected events text=%q thinking=%q start=%+v args=%q", text, thinking, start, args)
	}
	if textPartID != "output:0:content:0:text" || textID != "msg" ||
		thinkingPartID != "item:reason:thinking:0" || thinkingID != "reason" || thinkingSignature != "signature" ||
		start.PartID != "item:fc" || start.ID != "fc" || start.ProviderDetails["namespace"] != "tools" ||
		argsPartID != start.PartID {
		t.Fatalf("unstable Responses part IDs: text=%q thinking=%q start=%q args=%q", textPartID, thinkingPartID, start.PartID, argsPartID)
	}
	if finish.ModelName != "gpt-5" || finish.Usage.Requests != 1 || finish.Usage.InputTokens != 5 ||
		finish.Usage.OutputTokens != 3 || finish.ProviderName != "openai" || finish.ProviderURL == "" ||
		finish.ProviderResponseID != "response-stream" || finish.FinishReason != ai.FinishReasonStop ||
		finish.State != ai.ModelResponseStateComplete || finish.Timestamp.IsZero() ||
		finish.ProviderDetails["finish_reason"] != "completed" || finish.ProviderDetails["timestamp"] == nil {
		t.Fatalf("unexpected finish %+v", finish)
	}
}

func TestResponsesStreamUsesAuthoritativeCompletedSnapshot(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.output_text.delta","item_id":"message","delta":"suffix"}`,
		`{"type":"response.completed","response":{"id":"response","model":"gpt-5","status":"completed","output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"full output"}]}],"usage":{}}}`,
	}))
	stream := ai.NewAgent[struct{}, string](model).RunStream(t.Context(), "go", struct{}{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	if result := stream.Result(); result == nil || result.Output != "full output" {
		t.Fatalf("completed snapshot did not replace partial deltas: %+v", result)
	}
}

func TestResponsesCompletedSnapshotErrorsAndConsumerStop(t *testing.T) {
	t.Run("sequence-only metadata", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, []string{
			`{"type":"response.created","sequence_number":1,"response":{"id":"job"}}`,
			`{"type":"response.completed","response":{"id":"job","status":"completed"}}`,
		}))
		events, err := collect(t, model, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		metadata, ok := events[0].(ai.ResponseMetadataEvent)
		if !ok || metadata.ProviderDetails["sequence_number"] != 1 {
			t.Fatalf("unexpected sequence metadata: %+v", events)
		}
	})

	t.Run("metadata consumer stop", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, []string{
			`{"type":"response.created","response":{"id":"job","status":"queued","background":true}}`,
			`{"type":"mystery"}`,
		}))
		stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		for range stream {
			break
		}
	})

	t.Run("invalid snapshot", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, []string{
			`{"type":"response.completed","response":{"status":"completed","output":[{"type":"function_call","name":"work","arguments":"bad"}]}}`,
		}))
		if _, err := collect(t, model, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected completed snapshot error")
		}
	})

	t.Run("consumer stop", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, []string{
			`{"type":"response.completed","response":{"status":"completed","output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]}]}}`,
		}))
		stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		for range stream {
			break
		}
	})

	t.Run("invalid pending snapshot", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, []string{
			`{"type":"response.created","response":{"id":"job","status":"queued","background":true,"output":[{"type":"function_call","name":"work","arguments":"bad"}]}}`,
		}))
		if _, err := collect(t, model, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected pending snapshot error")
		}
	})

	t.Run("pending consumer stop", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, []string{
			`{"type":"response.created","response":{"id":"job","status":"queued","background":true,"output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"partial"}]}]}}`,
		}))
		stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		for event := range stream {
			if _, ok := event.(ai.TextDeltaEvent); ok {
				break
			}
		}
	})
}

func TestResponsesStreamDetachPreservesBackgroundJob(t *testing.T) {
	var paths []string
	model := newResponsesServerWithOptions(t, func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		sseHandler(t, []string{
			`{"type":"response.created","sequence_number":4,"response":{"id":"job","model":"gpt-5","status":"queued","background":true,"usage":{"input_tokens":2}}}`,
			`{"type":"response.output_text.delta","sequence_number":5,"item_id":"message","delta":"partial"}`,
		})(w, r)
	}, openai.WithBackgroundMode(true), openai.WithBackgroundPollInterval(0))
	stream := ai.NewAgent[struct{}, string](model).RunStream(t.Context(), "go", struct{}{})
	for range stream.Events() {
		break
	}
	suspended := stream.Suspended()
	if suspended == nil || suspended.Response().ProviderResponseID != "job" ||
		suspended.Response().Text() != "partial" || suspended.Response().ProviderDetails["sequence_number"] != 5 {
		t.Fatalf("unexpected OpenAI detached snapshot: %+v", suspended)
	}
	if len(paths) != 1 || paths[0] != "/responses" {
		t.Fatalf("detach polled or canceled the background job: %v", paths)
	}

	resumeModel := newResponsesServerWithOptions(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/responses/job" {
			t.Fatalf("unexpected resume request %s %s", r.Method, r.URL.Path)
		}
		_, _ = w.Write([]byte(`{
			"id":"job","model":"gpt-5","status":"completed","background":true,
			"output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]}],
			"usage":{"input_tokens":2,"output_tokens":1}
		}`))
	}, openai.WithBackgroundPollInterval(0))
	result, err := ai.NewAgent[struct{}, string](resumeModel).Resume(t.Context(), suspended.Messages(), struct{}{})
	if err != nil || result.Output != "done" {
		t.Fatalf("OpenAI detached history did not resume: result=%+v err=%v", result, err)
	}
}

func TestResponsesResumedStreamDetachesWithoutCreatedEvent(t *testing.T) {
	model := newResponsesServerWithOptions(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Query().Get("starting_after") != "1" {
			t.Fatalf("unexpected resumed stream request: %s %s", r.Method, r.URL.String())
		}
		sseHandler(t, []string{
			`{"type":"response.output_text.delta","sequence_number":2,"item_id":"message","delta":"partial"}`,
		})(w, r)
	}, openai.WithBackgroundPollInterval(0))
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "go"}}},
		ai.ModelResponse{
			ProviderName: "openai", ProviderResponseID: "job", ModelName: "gpt-5",
			ProviderDetails: map[string]any{"background": true, "sequence_number": 1},
			State:           ai.ModelResponseStateSuspended,
		},
	}
	stream := ai.NewAgent[struct{}, string](model).ResumeStream(t.Context(), history, struct{}{})
	for range stream.Events() {
		break
	}
	suspended := stream.Suspended()
	if suspended == nil || suspended.Response().ProviderResponseID != "job" ||
		suspended.Response().ProviderDetails["sequence_number"] != 2 || suspended.Response().Text() != "partial" {
		t.Fatalf("unexpected resumed detached snapshot: %+v", suspended)
	}
}

func TestResponsesStreamContinuesBackgroundJob(t *testing.T) {
	var methods []string
	var queries []string
	model := newResponsesServerWithOptions(t, func(w http.ResponseWriter, r *http.Request) {
		methods = append(methods, r.Method)
		queries = append(queries, r.URL.RawQuery)
		if r.Method == http.MethodPost {
			sseHandler(t, []string{
				`{"type":"response.created","sequence_number":4,"response":{"id":"job","status":"queued","background":true,"output":[{"id":"partial","type":"message","content":[{"type":"output_text","text":"partial"}]}],"usage":{"input_tokens":1}}}`,
			})(w, r)
			return
		}
		sseHandler(t, []string{
			`{"type":"response.output_text.delta","sequence_number":5,"item_id":"message","delta":"done"}`,
			`{"type":"response.completed","sequence_number":8,"response":{"id":"job","model":"gpt-5","status":"completed","background":true,"usage":{"input_tokens":1,"output_tokens":1}}}`,
		})(w, r)
	}, openai.WithBackgroundPollInterval(0))
	stream := ai.NewAgent[struct{}, string](model).RunStream(t.Context(), "go", struct{}{})
	var finishes []ai.FinishEvent
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(ai.FinishEvent); ok {
			finishes = append(finishes, event)
		}
	}
	result := stream.Result()
	if result == nil || result.Output != "done" || result.Usage().Requests != 1 || len(finishes) != 2 {
		t.Fatalf("unexpected streamed background result=%+v finishes=%+v", result, finishes)
	}
	if len(methods) != 2 || methods[0] != http.MethodPost || methods[1] != http.MethodGet ||
		!strings.Contains(queries[1], "starting_after=4") || !strings.Contains(queries[1], "stream=true") {
		t.Fatalf("unexpected stream retrieval methods=%v queries=%v", methods, queries)
	}
	if finishes[0].State != ai.ModelResponseStateSuspended || finishes[0].ProviderDetails["sequence_number"] != 4 {
		t.Fatalf("pending stream metadata was not retained: %+v", finishes[0])
	}
}

func TestResponsesStreamRetrievesWithoutSequenceAsStaticEvents(t *testing.T) {
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/responses/job" {
			t.Fatalf("unexpected static retrieval %s %s", r.Method, r.URL.Path)
		}
		_, _ = w.Write([]byte(`{
			"id":"job","model":"gpt-5","status":"completed","background":true,
			"output":[
				{"id":"reason","type":"reasoning","encrypted_content":"signature","summary":[{"text":"thinking"}]},
				{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]},
				{"id":"call","type":"function_call","call_id":"call","name":"work","arguments":{"x":1}}
			],
			"usage":{"input_tokens":1,"output_tokens":2}
		}`))
	})
	messages := []ai.ModelMessage{ai.ModelResponse{
		ProviderName: "openai", ProviderResponseID: "job", State: ai.ModelResponseStateSuspended,
		ProviderDetails: map[string]any{"background": true},
	}}
	events, err := model.StreamRequest(t.Context(), messages, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var got []ai.ModelStreamEvent
	for event, err := range events {
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, event)
	}
	if len(got) != 5 {
		t.Fatalf("unexpected static event count: %d (%+v)", len(got), got)
	}
	if _, ok := got[0].(ai.ThinkingDeltaEvent); !ok {
		t.Fatalf("missing static thinking event: %+v", got)
	}
	if _, ok := got[1].(ai.TextDeltaEvent); !ok {
		t.Fatalf("missing static text event: %+v", got)
	}
	if _, ok := got[2].(ai.ToolCallStartEvent); !ok {
		t.Fatalf("missing static tool start: %+v", got)
	}
	if finish, ok := got[4].(ai.FinishEvent); !ok || finish.State != ai.ModelResponseStateComplete {
		t.Fatalf("missing static finish: %+v", got)
	}
}

func TestResponsesStaticStreamSupportsEarlyConsumerStops(t *testing.T) {
	for stopAt := 1; stopAt <= 4; stopAt++ {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{
				"id":"job","model":"gpt-5","status":"completed",
				"output":[
					{"id":"reason","type":"reasoning","summary":[{"text":"thinking"}]},
					{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]},
					{"id":"call","type":"function_call","call_id":"call","name":"work","arguments":{}}
				]
			}`))
		})
		messages := []ai.ModelMessage{ai.ModelResponse{
			ProviderName: "openai", ProviderResponseID: "job", State: ai.ModelResponseStateSuspended,
			ProviderDetails: map[string]any{"background": true},
		}}
		stream, err := model.StreamRequest(t.Context(), messages, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		count := 0
		for range stream {
			count++
			if count == stopAt {
				break
			}
		}
	}
}

func TestResponsesStreamAcceptsSerializedSequenceNumber(t *testing.T) {
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("starting_after") != "7" {
			t.Fatalf("unexpected sequence query: %s", r.URL.RawQuery)
		}
		sseHandler(t, []string{
			`{"type":"response.completed","sequence_number":8,"response":{"id":"job","model":"gpt-5","background":false,"output":[{"id":"message","type":"message","content":[{"type":"output_text","text":"done"}]}],"usage":{}}}`,
		})(w, r)
	})
	messages := []ai.ModelMessage{ai.ModelResponse{
		ProviderName: "openai", ProviderResponseID: "job", State: ai.ModelResponseStateSuspended,
		ProviderDetails: map[string]any{"background": true, "sequence_number": float64(7)},
	}}
	stream, err := model.StreamRequest(t.Context(), messages, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var text string
	for event, err := range stream {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(ai.TextDeltaEvent); ok {
			text += event.Delta
		}
	}
	if text != "done" {
		t.Fatalf("completed retrieval output was not replayed: %q", text)
	}
}

func TestResponsesBackgroundStreamRequestFailures(t *testing.T) {
	messages := func(sequence bool) []ai.ModelMessage {
		details := map[string]any{"background": true}
		if sequence {
			details["sequence_number"] = 1
		}
		return []ai.ModelMessage{ai.ModelResponse{
			ProviderName: "openai", ProviderResponseID: "job", State: ai.ModelResponseStateSuspended,
			ProviderDetails: details,
		}}
	}
	t.Run("seed consumer stop", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, []string{`{"type":"mystery"}`}))
		stream, err := model.StreamRequest(t.Context(), messages(true), ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		for range stream {
			break
		}
	})

	t.Run("static retrieve", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "missing", http.StatusNotFound)
		})
		if _, err := model.StreamRequest(t.Context(), messages(false), ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected static retrieve error")
		}
	})
	t.Run("invalid URL", func(t *testing.T) {
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL("http://[::1"))
		if _, err := model.StreamRequest(t.Context(), messages(true), ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected stream retrieve URL error")
		}
	})
	t.Run("transport", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		server.Close()
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL(server.URL))
		if _, err := model.StreamRequest(t.Context(), messages(true), ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected stream retrieve transport error")
		}
	})
	t.Run("HTTP", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "bad", http.StatusBadRequest)
		})
		if _, err := model.StreamRequest(t.Context(), messages(true), ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected stream retrieve API error")
		}
	})
	t.Run("truncated HTTP", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "100")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte("short"))
		})
		if _, err := model.StreamRequest(t.Context(), messages(true), ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected stream retrieve read error")
		}
	})
}

func TestResponsesStreamUsesNativeDeferredToolSearch(t *testing.T) {
	var gotBody map[string]any
	model := newResponsesServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Error(err)
		}
		sseHandler(t, []string{
			`{"type":"response.completed","response":{"model":"gpt-5","status":"completed","usage":{}}}`,
			`[DONE]`,
		})(w, r)
	})
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	_, err := collect(t, model, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{
			Name: ai.ToolSearchName, Schema: schema, ToolKind: ai.ToolPartKindToolSearch,
		}},
		DeferredTools: []ai.ToolDefinition{{Name: "hidden", Schema: schema, DeferLoading: true}},
	})
	if err != nil {
		t.Fatal(err)
	}
	tools := gotBody["tools"].([]any)
	if len(tools) != 2 || tools[0].(map[string]any)["type"] != "function" ||
		tools[0].(map[string]any)["name"] != "hidden" || tools[0].(map[string]any)["defer_loading"] != true ||
		tools[1].(map[string]any)["type"] != "tool_search" ||
		tools[1].(map[string]any)["execution"] != "client" {
		t.Fatalf("stream did not use native deferred search: %+v", tools)
	}
}

func TestResponsesStreamClientToolSearchCall(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"search-item","type":"tool_search_call","call_id":"provisional","execution":"client","arguments":null}}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"id":"search-item","type":"tool_search_call","call_id":"search-final","execution":"client","arguments":{"query":"weather"},"status":"completed"}}`,
		`{"type":"response.completed","response":{"model":"gpt-5","status":"completed","usage":{}}}`,
		`[DONE]`,
	}))
	events, err := collect(t, model, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	start, ok := events[0].(ai.ToolCallStartEvent)
	if !ok || start.ToolName != ai.ToolSearchName || start.ToolKind != ai.ToolPartKindToolSearch ||
		start.ToolCallID != "" || start.PartID != "item:search-item" ||
		start.ProviderDetails["execution"] != "client" {
		t.Fatalf("unexpected client search start: %+v", events[0])
	}
	delta, ok := events[1].(ai.ToolCallDeltaEvent)
	if !ok || delta.PartID != start.PartID || delta.ToolCallID != "search-final" ||
		delta.ArgsDelta != `{"query":"weather"}` {
		t.Fatalf("unexpected client search completion: %+v", events[1])
	}
}

func TestResponsesStreamToolSearchEdgeCases(t *testing.T) {
	for name, events := range map[string][]string{
		"invalid function arguments": {
			`{"type":"response.output_item.added","item":{"id":"call","type":"function_call","call_id":"call","name":"work","arguments":"bad"}}`,
		},
		"server search": {
			`{"type":"response.output_item.added","item":{"id":"search","type":"tool_search_call","execution":"server"}}`,
		},
		"invalid search arguments": {
			`{"type":"response.output_item.added","item":{"id":"search","type":"tool_search_call","execution":"client"}}`,
			`{"type":"response.output_item.done","item":{"id":"search","type":"tool_search_call","execution":"client","arguments":"bad"}}`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			model := newResponsesServer(t, sseHandler(t, events))
			if _, err := collect(t, model, ai.ModelRequestParams{}); err == nil {
				t.Fatal("expected tool-search stream error")
			}
		})
	}

	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.output_item.added","item":{"id":"search","type":"tool_search_call","execution":"client"}}`,
		`{"type":"response.output_item.done","item":{"id":"search","type":"tool_search_call","execution":"client","arguments":{"query":"x"}}}`,
		`{"type":"response.completed","response":{"status":"completed","usage":{}}}`,
	}))
	events, err := collect(t, model, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if delta := events[1].(ai.ToolCallDeltaEvent); delta.ToolCallID != "search" {
		t.Fatalf("item ID was not used as call ID: %+v", delta)
	}
}

func TestResponsesStreamConsumerBreakOnToolSearch(t *testing.T) {
	for name, stopAt := range map[string]string{"start": "start", "delta": "delta"} {
		t.Run(name, func(t *testing.T) {
			model := newResponsesServer(t, sseHandler(t, []string{
				`{"type":"response.output_item.added","item":{"id":"search","type":"tool_search_call","execution":"client"}}`,
				`{"type":"response.output_item.done","item":{"id":"search","type":"tool_search_call","call_id":"search","execution":"client","arguments":{"query":"x"}}}`,
				`{"type":"mystery"}`,
			}))
			stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			for event, err := range stream {
				if err != nil {
					t.Fatal(err)
				}
				if stopAt == "start" {
					if _, ok := event.(ai.ToolCallStartEvent); ok {
						break
					}
				} else if _, ok := event.(ai.ToolCallDeltaEvent); ok {
					break
				}
			}
		})
	}
}

func TestResponsesStreamConsumerBreakOnEncryptedReasoning(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.output_item.added","item":{"id":"reason","type":"reasoning","encrypted_content":"signature"}}`,
		`{"type":"mystery"}`,
	}))
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for event, err := range stream {
		if err != nil {
			t.Fatal(err)
		}
		if event.(ai.ThinkingDeltaEvent).SignatureDelta != "signature" {
			t.Fatalf("unexpected reasoning event: %+v", event)
		}
		break
	}
}

func TestResponsesStreamEndToEnd(t *testing.T) {
	model := newResponsesServer(t, sseHandler(t, []string{
		`{"type":"response.output_text.delta","delta":"hello"}`,
		`{"type":"response.completed","response":{"usage":{"input_tokens":2,"output_tokens":1}}}`,
	}))
	agent := ai.NewAgent[struct{}, string](model)
	stream := agent.RunStream(t.Context(), "go", struct{}{})
	var text string
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		text += normalizedText(event)
	}
	if text != "hello" || stream.Result().Output != "hello" {
		t.Fatalf("unexpected text %q and result %+v", text, stream.Result())
	}
}

func TestResponsesStreamProtocolErrors(t *testing.T) {
	tests := []struct {
		name   string
		chunks []string
		want   string
	}{
		{name: "malformed", chunks: []string{`not json`}, want: "parse Responses stream"},
		{name: "error", chunks: []string{`{"type":"error","error":{"code":"busy","message":"later"}}`}, want: "busy"},
		{name: "failed", chunks: []string{`{"type":"response.failed","response":{"status":"failed","error":{"code":"bad","message":"request"}}}`}, want: "bad: request"},
		{name: "incomplete", chunks: []string{`{"type":"response.incomplete","response":{"status":"incomplete"}}`}, want: "incomplete"},
		{name: "unknown", chunks: []string{`{"type":"mystery"}`}, want: "unknown Responses stream"},
		{name: "missing completed", chunks: []string{`{"type":"response.in_progress"}`, `[DONE]`}, want: "without response.completed"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := newResponsesServer(t, sseHandler(t, test.chunks))
			_, err := collect(t, model, ai.ModelRequestParams{})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("expected %q error, got %v", test.want, err)
			}
		})
	}
}

func TestResponsesStreamRequestErrors(t *testing.T) {
	t.Run("bad payload", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, nil))
		params := ai.ModelRequestParams{Tools: []ai.ToolDefinition{{Name: "bad", Schema: map[string]any{
			"type": "object", "properties": map[string]any{}, "required": []string{}, "bad": make(chan int),
		}}}}
		if _, err := collect(t, model, params); err == nil {
			t.Fatal("expected marshal error")
		}
	})
	t.Run("bad message", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, nil))
		if _, err := model.StreamRequest(t.Context(), []ai.ModelMessage{nil}, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected message error")
		}
	})
	t.Run("native output", func(t *testing.T) {
		model := newResponsesServer(t, sseHandler(t, nil))
		if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{OutputSchema: map[string]any{"type": "object"}}); err == nil {
			t.Fatal("expected native output error")
		}
		prompted := newResponsesServer(t, sseHandler(t, []string{
			`{"type":"response.completed","response":{"model":"gpt-5","status":"completed","usage":{}}}`,
			`[DONE]`,
		}))
		if _, err := collect(t, prompted, ai.ModelRequestParams{
			OutputSchema: map[string]any{"type": "object"}, OutputMode: ai.OutputModePrompted,
		}); err != nil {
			t.Fatalf("prompted output should not request native mode: %v", err)
		}
	})
	t.Run("invalid URL", func(t *testing.T) {
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL("http://[::1"))
		if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected URL error")
		}
	})
	t.Run("transport", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		server.Close()
		model := openai.NewResponsesModel("gpt-5", openai.WithBaseURL(server.URL))
		if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected transport error")
		}
	})
	t.Run("http error", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte("bad"))
		})
		if _, err := collect(t, model, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected API error")
		}
	})
	t.Run("truncated error", func(t *testing.T) {
		model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "100")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte("short"))
		})
		if _, err := collect(t, model, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected read error")
		}
	})
}

func TestResponsesStreamScannerError(t *testing.T) {
	model := newResponsesServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("data: "))
		_, _ = w.Write(make([]byte, 2*1024*1024))
	})
	if _, err := collect(t, model, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected scanner error")
	}
}

func TestResponsesStreamEarlyBreak(t *testing.T) {
	chunks := []string{
		`{"type":"response.reasoning_summary_part.added","part":{"text":""}}`,
		`{"type":"response.reasoning_summary_part.added","part":{"text":"p"}}`,
		`{"type":"response.output_text.delta","delta":"a"}`,
		`{"type":"response.reasoning_summary_text.delta","delta":"b"}`,
		`{"type":"response.output_item.added","item":{"type":"function_call","call_id":"c","name":"work","arguments":"{}"}}`,
		`{"type":"response.function_call_arguments.delta","delta":"{}"}`,
		`{"type":"response.completed","response":{"model":"gpt-5","usage":{}}}`,
	}
	for breakAt := 1; breakAt <= 7; breakAt++ {
		model := newResponsesServer(t, sseHandler(t, chunks))
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
