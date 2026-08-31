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

func TestResponsesStreamUsesPortableDeferredToolFallback(t *testing.T) {
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
	if len(tools) != 1 || tools[0].(map[string]any)["type"] != "function" ||
		tools[0].(map[string]any)["name"] != ai.ToolSearchName {
		t.Fatalf("stream did not use local deferred fallback: %+v", tools)
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
