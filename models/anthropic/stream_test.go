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
) ([]ai.StreamEvent, error) {
	t.Helper()
	stream, err := model.StreamRequest(t.Context(), nil, params)
	if err != nil {
		return nil, err
	}
	var events []ai.StreamEvent
	for event, err := range stream {
		if err != nil {
			return events, err
		}
		events = append(events, event)
	}
	return events, nil
}

func TestStreamEvents(t *testing.T) {
	var gotStream bool
	var gotAccept string
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotAccept = r.Header.Get("Accept")
		var body struct {
			Stream bool `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		gotStream = body.Stream
		anthropicSSE(t, []string{
			`{"type":"message_start","message":{"model":"claude-stream","usage":{"input_tokens":5,"output_tokens":1}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"H"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"i"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"content_block_start","index":1,"content_block":{"type":"thinking","thinking":"A"}}`,
			`{"type":"content_block_delta","index":1,"delta":{"type":"thinking_delta","thinking":"B"}}`,
			`{"type":"content_block_delta","index":1,"delta":{"type":"signature_delta","signature":"ignored"}}`,
			`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"c1","name":"work","input":{}}}`,
			`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"x\":"}}`,
			`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"1}"}}`,
			`{"type":"ping"}`,
			`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":8}}`,
			`{"type":"message_stop"}`,
		})(w, r)
	})
	events, err := collectAnthropicStream(t, model, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if !gotStream || gotAccept != "text/event-stream" {
		t.Fatalf("stream request not configured: stream=%v accept=%q", gotStream, gotAccept)
	}
	var text, thinking, args string
	var textPartID, thinkingPartID, argsPartID string
	var start ai.ToolCallStartEvent
	var finish ai.FinishEvent
	for _, event := range events {
		switch event := event.(type) {
		case ai.TextDeltaEvent:
			text += event.Delta
			textPartID = event.PartID
		case ai.ThinkingDeltaEvent:
			thinking += event.Delta
			thinkingPartID = event.PartID
		case ai.ToolCallStartEvent:
			start = event
		case ai.ToolCallDeltaEvent:
			args += event.ArgsDelta
			argsPartID = event.PartID
		case ai.FinishEvent:
			finish = event
		}
	}
	if text != "Hi" || thinking != "AB" || start.ToolName != "work" || start.ToolCallID != "c1" || args != `{"x":1}` {
		t.Fatalf("unexpected events: text=%q thinking=%q start=%+v args=%q", text, thinking, start, args)
	}
	if textPartID != "0" || thinkingPartID != "1" || start.PartID != "2" || argsPartID != "2" {
		t.Fatalf("unstable Anthropic part IDs: text=%q thinking=%q start=%q args=%q", textPartID, thinkingPartID, start.PartID, argsPartID)
	}
	if finish.ModelName != "claude-stream" || finish.Usage.Requests != 1 || finish.Usage.InputTokens != 5 || finish.Usage.OutputTokens != 8 {
		t.Fatalf("unexpected finish %+v", finish)
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
		if delta, ok := event.(ai.TextDeltaEvent); ok {
			text += delta.Delta
		}
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
