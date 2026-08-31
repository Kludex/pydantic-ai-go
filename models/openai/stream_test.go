package openai_test

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func sseHandler(t *testing.T, chunks []string) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, chunk := range chunks {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", chunk)
		}
	}
}

func collect(t *testing.T, model ai.StreamingModel, params ai.ModelRequestParams) ([]ai.ModelStreamEvent, error) {
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

func normalizedText(event ai.StreamEvent) string {
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

func TestStreamTextDeltas(t *testing.T) {
	model := newServer(t, sseHandler(t, []string{
		`{"id":"chat-stream","model":"gpt-5","created":1735689600,"service_tier":"default","system_fingerprint":"fp-1","choices":[{"delta":{"content":"Hel"},"logprobs":{"content":[{"token":"Hel","logprob":-0.1}]}}]}`,
		`{"model":"gpt-5","choices":[{"delta":{"content":"lo"},"finish_reason":"stop","logprobs":{"content":[{"token":"lo","logprob":-0.2}]}}]}`,
		`{"model":"gpt-5","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}`,
		`[DONE]`,
	}))
	events, err := collect(t, model, ai.ModelRequestParams{AllowText: true})
	if err != nil {
		t.Fatal(err)
	}
	var text, textPartID string
	var finish ai.FinishEvent
	for _, e := range events {
		switch ev := e.(type) {
		case ai.TextDeltaEvent:
			text += ev.Delta
			textPartID = ev.PartID
		case ai.FinishEvent:
			finish = ev
		}
	}
	if text != "Hello" || textPartID != "text" {
		t.Fatalf("unexpected text %q with part ID %q", text, textPartID)
	}
	if finish.Usage.InputTokens != 5 || finish.Usage.OutputTokens != 2 || finish.Usage.Requests != 1 {
		t.Fatalf("unexpected usage %+v", finish.Usage)
	}
	if finish.ModelName != "gpt-5" || finish.ProviderName != "openai" || finish.ProviderURL == "" ||
		finish.ProviderResponseID != "chat-stream" || finish.FinishReason != ai.FinishReasonStop ||
		finish.Timestamp.IsZero() || finish.ProviderDetails["service_tier"] != "default" ||
		finish.ProviderDetails["system_fingerprint"] != "fp-1" ||
		finish.ProviderDetails["finish_reason"] != "stop" ||
		len(finish.ProviderDetails["logprobs"].([]map[string]any)) != 2 {
		t.Fatalf("unexpected finish metadata %+v", finish)
	}
}

func TestStreamToolCalls(t *testing.T) {
	model := newServer(t, sseHandler(t, []string{
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"get_weather","arguments":""}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"SF\"}"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c2","function":{"name":"other","arguments":"{}"}}]}}]}`,
		`[DONE]`,
	}))
	events, err := collect(t, model, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var starts []ai.ToolCallStartEvent
	var args string
	for _, e := range events {
		switch ev := e.(type) {
		case ai.ToolCallStartEvent:
			starts = append(starts, ev)
		case ai.ToolCallDeltaEvent:
			if len(starts) == 1 {
				args += ev.ArgsDelta
			}
		}
	}
	if len(starts) != 2 || starts[0].ToolName != "get_weather" || starts[1].ToolCallID != "c2" {
		t.Fatalf("unexpected starts %+v", starts)
	}
	if args != `{"city":"SF"}` {
		t.Fatalf("unexpected args %q", args)
	}
}

func TestStreamInterleavedToolCallDeltas(t *testing.T) {
	model := newServer(t, sseHandler(t, []string{
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"one","arguments":"{\"x\":"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c2","function":{"name":"two","arguments":"{\"y\":"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"2}"}}]}}]}`,
		`[DONE]`,
	}))
	events, err := collect(t, model, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	starts := map[string]int{}
	args := map[string]string{}
	for _, event := range events {
		switch event := event.(type) {
		case ai.ToolCallStartEvent:
			starts[event.PartID]++
		case ai.ToolCallDeltaEvent:
			args[event.PartID] += event.ArgsDelta
		}
	}
	if starts["tool:0"] != 1 || starts["tool:1"] != 1 {
		t.Fatalf("tool part IDs were not stable: %v", starts)
	}
	if args["tool:0"] != `{"x":1}` || args["tool:1"] != `{"y":2}` {
		t.Fatalf("interleaved arguments were not keyed: %v", args)
	}
}

func TestStreamErrors(t *testing.T) {
	t.Run("http error", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte("bad request"))
		})
		if _, err := collect(t, model, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected API error")
		}
	})
	t.Run("malformed chunk", func(t *testing.T) {
		model := newServer(t, sseHandler(t, []string{`not json`}))
		if _, err := collect(t, model, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected parse error")
		}
	})
	t.Run("missing DONE", func(t *testing.T) {
		model := newServer(t, sseHandler(t, []string{`{"choices":[{"delta":{"content":"hi"}}]}`}))
		_, err := collect(t, model, ai.ModelRequestParams{})
		if err == nil || !strings.Contains(err.Error(), "[DONE]") {
			t.Fatalf("expected missing DONE error, got %v", err)
		}
	})
	t.Run("bad payload", func(t *testing.T) {
		model := newServer(t, sseHandler(t, nil))
		params := ai.ModelRequestParams{Tools: []ai.ToolDefinition{{Name: "t", Schema: map[string]any{
			"type": "object", "properties": map[string]any{}, "required": []string{}, "bad": make(chan int),
		}}}}
		if _, err := collect(t, model, params); err == nil {
			t.Fatal("expected marshal error")
		}
	})
}

func TestStreamEndToEnd(t *testing.T) {
	model := newServer(t, sseHandler(t, []string{
		`{"model":"gpt-5","choices":[{"delta":{"content":"Hi "}}]}`,
		`{"model":"gpt-5","choices":[{"delta":{"content":"there"}}]}`,
		`{"model":"gpt-5","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}`,
		`[DONE]`,
	}))
	agent := ai.NewAgent[struct{}, string](model)
	stream := agent.RunStream(t.Context(), "hello", struct{}{})
	var text string
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		text += normalizedText(event)
	}
	if text != "Hi there" || stream.Result().Output != "Hi there" {
		t.Fatalf("text %q, result %+v", text, stream.Result())
	}
}

func TestStreamRequestSetupErrors(t *testing.T) {
	t.Run("bad message", func(t *testing.T) {
		model := newServer(t, sseHandler(t, nil))
		if _, err := model.StreamRequest(t.Context(), []ai.ModelMessage{nil}, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected payload error")
		}
	})
	t.Run("invalid URL", func(t *testing.T) {
		model := openai.NewModel("gpt-5", openai.WithBaseURL("http://[::1"))
		if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected URL error")
		}
	})
	t.Run("transport error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		server.Close()
		model := openai.NewModel("gpt-5", openai.WithBaseURL(server.URL))
		if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected transport error")
		}
	})
	t.Run("truncated error body", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "1000")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte("partial"))
		})
		if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected read error")
		}
	})
}

func TestStreamEarlyBreaks(t *testing.T) {
	chunks := []string{
		`{"choices":[{"delta":{"content":"a"}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"t","arguments":"{}"}}]}}]}`,
		`{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}`,
		`[DONE]`,
	}
	for breakAt := 1; breakAt <= 3; breakAt++ {
		model := newServer(t, sseHandler(t, chunks))
		stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
		if err != nil {
			t.Fatal(err)
		}
		count := 0
		for _, err := range stream {
			if err != nil {
				t.Fatal(err)
			}
			count++
			if count == breakAt {
				break
			}
		}
	}
}

func TestStreamScannerError(t *testing.T) {
	// A single line larger than the scanner's 1MB buffer cap.
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("data: "))
		_, _ = w.Write(make([]byte, 2*1024*1024))
	})
	_, err := collect(t, model, ai.ModelRequestParams{})
	if err == nil {
		t.Fatal("expected scanner error")
	}
}
