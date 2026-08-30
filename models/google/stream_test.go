package google_test

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/google"
)

func googleSSE(t *testing.T, chunks []string) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, chunk := range chunks {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", chunk)
		}
	}
}

func collectGoogleStream(
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

func normalizedGoogleText(event ai.StreamEvent) string {
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
	var path, query, accept string
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		path, query, accept = r.URL.Path, r.URL.RawQuery, r.Header.Get("Accept")
		googleSSE(t, []string{
			`{"modelVersion":"gemini-stream","candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"thought":true,"text":"plan"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"functionCall":{"id":"c1","name":"work","args":{"x":1}}}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"lo"}]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3}}`,
		})(w, r)
	})
	events, err := collectGoogleStream(t, model, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(path, ":streamGenerateContent") || query != "alt=sse" || accept != "text/event-stream" {
		t.Fatalf("unexpected request path=%q query=%q accept=%q", path, query, accept)
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
	if text != "Hello" || thinking != "plan" || start.ToolName != "work" || start.ToolCallID != "c1" || args != `{"x":1}` {
		t.Fatalf("unexpected events text=%q thinking=%q start=%+v args=%q", text, thinking, start, args)
	}
	if textPartID != "text:0" || thinkingPartID != "thinking:0" || start.PartID != "tool:0" || argsPartID != "tool:0" {
		t.Fatalf("unstable Gemini part IDs: text=%q thinking=%q start=%q args=%q", textPartID, thinkingPartID, start.PartID, argsPartID)
	}
	if finish.ModelName != "gemini-stream" || finish.Usage.Requests != 1 || finish.Usage.InputTokens != 5 || finish.Usage.OutputTokens != 3 {
		t.Fatalf("unexpected finish %+v", finish)
	}
}

func TestStreamEndToEnd(t *testing.T) {
	model := newServer(t, googleSSE(t, []string{
		`{"modelVersion":"gemini","candidates":[{"content":{"parts":[{"text":"hello"}]}}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1}}`,
	}))
	agent := ai.NewAgent[struct{}, string](model)
	stream := agent.RunStream(t.Context(), "go", struct{}{})
	var text string
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		text += normalizedGoogleText(event)
	}
	if text != "hello" || stream.Result().Output != "hello" {
		t.Fatalf("unexpected stream text %q and result %+v", text, stream.Result())
	}
}

func TestStreamProtocolErrors(t *testing.T) {
	tests := []struct {
		name   string
		chunks []string
		want   string
	}{
		{name: "malformed", chunks: []string{`not json`}, want: "parse stream chunk"},
		{name: "binary", chunks: []string{`{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"aGk="}}]}}]}`}, want: "binary output"},
		{name: "empty", want: "without a response"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := newServer(t, googleSSE(t, test.chunks))
			_, err := collectGoogleStream(t, model, ai.ModelRequestParams{})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("expected %q error, got %v", test.want, err)
			}
		})
	}
}

func TestStreamRequestErrors(t *testing.T) {
	t.Run("bad payload", func(t *testing.T) {
		model := newServer(t, googleSSE(t, nil))
		params := ai.ModelRequestParams{Tools: []ai.ToolDefinition{{Name: "bad", Schema: map[string]any{"x": make(chan int)}}}}
		if _, err := collectGoogleStream(t, model, params); err == nil {
			t.Fatal("expected marshal error")
		}
	})
	t.Run("bad message", func(t *testing.T) {
		model := newServer(t, googleSSE(t, nil))
		if _, err := model.StreamRequest(t.Context(), []ai.ModelMessage{nil}, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected message error")
		}
	})
	t.Run("invalid URL", func(t *testing.T) {
		model := google.NewModel("gemini", google.WithBaseURL("http://[::1"))
		if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected URL error")
		}
	})
	t.Run("transport", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		server.Close()
		model := google.NewModel("gemini", google.WithBaseURL(server.URL))
		if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected transport error")
		}
	})
	t.Run("http error", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte("bad"))
		})
		if _, err := collectGoogleStream(t, model, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected API error")
		}
	})
	t.Run("truncated error", func(t *testing.T) {
		model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Length", "100")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte("short"))
		})
		if _, err := collectGoogleStream(t, model, ai.ModelRequestParams{}); err == nil {
			t.Fatal("expected read error")
		}
	})
}

func TestStreamScannerError(t *testing.T) {
	model := newServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("data: "))
		_, _ = w.Write(make([]byte, 2*1024*1024))
	})
	if _, err := collectGoogleStream(t, model, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected scanner error")
	}
}

func TestStreamEarlyBreak(t *testing.T) {
	chunks := []string{
		`{"candidates":[]}`,
		`{"candidates":[{"content":{"parts":[{}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"thought":true,"text":"a"}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"text":"b"}]}}]}`,
		`{"candidates":[{"content":{"parts":[{"functionCall":{"id":"c","name":"work","args":{"x":1}}}]}}]}`,
	}
	for breakAt := 1; breakAt <= 4; breakAt++ {
		model := newServer(t, googleSSE(t, chunks))
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
