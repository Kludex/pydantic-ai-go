package mistral_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/mistral"
)

func TestStreamStopsAtEachEventKind(t *testing.T) {
	tests := []struct {
		name       string
		body       string
		stopAtKind string
	}{
		{
			name: "thinking", stopAtKind: "thinking",
			body: `data: {"choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"reason"}]}]}}]}` + "\n\ndata: [DONE]\n\n",
		},
		{
			name: "tool start", stopAtKind: "start",
			body: `data: {"choices":[{"delta":{"tool_calls":[{"id":"call","function":{"name":"x","arguments":{}}}]}}]}` + "\n\ndata: [DONE]\n\n",
		},
		{
			name: "tool delta", stopAtKind: "delta",
			body: `data: {"choices":[{"delta":{"tool_calls":[{"id":"call","function":{"name":"x","arguments":{}}}]}}]}` + "\n\ndata: [DONE]\n\n",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
				_, _ = io.WriteString(response, test.body)
			}))
			defer server.Close()
			model := mistral.NewModel("model", mistral.WithBaseURL(server.URL), mistral.WithHTTPClient(server.Client()))
			stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			stopped := false
			stream(func(event ai.ModelStreamEvent, eventErr error) bool {
				if eventErr != nil {
					t.Fatal(eventErr)
				}
				kind := ""
				switch event.(type) {
				case ai.ThinkingDeltaEvent:
					kind = "thinking"
				case ai.ToolCallStartEvent:
					kind = "start"
				case ai.ToolCallDeltaEvent:
					kind = "delta"
				}
				if kind == test.stopAtKind {
					stopped = true
					return false
				}
				return true
			})
			if !stopped {
				t.Fatal("target event was not emitted")
			}
		})
	}
}

func TestStreamAppendsFullToolSnapshots(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(response,
			`data: {"choices":[{"delta":{"tool_calls":[{"id":"call","function":{"name":"x","arguments":"{"}}]}}]}`+"\n\n"+
				`data: {"choices":[{"delta":{"tool_calls":[{"id":"call","function":{"name":"x","arguments":"{}"}}]}}]}`+"\n\n"+
				"data: [DONE]\n\n")
	}))
	defer server.Close()
	model := mistral.NewModel("model", mistral.WithBaseURL(server.URL), mistral.WithHTTPClient(server.Client()))
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var deltas string
	for event, eventErr := range stream {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		if delta, ok := event.(ai.ToolCallDeltaEvent); ok {
			deltas += delta.ArgsDelta
		}
	}
	if deltas != "{}" {
		t.Fatalf("unexpected arguments: %q", deltas)
	}
}
