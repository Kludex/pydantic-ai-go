package mistral_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
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
		{
			name: "nameless tool start", stopAtKind: "start",
			body: `data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"","arguments":"{}"}}]}}]}` + "\n\ndata: [DONE]\n\n",
		},
		{
			name: "nameless tool delta", stopAtKind: "delta",
			body: `data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"","arguments":"{}"}}]}}]}` + "\n\ndata: [DONE]\n\n",
		},
		{
			name: "empty arguments", stopAtKind: "delta",
			body: `data: {"choices":[{"delta":{"tool_calls":[{"id":"call","function":{"name":"x","arguments":""}}]}}]}` + "\n\ndata: [DONE]\n\n",
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

func TestStreamAccumulatesToolCallFragments(t *testing.T) {
	type call struct{ name, args, id string }
	chunk := func(deltas ...string) string {
		return `data: {"choices":[{"delta":{"tool_calls":[` + strings.Join(deltas, ",") + `]}}]}` + "\n\n"
	}
	for _, test := range []struct {
		name   string
		chunks []string
		want   []call
	}{
		{name: "fragments name first", chunks: []string{
			chunk(`{"index":0,"id":"c1","function":{"name":"search","arguments":""}}`),
			chunk(`{"index":0,"id":"null","function":{"name":"","arguments":"{\"query\": \"ver"}}`),
			chunk(`{"index":0,"id":"null","function":{"name":"","arguments":"rous\"}"}}`),
		}, want: []call{{"search", `{"query": "verrous"}`, "c1"}}},
		{name: "fragments name later", chunks: []string{
			chunk(`{"index":0,"id":"null","function":{"name":"","arguments":"{\"query\":"}}`),
			chunk(`{"index":0,"id":"c1","function":{"name":"search","arguments":" \"verrous\"}"}}`),
		}, want: []call{{"search", `{"query": "verrous"}`, "c1"}}},
		{name: "interleaved fragments", chunks: []string{
			chunk(`{"index":0,"id":"c1","function":{"name":"search","arguments":"{\"query\": "}}`,
				`{"index":1,"id":"c2","function":{"name":"lookup","arguments":"{\"city\": "}}`),
			chunk(`{"index":0,"id":"null","function":{"name":"","arguments":"\"verrous\"}"}}`,
				`{"index":1,"id":"null","function":{"name":"","arguments":"\"Paris\"}"}}`),
		}, want: []call{{"search", `{"query": "verrous"}`, "c1"}, {"lookup", `{"city": "Paris"}`, "c2"}}},
		{name: "whole call without arguments", chunks: []string{
			chunk(`{"index":0,"id":"c1","function":{"name":"ping","arguments":""}}`),
		}, want: []call{{"ping", `{}`, "c1"}}},
		{name: "nameless call", chunks: []string{
			chunk(`{"index":0,"id":"c1","function":{"name":"","arguments":"{}"}}`),
		}, want: []call{{"", `{}`, "c1"}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
				_, _ = io.WriteString(response, strings.Join(test.chunks, "")+"data: [DONE]\n\n")
			}))
			defer server.Close()
			model := mistral.NewModel("model", mistral.WithBaseURL(server.URL), mistral.WithHTTPClient(server.Client()))
			stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			response, err := ai.CollectModelStream(stream, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			var got []call
			for _, part := range response.Parts {
				toolCall := part.(ai.ToolCallPart)
				got = append(got, call{toolCall.ToolName, string(toolCall.Args), toolCall.ToolCallID})
			}
			if !slices.Equal(got, test.want) {
				t.Fatalf("got %#v, want %#v", got, test.want)
			}
		})
	}
}
