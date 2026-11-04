package cohere_test

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/cohere"
)

// Verify Cohere implements streaming.
var _ ai.StreamingModel = (*cohere.Model)(nil)

func TestStreamContentAndThinking(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(response, strings.Join([]string{
			`data: {"type":"message-start","message":{"id":"cohere-1"}}`,
			`data: {"type":"content-delta","content_block":{"type":"text"},"delta":{"message":{"content":{"text":"hello "}}}}`,
			`data: {"type":"content-delta","content_block":{"type":"text"},"delta":{"message":{"content":{"text":"world"}}}}`,
			`data: {"type":"content-delta","content_block":{"type":"thinking"},"delta":{"message":{"content":{"thinking":"thinking..."}}}}`,
			`data: {"type":"message-end","message":{"finish_reason":"COMPLETE","usage":{"billed_units":{"input_tokens":1,"output_tokens":2},"tokens":{"input_tokens":10,"output_tokens":20},"cached_tokens":0}}}`,
			"data: [DONE]",
			"",
		}, "\n"))
	}))
	defer server.Close()
	model := cohere.NewModel("command-r7b-12-2024",
		cohere.WithBaseURL(server.URL),
		cohere.WithHTTPClient(server.Client()),
		cohere.WithAPIKey("key"),
	)
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	events := collectEvents(t, stream)
	if len(events) != 4 {
		t.Fatalf("expected 4 events, got %d: %#v", len(events), events)
	}
	if text, ok := events[0].(ai.TextDeltaEvent); !ok || text.Delta != "hello " {
		t.Fatalf("unexpected first event: %#v", events[0])
	}
	if text, ok := events[1].(ai.TextDeltaEvent); !ok || text.Delta != "world" {
		t.Fatalf("unexpected second event: %#v", events[1])
	}
	if thinking, ok := events[2].(ai.ThinkingDeltaEvent); !ok || thinking.Delta != "thinking..." {
		t.Fatalf("unexpected third event: %#v", events[2])
	}
	finish, ok := events[3].(ai.FinishEvent)
	if !ok {
		t.Fatalf("expected finish event: %#v", events[3])
	}
	if finish.Usage.OutputTokens != 20 || finish.Usage.InputTokens != 10 || finish.FinishReason != ai.FinishReasonStop {
		t.Fatalf("unexpected finish: %+v", finish)
	}
	if finish.ProviderResponseID != "cohere-1" {
		t.Fatalf("expected response id: %+v", finish)
	}
}

func TestStreamToolCallStartAndDelta(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		body := strings.Join([]string{
			`data: {"type":"tool-call-start","tool_call":{"index":1,"id":"call-cohere","delta":{"message":{"tool_calls":{"function":{"name":"lookup"}}}}}}`,
			`data: {"type":"tool-call-delta","delta":{"message":{"tool_calls":{"index":1,"id":"call-cohere","function":{"arguments":"{\"a\":1}"}}}}}`,
			`data: {"type":"message-end","message":{"finish_reason":"TOOL_CALL"}}`,
			"data: [DONE]",
			"",
		}, "\n")
		_, _ = io.WriteString(response, body)
	}))
	defer server.Close()
	model := cohere.NewModel("command-r7b-12-2024",
		cohere.WithBaseURL(server.URL),
		cohere.WithHTTPClient(server.Client()),
		cohere.WithAPIKey("key"),
	)
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	events := collectEvents(t, stream)
	if len(events) < 2 {
		t.Fatalf("expected at least 2 events, got %d", len(events))
	}
	start, ok := events[0].(ai.ToolCallStartEvent)
	if !ok {
		t.Fatalf("expected tool-call-start, got %T", events[0])
	}
	if start.ToolName != "lookup" || start.ToolCallID != "call-cohere" || start.PartID != "tool:1" {
		t.Fatalf("unexpected start %+v", start)
	}
	if len(events) >= 2 {
		if delta, ok := events[1].(ai.ToolCallDeltaEvent); !ok || delta.ArgsDelta != `{"a":1}` {
			t.Fatalf("unexpected tool delta: %#v", events[1])
		}
	}
}

func TestStreamToolCallContinuation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(response, strings.Join([]string{
			`data: {"type":"tool-call-start","tool_call":{"index":0,"id":"call-1","delta":{"message":{"tool_calls":{"function":{"name":"lookup","arguments":"{\"a\":"}}}}}}`,
			`data: {"type":"tool-call-start","tool_call":{"index":0,"id":"call-1","delta":{"message":{"tool_calls":{"function":{"arguments":"1}"}}}}}}`,
			`data: {"type":"message-end","message":{"finish_reason":"TOOL_CALL"}}`,
			"data: [DONE]",
			"",
		}, "\n"))
	}))
	defer server.Close()
	model := cohere.NewModel("command",
		cohere.WithBaseURL(server.URL), cohere.WithHTTPClient(server.Client()),
		cohere.WithAPIKey("key"),
	)
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	events := collectEvents(t, stream)
	if len(events) == 0 {
		t.Fatal("expected events")
	}
	if start, ok := events[0].(ai.ToolCallStartEvent); !ok || start.ToolName != "lookup" {
		t.Fatalf("unexpected start: %#v", events[0])
	}
}

func TestStreamToolCallDeltaNewTool(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(response, strings.Join([]string{
			`data: {"type":"tool-call-delta","delta":{"message":{"tool_calls":{"index":2,"id":"call-direct","function":{"name":"lookup","arguments":"{\"a\":1}"}}}}}`,
			`data: {"type":"message-end","message":{"finish_reason":"TOOL_CALL"}}`,
			"data: [DONE]",
			"",
		}, "\n"))
	}))
	defer server.Close()
	model := cohere.NewModel("command",
		cohere.WithBaseURL(server.URL), cohere.WithHTTPClient(server.Client()),
		cohere.WithAPIKey("key"),
	)
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	events := collectEvents(t, stream)
	if len(events) == 0 {
		t.Fatal("expected events")
	}
	if start, ok := events[0].(ai.ToolCallStartEvent); !ok || start.ToolName != "lookup" || start.ToolCallID != "call-direct" {
		t.Fatalf("unexpected start: %#v", events[0])
	}
}

func TestStreamSyntheticToolIdentity(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(response, strings.Join([]string{
			`data: {"type":"tool-call-start","tool_call":{"index":0,"delta":{"message":{"tool_calls":{"function":{"name":"lookup","arguments":"{"}}}}}}`,
		}, "\n"))
	}))
	defer server.Close()
	model := cohere.NewModel("command-r7b-12-2024",
		cohere.WithBaseURL(server.URL),
		cohere.WithHTTPClient(server.Client()),
		cohere.WithAPIKey("key"),
	)
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	events := collectEvents(t, stream)
	if len(events) == 0 {
		t.Fatal("expected events")
	}
	start, ok := events[0].(ai.ToolCallStartEvent)
	if !ok {
		t.Fatalf("expected tool-call-start, got %T", events[0])
	}
	if start.ToolCallID == "" || !strings.HasPrefix(start.ToolCallID, "cohere-") {
		t.Fatalf("unexpected synthetic tool id %q", start.ToolCallID)
	}
}

func TestStreamEmptyDataAndErrorEvent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(response, strings.Join([]string{
			"data: ",
			`data: {"type":"message-start"}`,
			`data: {"type":"error","error":{"message":"upstream broke"}}`,
		}, "\n"))
	}))
	defer server.Close()
	model := cohere.NewModel("command",
		cohere.WithBaseURL(server.URL), cohere.WithHTTPClient(server.Client()),
		cohere.WithAPIKey("key"),
	)
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var streamErr error
	for _, err := range stream {
		if err != nil {
			streamErr = err
		}
	}
	if streamErr == nil || !strings.Contains(streamErr.Error(), "upstream broke") {
		t.Fatalf("expected stream error, got: %v", streamErr)
	}
}

func TestStreamUnknownEventType(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(response, strings.Join([]string{
			`data: {"type":"unknown-event","payload":"ignored"}`,
			`data: {"type":"message-end","message":{"finish_reason":"COMPLETE"}}`,
			"data: [DONE]",
			"",
		}, "\n"))
	}))
	defer server.Close()
	model := cohere.NewModel("command",
		cohere.WithBaseURL(server.URL), cohere.WithHTTPClient(server.Client()),
		cohere.WithAPIKey("key"),
	)
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	events := collectEvents(t, stream)
	if len(events) != 1 {
		t.Fatalf("expected 1 event, got %d: %#v", len(events), events)
	}
	if finish, ok := events[0].(ai.FinishEvent); !ok || finish.FinishReason != ai.FinishReasonStop {
		t.Fatalf("expected single FinishEvent with stop reason: %#v", events[0])
	}
}

func TestStreamProtocolFailures(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{name: "malformed chunk", body: "data: {\n\n"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
				_, _ = io.WriteString(response, test.body)
			}))
			defer server.Close()
			model := cohere.NewModel("command", cohere.WithBaseURL(server.URL), cohere.WithHTTPClient(server.Client()))
			stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			var streamErr error
			for _, err := range stream {
				if err != nil {
					streamErr = err
				}
			}
			if streamErr == nil {
				t.Fatal("expected stream error")
			}
		})
	}
}

func TestStreamSetupFailures(t *testing.T) {
	model := cohere.NewModel("command", cohere.WithBaseURL("://bad"), cohere.WithAPIKey("key"))
	if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected URL error")
	}
	model = cohere.NewModel("command",
		cohere.WithProvider(cohere.ProviderConfig{
			Name: "cohere", BaseURL: "https://example.com/v2",
			PrepareRequest: func(*http.Request) error { return errors.New("prepare") },
		}),
	)
	if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil ||
		!strings.Contains(err.Error(), "prepare") {
		t.Fatalf("expected prepare error, got: %v", err)
	}
	model = cohere.NewModel("command", cohere.WithProvider(cohere.ProviderConfig{
		Name: "cohere", BaseURL: "https://example.com/v2",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return nil, errors.New("network")
		})},
	}))
	if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected transport error")
	}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(response, "denied")
	}))
	defer server.Close()
	model = cohere.NewModel("command", cohere.WithBaseURL(server.URL), cohere.WithHTTPClient(server.Client()))
	if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected status error")
	}
}

func collectEvents(t *testing.T, stream func(yield func(ai.ModelStreamEvent, error) bool)) []ai.ModelStreamEvent {
	t.Helper()
	var events []ai.ModelStreamEvent
	var streamErr error
	for event, err := range stream {
		if err != nil {
			streamErr = err
			continue
		}
		events = append(events, event)
	}
	if streamErr != nil {
		t.Fatalf("unexpected stream error: %v", streamErr)
	}
	return events
}
