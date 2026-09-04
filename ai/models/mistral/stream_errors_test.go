package mistral_test

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/mistral"
)

func TestStreamSetupFailures(t *testing.T) {
	buildModel, _ := noRequestModel(t)
	if _, err := buildModel.StreamRequest(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
	}); err == nil {
		t.Fatal("expected build error")
	}
	if _, err := buildModel.StreamRequest(t.Context(), nil, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{Name: "bad", Schema: map[string]any{"value": make(chan int)}}},
	}); err == nil {
		t.Fatal("expected marshal error")
	}
	model := mistral.NewModel("model", mistral.WithBaseURL("://bad"))
	if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected URL error")
	}
	model = mistral.NewModel("model", mistral.WithProvider(mistral.ProviderConfig{
		Name: "mistral", BaseURL: "https://example.com/v1", PrepareRequest: func(*http.Request) error {
			return errors.New("prepare")
		},
	}))
	if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected preparation error")
	}
	model = mistral.NewModel("model", mistral.WithProvider(mistral.ProviderConfig{
		Name: "mistral", BaseURL: "https://example.com/v1", HTTPClient: &http.Client{Transport: roundTripFunc(
			func(*http.Request) (*http.Response, error) { return nil, errors.New("network") },
		)},
	}))
	if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected transport error")
	}
	for _, test := range []struct {
		name string
		body io.ReadCloser
	}{
		{name: "status", body: io.NopCloser(strings.NewReader("denied"))},
		{name: "status read", body: failingBody{}},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := mistral.NewModel("model", mistral.WithProvider(mistral.ProviderConfig{
				Name: "mistral", BaseURL: "https://example.com/v1", HTTPClient: &http.Client{Transport: roundTripFunc(
					func(*http.Request) (*http.Response, error) {
						return &http.Response{StatusCode: http.StatusBadRequest, Header: make(http.Header), Body: test.body}, nil
					},
				)},
			}))
			if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
				t.Fatal("expected status error")
			}
		})
	}
}

func TestStreamProtocolFailures(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{name: "malformed chunk", body: "data: {\n\n"},
		{name: "malformed content", body: `data: {"choices":[{"delta":{"content":{}}}]}` + "\n\n"},
		{name: "tool type", body: `data: {"choices":[{"delta":{"tool_calls":[{"type":"custom","function":{"name":"x","arguments":{}}}]}}]}` + "\n\n"},
		{name: "replaced arguments", body: strings.Join([]string{
			`data: {"choices":[{"delta":{"tool_calls":[{"index":2,"id":"call","function":{"name":"x","arguments":{"a":1}}}]}}]}`,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":2,"id":"call","function":{"name":"x","arguments":{"b":2}}}]}}]}`,
		}, "\n\n") + "\n\n"},
		{name: "missing done", body: `data: {"choices":[]}` + "\n\n"},
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
			var streamErr error
			for _, eventErr := range stream {
				if eventErr != nil {
					streamErr = eventErr
				}
			}
			if streamErr == nil {
				t.Fatal("expected stream error")
			}
		})
	}
}

func TestStreamGeneratedToolIdentityAndNoTimestamp(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(response,
			"not-data\n"+
				`data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"x","arguments":"{\"a\":1}"}}]}}]}`+"\n\n"+
				"data: [DONE]\n\n")
	}))
	defer server.Close()
	model := mistral.NewModel("model", mistral.WithBaseURL(server.URL), mistral.WithHTTPClient(server.Client()))
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var events []ai.ModelStreamEvent
	for event, eventErr := range stream {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		events = append(events, event)
	}
	if len(events) != 3 || events[0].(ai.ToolCallStartEvent).ToolCallID == "" ||
		events[1].(ai.ToolCallDeltaEvent).ArgsDelta != `{"a":1}` {
		t.Fatalf("unexpected events: %#v", events)
	}
	finish := events[2].(ai.FinishEvent)
	if _, exists := finish.ProviderDetails["timestamp"]; exists {
		t.Fatalf("unexpected provider timestamp: %#v", finish.ProviderDetails)
	}
}

func TestStreamReadFailure(t *testing.T) {
	model := mistral.NewModel("model", mistral.WithProvider(mistral.ProviderConfig{
		Name: "mistral", BaseURL: "https://example.com/v1", HTTPClient: &http.Client{Transport: roundTripFunc(
			func(*http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK, Header: make(http.Header), Body: &scanFailBody{},
				}, nil
			},
		)},
	}))
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var streamErr error
	for _, err := range stream {
		streamErr = err
	}
	if streamErr == nil {
		t.Fatal("expected stream read error")
	}
}

type scanFailBody struct{ sent bool }

func (body *scanFailBody) Read(buffer []byte) (int, error) {
	if !body.sent {
		body.sent = true
		return copy(buffer, "data: {\"choices\":[]}"), nil
	}
	return 0, errors.New("scan")
}

func (*scanFailBody) Close() error { return nil }
