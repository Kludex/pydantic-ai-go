package mistral_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/mistral"
)

func TestStreamRequest(t *testing.T) {
	var payload map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			t.Error(err)
		}
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(response,
			"event: message\n"+
				"data: "+`{"id":"stream-id","model":"resolved","created":1704067200,"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2}}`+"\n\n"+
				"data: "+`{"choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"reason"}]},{"type":"text","text":"hello "}]}}],"usage":{"prompt_tokens":3,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":2}}}`+"\n\n"+
				"data: "+`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call","type":"function","function":{"name":"tool","arguments":{"x":1}}}]}}]}`+"\n\n"+
				"data: "+`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call","type":"function","function":{"name":"tool","arguments":{"x":1}}}]}}]}`+"\n\n"+
				"data: "+`{"choices":[{"delta":{"content":"world"},"finish_reason":"tool_calls"}]}`+"\n\n"+
				"data: [DONE]\n\n")
	}))
	defer server.Close()
	settings, err := (mistral.Settings{Common: ai.ModelSettings{
		Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled},
	}, PromptCacheKey: "cache"}).Build()
	if err != nil {
		t.Fatal(err)
	}
	model := mistral.NewModel(
		"mistral-medium-latest", mistral.WithBaseURL(server.URL), mistral.WithAPIKey("key"),
		mistral.WithHTTPClient(server.Client()),
	)
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{
		Tools:     []ai.ToolDefinition{{Name: "tool", Schema: map[string]any{"type": "object"}}},
		AllowText: true, Settings: settings,
	})
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
	if len(events) != 6 {
		t.Fatalf("unexpected event count: %d %#v", len(events), events)
	}
	if events[0].(ai.ThinkingDeltaEvent).Delta != "reason" ||
		events[1].(ai.TextDeltaEvent).Delta != "hello " ||
		events[2].(ai.ToolCallStartEvent).ToolCallID != "call" ||
		events[3].(ai.ToolCallDeltaEvent).ArgsDelta != `{"x":1}` ||
		events[4].(ai.TextDeltaEvent).Delta != "world" {
		t.Fatalf("unexpected deltas: %#v", events)
	}
	finish := events[5].(ai.FinishEvent)
	if finish.Usage.Requests != 1 || finish.Usage.InputTokens != 4 || finish.Usage.OutputTokens != 6 ||
		finish.Usage.CacheReadTokens != 2 || finish.FinishReason != ai.FinishReasonToolCall ||
		finish.ProviderResponseID != "stream-id" || finish.ModelName != "resolved" ||
		!finish.Timestamp.Equal(time.Unix(1704067200, 0)) {
		t.Fatalf("unexpected finish: %#v", finish)
	}
	if payload["stream"] != true || payload["reasoning_effort"] != "none" || payload["prompt_cache_key"] != "cache" ||
		payload["tool_choice"] != "auto" || payload["top_p"] != float64(1) {
		t.Fatalf("unexpected stream request: %#v", payload)
	}
}

func TestStreamConsumerStops(t *testing.T) {
	closed := make(chan struct{})
	model := mistral.NewModel("model", mistral.WithProvider(mistral.ProviderConfig{
		Name: "mistral", BaseURL: "https://example.com/v1", HTTPClient: &http.Client{Transport: roundTripFunc(
			func(*http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK, Header: make(http.Header),
					Body: &closingBody{ReadCloser: io.NopCloser(&endlessReader{}), closed: closed},
				}, nil
			},
		)},
	}))
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{AllowText: true})
	if err != nil {
		t.Fatal(err)
	}
	for range stream {
		break
	}
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("stream body was not closed")
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

type closingBody struct {
	io.ReadCloser
	closed chan struct{}
}

func (body *closingBody) Close() error {
	select {
	case <-body.closed:
	default:
		close(body.closed)
	}
	return body.ReadCloser.Close()
}

type endlessReader struct{ sent bool }

func (reader *endlessReader) Read(buffer []byte) (int, error) {
	if reader.sent {
		return 0, io.EOF
	}
	reader.sent = true
	return copy(buffer, "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"), nil
}
