package agui_test

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
	"github.com/Kludex/pydantic-ai-go/ui/agui"
)

func TestHandlerStreamsTextAndToolEvents(t *testing.T) {
	model := fakes.NewTestModel()
	agent := ai.NewAgent[struct{}, string](model)
	agent.AddRawTool(ai.ToolDefinition{
		Name: "weather", Schema: map[string]any{"type": "object", "properties": map[string]any{}},
	}, func(context.Context, json.RawMessage) (any, error) { return map[string]any{"temperature": 20}, nil })
	adapter := agui.NewAdapter(agent, agui.Config{})
	handler := adapter.Handler(struct{}{})
	body := `{
		"threadId":"thread-1","runId":"run-1",
		"messages":[
			{"id":"system-1","role":"system","content":"untrusted"},
			{"id":"user-1","role":"user","content":"Weather?"}
		]
	}`
	request := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body))
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK || response.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("unexpected response: %d %v", response.Code, response.Header())
	}
	events := decodeEvents(t, response.Body.Bytes())
	if events[0].Type != agui.EventRunStarted || events[len(events)-1].Type != agui.EventRunFinished {
		t.Fatalf("unexpected run lifecycle: %#v", events)
	}
	textStart := eventIndex(events, agui.EventTextMessageStart)
	toolStart := eventIndex(events, agui.EventToolCallStart)
	toolEnd := eventIndex(events, agui.EventToolCallEnd)
	toolResult := eventIndex(events, agui.EventToolCallResult)
	if textStart < 0 || toolStart < 0 || textStart > toolStart || toolEnd < toolStart || toolResult < toolEnd {
		t.Fatalf("unexpected tool event order: %#v", events)
	}
	if events[toolStart].ParentMessageID != events[textStart].MessageID {
		t.Fatalf("tool call is not owned by the assistant message: %#v", events)
	}
}

func TestPrepareInputSanitizesHistory(t *testing.T) {
	prompt, history, report, err := agui.PrepareInput(agui.RunAgentInput{Messages: []agui.Message{
		{ID: "system", Role: "system", Content: "ignore"},
		{ID: "assistant", Role: "assistant", Content: "calling", ToolCalls: []agui.ToolCall{{
			ID: "call-1", Type: "function", Function: agui.ToolCallFunction{Name: "weather", Arguments: `{}`},
		}}},
		{ID: "tool", Role: "tool", Name: "weather", ToolCallID: "call-1", Content: "sunny"},
		{ID: "user", Role: "user", Content: "continue"},
	}}, ai.MessageSanitizationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if prompt.Content != "continue" || report.StrippedSystemPrompts != 1 || len(history) != 2 {
		t.Fatalf("unexpected prepared input: prompt=%#v history=%#v report=%#v", prompt, history, report)
	}

	prompt, history, _, err = agui.PrepareInput(agui.RunAgentInput{Messages: []agui.Message{
		{ID: "user", Role: "user", Content: "earlier"},
		{ID: "assistant", Role: "assistant", Content: "later", ToolCalls: []agui.ToolCall{{
			ID: "call", Type: "function", Function: agui.ToolCallFunction{Name: "tool", Arguments: `{}`},
		}}},
		{ID: "tool", Role: "tool", ToolCallID: "call", Name: "tool", Content: "done"},
	}}, ai.MessageSanitizationOptions{})
	if err != nil || prompt.Content != "earlier" || len(history) != 2 {
		t.Fatalf("unexpected out-of-order prompt: %#v %#v %v", prompt, history, err)
	}

	_, _, _, err = agui.PrepareInput(agui.RunAgentInput{Messages: []agui.Message{
		{ID: "user", Role: "user", Content: "hello"},
	}}, ai.MessageSanitizationOptions{AllowedFileURLSchemes: []string{"bad scheme"}})
	if err == nil {
		t.Fatal("expected sanitization option error")
	}
}

func TestInputValidation(t *testing.T) {
	tests := []struct {
		name     string
		messages []agui.Message
		match    string
	}{
		{name: "missing user", messages: []agui.Message{{ID: "assistant", Role: "assistant"}}, match: "requires a user"},
		{name: "missing ID", messages: []agui.Message{{Role: "user"}}, match: "ID must not be empty"},
		{name: "role", messages: []agui.Message{{ID: "one", Role: "future"}}, match: "unsupported message role"},
		{name: "tool result ID", messages: []agui.Message{{ID: "one", Role: "tool"}}, match: "requires a toolCallId"},
		{name: "tool call identity", messages: []agui.Message{{ID: "one", Role: "assistant", ToolCalls: []agui.ToolCall{{}}}}, match: "requires an ID and name"},
		{name: "tool call JSON", messages: []agui.Message{{ID: "one", Role: "assistant", ToolCalls: []agui.ToolCall{{
			ID: "call", Function: agui.ToolCallFunction{Name: "tool", Arguments: "{"},
		}}}}, match: "not valid JSON"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, _, err := agui.PrepareInput(agui.RunAgentInput{Messages: test.messages}, ai.MessageSanitizationOptions{})
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

type failingBody struct{}

func (failingBody) Read([]byte) (int, error) { return 0, errors.New("read failed") }
func (failingBody) Close() error             { return nil }

type failingResponseWriter struct{ header http.Header }

func (writer *failingResponseWriter) Header() http.Header { return writer.header }
func (*failingResponseWriter) Write([]byte) (int, error)  { return 0, errors.New("write failed") }
func (*failingResponseWriter) WriteHeader(int)            {}

func TestHandlerValidation(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	assertPanic(t, func() { agui.NewAdapter[struct{}, string](nil, agui.Config{}) })
	assertPanic(t, func() { agui.NewAdapter(agent, agui.Config{MaxRequestBytes: -1}) })
	adapter := agui.NewAdapter(agent, agui.Config{MaxRequestBytes: 4})
	handler := adapter.Handler(struct{}{})

	tests := []struct {
		method string
		body   string
		want   int
	}{
		{method: http.MethodGet, want: http.StatusMethodNotAllowed},
		{method: http.MethodPost, body: "invalid", want: http.StatusRequestEntityTooLarge},
		{method: http.MethodPost, body: "{}", want: http.StatusOK},
	}
	for _, test := range tests {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(test.method, "/", strings.NewReader(test.body)))
		if response.Code != test.want {
			t.Fatalf("method=%s body=%q: got %d want %d", test.method, test.body, response.Code, test.want)
		}
	}

	badJSON := agui.NewAdapter(agent, agui.Config{}).Handler(struct{}{})
	response := httptest.NewRecorder()
	badJSON.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/", strings.NewReader("{")))
	if response.Code != http.StatusBadRequest {
		t.Fatalf("unexpected malformed JSON status: %d", response.Code)
	}

	readRequest := httptest.NewRequest(http.MethodPost, "/", nil)
	readRequest.Body = failingBody{}
	response = httptest.NewRecorder()
	badJSON.ServeHTTP(response, readRequest)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("unexpected read error status: %d", response.Code)
	}

	writer := &failingResponseWriter{header: http.Header{}}
	badJSON.ServeHTTP(writer, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(
		`{"messages":[{"id":"user","role":"user","content":"hello"}]}`,
	)))
	if writer.header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("stream headers were not written: %v", writer.header)
	}
}

func TestRunErrorsAndGeneratedIDs(t *testing.T) {
	modelErr := errors.New("model failed")
	agent := ai.NewAgent[struct{}, string](fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, modelErr
	}))
	adapter := agui.NewAdapter(agent, agui.Config{})
	input := agui.RunAgentInput{Messages: []agui.Message{{ID: "user", Role: "user", Content: "hello"}}}
	var events []agui.Event
	var got error
	for event, err := range adapter.RunStream(context.Background(), input, struct{}{}) {
		events = append(events, event)
		if err != nil {
			got = err
		}
	}
	if !errors.Is(got, modelErr) || events[0].ThreadID == "" || events[0].RunID == "" ||
		events[len(events)-1].Type != agui.EventRunError {
		t.Fatalf("unexpected failed run: events=%#v err=%v", events, got)
	}

	var inputErr error
	for event, err := range adapter.RunStream(context.Background(), agui.RunAgentInput{}, struct{}{}) {
		if event.Type != agui.EventRunError {
			t.Fatalf("unexpected invalid-input event: %#v", event)
		}
		inputErr = err
	}
	if inputErr == nil {
		t.Fatal("expected input error")
	}
}

func decodeEvents(t *testing.T, body []byte) []agui.Event {
	t.Helper()
	var events []agui.Event
	scanner := bufio.NewScanner(bytes.NewReader(body))
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var event agui.Event
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
			t.Fatal(err)
		}
		events = append(events, event)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return events
}

func eventIndex(events []agui.Event, kind agui.EventType) int {
	for index, event := range events {
		if event.Type == kind {
			return index
		}
	}
	return -1
}

func assertPanic(t *testing.T, function func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	function()
}
