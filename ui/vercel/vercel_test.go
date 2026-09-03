package vercel_test

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
	"github.com/Kludex/pydantic-ai-go/ui/vercel"
)

func TestHandlerStreamsTextAndTools(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	agent.AddRawTool(ai.ToolDefinition{
		Name: "weather", Schema: map[string]any{"type": "object", "properties": map[string]any{}},
	}, func(context.Context, json.RawMessage) (any, error) { return map[string]any{"temperature": 20}, nil })
	adapter := vercel.NewAdapter(agent, vercel.Config{ServerMessageID: "server-message"})
	request := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{
		"trigger":"submit-message","id":"chat-1","messages":[
			{"id":"system","role":"system","parts":[{"type":"text","text":"untrusted"}]},
			{"id":"user","role":"user","parts":[{"type":"text","text":"Weather?"}]}
		]
	}`))
	response := httptest.NewRecorder()
	adapter.Handler(struct{}{}).ServeHTTP(response, request)
	if response.Code != http.StatusOK || response.Header().Get("x-vercel-ai-ui-message-stream") != "v1" {
		t.Fatalf("unexpected response: %d %v", response.Code, response.Header())
	}
	chunks := decodeChunks(t, response.Body.Bytes())
	if chunks[0].Type != vercel.ChunkStart || chunks[0].MessageID != "server-message" ||
		chunks[len(chunks)-1].Type != vercel.ChunkDone {
		t.Fatalf("unexpected lifecycle: %#v", chunks)
	}
	for _, kind := range []vercel.ChunkType{
		vercel.ChunkStartStep, vercel.ChunkToolInputStart, vercel.ChunkToolInputAvailable,
		vercel.ChunkToolOutputAvailable, vercel.ChunkTextStart, vercel.ChunkTextDelta,
		vercel.ChunkFinishStep, vercel.ChunkFinish,
	} {
		if chunkIndex(chunks, kind) < 0 {
			t.Fatalf("missing %s: %#v", kind, chunks)
		}
	}
}

func TestPrepareInput(t *testing.T) {
	prompt, history, report, err := vercel.PrepareInput(vercel.RequestData{
		Trigger: "submit-message", ID: "chat", Messages: []vercel.UIMessage{
			{ID: "system", Role: "system", Parts: []vercel.UIMessagePart{{Type: "text", Text: "untrusted"}}},
			{ID: "assistant", Role: "assistant", Parts: []vercel.UIMessagePart{
				{Type: "text", Text: "calling"}, {Type: "reasoning", Text: "thinking"},
				{Type: "tool-weather", ToolCallID: "call", State: "output-available", Input: []byte(`{}`), Output: []byte(`{"ok":true}`)},
				{Type: "tool-fail", ToolCallID: "fail", State: "output-error", ErrorText: "failed"},
				{Type: "tool-deny", ToolCallID: "deny", State: "output-denied"},
			}},
			{ID: "user", Role: "user", Parts: []vercel.UIMessagePart{{Type: "text", Text: "continue"}}},
		},
	}, ai.MessageSanitizationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if prompt.Content != "continue" || report.StrippedSystemPrompts != 1 || len(history) != 4 {
		t.Fatalf("unexpected prepared input: %#v %#v %#v", prompt, history, report)
	}

	prompt, history, _, err = vercel.PrepareInput(vercel.RequestData{
		Trigger: "regenerate-message", ID: "chat", Messages: []vercel.UIMessage{
			{ID: "user", Role: "user", Parts: []vercel.UIMessagePart{{Type: "text", Text: "first"}, {Type: "text", Text: " second"}}},
			{ID: "assistant", Role: "assistant", Parts: []vercel.UIMessagePart{
				{Type: "text", Text: "answer"},
				{Type: "tool-later", ToolCallID: "later", State: "output-available", Output: []byte(`null`)},
			}},
		},
	}, ai.MessageSanitizationOptions{})
	if err != nil || prompt.Content != "first second" || len(history) != 2 {
		t.Fatalf("unexpected regeneration input: %#v %#v %v", prompt, history, err)
	}
}

func TestInputValidation(t *testing.T) {
	tests := []struct {
		name  string
		input vercel.RequestData
		match string
	}{
		{name: "trigger", input: vercel.RequestData{}, match: "unsupported trigger"},
		{name: "missing user", input: requestWith(vercel.UIMessage{ID: "assistant", Role: "assistant"}), match: "requires a user"},
		{name: "missing ID", input: requestWith(vercel.UIMessage{Role: "user"}), match: "ID must not be empty"},
		{name: "role", input: requestWith(vercel.UIMessage{ID: "one", Role: "tool"}), match: "unsupported message role"},
		{name: "user part", input: requestWith(vercel.UIMessage{ID: "one", Role: "user", Parts: []vercel.UIMessagePart{{Type: "file"}}}), match: "support only text"},
		{name: "tool ID", input: requestWith(vercel.UIMessage{ID: "one", Role: "assistant", Parts: []vercel.UIMessagePart{{Type: "tool-weather"}}}), match: "requires a toolCallId"},
		{name: "tool input", input: requestWith(vercel.UIMessage{ID: "one", Role: "assistant", Parts: []vercel.UIMessagePart{{Type: "tool-weather", ToolCallID: "call", Input: []byte("{")}}}), match: "not valid JSON"},
		{name: "tool output", input: requestWith(vercel.UIMessage{ID: "one", Role: "assistant", Parts: []vercel.UIMessagePart{{Type: "tool-weather", ToolCallID: "call", State: "output-available", Output: []byte("{")}}}), match: "decode tool"},
		{name: "assistant part", input: requestWith(vercel.UIMessage{ID: "one", Role: "assistant", Parts: []vercel.UIMessagePart{{Type: "future"}}}), match: "unsupported assistant part"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, _, err := vercel.PrepareInput(test.input, ai.MessageSanitizationOptions{})
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}

	_, _, _, err := vercel.PrepareInput(requestWith(vercel.UIMessage{
		ID: "user", Role: "user", Parts: []vercel.UIMessagePart{{Type: "text", Text: "hello"}},
	}), ai.MessageSanitizationOptions{AllowedFileURLSchemes: []string{"bad scheme"}})
	if err == nil {
		t.Fatal("expected sanitization error")
	}
}

func requestWith(messages ...vercel.UIMessage) vercel.RequestData {
	return vercel.RequestData{Trigger: "submit-message", ID: "chat", Messages: messages}
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
	adapter := vercel.NewAdapter(agent, vercel.Config{MaxRequestBytes: 4})
	handler := adapter.Handler(struct{}{})
	tests := []struct {
		method string
		body   string
		want   int
	}{
		{method: http.MethodGet, want: http.StatusMethodNotAllowed},
		{method: http.MethodPost, body: "large", want: http.StatusRequestEntityTooLarge},
		{method: http.MethodPost, body: "{", want: http.StatusBadRequest},
	}
	for _, test := range tests {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(test.method, "/", strings.NewReader(test.body)))
		if response.Code != test.want {
			t.Fatalf("method=%s: got %d want %d", test.method, response.Code, test.want)
		}
	}
	request := httptest.NewRequest(http.MethodPost, "/", nil)
	request.Body = failingBody{}
	response := httptest.NewRecorder()
	vercel.NewAdapter(agent, vercel.Config{}).Handler(struct{}{}).ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("unexpected read status: %d", response.Code)
	}
	writer := &failingResponseWriter{header: http.Header{}}
	vercel.NewAdapter(agent, vercel.Config{}).Handler(struct{}{}).ServeHTTP(
		writer, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(
			`{"trigger":"submit-message","id":"chat","messages":[{"id":"user","role":"user","parts":[{"type":"text","text":"hello"}]}]}`,
		)),
	)
	if writer.header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("stream headers missing: %v", writer.header)
	}

	failedAgent := ai.NewAgent[struct{}, string](fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, errors.New("failed")
	}))
	response = httptest.NewRecorder()
	vercel.NewAdapter(failedAgent, vercel.Config{}).Handler(struct{}{}).ServeHTTP(
		response, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(
			`{"trigger":"submit-message","id":"chat","messages":[{"id":"user","role":"user","parts":[{"type":"text","text":"hello"}]}]}`,
		)),
	)
	if !strings.Contains(response.Body.String(), `"type":"error"`) {
		t.Fatalf("handler did not encode model error: %s", response.Body.String())
	}
}

func TestAdapterValidationAndErrors(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	assertPanic(t, func() { vercel.NewAdapter[struct{}, string](nil, vercel.Config{}) })
	assertPanic(t, func() { vercel.NewAdapter(agent, vercel.Config{MaxRequestBytes: -1}) })
	assertPanic(t, func() { vercel.NewAdapter(agent, vercel.Config{SDKVersion: 4}) })
	assertPanic(t, func() { vercel.NewAdapter(agent, vercel.Config{SDKVersion: 8}) })
	for _, version := range []int{5, 6, 7} {
		vercel.NewAdapter(agent, vercel.Config{SDKVersion: version})
	}

	modelErr := errors.New("model failed")
	failed := vercel.NewAdapter(ai.NewAgent[struct{}, string](fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, modelErr
	})), vercel.Config{})
	input := requestWith(vercel.UIMessage{ID: "user", Role: "user", Parts: []vercel.UIMessagePart{{Type: "text", Text: "hello"}}})
	var got error
	for chunk, err := range failed.RunStream(context.Background(), input, struct{}{}) {
		if err != nil {
			got = err
			if chunk.Type != vercel.ChunkError {
				t.Fatalf("unexpected error chunk: %#v", chunk)
			}
		}
	}
	if !errors.Is(got, modelErr) {
		t.Fatalf("unexpected model error: %v", got)
	}

	for chunk, err := range failed.RunStream(context.Background(), vercel.RequestData{}, struct{}{}) {
		if err == nil || chunk.Type != vercel.ChunkError {
			t.Fatalf("unexpected input error: %#v %v", chunk, err)
		}
	}
}

func decodeChunks(t *testing.T, body []byte) []vercel.Chunk {
	t.Helper()
	var chunks []vercel.Chunk
	scanner := bufio.NewScanner(bytes.NewReader(body))
	for scanner.Scan() {
		data, ok := strings.CutPrefix(scanner.Text(), "data: ")
		if !ok {
			continue
		}
		if data == "[DONE]" {
			chunks = append(chunks, vercel.Chunk{Type: vercel.ChunkDone})
			continue
		}
		var chunk vercel.Chunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return chunks
}

func chunkIndex(chunks []vercel.Chunk, kind vercel.ChunkType) int {
	for index, chunk := range chunks {
		if chunk.Type == kind {
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
