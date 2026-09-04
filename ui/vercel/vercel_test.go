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

func TestPrepareInputFiles(t *testing.T) {
	prompt, history, _, err := vercel.PrepareInput(vercel.RequestData{
		Trigger: "submit-message", ID: "chat", Messages: []vercel.UIMessage{
			{ID: "assistant", Role: "assistant", Parts: []vercel.UIMessagePart{{
				Type: "file", URL: "data:image/png;base64,b2xk", MediaType: "image/png",
			}}},
			{ID: "user", Role: "user", Parts: []vercel.UIMessagePart{
				{Type: "text", Text: "Describe these files."},
				{Type: "file", URL: "data:application/pdf;base64,cGRm", MediaType: "application/pdf"},
				{Type: "file", URL: "https://example.com/image.png", MediaType: "image/png"},
				{Type: "file", URL: "https://example.com/video.mp4", MediaType: "video/mp4"},
				{Type: "file", URL: "https://example.com/audio.mp3", MediaType: "audio/mpeg"},
				{Type: "file", URL: "https://example.com/file.pdf", MediaType: "application/pdf"},
			}},
		},
	}, ai.MessageSanitizationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(prompt.Contents) != 6 || len(history) != 1 {
		t.Fatalf("unexpected file input: prompt=%#v history=%#v", prompt, history)
	}
	if prompt.Contents[0].(ai.TextContent).Text != "Describe these files." ||
		string(prompt.Contents[1].(ai.BinaryContent).Data) != "pdf" ||
		prompt.Contents[2].(ai.ImageURL).URL != "https://example.com/image.png" ||
		prompt.Contents[3].(ai.VideoURL).URL != "https://example.com/video.mp4" ||
		prompt.Contents[4].(ai.AudioURL).URL != "https://example.com/audio.mp3" ||
		prompt.Contents[5].(ai.DocumentURL).URL != "https://example.com/file.pdf" {
		t.Fatalf("unexpected file content: %#v", prompt.Contents)
	}
	response := history[0].(ai.ModelResponse)
	file := response.Parts[0].(ai.FilePart)
	if string(file.Content.Data) != "old" || file.Content.MediaType != "image/png" {
		t.Fatalf("unexpected assistant file: %#v", file)
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
		{name: "system part", input: requestWith(
			vercel.UIMessage{ID: "system", Role: "system", Parts: []vercel.UIMessagePart{{Type: "file"}}},
			vercel.UIMessage{ID: "user", Role: "user", Parts: []vercel.UIMessagePart{{Type: "text", Text: "next"}}},
		), match: "support only text"},
		{name: "user file URL", input: requestWith(vercel.UIMessage{ID: "one", Role: "user", Parts: []vercel.UIMessagePart{{Type: "file"}}}), match: "file URL"},
		{name: "user part", input: requestWith(vercel.UIMessage{ID: "one", Role: "user", Parts: []vercel.UIMessagePart{{Type: "source-url"}}}), match: "unsupported user part"},
		{name: "data encoding", input: requestWith(vercel.UIMessage{ID: "one", Role: "user", Parts: []vercel.UIMessagePart{{Type: "file", URL: "data:text/plain,hello"}}}), match: "base64 data"},
		{name: "data media type", input: requestWith(vercel.UIMessage{ID: "one", Role: "user", Parts: []vercel.UIMessagePart{{Type: "file", URL: "data:;base64,aGVsbG8="}}}), match: "media type"},
		{name: "data base64", input: requestWith(vercel.UIMessage{ID: "one", Role: "user", Parts: []vercel.UIMessagePart{{Type: "file", URL: "data:text/plain;base64,!"}}}), match: "decode file data URL"},
		{name: "assistant file URL", input: requestWith(
			vercel.UIMessage{ID: "assistant", Role: "assistant", Parts: []vercel.UIMessagePart{{Type: "file", URL: "https://example.com/file.png"}}},
			vercel.UIMessage{ID: "user", Role: "user", Parts: []vercel.UIMessagePart{{Type: "text", Text: "next"}}},
		), match: "decode assistant file"},
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

func TestApprovalRequestAndResume(t *testing.T) {
	executions := 0
	var received map[string]any
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	agent.AddRawTool(ai.ToolDefinition{
		Name: "approve", Schema: map[string]any{
			"type": "object", "properties": map[string]any{"value": map[string]any{"type": "string"}},
		},
	}, func(_ context.Context, args json.RawMessage) (any, error) {
		executions++
		if err := json.Unmarshal(args, &received); err != nil {
			return nil, err
		}
		return "done", nil
	}, ai.WithApprovalRequired())
	adapter := vercel.NewAdapter(agent, vercel.Config{SDKVersion: 6})
	input := requestWith(vercel.UIMessage{
		ID: "user", Role: "user", Parts: []vercel.UIMessagePart{{Type: "text", Text: "run"}},
	})
	approvalSeen := false
	for chunk, err := range adapter.RunStream(context.Background(), input, struct{}{}) {
		if err != nil {
			t.Fatal(err)
		}
		if chunk.Type == vercel.ChunkToolApprovalRequest {
			approvalSeen = chunk.ApprovalID == "call_approve"
		}
	}
	if executions != 0 || !approvalSeen {
		t.Fatalf("unexpected approval request: executions=%d seen=%v", executions, approvalSeen)
	}
	approved := true
	resume := requestWith(
		vercel.UIMessage{ID: "user", Role: "user", Parts: []vercel.UIMessagePart{{Type: "text", Text: "run"}}},
		vercel.UIMessage{ID: "assistant", Role: "assistant", Parts: []vercel.UIMessagePart{{
			Type: "tool-approve", ToolCallID: "call_approve", State: "approval-responded",
			Input:    []byte(`{"value":"edited"}`),
			Approval: &vercel.ToolApproval{ID: "call_approve", Approved: &approved},
		}}},
	)
	for _, err := range adapter.RunStream(context.Background(), resume, struct{}{}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if executions != 1 || received["value"] != "edited" {
		t.Fatalf("approval did not execute edited arguments: executions=%d args=%#v", executions, received)
	}
}

func TestApprovalResumeValidation(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	agent.AddRawTool(ai.ToolDefinition{
		Name: "approve", Schema: map[string]any{"type": "object", "properties": map[string]any{}},
	}, func(context.Context, json.RawMessage) (any, error) { return "done", nil }, ai.WithApprovalRequired())
	denied := false
	base := requestWith(
		vercel.UIMessage{ID: "user", Role: "user", Parts: []vercel.UIMessagePart{{Type: "text", Text: "run"}}},
		vercel.UIMessage{ID: "assistant", Role: "assistant", Parts: []vercel.UIMessagePart{{
			Type: "tool-approve", ToolCallID: "call_approve", State: "approval-responded", Input: []byte(`{}`),
			Approval: &vercel.ToolApproval{ID: "call_approve", Approved: &denied, Reason: "no"},
		}}},
	)
	for _, err := range vercel.NewAdapter(agent, vercel.Config{SDKVersion: 6}).RunStream(
		context.Background(), base, struct{}{},
	) {
		if err != nil {
			t.Fatal(err)
		}
	}
	base.Messages[1].Parts[0].Approval = nil
	base.Messages[1].Parts[0].State = "approval-responded"
	for _, err := range vercel.NewAdapter(agent, vercel.Config{SDKVersion: 6}).RunStream(
		context.Background(), base, struct{}{},
	) {
		if err != nil {
			t.Fatal(err)
		}
	}

	tests := []struct {
		name   string
		input  vercel.RequestData
		config vercel.Config
	}{
		{name: "missing tool ID", input: requestWith(vercel.UIMessage{
			ID: "assistant", Role: "assistant", Parts: []vercel.UIMessagePart{{Type: "tool-x", State: "approval-responded"}},
		})},
		{name: "invalid message", input: requestWith(vercel.UIMessage{
			Role: "assistant", Parts: []vercel.UIMessagePart{{Type: "tool-x", ToolCallID: "call", State: "approval-responded"}},
		})},
		{name: "sanitization", input: base, config: vercel.Config{Sanitization: ai.MessageSanitizationOptions{
			AllowedFileURLSchemes: []string{"bad scheme"},
		}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var got error
			for _, err := range vercel.NewAdapter(agent, test.config).RunStream(context.Background(), test.input, struct{}{}) {
				got = err
			}
			if got == nil {
				t.Fatal("expected approval resume error")
			}
		})
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
