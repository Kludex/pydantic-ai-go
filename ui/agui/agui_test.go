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

func TestPrepareMultimodalInput(t *testing.T) {
	metadata := map[string]any{
		"vendor_metadata": map[string]any{"source": "client"}, "force_download": "safe",
	}
	prompt, history, _, err := agui.PrepareInput(agui.RunAgentInput{Messages: []agui.Message{{
		ID: "user", Role: "user", Content: []agui.InputContent{
			{Type: "text", Text: "inspect"},
			{Type: "binary", Data: "YmluYXJ5", MimeType: "application/pdf"},
			{Type: "binary", URL: "data:text/plain;base64,aW5saW5l"},
			{Type: "binary", URL: "https://example.com/legacy.png", MimeType: "image/png"},
			{Type: "image", Source: &agui.InputContentSource{
				Type: "url", Value: "https://example.com/image.png", MimeType: "image/png",
			}, Metadata: metadata},
			{Type: "audio", Source: &agui.InputContentSource{
				Type: "url", Value: "https://example.com/audio.mp3", MimeType: "audio/mpeg",
			}},
			{Type: "video", Source: &agui.InputContentSource{
				Type: "url", Value: "https://example.com/video.mp4", MimeType: "video/mp4",
			}},
			{Type: "document", Source: &agui.InputContentSource{
				Type: "url", Value: "https://example.com/document.pdf", MimeType: "application/pdf",
			}},
			{Type: "image", Source: &agui.InputContentSource{Type: "data", Value: "aW1hZ2U=", MimeType: "image/png"}},
		},
	}}}, ai.MessageSanitizationOptions{AllowedFileDownloadModes: []ai.FileDownloadMode{ai.FileDownloadSafe}})
	if err != nil {
		t.Fatal(err)
	}
	if len(history) != 0 || len(prompt.Contents) != 9 || prompt.Contents[0].(ai.TextContent).Text != "inspect" ||
		string(prompt.Contents[1].(ai.BinaryContent).Data) != "binary" ||
		string(prompt.Contents[2].(ai.BinaryContent).Data) != "inline" ||
		prompt.Contents[3].(ai.ImageURL).URL != "https://example.com/legacy.png" ||
		prompt.Contents[4].(ai.ImageURL).ForceDownload != ai.FileDownloadSafe ||
		prompt.Contents[4].(ai.ImageURL).VendorMetadata["source"] != "client" ||
		prompt.Contents[5].(ai.AudioURL).URL != "https://example.com/audio.mp3" ||
		prompt.Contents[6].(ai.VideoURL).URL != "https://example.com/video.mp4" ||
		prompt.Contents[7].(ai.DocumentURL).URL != "https://example.com/document.pdf" ||
		string(prompt.Contents[8].(ai.BinaryContent).Data) != "image" {
		t.Fatalf("unexpected multimodal prompt: %#v", prompt)
	}
	metadata["vendor_metadata"].(map[string]any)["source"] = "changed"
	if prompt.Contents[4].(ai.ImageURL).VendorMetadata["source"] != "client" {
		t.Fatal("multimodal metadata shares client input")
	}
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	streamInput := agui.RunAgentInput{Messages: []agui.Message{{
		ID: "user", Role: "user", Content: []agui.InputContent{{Type: "text", Text: "hello"}},
	}}}
	finished := false
	for event, err := range agui.NewAdapter(agent, agui.Config{}).RunStream(t.Context(), streamInput, struct{}{}) {
		if err != nil {
			t.Fatal(err)
		}
		finished = finished || event.Type == agui.EventRunFinished
	}
	if !finished {
		t.Fatal("multimodal run did not finish")
	}
	empty, _, _, err := agui.PrepareInput(agui.RunAgentInput{Messages: []agui.Message{{
		ID: "empty", Role: "user",
	}}}, ai.MessageSanitizationOptions{})
	if err != nil || empty.Content != "" || len(empty.Contents) != 0 {
		t.Fatalf("unexpected empty prompt: %#v %v", empty, err)
	}
}

func TestPrepareReasoningInput(t *testing.T) {
	encrypted := `{"id":"thinking-id","signature":"signature","provider_name":"anthropic","provider_details":{"redacted":false}}`
	prompt, history, _, err := agui.PrepareInput(agui.RunAgentInput{Messages: []agui.Message{
		{ID: "reasoning", Role: "reasoning", Content: "thought", EncryptedValue: encrypted},
		{ID: "assistant", Role: "assistant", Content: "answer"},
		{ID: "user", Role: "user", Content: "continue"},
	}}, ai.MessageSanitizationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if prompt.Content != "continue" || len(history) != 1 {
		t.Fatalf("unexpected reasoning input: prompt=%#v history=%#v", prompt, history)
	}
	response := history[0].(ai.ModelResponse)
	thinking := response.Parts[0].(ai.ThinkingPart)
	text := response.Parts[1].(ai.TextPart)
	if thinking.Content != "thought" || thinking.ID != "thinking-id" || thinking.Signature != "signature" ||
		thinking.ProviderName != "anthropic" || thinking.ProviderDetails["redacted"] != false || text.Content != "answer" {
		t.Fatalf("unexpected reasoning history: %#v", response)
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
		{name: "system content", messages: []agui.Message{{ID: "one", Role: "system", Content: 1}}, match: "system message content"},
		{name: "assistant content", messages: []agui.Message{{ID: "one", Role: "assistant", Content: 1}}, match: "assistant message content"},
		{name: "reasoning content", messages: []agui.Message{{ID: "one", Role: "reasoning", Content: 1}}, match: "reasoning message content"},
		{name: "user encoding", messages: []agui.Message{{ID: "one", Role: "user", Content: func() {}}}, match: "encode user content"},
		{name: "user decoding", messages: []agui.Message{{ID: "one", Role: "user", Content: 1}}, match: "decode user content"},
		{name: "content type", messages: []agui.Message{{ID: "one", Role: "user", Content: []agui.InputContent{{Type: "future"}}}}, match: "unsupported user content"},
		{name: "binary source", messages: []agui.Message{{ID: "one", Role: "user", Content: []agui.InputContent{{Type: "binary"}}}}, match: "requires url or data"},
		{name: "binary data", messages: []agui.Message{{ID: "one", Role: "user", Content: []agui.InputContent{{Type: "binary", Data: "!"}}}}, match: "decode binary input"},
		{name: "data URL", messages: []agui.Message{{ID: "one", Role: "user", Content: []agui.InputContent{{Type: "binary", URL: "data:text/plain,hi"}}}}, match: "must contain base64"},
		{name: "data URL base64", messages: []agui.Message{{ID: "one", Role: "user", Content: []agui.InputContent{{Type: "binary", URL: "data:text/plain;base64,!"}}}}, match: "decode binary data URL"},
		{name: "typed source", messages: []agui.Message{{ID: "one", Role: "user", Content: []agui.InputContent{{Type: "image"}}}}, match: "requires a url or data source"},
		{name: "typed data", messages: []agui.Message{{ID: "one", Role: "user", Content: []agui.InputContent{{Type: "image", Source: &agui.InputContentSource{Type: "data", Value: "!"}}}}}, match: "decode image input"},
		{name: "force download", messages: []agui.Message{{ID: "one", Role: "user", Content: []agui.InputContent{{
			Type: "image", Source: &agui.InputContentSource{Type: "url", Value: "https://example.com", MimeType: "image/png"},
			Metadata: map[string]any{"force_download": "always"},
		}}}}, match: "multimodal force download"},
		{name: "tool encoding", messages: []agui.Message{{ID: "one", Role: "tool", ToolCallID: "call", Content: func() {}}}, match: "encode tool result"},
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
	for _, version := range []string{"invalid", "0..1", "999999999999999999999999999999999"} {
		assertPanic(t, func() { agui.NewAdapter(agent, agui.Config{Version: version}) })
	}
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

func TestApprovalInterruptAndResume(t *testing.T) {
	executions := 0
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	agent.AddRawTool(ai.ToolDefinition{
		Name: "approve", Schema: map[string]any{"type": "object", "properties": map[string]any{}},
	}, func(context.Context, json.RawMessage) (any, error) {
		executions++
		return "approved", nil
	}, ai.WithApprovalRequired(), ai.WithApprovalMetadata(map[string]any{"risk": "low"}))
	adapter := agui.NewAdapter(agent, agui.Config{})
	input := agui.RunAgentInput{ThreadID: "thread", RunID: "first", Messages: []agui.Message{
		{ID: "user", Role: "user", Content: "run"},
	}}
	var outcome *agui.RunOutcome
	for event, err := range adapter.RunStream(context.Background(), input, struct{}{}) {
		if err != nil {
			t.Fatal(err)
		}
		if event.Type == agui.EventRunFinished {
			outcome = event.Outcome
		}
	}
	if executions != 0 || outcome == nil || outcome.Type != "interrupt" || len(outcome.Interrupts) != 1 ||
		outcome.Interrupts[0].ID != "int-call_approve" || outcome.Interrupts[0].Metadata["risk"] != "low" {
		t.Fatalf("unexpected interrupt: executions=%d outcome=%#v", executions, outcome)
	}
	resume := agui.RunAgentInput{ThreadID: "thread", RunID: "second", Messages: []agui.Message{
		{ID: "user", Role: "user", Content: "run"},
		{ID: "assistant", Role: "assistant", ToolCalls: []agui.ToolCall{{
			ID: "call_approve", Type: "function", Function: agui.ToolCallFunction{Name: "approve", Arguments: `{}`},
		}}},
	}, Resume: []agui.ResumeEntry{{
		InterruptID: "int-call_approve", Status: "completed", Payload: []byte(`{"approved":true,"editedArgs":{}}`),
	}}}
	resultSeen := false
	for event, err := range adapter.RunStream(context.Background(), resume, struct{}{}) {
		if err != nil {
			t.Fatal(err)
		}
		if event.Type == agui.EventToolCallResult {
			resultSeen = true
		}
	}
	if executions != 1 || !resultSeen {
		t.Fatalf("approval did not resume the tool: executions=%d result=%v", executions, resultSeen)
	}
}

func TestApprovalResumeValidation(t *testing.T) {
	executions := 0
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	agent.AddRawTool(ai.ToolDefinition{
		Name: "approve", Schema: map[string]any{"type": "object", "properties": map[string]any{}},
	}, func(context.Context, json.RawMessage) (any, error) {
		executions++
		return "done", nil
	}, ai.WithApprovalRequired())
	base := agui.RunAgentInput{ThreadID: "thread", Messages: []agui.Message{
		{ID: "user", Role: "user", Content: "run"},
		{ID: "assistant", Role: "assistant", ToolCalls: []agui.ToolCall{{
			ID: "call_approve", Type: "function", Function: agui.ToolCallFunction{Name: "approve", Arguments: `{}`},
		}}},
	}}
	tests := []struct {
		name      string
		entry     agui.ResumeEntry
		executes  bool
		wantError bool
	}{
		{name: "approved", entry: agui.ResumeEntry{InterruptID: "int-call_approve", Payload: []byte(`{"approved":true}`)}, executes: true},
		{name: "cancelled", entry: agui.ResumeEntry{InterruptID: "int-call_approve", Status: "cancelled"}},
		{name: "malformed", entry: agui.ResumeEntry{InterruptID: "int-call_approve", Payload: []byte(`{`)}},
		{name: "missing decision", entry: agui.ResumeEntry{InterruptID: "int-call_approve", Payload: []byte(`{}`)}},
		{name: "bad edits", entry: agui.ResumeEntry{InterruptID: "int-call_approve", Payload: []byte(`{"approved":true,"editedArgs":[]}`)}},
		{name: "denied", entry: agui.ResumeEntry{InterruptID: "int-call_approve", Payload: []byte(`{"approved":false,"reason":"no"}`)}},
		{name: "invalid ID", entry: agui.ResumeEntry{InterruptID: "bad", Payload: []byte(`{"approved":true}`)}, wantError: true},
		{name: "empty ID", entry: agui.ResumeEntry{InterruptID: "int-", Payload: []byte(`{"approved":true}`)}, wantError: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			before := executions
			input := base
			input.Resume = []agui.ResumeEntry{test.entry}
			var got error
			for _, err := range agui.NewAdapter(agent, agui.Config{}).RunStream(context.Background(), input, struct{}{}) {
				if err != nil {
					got = err
				}
			}
			if (got != nil) != test.wantError {
				t.Fatalf("unexpected error: %v", got)
			}
			if (executions > before) != test.executes {
				t.Fatalf("unexpected execution count: before=%d after=%d", before, executions)
			}
		})
	}

	invalidMessage := base
	invalidMessage.Messages[0].ID = ""
	invalidMessage.Resume = []agui.ResumeEntry{{InterruptID: "int-call_approve", Payload: []byte(`{"approved":true}`)}}
	for _, config := range []agui.Config{
		{},
		{Sanitization: ai.MessageSanitizationOptions{AllowedFileURLSchemes: []string{"bad scheme"}}},
	} {
		var got error
		for _, err := range agui.NewAdapter(agent, config).RunStream(context.Background(), invalidMessage, struct{}{}) {
			got = err
		}
		if got == nil {
			t.Fatal("expected resume preparation error")
		}
		invalidMessage.Messages[0].ID = "user"
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
