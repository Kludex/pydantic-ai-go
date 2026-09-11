package agui_test

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
	"github.com/Kludex/pydantic-ai-go/ai/ui/agui"
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

func TestPrepareInputRehydratesToolReturnContent(t *testing.T) {
	input := agui.RunAgentInput{Messages: []agui.Message{
		{ID: "assistant", Role: "assistant", ToolCalls: []agui.ToolCall{{
			ID: "call", Type: "function", Function: agui.ToolCallFunction{Name: "files", Arguments: `{}`},
		}}},
		{ID: "tool", Role: "tool", ToolCallID: "call", Content: []any{
			map[string]any{
				"kind": "binary", "media_type": "application/octet-stream", "data": "AAE=",
			},
			map[string]any{"kind": "binary", "media_type": "text/plain", "label": "keep"},
		}},
		{ID: "user", Role: "user", Content: "continue"},
	}}
	_, history, _, err := agui.PrepareInput(input, ai.MessageSanitizationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	result := history[1].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart).Content.([]any)
	binary, ok := result[0].(ai.BinaryContent)
	if !ok || !slices.Equal(binary.Data, []byte{0, 1}) {
		t.Fatalf("binary tool return was not restored: %#v", result[0])
	}
	if mapping, ok := result[1].(map[string]any); !ok || mapping["label"] != "keep" {
		t.Fatalf("kind-colliding application data was lost: %#v", result[1])
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

func TestPrepareTypedToolHistory(t *testing.T) {
	input := agui.RunAgentInput{Messages: []agui.Message{
		{ID: "developer", Role: "developer", Content: "trusted instruction"},
		{ID: "assistant", Role: "assistant", ToolCalls: []agui.ToolCall{
			{ID: "local", Function: agui.ToolCallFunction{Name: "search_tools", Arguments: `{}`},
				EncryptedValue: `{"pydantic_ai":{"tool_kind":"tool-search"}}`},
			{ID: "pyd_ai_builtin|openai|native", Function: agui.ToolCallFunction{Name: "web_search", Arguments: `{}`},
				EncryptedValue: `{"pydantic_ai":{"tool_kind":"web-search"}}`},
		}},
		{ID: "local-result", Role: "tool", ToolCallID: "local", Content: "found"},
		{ID: "native-result", Role: "tool", ToolCallID: "pyd_ai_builtin|openai|native", Content: "blocked",
			EncryptedValue: `{"pydantic_ai":{"tool_kind":"web-search","outcome":"denied"}}`},
		{ID: "user", Role: "user", Content: "continue"},
	}}
	prompt, history, _, err := agui.PrepareInput(input, ai.MessageSanitizationOptions{AllowSystemPrompts: true})
	if err != nil {
		t.Fatal(err)
	}
	if prompt.Content != "continue" || len(history) < 3 {
		t.Fatalf("unexpected typed history: %#v %#v", prompt, history)
	}
	instruction := history[0].(ai.ModelRequest).Parts[0].(ai.SystemPromptPart)
	var localCall ai.ToolCallPart
	var nativeCall ai.NativeToolCallPart
	var localResult ai.ToolReturnPart
	var nativeResult ai.NativeToolReturnPart
	for _, message := range history {
		switch message := message.(type) {
		case ai.ModelResponse:
			for _, part := range message.Parts {
				switch part := part.(type) {
				case ai.ToolCallPart:
					localCall = part
				case ai.NativeToolCallPart:
					nativeCall = part
				case ai.NativeToolReturnPart:
					nativeResult = part
				}
			}
		case ai.ModelRequest:
			for _, part := range message.Parts {
				if value, ok := part.(ai.ToolReturnPart); ok {
					localResult = value
				}
			}
		}
	}
	if instruction.Content != "trusted instruction" || localCall.ToolKind != ai.ToolPartKindToolSearch ||
		nativeCall.ToolCallID != "native" || nativeCall.ProviderName != "openai" ||
		nativeCall.ToolKind != ai.ToolPartKindWebSearch || localResult.ToolName != "search_tools" ||
		localResult.ToolKind != ai.ToolPartKindToolSearch || nativeResult.ToolName != "web_search" ||
		nativeResult.Outcome != ai.ToolReturnOutcomeDenied || nativeResult.ToolKind != "" {
		t.Fatalf("unexpected typed parts: %#v %#v %#v %#v", localCall, nativeCall, localResult, nativeResult)
	}

	input.Messages[1].ToolCalls[0].EncryptedValue = `not-json`
	input.Messages[1].ToolCalls[1].ID = "pyd_ai_builtin|broken"
	input.Messages[2].EncryptedValue = `{"pydantic_ai":{"tool_kind":"unknown","outcome":"unknown"}}`
	input.Messages[3].ToolCallID = "pyd_ai_builtin|broken"
	input.Messages[3].EncryptedValue = ""
	input.Messages[3].Error = "failed"
	_, history, _, err = agui.PrepareInput(input, ai.MessageSanitizationOptions{AllowSystemPrompts: true})
	if err != nil {
		t.Fatal(err)
	}
	var calls []ai.ToolCallPart
	var returns []ai.ToolReturnPart
	for _, message := range history {
		switch message := message.(type) {
		case ai.ModelResponse:
			for _, part := range message.Parts {
				if value, ok := part.(ai.ToolCallPart); ok {
					calls = append(calls, value)
				}
			}
		case ai.ModelRequest:
			for _, part := range message.Parts {
				if value, ok := part.(ai.ToolReturnPart); ok {
					returns = append(returns, value)
				}
			}
		}
	}
	if len(calls) != 2 || len(returns) != 2 || calls[0].ToolKind != "" ||
		returns[0].Outcome != ai.ToolReturnOutcomeSuccess || returns[1].Outcome != ai.ToolReturnOutcomeFailed {
		t.Fatalf("malformed metadata did not degrade safely: %#v", history)
	}
	_, _, _, err = agui.PrepareInput(agui.RunAgentInput{Messages: []agui.Message{
		{ID: "orphan", Role: "tool", Name: "fallback", ToolCallID: "orphan", Content: "result"},
		{ID: "user", Role: "user", Content: "continue"},
	}}, ai.MessageSanitizationOptions{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestPrepareActivities(t *testing.T) {
	compaction := map[string]any{
		"content": "summary", "id": "compact", "provider_name": "openai",
		"provider_details": map[string]any{"encrypted": "value"},
	}
	prompt, history, _, err := agui.PrepareInput(agui.RunAgentInput{Messages: []agui.Message{
		{ID: "compact", Role: "activity", ActivityType: "pydantic_ai_compaction", Content: compaction},
		{ID: "custom", Role: "activity", ActivityType: "application_custom", Content: map[string]any{"ignored": true}},
		{ID: "tools", Role: "activity", ActivityType: "pydantic_ai_tool_availability_delta", Content: map[string]any{
			"added": []any{"search", 1, ""}, "tool_call_id": "call",
		}},
		{ID: "tools-typed", Role: "activity", ActivityType: "pydantic_ai_tool_availability_delta", Content: map[string]any{
			"added": []string{"fetch"},
		}},
		{ID: "user", Role: "user", Content: "continue"},
	}}, ai.MessageSanitizationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if prompt.Content != "continue" || len(history) != 3 {
		t.Fatalf("unexpected activity input: prompt=%#v history=%#v", prompt, history)
	}
	part := history[0].(ai.ModelResponse).Parts[0].(ai.CompactionPart)
	first := history[1].(ai.ModelRequest).Parts[0].(ai.ToolAvailabilityDeltaPart)
	second := history[2].(ai.ModelRequest).Parts[0].(ai.ToolAvailabilityDeltaPart)
	if part.Content != "summary" || part.ProviderDetails["encrypted"] != "value" ||
		len(first.ToolsAdded) != 1 || first.ToolsAdded[0] != "search" || first.ToolCallID != "call" ||
		len(second.ToolsAdded) != 1 || second.ToolsAdded[0] != "fetch" {
		t.Fatalf("unexpected activity parts: %#v %#v %#v", part, first, second)
	}
	compaction["provider_details"].(map[string]any)["encrypted"] = "changed"
	if part.ProviderDetails["encrypted"] != "value" {
		t.Fatal("activity metadata shares client input")
	}
	for _, malformed := range []map[string]any{
		{"content": 1}, {"id": 1}, {"provider_name": 1}, {"provider_details": 1},
		{"id": "compact"}, {},
	} {
		_, malformedHistory, _, err := agui.PrepareInput(agui.RunAgentInput{Messages: []agui.Message{
			{ID: "activity", Role: "activity", ActivityType: "pydantic_ai_compaction", Content: malformed},
			{ID: "user", Role: "user", Content: "continue"},
		}}, ai.MessageSanitizationOptions{})
		if err != nil || len(malformedHistory) != 0 {
			t.Fatalf("malformed compaction was retained: %#v %v", malformedHistory, err)
		}
	}
}

func TestPreserveFileActivities(t *testing.T) {
	var captured []ai.ModelMessage
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		captured = messages
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	input := agui.RunAgentInput{Messages: []agui.Message{
		{ID: "file", Role: "activity", ActivityType: "pydantic_ai_file", Content: map[string]any{
			"url": "data:image/png;base64,aW1hZ2U=", "id": "generated", "provider_name": "openai",
			"provider_details": map[string]any{"detail": "value"},
			"vendor_metadata":  map[string]any{"vendor": "value"},
		}},
		{ID: "uploaded", Role: "activity", ActivityType: "pydantic_ai_uploaded_file", Content: map[string]any{
			"file_id": "file-1", "provider_name": "openai", "media_type": "application/pdf",
			"identifier": "document", "vendor_metadata": map[string]any{"purpose": "input"},
		}},
		{ID: "user", Role: "user", Content: "continue"},
	}}
	config := agui.Config{PreserveFileData: true}
	config.Sanitization.AllowUploadedFiles = true
	for _, err := range agui.NewAdapter(ai.NewAgent[struct{}, string](model), config).RunStream(t.Context(), input, struct{}{}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(captured) != 3 {
		t.Fatalf("unexpected preserved history: %#v", captured)
	}
	file := captured[0].(ai.ModelResponse).Parts[0].(ai.FilePart)
	uploaded := captured[1].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Contents[0].(ai.UploadedFile)
	if string(file.Content.Data) != "image" || file.ID != "generated" || file.ProviderName != "openai" ||
		file.ProviderDetails["detail"] != "value" || file.Content.VendorMetadata["vendor"] != "value" ||
		uploaded.FileID != "file-1" || uploaded.MediaType != "application/pdf" || uploaded.Identifier != "document" ||
		uploaded.VendorMetadata["purpose"] != "input" {
		t.Fatalf("unexpected preserved files: %#v %#v", file, uploaded)
	}

	for _, malformed := range []agui.Message{
		{ID: "file", Role: "activity", ActivityType: "pydantic_ai_file", Content: map[string]any{}},
		{ID: "file", Role: "activity", ActivityType: "pydantic_ai_file", Content: map[string]any{"url": "invalid"}},
		{ID: "upload", Role: "activity", ActivityType: "pydantic_ai_uploaded_file", Content: map[string]any{}},
	} {
		invalid := input
		invalid.Messages = []agui.Message{malformed, {ID: "user", Role: "user", Content: "continue"}}
		var got error
		for _, err := range agui.NewAdapter(ai.NewAgent[struct{}, string](model), config).RunStream(
			t.Context(), invalid, struct{}{},
		) {
			got = err
		}
		if got == nil {
			t.Fatalf("malformed file activity was accepted: %#v", malformed)
		}
	}

	ignored := input
	ignored.Messages[0].Content = map[string]any{}
	for _, err := range agui.NewAdapter(ai.NewAgent[struct{}, string](model), agui.Config{}).RunStream(
		t.Context(), ignored, struct{}{},
	) {
		if err != nil {
			t.Fatal(err)
		}
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
		{name: "activity content", messages: []agui.Message{{ID: "one", Role: "activity", Content: "invalid"}}, match: "activity content"},
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

type forwardedDeps struct {
	input agui.ForwardedInput
	err   error
	emit  bool
}

func (deps *forwardedDeps) SetAGUIRunInput(input agui.ForwardedInput) error {
	deps.input = input
	if deps.emit {
		if err := input.Events.EmitStateSnapshot(map[string]any{"count": 1}); err != nil {
			return err
		}
		if err := input.Events.EmitStateDelta([]any{map[string]any{"op": "replace", "path": "/count", "value": 2}}); err != nil {
			return err
		}
		if err := input.Events.EmitCustom("notice", map[string]any{"text": "ready"}); err != nil {
			return err
		}
	}
	return deps.err
}

func TestForwardedInput(t *testing.T) {
	state := map[string]any{"count": 1}
	props := map[string]any{"tenant": "one"}
	deps := &forwardedDeps{}
	input := agui.RunAgentInput{
		State: state, Context: []agui.Context{{Description: "User", Value: "Ada"}}, ForwardedProps: props,
		Messages: []agui.Message{{ID: "user", Role: "user", Content: "run"}},
	}
	adapter := agui.NewAdapter(ai.NewAgent[*forwardedDeps, string](fakes.NewTestModel()), agui.Config{})
	for _, err := range adapter.RunStream(t.Context(), input, deps) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if deps.input.ThreadID == "" || deps.input.RunID == "" || deps.input.State.(map[string]any)["count"] != float64(1) ||
		deps.input.Context[0].Value != "Ada" || deps.input.ForwardedProps.(map[string]any)["tenant"] != "one" {
		t.Fatalf("unexpected forwarded input: %#v", deps.input)
	}
	deps.input.State.(map[string]any)["count"] = 2
	deps.input.Context[0].Value = "changed"
	deps.input.ForwardedProps.(map[string]any)["tenant"] = "changed"
	if state["count"] != 1 || input.Context[0].Value != "Ada" || props["tenant"] != "one" {
		t.Fatal("forwarded input shares client-owned data")
	}

	deps.err = errors.New("rejected")
	var got error
	for _, err := range adapter.RunStream(t.Context(), input, deps) {
		got = err
	}
	if got == nil || !strings.Contains(got.Error(), "apply forwarded input") {
		t.Fatalf("unexpected receiver error: %v", got)
	}
	deps.err = nil
	cyclic := map[string]any{}
	cyclic["self"] = cyclic
	invalidState := input
	invalidState.State = cyclic
	for _, err := range adapter.RunStream(t.Context(), invalidState, deps) {
		got = err
	}
	if got == nil || !strings.Contains(got.Error(), "clone state") {
		t.Fatalf("unexpected state clone error: %v", got)
	}
	invalidProps := input
	invalidProps.State = nil
	invalidProps.ForwardedProps = make(chan int)
	for _, err := range adapter.RunStream(t.Context(), invalidProps, deps) {
		got = err
	}
	if got == nil || !strings.Contains(got.Error(), "clone forwarded properties") {
		t.Fatalf("unexpected props clone error: %v", got)
	}

	deps.emit = true
	input.State = nil
	input.ForwardedProps = nil
	var emitted []agui.Event
	for event, err := range adapter.RunStream(t.Context(), input, deps) {
		if err != nil {
			t.Fatal(err)
		}
		if event.Type == agui.EventStateSnapshot || event.Type == agui.EventStateDelta || event.Type == agui.EventCustom {
			emitted = append(emitted, event)
		}
	}
	if len(emitted) != 3 || emitted[0].Snapshot.(map[string]any)["count"] != float64(1) ||
		emitted[1].Delta.([]any)[0].(map[string]any)["op"] != "replace" ||
		emitted[2].Name != "notice" || emitted[2].Value.(map[string]any)["text"] != "ready" ||
		emitted[0].Timestamp == 0 {
		t.Fatalf("unexpected application events: %#v", emitted)
	}
	for event := range adapter.RunStream(t.Context(), input, deps) {
		if event.Type == agui.EventStateSnapshot {
			break
		}
	}
	deps.emit = false
	for event := range adapter.RunStream(t.Context(), input, deps) {
		switch {
		case event.Type == agui.EventTextMessageEnd:
			if err := deps.input.Events.EmitCustom("before-finish", true); err != nil {
				t.Fatal(err)
			}
		case event.Type == agui.EventCustom && event.Name == "before-finish":
			if err := deps.input.Events.EmitCustom("at-finish", true); err != nil {
				t.Fatal(err)
			}
		}
		if event.Type == agui.EventCustom && event.Name == "at-finish" {
			break
		}
	}

	queue := &agui.EventQueue{}
	cycleMap := map[string]any{}
	cycleMap["self"] = cycleMap
	cycleSlice := []any{nil}
	cycleSlice[0] = cycleSlice
	for name, err := range map[string]error{
		"snapshot": queue.EmitStateSnapshot(cycleMap),
		"delta":    queue.EmitStateDelta(cycleSlice),
		"name":     queue.EmitCustom("", true),
		"custom":   queue.EmitCustom("event", cycleMap),
	} {
		if err == nil {
			t.Fatalf("invalid %s event was accepted", name)
		}
	}
}

func TestFrontendTools(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	input := agui.RunAgentInput{
		ThreadID: "thread", RunID: "run",
		Messages: []agui.Message{{ID: "user", Role: "user", Content: "show weather"}},
		Tools: []agui.FrontendTool{{
			Name: "weather", Description: "Read browser weather.",
			Parameters: map[string]any{
				"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}},
			},
		}},
	}
	var events []agui.Event
	for event, err := range agui.NewAdapter(agent, agui.Config{}).RunStream(t.Context(), input, struct{}{}) {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, event)
	}
	start := events[eventIndex(events, agui.EventToolCallStart)]
	if start.ToolCallName != "weather" || eventIndex(events, agui.EventToolCallEnd) < 0 ||
		eventIndex(events, agui.EventToolCallResult) >= 0 {
		t.Fatalf("unexpected frontend tool lifecycle: %#v", events)
	}
	empty := input
	empty.Tools = []agui.FrontendTool{{}}
	var got error
	for _, err := range agui.NewAdapter(agent, agui.Config{}).RunStream(t.Context(), empty, struct{}{}) {
		got = err
	}
	if got == nil || !strings.Contains(got.Error(), "frontend tool name") {
		t.Fatalf("unexpected frontend tool error: %v", got)
	}
}

func TestExternalInterruptAndResume(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	agent.AddTool(ai.NewRawExternalTool[struct{}](ai.ToolDefinition{
		Name: "remote", Schema: map[string]any{"type": "object", "properties": map[string]any{}},
	}))
	adapter := agui.NewAdapter(agent, agui.Config{})
	initial := agui.RunAgentInput{Messages: []agui.Message{{ID: "user", Role: "user", Content: "run"}}}
	var outcome *agui.RunOutcome
	for event, err := range adapter.RunStream(t.Context(), initial, struct{}{}) {
		if err != nil {
			t.Fatal(err)
		}
		if event.Type == agui.EventRunFinished {
			outcome = event.Outcome
		}
	}
	if outcome == nil || outcome.Type != "interrupt" || len(outcome.Interrupts) != 1 ||
		outcome.Interrupts[0].ID != "ext-call_remote" ||
		outcome.Interrupts[0].ResponseSchema["type"] != "object" {
		t.Fatalf("unexpected external interrupt: %#v", outcome)
	}
	base := agui.RunAgentInput{Messages: []agui.Message{
		{ID: "user", Role: "user", Content: "run"},
		{ID: "assistant", Role: "assistant", ToolCalls: []agui.ToolCall{{
			ID: "call_remote", Type: "function", Function: agui.ToolCallFunction{Name: "remote", Arguments: `{}`},
		}}},
	}}
	resolved := base
	resolved.Resume = []agui.ResumeEntry{{InterruptID: "ext-call_remote", Payload: []byte(`{"result":{"ok":true}}`)}}
	resultSeen := false
	for event, err := range adapter.RunStream(t.Context(), resolved, struct{}{}) {
		if err != nil {
			t.Fatal(err)
		}
		if event.Type == agui.EventToolCallResult && strings.Contains(event.Content.(string), `"ok":true`) {
			resultSeen = true
		}
	}
	if !resultSeen {
		t.Fatal("external result was not resumed")
	}

	for _, entry := range []agui.ResumeEntry{
		{InterruptID: "ext-call_remote", Payload: []byte(`{}`)},
		{InterruptID: "ext-call_remote", Payload: []byte(`{`)},
	} {
		input := base
		input.Resume = []agui.ResumeEntry{entry}
		var got error
		for _, err := range adapter.RunStream(t.Context(), input, struct{}{}) {
			got = err
		}
		if got == nil || !strings.Contains(got.Error(), "requires a result") {
			t.Fatalf("unexpected invalid external result error: %v", got)
		}
	}
	cancelled := base
	cancelled.Resume = []agui.ResumeEntry{{InterruptID: "ext-call_remote", Status: "cancelled"}}
	for _, err := range adapter.RunStream(t.Context(), cancelled, struct{}{}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	duplicate := base
	duplicate.Resume = []agui.ResumeEntry{{InterruptID: "ext-call_remote"}, {InterruptID: "int-call_remote"}}
	var got error
	for _, err := range adapter.RunStream(t.Context(), duplicate, struct{}{}) {
		got = err
	}
	if got == nil || !strings.Contains(got.Error(), "duplicate resume") {
		t.Fatalf("unexpected duplicate error: %v", got)
	}
	missing := base
	missing.Messages[1].ToolCalls = nil
	missing.Resume = []agui.ResumeEntry{{InterruptID: "ext-call_remote", Payload: []byte(`{"result":true}`)}}
	got = nil
	for _, err := range adapter.RunStream(t.Context(), missing, struct{}{}) {
		got = err
	}
	if got == nil {
		t.Fatal("resume without a matching tool call was accepted")
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
