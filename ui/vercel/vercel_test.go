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
	"time"

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
				{Type: "source-url", SourceID: "source", URL: "https://example.com"},
				{Type: "source-document", SourceID: "document", MediaType: "application/pdf", Title: "Document"},
				{Type: "step-start"},
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

func TestPrepareDynamicToolAndStreamingReasoning(t *testing.T) {
	providerMetadata := map[string]any{"pydantic_ai": map[string]any{"signature": "untrusted-stream-signature"}}
	denied := false
	prompt, history, _, err := vercel.PrepareInput(vercel.RequestData{
		Trigger: "submit-message", ID: "chat", Messages: []vercel.UIMessage{
			{ID: "assistant", Role: "assistant", Parts: []vercel.UIMessagePart{
				{Type: "reasoning", Text: "partial", State: "streaming", ProviderMetadata: providerMetadata},
				{
					Type: "dynamic-tool", ToolName: "lookup", ToolCallID: "dynamic", State: "output-denied",
					Input: []byte(`{"query":"go"}`), Approval: &vercel.ToolApproval{
						ID: "approval", Approved: &denied, Reason: "policy",
					},
				},
			}},
			{ID: "user", Role: "user", Parts: []vercel.UIMessagePart{{Type: "text", Text: "continue"}}},
		},
	}, ai.MessageSanitizationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if prompt.Content != "continue" || len(history) != 2 {
		t.Fatalf("unexpected input: prompt=%#v history=%#v", prompt, history)
	}
	response := history[0].(ai.ModelResponse)
	thinking := response.Parts[0].(ai.ThinkingPart)
	call := response.Parts[1].(ai.ToolCallPart)
	result := history[1].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart)
	if thinking.Signature != "" || call.ToolName != "lookup" || result.Content != "policy" ||
		result.Outcome != ai.ToolReturnOutcomeDenied {
		t.Fatalf("unexpected dynamic history: thinking=%#v call=%#v result=%#v", thinking, call, result)
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

func TestPrepareInputMetadata(t *testing.T) {
	providerMetadata := func(values map[string]any) map[string]any {
		return map[string]any{"pydantic_ai": values}
	}
	providerExecuted := true
	input := vercel.RequestData{
		Trigger: "submit-message", ID: "chat", Messages: []vercel.UIMessage{
			{ID: "files", Role: "user", Metadata: map[string]any{
				"application": map[string]any{"nested": "value"},
				"pydantic_ai": map[string]any{"timestamp": "2026-09-04T10:00:00Z"},
			}, Parts: []vercel.UIMessagePart{
				{Type: "file", URL: "provider-file", MediaType: "application/pdf", ProviderMetadata: providerMetadata(map[string]any{
					"file_id": "provider-file", "provider_name": "openai", "identifier": "uploaded-id",
					"vendor_metadata": map[string]any{"purpose": "assistants"},
				})},
				{Type: "file", URL: "https://example.com/image.png", MediaType: "image/png", ProviderMetadata: providerMetadata(map[string]any{
					"identifier": "image-id", "force_download": "safe",
					"vendor_metadata": map[string]any{"detail": "high"},
				})},
				{Type: "file", URL: "data:text/plain;base64,aGVsbG8=", ProviderMetadata: providerMetadata(map[string]any{
					"identifier": "binary-id", "vendor_metadata": map[string]any{"source": "inline"},
				})},
			}},
			{ID: "assistant", Role: "assistant", Metadata: map[string]any{
				"response": true, "pydantic_ai": map[string]any{"timestamp": "2026-09-04T10:01:00Z"},
			}, Parts: []vercel.UIMessagePart{
				{Type: "text", Text: "answer", ProviderMetadata: providerMetadata(map[string]any{
					"id": "text-id", "provider_name": "openai", "provider_details": map[string]any{"phase": "final"},
				})},
				{Type: "reasoning", Text: "thought", ProviderMetadata: providerMetadata(map[string]any{
					"id": "thinking-id", "signature": "signature", "provider_name": "openai",
				})},
				{Type: "file", URL: "data:image/png;base64,aW1hZ2U=", ProviderMetadata: providerMetadata(map[string]any{
					"id": "file-id", "provider_name": "openai", "provider_details": map[string]any{"kind": "image"},
					"vendor_metadata": map[string]any{"quality": "high"},
				})},
				{Type: "tool-search", ToolCallID: "native", State: "output-available", Input: []byte(`{"query":"go"}`),
					Output: []byte(`{"result":"found"}`), ProviderExecuted: &providerExecuted,
					CallProviderMetadata: providerMetadata(map[string]any{
						"id": "call-id", "provider_name": "openai", "tool_kind": "web-search",
						"provider_details": map[string]any{"status": "complete"},
					}),
				},
				{Type: "tool-future", ToolCallID: "future", State: "input-available", Input: []byte(`{}`), ProviderExecuted: &providerExecuted,
					CallProviderMetadata: providerMetadata(map[string]any{"tool_kind": "future"})},
			}},
			{ID: "user", Role: "user", Parts: []vercel.UIMessagePart{{Type: "text", Text: "continue"}}},
		},
	}
	prompt, history, _, err := vercel.PrepareInput(input, ai.MessageSanitizationOptions{
		AllowUploadedFiles: true, AllowedFileDownloadModes: []ai.FileDownloadMode{ai.FileDownloadSafe},
	})
	if err != nil {
		t.Fatal(err)
	}
	if prompt.Content != "continue" || len(history) != 2 {
		t.Fatalf("unexpected metadata history: prompt=%#v history=%#v", prompt, history)
	}
	request := history[0].(ai.ModelRequest)
	if request.Timestamp.Format(time.RFC3339) != "2026-09-04T10:00:00Z" ||
		request.Metadata["application"].(map[string]any)["nested"] != "value" {
		t.Fatalf("unexpected request metadata: %#v", request)
	}
	contents := request.Parts[0].(ai.UserPromptPart).Contents
	uploaded := contents[0].(ai.UploadedFile)
	image := contents[1].(ai.ImageURL)
	binary := contents[2].(ai.BinaryContent)
	if uploaded.FileID != "provider-file" || uploaded.Identifier != "uploaded-id" ||
		uploaded.VendorMetadata["purpose"] != "assistants" || image.Identifier != "image-id" ||
		image.ForceDownload != ai.FileDownloadSafe || image.VendorMetadata["detail"] != "high" ||
		binary.Identifier != "binary-id" || binary.VendorMetadata["source"] != "inline" {
		t.Fatalf("unexpected file metadata: %#v", contents)
	}
	response := history[1].(ai.ModelResponse)
	if response.Timestamp.Format(time.RFC3339) != "2026-09-04T10:01:00Z" || response.Metadata["response"] != true {
		t.Fatalf("unexpected response metadata: %#v", response)
	}
	text := response.Parts[0].(ai.TextPart)
	thinking := response.Parts[1].(ai.ThinkingPart)
	file := response.Parts[2].(ai.FilePart)
	call := response.Parts[3].(ai.NativeToolCallPart)
	returned := response.Parts[4].(ai.NativeToolReturnPart)
	future := response.Parts[5].(ai.NativeToolCallPart)
	if text.ID != "text-id" || text.ProviderDetails["phase"] != "final" ||
		thinking.Signature != "signature" || file.Content.VendorMetadata["quality"] != "high" ||
		call.ToolKind != ai.ToolPartKindWebSearch || returned.Outcome != ai.ToolReturnOutcomeSuccess ||
		returned.Content.(map[string]any)["result"] != "found" || future.ToolKind != "" {
		t.Fatalf("unexpected response parts: %#v", response.Parts)
	}
	input.Messages[0].Metadata["application"].(map[string]any)["nested"] = "changed"
	if request.Metadata["application"].(map[string]any)["nested"] != "value" {
		t.Fatal("message metadata was not detached")
	}
}

func TestPrepareInputDataParts(t *testing.T) {
	prompt, history, _, err := vercel.PrepareInput(vercel.RequestData{
		Trigger: "submit-message", ID: "chat", Messages: []vercel.UIMessage{
			{ID: "assistant", Role: "assistant", Parts: []vercel.UIMessagePart{
				{Type: string(vercel.ChunkDataCompaction), Data: map[string]any{
					"content": "summary", "id": "compact-1", "provider_name": "openai",
					"provider_details": map[string]any{"encrypted": "value"},
				}},
				{Type: "data-custom", Data: map[string]any{"ignored": true}},
			}},
			{ID: "user", Role: "user", Parts: []vercel.UIMessagePart{
				{Type: string(vercel.ChunkDataToolAvailability), Data: map[string]any{
					"added": []any{"search", "bad name", 1}, "tool_call_id": "reveal-1",
				}},
				{Type: "data-custom", Data: map[string]any{"ignored": true}},
				{Type: "text", Text: "Continue."},
			}},
		},
	}, ai.MessageSanitizationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if prompt.Contents[0].(ai.TextContent).Text != "Continue." || len(history) != 2 {
		t.Fatalf("unexpected data input: prompt=%#v history=%#v", prompt, history)
	}
	compaction := history[0].(ai.ModelResponse).Parts[0].(ai.CompactionPart)
	if compaction.Content != "summary" || compaction.ID != "compact-1" ||
		compaction.ProviderDetails["encrypted"] != "value" {
		t.Fatalf("unexpected compaction: %#v", compaction)
	}
	delta := history[1].(ai.ModelRequest).Parts[0].(ai.ToolAvailabilityDeltaPart)
	if len(delta.ToolsAdded) != 1 || delta.ToolsAdded[0] != "search" || delta.ToolCallID != "reveal-1" {
		t.Fatalf("unexpected availability delta: %#v", delta)
	}

	prompt, history, _, err = vercel.PrepareInput(vercel.RequestData{
		Trigger: "submit-message", ID: "chat", Messages: []vercel.UIMessage{
			{ID: "old", Role: "user", Parts: []vercel.UIMessagePart{{Type: "text", Text: "old"}}},
			{ID: "data", Role: "user", Parts: []vercel.UIMessagePart{{
				Type: string(vercel.ChunkDataToolAvailability), Data: map[string]any{"added": []string{"next"}},
			}}},
		},
	}, ai.MessageSanitizationOptions{})
	if err != nil || prompt.Content != "old" || len(history) != 1 ||
		history[0].(ai.ModelRequest).Parts[0].(ai.ToolAvailabilityDeltaPart).ToolsAdded[0] != "next" {
		t.Fatalf("unexpected data-only message: prompt=%#v history=%#v err=%v", prompt, history, err)
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
		{name: "system text state", input: requestWith(
			vercel.UIMessage{ID: "system", Role: "system", Parts: []vercel.UIMessagePart{{Type: "text", State: "future"}}},
			vercel.UIMessage{ID: "user", Role: "user", Parts: []vercel.UIMessagePart{{Type: "text", Text: "next"}}},
		), match: "unsupported text state"},
		{name: "user mixed text state", input: requestWith(vercel.UIMessage{
			ID: "one", Role: "user", Parts: []vercel.UIMessagePart{
				{Type: "text", State: "future"},
				{Type: "file", URL: "data:text/plain;base64,aGk=", MediaType: "text/plain"},
			},
		}), match: "unsupported text state"},
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
		{name: "dynamic tool name", input: requestWith(vercel.UIMessage{
			ID: "one", Role: "assistant", Parts: []vercel.UIMessagePart{{Type: "dynamic-tool", ToolCallID: "call"}},
		}), match: "requires a toolName"},
		{name: "tool input", input: requestWith(vercel.UIMessage{ID: "one", Role: "assistant", Parts: []vercel.UIMessagePart{{Type: "tool-weather", ToolCallID: "call", State: "input-available", Input: []byte("{")}}}), match: "not valid JSON"},
		{name: "tool state", input: requestWith(vercel.UIMessage{ID: "one", Role: "assistant", Parts: []vercel.UIMessagePart{{
			Type: "tool-weather", ToolCallID: "call", State: "future",
		}}}), match: "unsupported tool state"},
		{name: "tool output", input: requestWith(vercel.UIMessage{ID: "one", Role: "assistant", Parts: []vercel.UIMessagePart{{Type: "tool-weather", ToolCallID: "call", State: "output-available", Output: []byte("{")}}}), match: "decode tool"},
		{name: "assistant part", input: requestWith(vercel.UIMessage{ID: "one", Role: "assistant", Parts: []vercel.UIMessagePart{{Type: "future"}}}), match: "unsupported assistant part"},
		{name: "text state", input: requestWith(vercel.UIMessage{ID: "one", Role: "assistant", Parts: []vercel.UIMessagePart{{
			Type: "text", State: "future",
		}}}), match: "unsupported text state"},
		{name: "reasoning state", input: requestWith(vercel.UIMessage{ID: "one", Role: "assistant", Parts: []vercel.UIMessagePart{{
			Type: "reasoning", State: "future",
		}}}), match: "unsupported reasoning state"},
		{name: "source URL", input: requestWith(vercel.UIMessage{
			ID: "one", Role: "assistant", Parts: []vercel.UIMessagePart{{Type: "source-url"}},
		}), match: "source-url requires"},
		{name: "source document", input: requestWith(vercel.UIMessage{
			ID: "one", Role: "assistant", Parts: []vercel.UIMessagePart{{Type: "source-document"}},
		}), match: "source-document requires"},
		{name: "invalid compaction field", input: requestWith(
			vercel.UIMessage{ID: "assistant", Role: "assistant", Parts: []vercel.UIMessagePart{{
				Type: string(vercel.ChunkDataCompaction), Data: map[string]any{"content": 1},
			}}},
		), match: "requires a user"},
		{name: "invalid compaction provider", input: requestWith(
			vercel.UIMessage{ID: "assistant", Role: "assistant", Parts: []vercel.UIMessagePart{{
				Type: string(vercel.ChunkDataCompaction), Data: map[string]any{"id": "compact"},
			}}},
		), match: "requires a user"},
		{name: "empty compaction", input: requestWith(
			vercel.UIMessage{ID: "assistant", Role: "assistant", Parts: []vercel.UIMessagePart{{
				Type: string(vercel.ChunkDataCompaction), Data: map[string]any{},
			}}},
		), match: "requires a user"},
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
		vercel.UIMessage{ID: "assistant", Role: "assistant", Metadata: map[string]any{
			"pydantic_ai": map[string]any{"timestamp": "2026-09-04T10:00:00Z"},
		}, Parts: []vercel.UIMessagePart{{
			Type: "dynamic-tool", ToolName: "approve", ToolCallID: "call_approve", State: "approval-responded",
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

func TestExternalToolRequestAndResume(t *testing.T) {
	for _, test := range []struct {
		name      string
		state     string
		output    json.RawMessage
		errorText string
		dynamic   bool
	}{
		{name: "output", state: "output-available", output: json.RawMessage(`"remote result"`), dynamic: true},
		{name: "error", state: "output-error", errorText: "worker failed"},
	} {
		t.Run(test.name, func(t *testing.T) {
			agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
			ai.AddExternalTool[struct{}, string, struct{}, string](agent, "remote")
			adapter := vercel.NewAdapter(agent, vercel.Config{})
			input := requestWith(vercel.UIMessage{
				ID: "user", Role: "user", Parts: []vercel.UIMessagePart{{Type: "text", Text: "run"}},
			})
			var finishMetadata map[string]any
			for chunk, err := range adapter.RunStream(context.Background(), input, struct{}{}) {
				if err != nil {
					t.Fatal(err)
				}
				if chunk.Type == vercel.ChunkMessageMetadata {
					finishMetadata = chunk.MessageMetadata
				}
			}
			encoded, err := json.Marshal(finishMetadata)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(encoded, &finishMetadata); err != nil {
				t.Fatal(err)
			}
			part := vercel.UIMessagePart{
				Type: "tool-remote", ToolCallID: "call_remote", State: test.state,
				Input: []byte(`{}`), Output: test.output, ErrorText: test.errorText,
			}
			if test.dynamic {
				part.Type = "dynamic-tool"
				part.ToolName = "remote"
			}
			resume := requestWith(
				vercel.UIMessage{ID: "user", Role: "user", Parts: []vercel.UIMessagePart{{Type: "text", Text: "run"}}},
				vercel.UIMessage{ID: "assistant", Role: "assistant", Metadata: finishMetadata,
					Parts: []vercel.UIMessagePart{part}},
			)
			textSeen := false
			for chunk, err := range adapter.RunStream(context.Background(), resume, struct{}{}) {
				if err != nil {
					t.Fatal(err)
				}
				textSeen = textSeen || chunk.Type == vercel.ChunkTextDelta
			}
			if !textSeen {
				t.Fatal("resumed stream did not produce a model response")
			}
		})
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
		{name: "incomplete external result", input: requestWith(vercel.UIMessage{
			ID: "assistant", Role: "assistant", Metadata: map[string]any{
				"pydantic_ai": map[string]any{"external_tool_call_ids": []string{"call"}},
			}, Parts: []vercel.UIMessagePart{
				{Type: "text", Text: "waiting"},
				{Type: "tool-x", ToolCallID: "call", State: "input-available"},
			},
		})},
		{name: "missing external result", input: requestWith(vercel.UIMessage{
			ID: "assistant", Role: "assistant", Metadata: map[string]any{
				"pydantic_ai": map[string]any{"external_tool_call_ids": []string{"call"}},
			},
		})},
		{name: "invalid external result", input: requestWith(vercel.UIMessage{
			ID: "assistant", Role: "assistant", Metadata: map[string]any{
				"pydantic_ai": map[string]any{"external_tool_call_ids": []string{"call"}},
			}, Parts: []vercel.UIMessagePart{{
				Type: "tool-x", ToolCallID: "call", State: "output-available", Output: []byte(`{`),
			}},
		})},
		{name: "invalid external metadata type", input: requestWith(vercel.UIMessage{
			ID: "assistant", Role: "assistant", Metadata: map[string]any{
				"pydantic_ai": map[string]any{"external_tool_call_ids": "call"},
			},
		})},
		{name: "invalid external metadata value", input: requestWith(vercel.UIMessage{
			ID: "assistant", Role: "assistant", Metadata: map[string]any{
				"pydantic_ai": map[string]any{"external_tool_call_ids": []any{1}},
			},
		})},
		{name: "empty external metadata ID", input: requestWith(vercel.UIMessage{
			ID: "assistant", Role: "assistant", Metadata: map[string]any{
				"pydantic_ai": map[string]any{"external_tool_call_ids": []string{""}},
			},
		})},
		{name: "duplicate external metadata ID", input: requestWith(vercel.UIMessage{
			ID: "assistant", Role: "assistant", Metadata: map[string]any{
				"pydantic_ai": map[string]any{"external_tool_call_ids": []string{"call", "call"}},
			},
		})},
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
