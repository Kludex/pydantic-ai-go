package ai_test

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func TestMessagesRoundTrip(t *testing.T) {
	msgs := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "be nice"},
			ai.UserPromptPart{Content: "weather in SF?"},
		}},
		ai.ModelResponse{
			Parts: []ai.ResponsePart{
				ai.ThinkingPart{Content: "hmm"},
				ai.ToolCallPart{
					ToolName: "get_weather", Args: json.RawMessage(`{"city":"SF"}`), ToolCallID: "c1",
					ToolKind: ai.ToolPartKindToolSearch,
				},
			},
			Usage: ai.Usage{
				Requests: 1, InputTokens: 10, OutputTokens: 5,
				Details: map[string]int{"provider_units": 7},
			},
			ModelName: "test-model",
			Timestamp: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
		},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "get_weather", Content: "sunny", ToolCallID: "c1"},
			ai.RetryPromptPart{Content: "try again", ToolName: "get_weather", ToolCallID: "c1"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "It is sunny."}}},
	}

	data, err := ai.MarshalMessages(msgs)
	if err != nil {
		t.Fatal(err)
	}
	for _, expected := range []string{`"kind":"request"`, `"kind":"response"`, `"part_kind":"system-prompt"`,
		`"part_kind":"user-prompt"`, `"part_kind":"tool-call"`, `"part_kind":"tool-return"`,
		`"part_kind":"retry-prompt"`, `"part_kind":"thinking"`, `"part_kind":"text"`, `"model_name":"test-model"`} {
		if !strings.Contains(string(data), expected) {
			t.Fatalf("expected %s in %s", expected, data)
		}
	}

	back, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(back) != len(msgs) {
		t.Fatalf("expected %d messages, got %d", len(msgs), len(back))
	}
	resp := back[1].(ai.ModelResponse)
	if resp.ModelName != "test-model" || resp.Usage.InputTokens != 10 ||
		resp.Usage.Details["provider_units"] != 7 {
		t.Fatalf("unexpected response %+v", resp)
	}
	if resp.ToolCalls()[0].ToolName != "get_weather" ||
		resp.ToolCalls()[0].ToolKind != ai.ToolPartKindToolSearch {
		t.Fatal("tool call lost in round trip")
	}
	req := back[2].(ai.ModelRequest)
	if req.Parts[0].(ai.ToolReturnPart).Content != "sunny" {
		t.Fatal("tool return lost in round trip")
	}
	if back[3].(ai.ModelResponse).Text() != "It is sunny." {
		t.Fatal("text lost in round trip")
	}
}

func TestInterruptedRequestRoundTrip(t *testing.T) {
	messages := []ai.ModelMessage{ai.ModelRequest{
		State: ai.RequestStateInterrupted,
		Parts: []ai.RequestPart{ai.ToolReturnPart{
			ToolName: "work", ToolCallID: "call", Content: "interrupted",
			Outcome:  ai.ToolReturnOutcomeInterrupted,
			Metadata: map[string]any{ai.SynthesizedToolReturnMetadataKey: true},
		}},
	}}
	data, err := ai.MarshalMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"state":"interrupted"`) ||
		!strings.Contains(string(data), `"outcome":"interrupted"`) ||
		!strings.Contains(string(data), `"pydantic_ai_synthesized_tool_return":true`) {
		t.Fatalf("interrupted state was not serialized: %s", data)
	}
	decoded, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	request := decoded[0].(ai.ModelRequest)
	part := request.Parts[0].(ai.ToolReturnPart)
	if request.State != ai.RequestStateInterrupted || part.Outcome != ai.ToolReturnOutcomeInterrupted ||
		part.Metadata[ai.SynthesizedToolReturnMetadataKey] != true {
		t.Fatalf("interrupted state was not restored: %+v %+v", request, part)
	}
}

func TestRetryPromptStructuredErrorFailures(t *testing.T) {
	_, err := ai.MarshalMessages([]ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.RetryPromptPart{Errors: []ai.ValidationError{{Type: "custom", Input: make(chan int)}}},
	}}})
	if err == nil || !strings.Contains(err.Error(), "ai: marshal retry validation errors") {
		t.Fatalf("unexpected marshal error: %v", err)
	}

	_, err = ai.UnmarshalMessages([]byte(`[{"kind":"request","parts":[{"part_kind":"retry-prompt","content":{}}]}]`))
	if err == nil || !strings.Contains(err.Error(), "ai: unmarshal retry validation errors") {
		t.Fatalf("unexpected unmarshal error: %v", err)
	}
}

func TestRetryPromptModelResponseFormatting(t *testing.T) {
	plain := ai.RetryPromptPart{Content: "retry", ToolName: "tool"}.ModelResponse()
	if plain != "retry\n\nFix the errors and try again." {
		t.Fatalf("unexpected tool retry formatting: %q", plain)
	}
	plain = ai.RetryPromptPart{Content: "retry"}.ModelResponse()
	if plain != "Validation feedback:\nretry\n\nFix the errors and try again." {
		t.Fatalf("unexpected plain validation feedback: %q", plain)
	}
	structured := ai.RetryPromptPart{Errors: []ai.ValidationError{{
		Type: "required", Message: "missing", Input: map[string]any{"other": true},
		Context: map[string]any{"hidden": true},
	}}}.ModelResponse()
	if strings.Contains(structured, "other") || strings.Contains(structured, "hidden") ||
		!strings.Contains(structured, "1 validation error") {
		t.Fatalf("unexpected structured retry formatting: %s", structured)
	}
}

func TestUnmarshalUnknownKind(t *testing.T) {
	if _, err := ai.UnmarshalMessages([]byte(`[{"kind":"mystery"}]`)); err == nil {
		t.Fatal("expected error")
	}
}

func TestMarshalUnmarshalEdgeCases(t *testing.T) {
	if _, err := ai.UnmarshalMessages([]byte(`not json`)); err == nil {
		t.Fatal("expected error for invalid JSON")
	}
	if _, err := ai.UnmarshalMessages([]byte(`[42]`)); err == nil {
		t.Fatal("expected error for non-object message")
	}
	if _, err := ai.UnmarshalMessages([]byte(`[{"kind":"request","parts":[{"part_kind":"mystery"}]}]`)); err == nil {
		t.Fatal("expected error for unknown request part kind")
	}
	if _, err := ai.UnmarshalMessages([]byte(`[{"kind":"response","parts":[{"part_kind":"mystery"}]}]`)); err == nil {
		t.Fatal("expected error for unknown response part kind")
	}
	if _, err := ai.UnmarshalMessages([]byte(`[{"kind":"request","parts":42}]`)); err == nil {
		t.Fatal("expected error for malformed request")
	}
	if _, err := ai.UnmarshalMessages([]byte(`[{"kind":"response","parts":42}]`)); err == nil {
		t.Fatal("expected error for malformed response")
	}
	if _, err := ai.UnmarshalMessages([]byte(`[{"kind":"request","parts":[{"part_kind":"tool-return","content":42,"tool_name":"t"}]}]`)); err != nil {
		t.Fatalf("non-string tool return content should unmarshal: %v", err)
	}

	if _, err := ai.MarshalMessages([]ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.ToolReturnPart{ToolName: "t", Content: make(chan int)},
	}}}); err == nil {
		t.Fatal("expected error for unmarshallable tool return content")
	}
}

func TestUnmarshalUserContentForms(t *testing.T) {
	msgs, err := ai.UnmarshalMessages([]byte(`[{"kind":"request","parts":[{"part_kind":"user-prompt","content":[
		{"kind":"text-content","content":"look at this"},
		{"kind":"image-url","url":"https://example.com/cat.png"},
		{"kind":"binary","data":"aGk=","media_type":"image/png"},
		{"kind":"cache-point","ttl":"1h"},
		{"kind":"uploaded-file","file_id":"file-1","provider_name":"anthropic","media_type":"text/csv","identifier":"sales","vendor_metadata":{"purpose":"data"}}
	]}]}]`))
	if err != nil {
		t.Fatal(err)
	}
	part := msgs[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart)
	if len(part.Contents) != 5 {
		t.Fatalf("unexpected contents %+v", part.Contents)
	}
	if part.Contents[1].(ai.ImageURL).URL != "https://example.com/cat.png" {
		t.Fatalf("unexpected image %+v", part.Contents[1])
	}
	if string(part.Contents[2].(ai.BinaryContent).Data) != "hi" {
		t.Fatalf("unexpected binary %+v", part.Contents[2])
	}
	if part.Contents[3].(ai.CachePoint).TTL != ai.CachePointTTL1Hour {
		t.Fatalf("unexpected cache point %+v", part.Contents[3])
	}
	uploaded := part.Contents[4].(ai.UploadedFile)
	if uploaded.FileID != "file-1" || uploaded.ProviderName != "anthropic" ||
		uploaded.VendorMetadata["purpose"] != "data" {
		t.Fatalf("unexpected uploaded file %+v", uploaded)
	}

	if _, err := ai.UnmarshalMessages([]byte(`[{"kind":"request","parts":[{"part_kind":"user-prompt","content":42}]}]`)); err == nil {
		t.Fatal("expected error for invalid user content")
	}
	if _, err := ai.UnmarshalMessages([]byte(`[{"kind":"request","parts":[{"part_kind":"user-prompt","content":[{"kind":"mystery"}]}]}]`)); err == nil {
		t.Fatal("expected error for unknown content kind")
	}
}

func TestMarshalUserContentRoundTrip(t *testing.T) {
	msgs := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
		ai.TextContent{Text: "what is this?"},
		ai.ImageURL{URL: "https://example.com/cat.png"},
		ai.BinaryContent{Data: []byte("hi"), MediaType: "image/png"},
		ai.CachePoint{},
		ai.UploadedFile{
			FileID: "file-1", ProviderName: "openai", MediaType: "text/csv", Identifier: "sales",
			VendorMetadata: map[string]any{"purpose": "data"},
		},
	}}}}}
	data, err := ai.MarshalMessages(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"kind":"cache-point","ttl":"5m"`) {
		t.Fatalf("cache point is not wire compatible: %s", data)
	}
	back, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	part := back[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart)
	if len(part.Contents) != 5 || part.Contents[3].(ai.CachePoint).TTL != ai.CachePointTTL5Minutes ||
		part.Contents[4].(ai.UploadedFile).Identifier != "sales" {
		t.Fatalf("round trip lost contents: %+v", part.Contents)
	}
	invalid := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{
		Contents: []ai.UserContent{ai.TextContent{Text: "first"}, ai.CachePoint{TTL: "1d"}},
	}}}}
	if _, err := ai.MarshalMessages(invalid); err == nil || !strings.Contains(err.Error(), "invalid cache point TTL") {
		t.Fatalf("unexpected invalid cache-point marshal error: %v", err)
	}
	if _, err := ai.UnmarshalMessages([]byte(`[{"kind":"request","parts":[{"part_kind":"user-prompt","content":[{"kind":"cache-point","ttl":"1d"}]}]}]`)); err == nil || !strings.Contains(err.Error(), "invalid cache point TTL") {
		t.Fatalf("unexpected invalid cache-point unmarshal error: %v", err)
	}
}

func TestFilePartSerialization(t *testing.T) {
	messages := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.FilePart{
		Content: ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"},
		ID:      "file", ProviderName: "openai", ProviderDetails: map[string]any{"source": "code"},
	}}}}
	data, err := ai.MarshalMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	file := decoded[0].(ai.ModelResponse).Parts[0].(ai.FilePart)
	if string(file.Content.Data) != "image" || file.Content.MediaType != "image/png" ||
		file.ID != "file" || file.ProviderDetails["source"] != "code" {
		t.Fatalf("unexpected file round trip: %+v", file)
	}
	if _, err := ai.UnmarshalMessages([]byte(`[{"kind":"response","parts":[{"part_kind":"file","content":"bad"}]}]`)); err == nil || !strings.Contains(err.Error(), "unmarshal file content") {
		t.Fatalf("unexpected malformed file error: %v", err)
	}
}

func TestNativeToolReturnSerializationErrors(t *testing.T) {
	_, err := ai.MarshalMessages([]ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.NativeToolReturnPart{ToolName: "native", Content: make(chan int)},
	}}})
	if err == nil || !strings.Contains(err.Error(), "marshal native tool return content") {
		t.Fatalf("unexpected marshal error: %v", err)
	}
	_, err = ai.UnmarshalMessages([]byte(`[{"kind":"response","parts":[{"part_kind":"builtin-tool-return","tool_name":"native"}]}]`))
	if err == nil || !strings.Contains(err.Error(), "unmarshal native tool return content") {
		t.Fatalf("unexpected unmarshal error: %v", err)
	}
}
