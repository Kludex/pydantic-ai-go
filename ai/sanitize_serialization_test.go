package ai_test

import (
	"reflect"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func TestUnmarshalMessagesNarrowsNestedFileContentForSanitization(t *testing.T) {
	serialized := []byte(`[
		{"kind":"request","parts":[{"part_kind":"tool-return","tool_name":"lookup","tool_call_id":"1","content":[
			{"kind":"image-url","url":"https://example.com/image.png","media_type":"image/png","identifier":"image","force_download":"allow-local"},
			{"nested":{"kind":"document-url","url":"s3://bucket/file.pdf","media_type":"application/pdf","identifier":"document","force_download":false}},
			{"kind":"uploaded-file","file_id":"file-1","provider_name":"openai","media_type":"application/pdf"},
			{"kind":"binary","data":"ZGF0YQ==","media_type":"application/pdf","identifier":"binary"},
			{"kind":"image-url","note":"ordinary user data"}
		]}]},
		{"kind":"response","parts":[{"part_kind":"builtin-tool-return","tool_name":"search","tool_call_id":"2","content":{"kind":"audio-url","url":"https://example.com/audio.mp3","media_type":"audio/mpeg","identifier":"audio","force_download":true}}]}
	]`)
	messages, err := ai.UnmarshalMessages(serialized)
	if err != nil {
		t.Fatal(err)
	}
	content := messages[0].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart).Content.([]any)
	if _, ok := content[0].(ai.ImageURL); !ok {
		t.Fatalf("image was not narrowed: %T", content[0])
	}
	if _, ok := content[1].(map[string]any)["nested"].(ai.DocumentURL); !ok {
		t.Fatalf("document was not narrowed: %T", content[1].(map[string]any)["nested"])
	}
	if _, ok := content[2].(ai.UploadedFile); !ok {
		t.Fatalf("uploaded file was not narrowed: %T", content[2])
	}
	if binary, ok := content[3].(ai.BinaryContent); !ok || string(binary.Data) != "data" {
		t.Fatalf("binary was not narrowed: %#v", content[3])
	}
	if _, ok := content[4].(map[string]any); !ok {
		t.Fatalf("colliding user map was narrowed: %T", content[4])
	}
	if _, ok := messages[1].(ai.ModelResponse).Parts[0].(ai.NativeToolReturnPart).Content.(ai.AudioURL); !ok {
		t.Fatal("native return audio was not narrowed")
	}

	sanitized, report, err := ai.SanitizeMessages(messages, ai.MessageSanitizationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	sanitizedContent := sanitized[0].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart).Content.([]any)
	if len(sanitizedContent) != 4 || sanitizedContent[0].(ai.ImageURL).ForceDownload != ai.FileDownloadNever ||
		!reflect.DeepEqual(sanitizedContent[1], map[string]any{}) ||
		!reflect.DeepEqual(sanitizedContent[2], ai.BinaryContent{
			Data: []byte("data"), MediaType: "application/pdf", Identifier: "binary",
		}) || !reflect.DeepEqual(sanitizedContent[3], map[string]any{
		"kind": "image-url", "note": "ordinary user data",
	}) {
		t.Fatalf("unexpected sanitized content: %#v", sanitizedContent)
	}
	if !reflect.DeepEqual(report.DroppedFileURLSchemes, []string{"s3"}) ||
		!reflect.DeepEqual(report.DroppedUploadedFileProviders, []string{"openai"}) ||
		!reflect.DeepEqual(report.ResetFileDownloadModes, []ai.FileDownloadMode{
			ai.FileDownloadAllowLocal, ai.FileDownloadSafe,
		}) {
		t.Fatalf("unexpected report: %+v", report)
	}

	roundTrip, err := ai.MarshalMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	roundTripped, err := ai.UnmarshalMessages(roundTrip)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(roundTripped, messages) {
		t.Fatalf("nested content did not round trip:\nwant: %#v\ngot:  %#v", messages, roundTripped)
	}
}

func TestToolReturnURLWithoutMediaTypeRemainsApplicationData(t *testing.T) {
	serialized := []byte(`[{"kind":"request","parts":[{"part_kind":"tool-return","content":{
		"kind":"image-url","url":"https://example.com/report","force_download":"invalid"
	}}]}]`)
	messages, err := ai.UnmarshalMessages(serialized)
	if err != nil {
		t.Fatal(err)
	}
	content := messages[0].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart).Content
	if _, ok := content.(map[string]any); !ok {
		t.Fatalf("URL-shaped application data was narrowed without a media type: %T", content)
	}
	if _, err := ai.MarshalMessages(messages); err != nil {
		t.Fatalf("application data no longer round trips: %v", err)
	}
}

func TestTextContentMetadataRoundTripsWithoutBecomingModelText(t *testing.T) {
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{
		Contents: []ai.UserContent{ai.TextContent{
			Text: "inspect", Metadata: map[string]any{"source": "client"},
		}},
	}}}}
	encoded, err := ai.MarshalMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	roundTripped, err := ai.UnmarshalMessages(encoded)
	if err != nil {
		t.Fatal(err)
	}
	text := roundTripped[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Contents[0].(ai.TextContent)
	if text.Text != "inspect" || text.Metadata.(map[string]any)["source"] != "client" {
		t.Fatalf("unexpected text content: %+v", text)
	}
}

func TestUnmarshalMessagesRejectsInvalidMultimodalScalar(t *testing.T) {
	serialized := []byte(`[{"kind":"request","parts":[{"part_kind":"user-prompt","content":[42]}]}]`)
	if _, err := ai.UnmarshalMessages(serialized); err == nil {
		t.Fatal("expected invalid user content error")
	}
}

func TestUnmarshalMessagesRejectsInvalidNestedFileContent(t *testing.T) {
	serializedValues := [][]byte{
		[]byte(`[{"kind":"request","parts":[{"part_kind":"tool-return","content":[{
			"nested":{"kind":"image-url","url":"https://example.com/image.png","media_type":"image/png","force_download":"invalid"}
		}]}]}]`),
		[]byte(`[{"kind":"response","parts":[{"part_kind":"builtin-tool-return","content":{
			"kind":"audio-url","url":"https://example.com/audio.mp3","media_type":"audio/mpeg","force_download":"invalid"
		}}]}]`),
	}
	for _, serialized := range serializedValues {
		if _, err := ai.UnmarshalMessages(serialized); err == nil {
			t.Fatal("expected nested file validation error")
		}
	}
}
