package ai_test

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestFileContentProperties(t *testing.T) {
	for rawURL, want := range map[string]string{
		"https://example.com/photo.JPG?download=1": "image/jpeg",
		"https://example.com/photo.png":            "image/png",
	} {
		mediaType, err := (ai.ImageURL{URL: rawURL}).ResolvedMediaType()
		if err != nil || mediaType != want {
			t.Fatalf("image media type for %q = %q, %v; want %q", rawURL, mediaType, err, want)
		}
	}
	for rawURL, want := range map[string]string{
		"https://example.com/speech.aac":  "audio/aac",
		"https://example.com/speech.m4a":  "audio/mp4a-latm",
		"https://example.com/speech.opus": "audio/ogg",
		"https://example.com/speech.wav":  "audio/wav",
	} {
		mediaType, err := (ai.AudioURL{URL: rawURL}).ResolvedMediaType()
		if err != nil || mediaType != want {
			t.Fatalf("audio media type for %q = %q, %v; want %q", rawURL, mediaType, err, want)
		}
	}
	for rawURL, want := range map[string]string{
		"https://example.com/report.docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
		"https://example.com/report.md":   "text/markdown",
		"https://example.com/report.pdf":  "application/pdf",
	} {
		mediaType, err := (ai.DocumentURL{URL: rawURL}).ResolvedMediaType()
		if err != nil || mediaType != want {
			t.Fatalf("document media type for %q = %q, %v; want %q", rawURL, mediaType, err, want)
		}
	}

	image := ai.ImageURL{URL: "https://example.com/photo.jpg"}
	audio := ai.AudioURL{URL: "https://example.com/speech.mp3"}
	document := ai.DocumentURL{URL: "https://example.com/report.pdf"}
	binary := ai.BinaryContent{Data: []byte("document")}
	if image.ResolvedIdentifier() != "3b0283" || audio.ResolvedIdentifier() != "4e3096" ||
		document.ResolvedIdentifier() != "a5f6ba" || binary.ResolvedIdentifier() != "4f8278" {
		t.Fatalf(
			"unexpected identifiers: image=%q audio=%q document=%q binary=%q",
			image.ResolvedIdentifier(), audio.ResolvedIdentifier(), document.ResolvedIdentifier(), binary.ResolvedIdentifier(),
		)
	}
	if (ai.ImageURL{Identifier: "image"}).ResolvedIdentifier() != "image" ||
		(ai.AudioURL{Identifier: "audio"}).ResolvedIdentifier() != "audio" ||
		(ai.DocumentURL{Identifier: "document"}).ResolvedIdentifier() != "document" ||
		(ai.BinaryContent{Identifier: "binary"}).ResolvedIdentifier() != "binary" {
		t.Fatal("explicit identifiers were not preserved")
	}
	for name, resolve := range map[string]func() (string, error){
		"image":    (ai.ImageURL{URL: "://invalid"}).ResolvedMediaType,
		"audio":    (ai.AudioURL{URL: "https://example.com/speech"}).ResolvedMediaType,
		"document": (ai.DocumentURL{URL: "https://example.com/report"}).ResolvedMediaType,
	} {
		if _, err := resolve(); err == nil {
			t.Fatalf("%s media type was inferred", name)
		}
	}
	if mediaType, err := (ai.DocumentURL{URL: "invalid", MediaType: "application/custom"}).ResolvedMediaType(); err != nil || mediaType != "application/custom" {
		t.Fatalf("explicit media type = %q, %v", mediaType, err)
	}
}

func TestUpstreamFileContentFixtureRoundTrips(t *testing.T) {
	data, err := os.ReadFile("testdata/messages/upstream_file_content.json")
	if err != nil {
		t.Fatal(err)
	}
	messages, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	contents := messages[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Contents
	image := contents[0].(ai.ImageURL)
	audio := contents[1].(ai.AudioURL)
	document := contents[2].(ai.DocumentURL)
	binary := contents[3].(ai.BinaryContent)
	if image.ForceDownload != ai.FileDownloadSafe || image.MediaType != "image/jpeg" ||
		image.Identifier != "3b0283" || image.VendorMetadata["detail"] != "high" ||
		audio.MediaType != "audio/mpeg" || audio.Identifier != "4e3096" ||
		document.ForceDownload != ai.FileDownloadAllowLocal || document.Identifier != "report" ||
		string(binary.Data) != "document" || binary.Identifier != "4f8278" ||
		binary.VendorMetadata["name"] != "report.pdf" {
		t.Fatalf("unexpected file fixture: %+v", contents)
	}
	responseFile := messages[1].(ai.ModelResponse).Parts[0].(ai.FilePart)
	if string(responseFile.Content.Data) != "file" || responseFile.Content.Identifier != "971c41" ||
		responseFile.Content.VendorMetadata["name"] != "generated.pdf" || responseFile.ID != "file-part" ||
		responseFile.ProviderName != "provider" || responseFile.ProviderDetails["source"] != "generated" {
		t.Fatalf("unexpected response file fixture: %+v", responseFile)
	}
	encoded, err := ai.MarshalMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	var wire []map[string]any
	if err := json.Unmarshal(encoded, &wire); err != nil {
		t.Fatal(err)
	}
	wireContents := wire[0]["parts"].([]any)[0].(map[string]any)["content"].([]any)
	if wireContents[0].(map[string]any)["force_download"] != true ||
		wireContents[1].(map[string]any)["force_download"] != false ||
		wireContents[2].(map[string]any)["force_download"] != "allow-local" ||
		wireContents[3].(map[string]any)["identifier"] != "4f8278" {
		t.Fatalf("unexpected serialized file content: %s", encoded)
	}
}

func TestFileContentRunValuesAreDetached(t *testing.T) {
	metadata := func(kind string) map[string]any {
		return map[string]any{"nested": map[string]any{"kind": kind}}
	}
	textMetadata := metadata("text")
	imageMetadata := metadata("image")
	audioMetadata := metadata("audio")
	documentMetadata := metadata("document")
	binaryMetadata := metadata("binary")
	binaryData := []byte("data")
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		contents := messages[len(messages)-1].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Contents
		contents[0].(ai.TextContent).Metadata.(map[string]any)["nested"].(map[string]any)["kind"] = "changed"
		contents[1].(ai.ImageURL).VendorMetadata["nested"].(map[string]any)["kind"] = "changed"
		contents[2].(ai.AudioURL).VendorMetadata["nested"].(map[string]any)["kind"] = "changed"
		contents[3].(ai.DocumentURL).VendorMetadata["nested"].(map[string]any)["kind"] = "changed"
		binary := contents[4].(ai.BinaryContent)
		binary.Data[0] = 'X'
		binary.VendorMetadata["nested"].(map[string]any)["kind"] = "changed"
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	result, err := ai.NewAgent[struct{}, string](model).RunParts(t.Context(), []ai.UserContent{
		ai.TextContent{Text: "inspect", Metadata: textMetadata},
		ai.ImageURL{URL: "https://example.com/image.png", VendorMetadata: imageMetadata},
		ai.AudioURL{URL: "https://example.com/audio.mp3", VendorMetadata: audioMetadata},
		ai.DocumentURL{URL: "https://example.com/document.pdf", VendorMetadata: documentMetadata},
		ai.BinaryContent{Data: binaryData, MediaType: "application/pdf", VendorMetadata: binaryMetadata},
	}, struct{}{})
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected result=%+v err=%v", result, err)
	}
	for kind, value := range map[string]map[string]any{
		"text": textMetadata, "image": imageMetadata, "audio": audioMetadata,
		"document": documentMetadata, "binary": binaryMetadata,
	} {
		if value["nested"].(map[string]any)["kind"] != kind {
			t.Fatalf("caller %s metadata was mutated", kind)
		}
	}
	if string(binaryData) != "data" {
		t.Fatal("caller binary data was mutated")
	}
}

func TestBinaryResponseContentIsDetachedAndSerializable(t *testing.T) {
	metadata := map[string]any{"nested": map[string]any{"value": "original"}}
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.FilePart{Content: ai.BinaryContent{
				Data: []byte("file"), MediaType: "application/pdf", VendorMetadata: metadata,
			}},
			ai.TextPart{Content: "done"},
		}}, nil
	})
	response, err := ai.RequestModel(t.Context(), model, nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := ai.MarshalMessages([]ai.ModelMessage{*response})
	if err != nil {
		t.Fatal(err)
	}
	file := response.Parts[0].(ai.FilePart)
	file.Content.Data[0] = 'X'
	file.Content.VendorMetadata["nested"].(map[string]any)["value"] = "changed"
	if metadata["nested"].(map[string]any)["value"] != "original" {
		t.Fatal("response file metadata was not detached")
	}
	messages, err := ai.UnmarshalMessages(encoded)
	if err != nil {
		t.Fatal(err)
	}
	roundTripped := messages[0].(ai.ModelResponse).Parts[0].(ai.FilePart).Content
	if string(roundTripped.Data) != "file" || roundTripped.MediaType != "application/pdf" ||
		roundTripped.Identifier != "971c41" ||
		roundTripped.VendorMetadata["nested"].(map[string]any)["value"] != "original" {
		t.Fatalf("unexpected response file round trip: %+v", roundTripped)
	}
}

func TestFileContentSerializationErrors(t *testing.T) {
	for name, content := range map[string]ai.UserContent{
		"image media":    ai.ImageURL{URL: "https://example.com/image"},
		"image mode":     ai.ImageURL{URL: "https://example.com/image.png", ForceDownload: "invalid"},
		"audio media":    ai.AudioURL{URL: "https://example.com/audio"},
		"audio mode":     ai.AudioURL{URL: "https://example.com/audio.mp3", ForceDownload: "invalid"},
		"document media": ai.DocumentURL{URL: "https://example.com/document"},
		"document mode":  ai.DocumentURL{URL: "https://example.com/document.pdf", ForceDownload: "invalid"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := ai.MarshalMessages([]ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Contents: []ai.UserContent{content}},
			}}})
			if err == nil {
				t.Fatal("invalid file content serialized")
			}
		})
	}
	for name, kind := range map[string]string{
		"image": "image-url", "audio": "audio-url", "document": "document-url",
	} {
		t.Run(name, func(t *testing.T) {
			for _, forceDownload := range []string{`1`, `"invalid"`} {
				data := `[{"kind":"request","parts":[{"part_kind":"user-prompt","content":[{"kind":"` + kind +
					`","url":"https://example.com/file","media_type":"application/test","force_download":` +
					forceDownload + `}]}]}]`
				if _, err := ai.UnmarshalMessages([]byte(data)); err == nil {
					t.Fatal("invalid force_download decoded")
				}
			}
		})
	}
}
