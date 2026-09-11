package ai_test

import (
	"encoding/json"
	"fmt"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func TestFileContentStandardJSONRoundTrip(t *testing.T) {
	values := []any{
		ai.ImageURL{URL: "https://example.com/image.png", ForceDownload: ai.FileDownloadSafe},
		ai.VideoURL{URL: "https://example.com/video.mp4", ForceDownload: ai.FileDownloadAllowLocal},
		ai.AudioURL{URL: "https://example.com/audio.mp3"},
		ai.DocumentURL{URL: "https://example.com/document.pdf"},
		ai.UploadedFile{FileID: "file-1", ProviderName: "openai", MediaType: "application/pdf"},
		ai.BinaryContent{Data: []byte{0xfb, 0xff}, MediaType: "application/pdf"},
	}
	for _, value := range values {
		t.Run(fmt.Sprintf("%T", value), func(t *testing.T) {
			encoded, err := json.Marshal(value)
			if err != nil {
				t.Fatal(err)
			}
			var wire map[string]any
			if err := json.Unmarshal(encoded, &wire); err != nil {
				t.Fatal(err)
			}
			var decoded any
			var expectedKind string
			switch value.(type) {
			case ai.ImageURL:
				decoded, expectedKind = &ai.ImageURL{}, "image-url"
			case ai.VideoURL:
				decoded, expectedKind = &ai.VideoURL{}, "video-url"
			case ai.AudioURL:
				decoded, expectedKind = &ai.AudioURL{}, "audio-url"
			case ai.DocumentURL:
				decoded, expectedKind = &ai.DocumentURL{}, "document-url"
			case ai.UploadedFile:
				decoded, expectedKind = &ai.UploadedFile{}, "uploaded-file"
			case ai.BinaryContent:
				decoded, expectedKind = &ai.BinaryContent{}, "binary"
			}
			if wire["kind"] != expectedKind || expectedKind == "binary" && wire["data"] != "-_8=" {
				t.Fatalf("unexpected wire shape: %s", encoded)
			}
			if err := json.Unmarshal(encoded, decoded); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestUploadedFileResolvedMetadata(t *testing.T) {
	explicit := ai.UploadedFile{FileID: "file", MediaType: "text/custom", Identifier: "report"}
	if explicit.ResolvedMediaType() != "text/custom" || explicit.ResolvedIdentifier() != "report" {
		t.Fatalf("explicit metadata was not preserved: %+v", explicit)
	}
	inferred := ai.UploadedFile{FileID: "s3://bucket/report.PDF?version=1"}
	if inferred.ResolvedMediaType() != "application/pdf" || inferred.ResolvedIdentifier() == "" ||
		inferred.ResolvedIdentifier() != (ai.UploadedFile{FileID: inferred.FileID}).ResolvedIdentifier() {
		t.Fatalf("metadata was not inferred: type=%q id=%q", inferred.ResolvedMediaType(), inferred.ResolvedIdentifier())
	}
	for _, fileID := range []string{"opaque-file-id", "%"} {
		file := ai.UploadedFile{FileID: fileID}
		if mediaType := file.ResolvedMediaType(); mediaType != "application/octet-stream" {
			t.Fatalf("unexpected fallback media type %q", mediaType)
		}
		encoded, err := json.Marshal(file)
		if err != nil {
			t.Fatal(err)
		}
		var wire map[string]any
		if err := json.Unmarshal(encoded, &wire); err != nil {
			t.Fatal(err)
		}
		if wire["media_type"] != "application/octet-stream" || wire["identifier"] != file.ResolvedIdentifier() {
			t.Fatalf("uploaded file defaults were not serialized: %s", encoded)
		}
	}
}

func TestFileContentStandardJSONErrors(t *testing.T) {
	for name, value := range map[string]any{
		"image":    ai.ImageURL{URL: "https://example.com/image.png", ForceDownload: "invalid"},
		"video":    ai.VideoURL{URL: "https://example.com/video.mp4", ForceDownload: "invalid"},
		"audio":    ai.AudioURL{URL: "https://example.com/audio.mp3", ForceDownload: "invalid"},
		"document": ai.DocumentURL{URL: "https://example.com/document.pdf", ForceDownload: "invalid"},
	} {
		t.Run("marshal "+name, func(t *testing.T) {
			if _, err := json.Marshal(value); err == nil {
				t.Fatal("expected marshal error")
			}
		})
	}
	for name, destination := range map[string]any{
		"image":    &ai.ImageURL{},
		"video":    &ai.VideoURL{},
		"audio":    &ai.AudioURL{},
		"document": &ai.DocumentURL{},
		"uploaded": &ai.UploadedFile{},
		"binary":   &ai.BinaryContent{},
	} {
		t.Run("unmarshal "+name, func(t *testing.T) {
			if err := json.Unmarshal([]byte(`{"kind":"wrong"}`), destination); err == nil {
				t.Fatal("expected content kind error")
			}
		})
	}
	if err := (&ai.VideoURL{}).UnmarshalJSON([]byte(`{`)); err == nil {
		t.Fatal("expected malformed JSON error")
	}
	var legacy ai.BinaryContent
	if err := json.Unmarshal(
		[]byte(`{"kind":"binary","data":"+/8=","media_type":"application/pdf"}`), &legacy,
	); err != nil || len(legacy.Data) != 2 || legacy.Data[0] != 0xfb || legacy.Data[1] != 0xff {
		t.Fatalf("legacy standard base64 was not accepted: content=%+v err=%v", legacy, err)
	}
	for _, data := range []string{
		`{"kind":"binary","data":"***","media_type":"application/pdf"}`,
		`{"kind":"binary","data":42,"media_type":"application/pdf"}`,
	} {
		if err := json.Unmarshal([]byte(data), &ai.BinaryContent{}); err == nil {
			t.Fatal("expected invalid binary encoding error")
		}
	}
	var image ai.ImageURL
	if err := json.Unmarshal([]byte(`{"url":"https://example.com/image.png"}`), &image); err != nil ||
		image.URL != "https://example.com/image.png" {
		t.Fatalf("missing kind was not inferred: image=%+v err=%v", image, err)
	}
}
