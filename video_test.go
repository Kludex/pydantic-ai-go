package ai_test

import (
	"context"
	"os"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestVideoURLProperties(t *testing.T) {
	for rawURL, want := range map[string]string{
		"https://example.com/a.3gp":  "video/3gpp",
		"https://example.com/a.flv":  "video/x-flv",
		"https://example.com/a.mkv":  "video/x-matroska",
		"https://example.com/a.mov":  "video/quicktime",
		"https://example.com/a.MP4":  "video/mp4",
		"https://example.com/a.mpeg": "video/mpeg",
		"https://example.com/a.mpg":  "video/mpeg",
		"https://example.com/a.webm": "video/webm",
		"https://example.com/a.wmv":  "video/x-ms-wmv",
		"https://youtu.be/example":   "video/mp4",
	} {
		mediaType, err := (ai.VideoURL{URL: rawURL}).ResolvedMediaType()
		if err != nil || mediaType != want {
			t.Fatalf("media type for %q = %q, %v; want %q", rawURL, mediaType, err, want)
		}
	}
	explicit := ai.VideoURL{URL: "not a URL", MediaType: "video/custom", Identifier: "video"}
	if mediaType, err := explicit.ResolvedMediaType(); err != nil || mediaType != "video/custom" ||
		explicit.ResolvedIdentifier() != "video" {
		t.Fatalf("unexpected explicit video properties: media=%q identifier=%q err=%v", mediaType, explicit.ResolvedIdentifier(), err)
	}
	video := ai.VideoURL{URL: "https://example.com/video.mp4"}
	if video.ResolvedIdentifier() != "8cb95e" ||
		!(ai.VideoURL{URL: "https://www.youtube.com/watch?v=1"}).IsYouTube() {
		t.Fatalf("unexpected video identity or YouTube detection: %q", video.ResolvedIdentifier())
	}
	for _, rawURL := range []string{
		"https://music.youtube.com/watch?v=1", "https://youtube.com.example/video", "://invalid",
	} {
		if (ai.VideoURL{URL: rawURL}).IsYouTube() {
			t.Fatalf("URL %q was classified as YouTube", rawURL)
		}
	}
	for _, rawURL := range []string{"https://example.com/video", "://invalid"} {
		if _, err := (ai.VideoURL{URL: rawURL}).ResolvedMediaType(); err == nil {
			t.Fatalf("media type for %q was inferred", rawURL)
		}
	}
	for _, mode := range []ai.FileDownloadMode{
		ai.FileDownloadNever, ai.FileDownloadSafe, ai.FileDownloadAllowLocal,
	} {
		if err := mode.Validate(); err != nil {
			t.Fatalf("valid download mode %q failed: %v", mode, err)
		}
	}
	if err := (ai.FileDownloadMode("invalid")).Validate(); err == nil {
		t.Fatal("invalid download mode succeeded")
	}
}

func TestUpstreamVideoURLFixtureRoundTrips(t *testing.T) {
	data, err := os.ReadFile("testdata/messages/upstream_video_url.json")
	if err != nil {
		t.Fatal(err)
	}
	messages, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	contents := messages[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Contents
	first := contents[0].(ai.VideoURL)
	second := contents[1].(ai.VideoURL)
	if first.URL != "https://example.com/video.mp4" || first.MediaType != "video/mp4" ||
		first.Identifier != "8cb95e" || first.ForceDownload != ai.FileDownloadNever ||
		first.VendorMetadata["fps"] != float64(24) || second.ForceDownload != ai.FileDownloadAllowLocal {
		t.Fatalf("unexpected video fixture: first=%+v second=%+v", first, second)
	}
	encoded, err := ai.MarshalMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	roundTripped, err := ai.UnmarshalMessages(encoded)
	if err != nil {
		t.Fatal(err)
	}
	cloned := roundTripped[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Contents[0].(ai.VideoURL)
	cloned.VendorMetadata["fps"] = 30
	if first.VendorMetadata["fps"] != float64(24) {
		t.Fatal("video metadata was not detached")
	}
}

func TestVideoURLRunContentIsDetached(t *testing.T) {
	metadata := map[string]any{"nested": map[string]any{"value": "original"}}
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		video := messages[len(messages)-1].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Contents[0].(ai.VideoURL)
		video.VendorMetadata["nested"].(map[string]any)["value"] = "changed"
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	result, err := ai.NewAgent[struct{}, string](model).RunParts(
		t.Context(), []ai.UserContent{ai.VideoURL{
			URL: "https://example.com/video.mp4", VendorMetadata: metadata,
		}}, struct{}{},
	)
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected video run result=%+v err=%v", result, err)
	}
	if metadata["nested"].(map[string]any)["value"] != "original" {
		t.Fatal("model mutation changed caller video metadata")
	}
}

func TestVideoURLSerializationValidation(t *testing.T) {
	encoded, err := ai.MarshalMessages([]ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{ai.VideoURL{
			URL: "https://example.com/video.mp4", ForceDownload: ai.FileDownloadSafe,
		}}},
	}}})
	if err != nil || !strings.Contains(string(encoded), `"force_download":true`) {
		t.Fatalf("safe video did not serialize: data=%s err=%v", encoded, err)
	}
	for name, video := range map[string]ai.VideoURL{
		"media type": {URL: "https://example.com/video"},
		"mode":       {URL: "https://example.com/video.mp4", ForceDownload: "invalid"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := ai.MarshalMessages([]ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Contents: []ai.UserContent{video}},
			}}})
			if err == nil {
				t.Fatal("invalid video serialized")
			}
		})
	}
	for name, data := range map[string]string{
		"safe":         `[{"kind":"request","parts":[{"part_kind":"user-prompt","content":[{"kind":"video-url","url":"https://example.com/video.mp4","media_type":"video/mp4","force_download":true}]}]}]`,
		"invalid type": `[{"kind":"request","parts":[{"part_kind":"user-prompt","content":[{"kind":"video-url","url":"https://example.com/video.mp4","media_type":"video/mp4","force_download":1}]}]}]`,
		"invalid mode": `[{"kind":"request","parts":[{"part_kind":"user-prompt","content":[{"kind":"video-url","url":"https://example.com/video.mp4","media_type":"video/mp4","force_download":"invalid"}]}]}]`,
	} {
		t.Run(name, func(t *testing.T) {
			messages, err := ai.UnmarshalMessages([]byte(data))
			if name == "safe" {
				if err != nil || messages[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).
					Contents[0].(ai.VideoURL).ForceDownload != ai.FileDownloadSafe {
					t.Fatalf("safe force_download did not decode: messages=%+v err=%v", messages, err)
				}
			} else if err == nil {
				t.Fatal("invalid force_download value decoded")
			}
		})
	}
}
