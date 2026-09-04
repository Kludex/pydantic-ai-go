package mistral_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/mistral"
)

func TestDownloadedAndDirectContent(t *testing.T) {
	var payload map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/image":
			response.Header().Set("Content-Type", "image/png")
			_, _ = io.WriteString(response, "image")
		case "/text":
			response.Header().Set("Content-Type", "text/plain")
			_, _ = io.WriteString(response, "downloaded text")
		case "/pdf":
			response.Header().Set("Content-Type", "application/pdf")
			_, _ = io.WriteString(response, "pdf")
		case "/octet.png", "/octet-text", "/octet-pdf", "/octet":
			response.Header().Set("Content-Type", "application/octet-stream")
			_, _ = io.WriteString(response, "octets")
		case "/fail":
			http.Error(response, "no", http.StatusBadGateway)
		case "/chat/completions":
			if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
				t.Error(err)
			}
			_, _ = io.WriteString(response, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
		}
	}))
	defer server.Close()
	model := mistral.NewModel("model", mistral.WithBaseURL(server.URL), mistral.WithHTTPClient(server.Client()))
	_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Content: "plain"},
		ai.UserPromptPart{Contents: []ai.UserContent{
			ai.ImageURL{URL: server.URL + "/image", ForceDownload: ai.FileDownloadAllowLocal},
			ai.ImageURL{URL: "https://example.com/direct.png", VendorMetadata: map[string]any{"detail": ""}},
			ai.DocumentURL{
				URL: server.URL + "/text", MediaType: "text/plain", ForceDownload: ai.FileDownloadAllowLocal,
			},
			ai.DocumentURL{
				URL: server.URL + "/pdf", MediaType: "application/pdf", ForceDownload: ai.FileDownloadAllowLocal,
			},
			ai.DocumentURL{URL: "https://example.com/direct.pdf", MediaType: "application/pdf"},
			ai.ImageURL{
				URL: server.URL + "/octet.png", ForceDownload: ai.FileDownloadAllowLocal,
			},
			ai.DocumentURL{
				URL: server.URL + "/octet-text", MediaType: "text/plain", ForceDownload: ai.FileDownloadAllowLocal,
			},
			ai.DocumentURL{
				URL: server.URL + "/octet-pdf", MediaType: "application/pdf", ForceDownload: ai.FileDownloadAllowLocal,
			},
		}},
	}}}, ai.ModelRequestParams{AllowText: true})
	if err != nil {
		t.Fatal(err)
	}
	messages := payload["messages"].([]any)
	if messages[0].(map[string]any)["content"] != "plain" {
		t.Fatalf("plain content was not preserved: %#v", messages)
	}
	content := messages[1].(map[string]any)["content"].([]any)
	if len(content) != 8 ||
		!strings.HasPrefix(content[0].(map[string]any)["image_url"].(map[string]any)["url"].(string), "data:image/png;base64,") ||
		content[1].(map[string]any)["image_url"].(map[string]any)["detail"] != "auto" ||
		!strings.Contains(content[2].(map[string]any)["text"].(string), "downloaded text") ||
		!strings.HasPrefix(content[3].(map[string]any)["document_url"].(string), "data:application/pdf;base64,") ||
		content[4].(map[string]any)["document_url"] != "https://example.com/direct.pdf" ||
		!strings.HasPrefix(content[5].(map[string]any)["image_url"].(map[string]any)["url"].(string), "data:image/png;base64,") ||
		!strings.Contains(content[6].(map[string]any)["text"].(string), "octets") ||
		!strings.HasPrefix(content[7].(map[string]any)["document_url"].(string), "data:application/pdf;base64,") {
		t.Fatalf("unexpected downloaded content: %#v", content)
	}
	for _, item := range []ai.UserContent{
		ai.DocumentURL{URL: server.URL + "/text", MediaType: "text/plain"},
		ai.ImageURL{URL: server.URL + "/octet", ForceDownload: ai.FileDownloadAllowLocal},
		ai.ImageURL{URL: server.URL + "/fail", MediaType: "image/png", ForceDownload: ai.FileDownloadAllowLocal},
		ai.DocumentURL{
			URL: server.URL + "/fail", MediaType: "text/plain", ForceDownload: ai.FileDownloadAllowLocal,
		},
		ai.DocumentURL{
			URL: server.URL + "/fail", MediaType: "application/pdf", ForceDownload: ai.FileDownloadAllowLocal,
		},
	} {
		_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.UserPromptPart{Contents: []ai.UserContent{item}},
		}}}, ai.ModelRequestParams{})
		if err == nil {
			t.Fatalf("expected download error for %T", item)
		}
	}
}

func TestUnsupportedContent(t *testing.T) {
	model, requests := noRequestModel(t)
	tests := []struct {
		name    string
		content ai.UserContent
	}{
		{name: "audio", content: ai.AudioURL{URL: "https://example.com/a.mp3"}},
		{name: "video", content: ai.VideoURL{URL: "https://example.com/v.mp4"}},
		{name: "uploaded", content: ai.UploadedFile{FileID: "file"}},
		{name: "binary", content: ai.BinaryContent{MediaType: "audio/mpeg"}},
		{name: "document", content: ai.DocumentURL{URL: "https://example.com/file.doc", MediaType: "application/msword"}},
		{name: "unknown document", content: ai.DocumentURL{URL: "https://example.com/file"}},
		{name: "download mode", content: ai.ImageURL{
			URL: "https://example.com/image.png", ForceDownload: ai.FileDownloadMode("invalid"),
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Contents: []ai.UserContent{test.content}},
			}}}, ai.ModelRequestParams{AllowText: true})
			if err == nil {
				t.Fatal("expected error")
			}
		})
	}
	for _, content := range []ai.UserContent{
		ai.AudioURL{URL: "https://example.com/a.mp3"},
		ai.VideoURL{URL: "https://example.com/v.mp4"},
		ai.UploadedFile{FileID: "file", Identifier: "uploaded"},
	} {
		_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "tool", ToolCallID: "call", Content: ai.ToolReturn{
				ReturnValue: "file", Content: []ai.UserContent{content},
			}},
		}}}, ai.ModelRequestParams{})
		if err == nil {
			t.Fatalf("expected rich content error for %T", content)
		}
	}
	if *requests != 0 {
		t.Fatalf("unexpected transport calls: %d", *requests)
	}
}

func TestRichToolContentIdentifiers(t *testing.T) {
	var payload map[string]any
	server := responseServer(t, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`, &payload)
	defer server.Close()
	model := mistral.NewModel("model", mistral.WithBaseURL(server.URL), mistral.WithHTTPClient(server.Client()))
	rich := &ai.ToolReturn{ReturnValue: "files", Content: []ai.UserContent{
		ai.ImageURL{URL: "https://example.com/i.png"},
		ai.DocumentURL{URL: "https://example.com/d.pdf", MediaType: "application/pdf"},
		ai.TextContent{Text: "extra"},
	}}
	_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.ToolReturnPart{ToolName: "tool", ToolCallID: "call", Content: rich},
		ai.ToolReturnPart{ToolName: "tool", ToolCallID: "call-2", Content: (*ai.ToolReturn)(nil)},
	}}}, ai.ModelRequestParams{AllowText: true})
	if err != nil {
		t.Fatal(err)
	}
	messages := payload["messages"].([]any)
	if len(messages) != 4 || messages[0].(map[string]any)["role"] != "tool" ||
		messages[1].(map[string]any)["role"] != "tool" || messages[2].(map[string]any)["role"] != "assistant" ||
		messages[3].(map[string]any)["role"] != "user" {
		t.Fatalf("unexpected rich tool messages: %#v", messages)
	}
}
