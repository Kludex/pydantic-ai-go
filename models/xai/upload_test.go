package xai_test

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/openai"
	"github.com/Kludex/pydantic-ai-go/models/xai"
)

func TestDocumentUploads(t *testing.T) {
	uploads := 0
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/document.pdf", "/":
			response.Header().Set("Content-Type", "application/pdf")
			_, _ = response.Write([]byte("remote document"))
		case "/files":
			if request.Header.Get("Authorization") != "Bearer key" || request.Header.Get("X-Provider") != "yes" ||
				request.Header.Get("X-Dynamic") != "yes" {
				t.Errorf("unexpected upload headers: %#v", request.Header)
			}
			if err := request.ParseMultipartForm(1024); err != nil {
				t.Fatal(err)
			}
			file, header, err := request.FormFile("file")
			if err != nil {
				t.Fatal(err)
			}
			data, _ := io.ReadAll(file)
			_ = file.Close()
			uploads++
			if uploads == 1 && (header.Filename != "inline.pdf" || string(data) != "inline document") {
				t.Errorf("unexpected inline upload: %q %q", header.Filename, data)
			}
			if uploads == 2 && (header.Filename != "document.pdf" || string(data) != "remote document") {
				t.Errorf("unexpected remote upload: %q %q", header.Filename, data)
			}
			if uploads == 3 && header.Filename != "document" {
				t.Errorf("unexpected custom upload filename: %q", header.Filename)
			}
			if uploads == 4 && header.Filename != "document.pdf" {
				t.Errorf("unexpected root upload filename: %q", header.Filename)
			}
			if uploads == 3 {
				_, _ = fmt.Fprintf(response, `{"id":"file-%d"}`, uploads)
			} else {
				_, _ = fmt.Fprintf(response, `{"file":{"id":"file-%d"}}`, uploads)
			}
		case "/responses":
			var body map[string]any
			if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			input := body["input"].([]any)[0].(map[string]any)["content"].([]any)
			if len(input) != 4 || input[0].(map[string]any)["file_id"] != "file-1" ||
				input[1].(map[string]any)["file_id"] != "file-2" || input[3].(map[string]any)["file_id"] != "file-4" {
				t.Errorf("unexpected uploaded inputs: %#v", input)
			}
			_, _ = io.WriteString(response, `{"id":"response","model":"grok","status":"completed","output":[]}`)
		default:
			t.Errorf("unexpected path %q", request.URL.Path)
			response.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	model := xai.NewModel("grok-4.3", xai.WithProvider(openai.ProviderConfig{
		BaseURL: server.URL, APIKey: "key", HTTPClient: server.Client(),
		Headers: http.Header{"X-Provider": []string{"yes"}},
		PrepareRequest: func(request *http.Request) error {
			request.Header.Set("X-Dynamic", "yes")
			return nil
		},
	}))
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
		ai.BinaryContent{Data: []byte("inline document"), MediaType: "application/pdf", Identifier: "inline.pdf"},
		ai.DocumentURL{URL: server.URL + "/document.pdf", ForceDownload: ai.FileDownloadAllowLocal},
		ai.BinaryContent{Data: []byte("custom"), MediaType: "application/x-custom"},
		ai.DocumentURL{URL: server.URL, ForceDownload: ai.FileDownloadAllowLocal},
	}}}}}
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	if uploads != 4 || !strings.HasPrefix(model.ProviderURL(), server.URL) {
		t.Fatalf("unexpected uploads: %d", uploads)
	}
}
