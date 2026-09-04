package xai_test

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
	"github.com/Kludex/pydantic-ai-go/models/xai"
)

func TestDocumentUploadErrors(t *testing.T) {
	messages := documentMessages(ai.BinaryContent{Data: []byte("document"), MediaType: "application/pdf"})
	failure := errors.New("failed")
	cases := []struct {
		name    string
		options []xai.Option
		want    string
	}{
		{name: "request", options: []xai.Option{xai.WithBaseURL("://")}, want: "missing protocol"},
		{name: "prepare", options: []xai.Option{xai.WithProvider(openai.ProviderConfig{
			BaseURL: "https://example.com/v1", PrepareRequest: func(*http.Request) error { return failure },
		})}, want: "prepare file upload"},
		{name: "transport", options: []xai.Option{xai.WithProvider(openai.ProviderConfig{
			BaseURL: "https://example.com/v1", HTTPClient: &http.Client{Transport: roundTripFunc(
				func(*http.Request) (*http.Response, error) { return nil, failure },
			)},
		})}, want: "upload file"},
		{name: "read", options: responseOptions(http.StatusOK, io.NopCloser(errorReader{})), want: "read file upload"},
		{name: "status", options: responseOptions(http.StatusBadRequest, io.NopCloser(strings.NewReader("bad"))), want: "Bad Request"},
		{name: "decode", options: responseOptions(http.StatusOK, io.NopCloser(strings.NewReader("{"))), want: "decode"},
		{name: "missing ID", options: responseOptions(http.StatusOK, io.NopCloser(strings.NewReader(`{}`))), want: "no file ID"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			model := xai.NewModel("grok-4.3", test.options...)
			if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err == nil ||
				!strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
	model := xai.NewModel("grok-4.3", xai.WithBaseURL("://"))
	if _, err := model.StreamRequest(t.Context(), messages, ai.ModelRequestParams{}); err == nil ||
		!strings.Contains(err.Error(), "missing protocol") {
		t.Fatalf("unexpected streamed upload error: %v", err)
	}
}

func TestDocumentDownloadErrors(t *testing.T) {
	messages := documentMessages(ai.DocumentURL{URL: "file:///secret", ForceDownload: ai.FileDownloadSafe})
	model := xai.NewModel("grok-4.3")
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err == nil ||
		!strings.Contains(err.Error(), "download document") {
		t.Fatalf("unexpected download error: %v", err)
	}
	if _, err := model.StreamRequest(t.Context(), messages, ai.ModelRequestParams{}); err == nil ||
		!strings.Contains(err.Error(), "download document") {
		t.Fatalf("unexpected streamed download error: %v", err)
	}
}

func documentMessages(content ai.UserContent) []ai.ModelMessage {
	return []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{content}},
	}}}
}

func responseOptions(status int, body io.ReadCloser) []xai.Option {
	return []xai.Option{xai.WithProvider(openai.ProviderConfig{
		BaseURL: "https://example.com/v1", HTTPClient: &http.Client{Transport: roundTripFunc(
			func(request *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: status, Status: http.StatusText(status), Body: body, Request: request}, nil
			},
		)},
	})}
}

type errorReader struct{}

func (errorReader) Read([]byte) (int, error) { return 0, errors.New("read failed") }
