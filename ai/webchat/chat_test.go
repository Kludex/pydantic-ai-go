package webchat_test

import (
	"net/http"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
	"github.com/Kludex/pydantic-ai-go/ai/webchat"
)

func TestChatRequestValidation(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	handler := newTestHandler(t, agent, webchat.Config{})
	for _, contentType := range []string{"", "text/plain", "not a media type"} {
		request := request(http.MethodPost, "/api/chat", `{}`)
		request.Header.Set("Content-Type", contentType)
		response := serve(handler, request)
		if response.Code != http.StatusUnsupportedMediaType || !strings.Contains(response.Body.String(), "application/json") {
			t.Fatalf("content type %q returned %d: %s", contentType, response.Code, response.Body.String())
		}
	}

	request := request(http.MethodPost, "/api/chat", `{`)
	request.Header.Set("Content-Type", "application/json; charset=utf-8")
	response := serve(handler, request)
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "invalid Vercel AI request") {
		t.Fatalf("unexpected malformed request response: %d %s", response.Code, response.Body.String())
	}

	request = jsonRequest(http.MethodPost, "/api/chat", `{}`)
	request.Body = errorReadCloser{}
	response = serve(handler, request)
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "read request body") {
		t.Fatalf("unexpected read error response: %d %s", response.Code, response.Body.String())
	}

	limited := newTestHandler(t, agent, webchat.Config{MaxRequestBytes: 4})
	response = serve(limited, jsonRequest(http.MethodPost, "/api/chat", `12345`))
	if response.Code != http.StatusRequestEntityTooLarge || !strings.Contains(response.Body.String(), "request body too large") {
		t.Fatalf("unexpected oversized response: %d %s", response.Code, response.Body.String())
	}

	response = serve(handler, jsonRequest(http.MethodPost, "/api/chat", `{}`))
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"type":"error"`) {
		t.Fatalf("adapter error was not streamed: %d %s", response.Code, response.Body.String())
	}
}
