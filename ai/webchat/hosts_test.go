package webchat_test

import (
	"net/http"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
	"github.com/Kludex/pydantic-ai-go/ai/webchat"
)

func TestHandlerHostSecurity(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	handler := newTestHandler(t, agent, webchat.Config{AllowedHosts: []string{
		" EXAMPLE.COM. ", "*.service.example", "bücher.example",
	}})
	allowed := []string{
		"localhost", "chat.localhost:8080", "127.0.0.1:8080", "[::1]:8080", "[::1]", "::1",
		"example.com:443", "child.service.example", "xn--bcher-kva.example",
	}
	for _, host := range allowed {
		request := request(http.MethodGet, "/api/health", "")
		request.Host = host
		response := serve(handler, request)
		if response.Code != http.StatusOK {
			t.Fatalf("host %q was rejected with %d", host, response.Code)
		}
	}
	blocked := []string{"blocked.example", "service.example", "bad:host:port", "", "bad_host"}
	for _, host := range blocked {
		request := request(http.MethodGet, "/api/health", "")
		request.Host = host
		response := serve(handler, request)
		if response.Code != http.StatusMisdirectedRequest {
			t.Fatalf("host %q returned %d", host, response.Code)
		}
	}

	wildcard := newTestHandler(t, agent, webchat.Config{AllowedHosts: []string{"*"}})
	request := request(http.MethodGet, "/api/health", "")
	request.Host = "any.example"
	if response := serve(wildcard, request); response.Code != http.StatusOK {
		t.Fatalf("wildcard host returned %d", response.Code)
	}
}
