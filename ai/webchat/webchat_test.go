package webchat_test

import (
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
	"github.com/Kludex/pydantic-ai-go/ai/webchat"
)

func TestHandlerServesUIAndChat(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	handler, err := webchat.NewHandler(agent, struct{}{}, webchat.Config{AllowedHosts: []string{"example.com"}})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodGet, "http://example.com/", nil)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "Agent chat") ||
		response.Header().Get("Content-Security-Policy") == "" {
		t.Fatalf("unexpected UI response: %d %v %s", response.Code, response.Header(), response.Body.String())
	}

	body := `{"trigger":"submit-message","id":"chat","messages":[{"id":"user","role":"user","parts":[{"type":"text","text":"hello"}]}]}`
	request = httptest.NewRequest(http.MethodPost, "http://example.com/api/chat", strings.NewReader(body))
	request.Host = "EXAMPLE.COM:80"
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"type":"text-delta"`) {
		t.Fatalf("unexpected chat response: %d %s", response.Code, response.Body.String())
	}
}

func TestHandlerRoutingAndConfiguration(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	if _, err := webchat.NewHandler[struct{}, string](nil, struct{}{}, webchat.Config{}); err == nil {
		t.Fatal("expected nil agent error")
	}
	if _, err := webchat.NewHandler(agent, struct{}{}, webchat.Config{MaxRequestBytes: -1}); err == nil {
		t.Fatal("expected maximum size error")
	}
	if _, err := webchat.NewHandler(agent, struct{}{}, webchat.Config{MCPConfigPath: t.TempDir() + "/missing"}); err == nil || !strings.Contains(err.Error(), "load MCP configuration") {
		t.Fatalf("unexpected MCP error: %v", err)
	}
	configPath := t.TempDir() + "/mcp.json"
	if err := os.WriteFile(configPath, []byte(`{"mcpServers":{}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	handler, err := webchat.NewHandler(agent, struct{}{}, webchat.Config{MCPConfigPath: configPath})
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		method string
		path   string
		want   int
	}{
		{method: http.MethodPost, path: "/", want: http.StatusMethodNotAllowed},
		{method: http.MethodGet, path: "/missing", want: http.StatusNotFound},
	}
	for _, test := range tests {
		request := httptest.NewRequest(test.method, "http://anything.example"+test.path, nil)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != test.want {
			t.Fatalf("%s %s: got %d want %d", test.method, test.path, response.Code, test.want)
		}
	}

	restricted, err := webchat.NewHandler(agent, struct{}{}, webchat.Config{AllowedHosts: []string{"allowed.example"}})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	restricted.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "http://blocked.example/", nil))
	if response.Code != http.StatusForbidden {
		t.Fatalf("unexpected blocked host status: %d", response.Code)
	}
}
