package webchat_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
	"github.com/Kludex/pydantic-ai-go/ai/webchat"
)

func TestHandlerServesOfficialUIAndAPIs(t *testing.T) {
	model := fakes.NewTestModel()
	handler := newTestHandler(t, ai.NewAgent[struct{}, string](model), webchat.Config{
		DefaultModelID: "test:model", DefaultModelName: "Test model",
		NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
	})
	for _, path := range []string{"/", "/conversation-id"} {
		response := serve(handler, request(http.MethodGet, path, ""))
		if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "Pydantic AI official UI") ||
			response.Header().Get("Cache-Control") != "public, max-age=3600" {
			t.Fatalf("unexpected UI response for %s: %d %v %s", path, response.Code, response.Header(), response.Body.String())
		}
	}

	response := serve(handler, request(http.MethodGet, "/api/configure", ""))
	var configuration struct {
		Models []struct {
			ID           string   `json:"id"`
			Name         string   `json:"name"`
			BuiltinTools []string `json:"builtinTools"`
		} `json:"models"`
		BuiltinTools []struct {
			ID   string `json:"id"`
			Name string `json:"name"`
		} `json:"builtinTools"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &configuration); err != nil {
		t.Fatal(err)
	}
	if response.Code != http.StatusOK || len(configuration.Models) != 1 ||
		configuration.Models[0].ID != "test:model" || configuration.Models[0].Name != "Test model" ||
		len(configuration.Models[0].BuiltinTools) != 1 || configuration.Models[0].BuiltinTools[0] != "web_search" ||
		len(configuration.BuiltinTools) != 1 || configuration.BuiltinTools[0].ID != "web_search" ||
		configuration.BuiltinTools[0].Name != "Web Search" {
		t.Fatalf("unexpected frontend configuration: %#v", configuration)
	}

	response = serve(handler, request(http.MethodGet, "/api/health", ""))
	if response.Code != http.StatusOK || response.Body.String() != "{\"ok\":true}\n" {
		t.Fatalf("unexpected health response: %d %s", response.Code, response.Body.String())
	}

	body := `{"trigger":"submit-message","id":"chat","model":"test:model","messages":[{"id":"user","role":"user","parts":[{"type":"text","text":"hello"}]}]}`
	response = serve(handler, jsonRequest(http.MethodPost, "/api/chat", body))
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"type":"text-delta"`) ||
		response.Header().Get("x-vercel-ai-ui-message-stream") != "v1" {
		t.Fatalf("unexpected chat response: %d %v %s", response.Code, response.Header(), response.Body.String())
	}
}

func TestHandlerRoutesAndMethods(t *testing.T) {
	handler := newTestHandler(t, ai.NewAgent[struct{}, string](fakes.NewTestModel()), webchat.Config{})
	tests := []struct {
		method string
		path   string
		want   int
		allow  string
	}{
		{method: http.MethodPost, path: "/", want: http.StatusMethodNotAllowed, allow: http.MethodGet},
		{method: http.MethodGet, path: "/nested/path", want: http.StatusNotFound},
		{method: http.MethodGet, path: "/api/missing", want: http.StatusNotFound},
		{method: http.MethodPost, path: "/api/configure", want: http.StatusMethodNotAllowed, allow: http.MethodGet},
		{method: http.MethodPost, path: "/api/health", want: http.StatusMethodNotAllowed, allow: http.MethodGet},
		{method: http.MethodGet, path: "/api/chat", want: http.StatusMethodNotAllowed, allow: "POST, OPTIONS"},
	}
	for _, test := range tests {
		response := serve(handler, request(test.method, test.path, ""))
		if response.Code != test.want || response.Header().Get("Allow") != test.allow {
			t.Fatalf("%s %s: got %d allow %q, want %d allow %q", test.method, test.path,
				response.Code, response.Header().Get("Allow"), test.want, test.allow)
		}
	}
	response := serve(handler, request(http.MethodOptions, "/api/chat", ""))
	if response.Code != http.StatusOK || response.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Fatalf("unexpected preflight response: %d %v", response.Code, response.Header())
	}
	emptyPathRequest := request(http.MethodGet, "/", "")
	emptyPathRequest.URL.Path = ""
	if response := serve(handler, emptyPathRequest); response.Code != http.StatusNotFound {
		t.Fatalf("empty path returned %d", response.Code)
	}
}

func TestHandlerValidatesConfiguration(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	var nilModel *namedModel
	tests := []struct {
		name   string
		config webchat.Config
	}{
		{name: "negative body size", config: webchat.Config{MaxRequestBytes: -1}},
		{name: "empty host", config: webchat.Config{AllowedHosts: []string{""}}},
		{name: "bad wildcard", config: webchat.Config{AllowedHosts: []string{"foo.*.example"}}},
		{name: "wildcard without domain", config: webchat.Config{AllowedHosts: []string{"*."}}},
		{name: "IP allowlist entry", config: webchat.Config{AllowedHosts: []string{"127.0.0.1"}}},
		{name: "invalid hostname", config: webchat.Config{AllowedHosts: []string{"bad_host"}}},
		{name: "hyphenated hostname", config: webchat.Config{AllowedHosts: []string{"-bad.example"}}},
		{name: "nil model", config: webchat.Config{Models: []webchat.ModelOption{{Model: nil}}}},
		{name: "typed nil model", config: webchat.Config{Models: []webchat.ModelOption{{Model: nilModel}}}},
		{name: "empty model ID", config: webchat.Config{Models: []webchat.ModelOption{{Model: namedModel{}}}}},
		{name: "duplicate model", config: webchat.Config{Models: []webchat.ModelOption{
			{ID: "same", Model: namedModel{name: "one"}}, {ID: "same", Model: namedModel{name: "two"}},
		}}},
		{name: "invalid native tool", config: webchat.Config{NativeTools: []ai.NativeTool{(*ai.WebSearchTool)(nil)}}},
		{name: "duplicate native tool", config: webchat.Config{NativeTools: []ai.NativeTool{
			ai.WebSearchTool{}, ai.WebSearchTool{},
		}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := webchat.NewHandler(agent, struct{}{}, test.config); err == nil {
				t.Fatal("expected configuration error")
			}
		})
	}
	if _, err := webchat.NewHandler[struct{}, string](nil, struct{}{}, webchat.Config{}); err == nil {
		t.Fatal("expected nil agent error")
	}
	modelLess := ai.NewAgent[struct{}, string](nil)
	if _, err := webchat.NewHandler(modelLess, struct{}{}, webchat.Config{
		DefaultModelID: "missing", HTMLSource: testHTML(t),
	}); err == nil {
		t.Fatal("expected default model label error")
	}
	if _, err := webchat.NewHandler(agent, struct{}{}, webchat.Config{MCPConfigPath: t.TempDir() + "/missing"}); err == nil || !strings.Contains(err.Error(), "load MCP configuration") {
		t.Fatalf("unexpected MCP error: %v", err)
	}
	configPath := filepath.Join(t.TempDir(), "mcp.json")
	if err := os.WriteFile(configPath, []byte(`{"mcpServers":{}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := webchat.NewHandler(agent, struct{}{}, webchat.Config{
		MCPConfigPath: configPath, HTMLSource: testHTML(t),
	}); err != nil {
		t.Fatal(err)
	}
}

func TestHandlerContinuesToolApproval(t *testing.T) {
	executions := 0
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	agent.AddRawTool(ai.ToolDefinition{
		Name: "approve", Schema: map[string]any{"type": "object", "properties": map[string]any{}},
	}, func(context.Context, json.RawMessage) (any, error) {
		executions++
		return "done", nil
	}, ai.WithApprovalRequired())
	handler := newTestHandler(t, agent, webchat.Config{})
	first := `{"trigger":"submit-message","id":"chat","messages":[{"id":"user","role":"user","parts":[{"type":"text","text":"run"}]}]}`
	response := serve(handler, jsonRequest(http.MethodPost, "/api/chat", first))
	if response.Code != http.StatusOK || executions != 0 || !strings.Contains(response.Body.String(), `"type":"tool-approval-request"`) {
		t.Fatalf("unexpected approval request: status=%d executions=%d body=%s", response.Code, executions, response.Body.String())
	}
	resume := `{"trigger":"submit-message","id":"chat","messages":[` +
		`{"id":"user","role":"user","parts":[{"type":"text","text":"run"}]},` +
		`{"id":"assistant","role":"assistant","parts":[{"type":"tool-approve","toolCallId":"call_approve",` +
		`"state":"approval-responded","input":{},"approval":{"id":"call_approve","approved":true}}]}]}`
	response = serve(handler, jsonRequest(http.MethodPost, "/api/chat", resume))
	if response.Code != http.StatusOK || executions != 1 || !strings.Contains(response.Body.String(), `"type":"text-delta"`) {
		t.Fatalf("unexpected approval continuation: status=%d executions=%d body=%s", response.Code, executions, response.Body.String())
	}
}

type namedModel struct{ name string }

func (model namedModel) Name() string { return model.name }

func (namedModel) Request(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "ok"}}}, nil
}

func newTestHandler[Output any](
	t *testing.T, agent *ai.Agent[struct{}, Output], config webchat.Config,
) http.Handler {
	t.Helper()
	if config.HTMLSource == "" {
		config.HTMLSource = testHTML(t)
	}
	handler, err := webchat.NewHandler(agent, struct{}{}, config)
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

func testHTML(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "index.html")
	if err := os.WriteFile(path, []byte("<title>Pydantic AI official UI</title>"), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func request(method, path, body string) *http.Request {
	request := httptest.NewRequest(method, "http://localhost"+path, strings.NewReader(body))
	request.Host = "localhost"
	return request
}

func jsonRequest(method, path, body string) *http.Request {
	request := request(method, path, body)
	request.Header.Set("Content-Type", "application/json")
	return request
}

func serve(handler http.Handler, request *http.Request) *httptest.ResponseRecorder {
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}
