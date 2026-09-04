package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

func writeMCPConfig(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "mcp.json")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadToolsetsFromStreamableHTTPConfig(t *testing.T) {
	server := mcpsdk.NewServer(&mcpsdk.Implementation{Name: "server", Version: "1"}, nil)
	server.AddTool(&mcpsdk.Tool{Name: "echo", InputSchema: map[string]any{
		"type": "object", "properties": map[string]any{"value": map[string]any{"type": "string"}},
	}}, func(_ context.Context, request *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
		var args map[string]any
		if err := json.Unmarshal(request.Params.Arguments, &args); err != nil {
			return nil, err
		}
		return &mcpsdk.CallToolResult{StructuredContent: args}, nil
	})
	var header string
	handler := mcpsdk.NewStreamableHTTPHandler(func(request *http.Request) *mcpsdk.Server {
		header = request.Header.Get("Authorization")
		return server
	}, &mcpsdk.StreamableHTTPOptions{Stateless: true, JSONResponse: true})
	httpServer := httptest.NewServer(handler)
	defer httpServer.Close()
	t.Setenv("MCP_ENDPOINT", httpServer.URL)

	path := writeMCPConfig(t, `{
		"mcpServers": {
			"remote": {
				"url": "${MCP_ENDPOINT}/mcp",
				"headers": {"Authorization": "Bearer ${MCP_TOKEN:-test-token}"}
			}
		}
	}`)
	toolsets, err := LoadToolsets[testDeps](path)
	if err != nil || len(toolsets) != 1 {
		t.Fatalf("load failed: toolsets=%d err=%v", len(toolsets), err)
	}
	requests := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			if len(params.Tools) != 1 || params.Tools[0].Name != "remote_echo" || params.Tools[0].ToolsetID != "remote" {
				t.Fatalf("loaded tool was not prefixed: %+v", params.Tools)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "remote_echo", ToolCallID: "call", Args: json.RawMessage(`{"value":"ok"}`),
			}}}, nil
		}
		returned := messages[len(messages)-1].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart)
		if returned.Content.(map[string]any)["value"] != "ok" {
			t.Fatalf("unexpected loaded MCP result: %+v", returned)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[testDeps, string](model)
	for _, toolset := range toolsets {
		agent.AddToolset(toolset)
	}
	result, err := agent.Run(t.Context(), "echo", testDeps{})
	if err != nil || result.Output != "done" || header != "Bearer test-token" {
		t.Fatalf("configured MCP run failed: result=%+v header=%q err=%v", result, header, err)
	}
}

func TestLoadToolsetsConfigValidation(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "missing.json")
	if _, err := LoadToolsets[testDeps](missing); err == nil {
		t.Fatal("expected missing file error")
	}
	for name, content := range map[string]string{
		"invalid JSON":       `{`,
		"non-object":         `[]`,
		"missing servers":    `{}`,
		"non-object servers": `{"mcpServers":[]}`,
		"invalid field type": `{"mcpServers":{"a":{"args":"bad"}}}`,
		"missing transport":  `{"mcpServers":{"a":{}}}`,
		"both transports":    `{"mcpServers":{"a":{"command":"cmd","url":"http://example.com"}}}`,
		"empty server name":  `{"mcpServers":{"":{"command":"cmd"}}}`,
		"invalid URL":        `{"mcpServers":{"a":{"url":"not a URL"}}}`,
		"invalid URL scheme": `{"mcpServers":{"a":{"url":"file:///tmp/socket"}}}`,
		"missing env":        `{"mcpServers":{"a":{"url":"https://${MCP_CONFIG_MISSING}/mcp"}}}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := LoadToolsets[testDeps](writeMCPConfig(t, content)); err == nil {
				t.Fatal("expected configuration error")
			}
		})
	}
}

func TestConfiguredMCPTransports(t *testing.T) {
	commandToolset, err := configuredToolset[testDeps]("local", serverConfig{
		Command: "echo", Args: []string{"ok"}, CWD: "/tmp", Env: map[string]string{"MCP_TEST": "set"},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := commandToolset.ForRun(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	command := resolved.(*runToolset[testDeps]).transport.(*mcpsdk.CommandTransport).Command
	if command.Path == "" || !reflect.DeepEqual(command.Args[1:], []string{"ok"}) || command.Dir != "/tmp" ||
		!containsEnvironment(command.Env, "MCP_TEST=set") {
		t.Fatalf("unexpected command transport: %+v", command)
	}

	for _, test := range []struct {
		url string
		sse bool
	}{
		{url: "http://localhost/sse", sse: true},
		{url: "http://localhost/sse/?token=x", sse: true},
		{url: "http://localhost/mcp"},
	} {
		toolset, err := configuredToolset[testDeps]("remote", serverConfig{URL: test.url}, nil)
		if err != nil {
			t.Fatal(err)
		}
		resolved, err := toolset.ForRun(t.Context(), nil)
		if err != nil {
			t.Fatal(err)
		}
		transport := resolved.(*runToolset[testDeps]).transport
		_, isSSE := transport.(*mcpsdk.SSEClientTransport)
		if isSSE != test.sse {
			t.Fatalf("unexpected transport %T for %q", transport, test.url)
		}
	}
}

func TestExpandEnvironment(t *testing.T) {
	t.Setenv("MCP_SET", "value")
	input := map[string]any{
		"set": "before-${MCP_SET}-after", "default": "${MCP_UNSET:-fallback}", "empty": "${MCP_UNSET:-}",
		"list": []any{"${MCP_SET}", float64(1)}, "boolean": true,
	}
	expanded, err := expandEnvironment(input)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{
		"set": "before-value-after", "default": "fallback", "empty": "",
		"list": []any{"value", float64(1)}, "boolean": true,
	}
	if !reflect.DeepEqual(expanded, want) {
		t.Fatalf("unexpected expansion: %#v", expanded)
	}
	if _, err := expandEnvironment([]any{map[string]any{"bad": "${MCP_UNSET}"}}); err == nil {
		t.Fatal("expected nested expansion error")
	}
	if _, err := expandEnvironment(map[string]any{"bad": []any{"${MCP_UNSET}"}}); err == nil {
		t.Fatal("expected map expansion error")
	}
}

func containsEnvironment(environment []string, want string) bool {
	for _, entry := range environment {
		if entry == want {
			return true
		}
	}
	return false
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) { return fn(request) }

func TestHeaderTransport(t *testing.T) {
	baseErr := errors.New("stop")
	base := roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.Header.Get("Authorization") != "Bearer token" || request.Header.Get("Original") != "yes" {
			t.Fatalf("headers not applied: %v", request.Header)
		}
		request.Header.Set("Mutated", "yes")
		return nil, baseErr
	})
	transport := headerTransport{base: base, headers: map[string]string{"Authorization": "Bearer token"}}
	request := httptest.NewRequest(http.MethodGet, "http://example.com", nil)
	request.Header.Set("Original", "yes")
	if _, err := transport.RoundTrip(request); !errors.Is(err, baseErr) {
		t.Fatalf("unexpected transport error: %v", err)
	}
	if request.Header.Get("Authorization") != "" || request.Header.Get("Mutated") != "" {
		t.Fatalf("original request was modified: %v", request.Header)
	}
	original := map[string]string{"a": "b"}
	cloned := cloneHeaders(original)
	cloned["a"] = "changed"
	if original["a"] != "b" {
		t.Fatal("headers were not cloned")
	}
}

func TestLoadToolsetsLoadsMultipleServers(t *testing.T) {
	path := writeMCPConfig(t, `{"mcpServers":{"z":{"command":"echo"},"a":{"command":"echo"}}}`)
	toolsets, err := LoadToolsets[testDeps](path, WithReadTimeout(time.Second))
	if err != nil || len(toolsets) != 2 {
		t.Fatalf("unexpected sorted load: %d %v", len(toolsets), err)
	}
}
