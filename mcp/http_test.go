package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

type localMCPModel struct {
	t        *testing.T
	requests int
}

func (m *localMCPModel) Name() string { return "local-mcp" }

func (*localMCPModel) SupportsNativeTool(ai.NativeTool) bool { return false }

func (m *localMCPModel) Request(
	_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	m.requests++
	if m.requests == 1 {
		if len(params.NativeTools) != 0 || len(params.Tools) != 1 || params.Tools[0].Name != "search" {
			m.t.Fatalf("unexpected MCP request tools: native=%#v local=%#v", params.NativeTools, params.Tools)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "search", ToolCallID: "search-1", Args: json.RawMessage(`{"query":"Go"}`),
		}}}, nil
	}
	request := messages[len(messages)-1].(ai.ModelRequest)
	result := request.Parts[0].(ai.ToolReturnPart)
	if result.Content != "result for Go" {
		m.t.Fatalf("unexpected MCP tool result: %#v", result)
	}
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
}

func TestHTTPToolsetInfersTransportAndValidatesURL(t *testing.T) {
	for _, test := range []struct {
		name       string
		url        string
		streamable bool
	}{
		{name: "streamable", url: "https://mcp.example.com/api", streamable: true},
		{name: "SSE", url: "https://mcp.example.com/api/sse/"},
	} {
		t.Run(test.name, func(t *testing.T) {
			owner := NewHTTPToolset[struct{}](HTTPToolsetConfig{URL: test.url})
			resolved, err := owner.ForRun(t.Context(), &ai.RunContext[struct{}]{})
			if err != nil {
				t.Fatal(err)
			}
			transport := resolved.(*runToolset[struct{}]).transport
			_, isStreamable := transport.(*mcpsdk.StreamableClientTransport)
			_, isSSE := transport.(*mcpsdk.SSEClientTransport)
			if isStreamable != test.streamable || isSSE == test.streamable {
				t.Fatalf("unexpected inferred transport %T", transport)
			}
		})
	}

	owner := NewHTTPToolset[struct{}](HTTPToolsetConfig{URL: "file:///tmp/mcp"})
	if _, err := owner.ForRun(t.Context(), &ai.RunContext[struct{}]{}); err == nil ||
		!strings.Contains(err.Error(), "invalid HTTP URL") {
		t.Fatalf("unexpected invalid URL error: %v", err)
	}
}

func TestHTTPToolsetDoesNotForwardHeadersAcrossOrigins(t *testing.T) {
	var targetAuthorization string
	target := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		targetAuthorization = request.Header.Get("Authorization")
		response.WriteHeader(http.StatusNoContent)
	}))
	defer target.Close()
	var sourceAuthorization string
	source := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		sourceAuthorization = request.Header.Get("Authorization")
		http.Redirect(response, request, target.URL, http.StatusTemporaryRedirect)
	}))
	defer source.Close()

	owner := NewHTTPToolset[struct{}](HTTPToolsetConfig{
		URL: source.URL, Headers: map[string]string{"Authorization": "Bearer secret"},
	})
	resolved, err := owner.ForRun(t.Context(), &ai.RunContext[struct{}]{})
	if err != nil {
		t.Fatal(err)
	}
	transport := resolved.(*runToolset[struct{}]).transport.(*mcpsdk.StreamableClientTransport)
	response, err := transport.HTTPClient.Get(source.URL)
	if err != nil {
		t.Fatal(err)
	}
	if err := response.Body.Close(); err != nil {
		t.Fatal(err)
	}
	if sourceAuthorization != "Bearer secret" || targetAuthorization != "" {
		t.Fatalf("headers crossed origins: source=%q target=%q", sourceAuthorization, targetAuthorization)
	}
}

func TestHTTPServerCapabilityUsesLocalFallback(t *testing.T) {
	server := mcpsdk.NewServer(&mcpsdk.Implementation{Name: "docs", Version: "1"}, nil)
	server.AddTool(&mcpsdk.Tool{Name: "search", InputSchema: map[string]any{
		"type": "object", "properties": map[string]any{"query": map[string]any{"type": "string"}},
	}}, func(_ context.Context, request *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
		var arguments struct {
			Query string `json:"query"`
		}
		if err := json.Unmarshal(request.Params.Arguments, &arguments); err != nil {
			return nil, err
		}
		return &mcpsdk.CallToolResult{Content: []mcpsdk.Content{
			&mcpsdk.TextContent{Text: "result for " + arguments.Query},
		}}, nil
	})
	server.AddTool(&mcpsdk.Tool{Name: "hidden", InputSchema: map[string]any{"type": "object"}},
		func(context.Context, *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
			return &mcpsdk.CallToolResult{}, nil
		})
	var mu sync.Mutex
	var authorization, tenant string
	protocol := mcpsdk.NewStreamableHTTPHandler(func(*http.Request) *mcpsdk.Server { return server }, nil)
	httpServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		mu.Lock()
		authorization = request.Header.Get("Authorization")
		tenant = request.Header.Get("X-Tenant")
		mu.Unlock()
		protocol.ServeHTTP(response, request)
	}))
	defer httpServer.Close()

	client := httpServer.Client()
	native := ai.MCPServerTool{
		ID: "docs", URL: httpServer.URL, AuthorizationToken: "Bearer secret",
		Headers:      map[string]string{"Authorization": "old", "X-Tenant": "acme"},
		AllowedTools: []string{"search"},
	}
	capability := NewHTTPServerCapability[struct{}](HTTPServerCapabilityConfig{
		Native: native, Client: client,
	})
	native.Headers["X-Tenant"] = "mutated"
	native.AllowedTools[0] = "hidden"
	client.Transport = errorRoundTripper{}

	model := &localMCPModel{t: t}
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(capability))
	result, err := agent.Run(t.Context(), "search", struct{}{})
	if err != nil || result.Output != "done" || model.requests != 2 {
		t.Fatalf("unexpected local MCP result=%#v requests=%d err=%v", result, model.requests, err)
	}
	mu.Lock()
	defer mu.Unlock()
	if authorization != "Bearer secret" || tenant != "acme" {
		t.Fatalf("local MCP credentials were not preserved: authorization=%q tenant=%q", authorization, tenant)
	}
}

type errorRoundTripper struct{}

func (errorRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, errors.New("mutated caller client")
}
