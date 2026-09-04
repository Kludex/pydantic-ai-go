package mcp_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	aimcp "github.com/Kludex/pydantic-ai-go/mcp"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"golang.org/x/oauth2"
)

type testOAuthHandler struct {
	mu    *sync.Mutex
	calls *int
}

func (handler testOAuthHandler) TokenSource(context.Context) (oauth2.TokenSource, error) {
	handler.mu.Lock()
	(*handler.calls)++
	handler.mu.Unlock()
	return oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "oauth-token"}), nil
}

func (testOAuthHandler) Authorize(context.Context, *http.Request, *http.Response) error { return nil }

func TestOAuthStreamableSessionsAndToolsets(t *testing.T) {
	server := mcpsdk.NewServer(&mcpsdk.Implementation{Name: "oauth", Version: "1"}, nil)
	var mu sync.Mutex
	var authorizations []string
	protocolHandler := mcpsdk.NewStreamableHTTPHandler(func(*http.Request) *mcpsdk.Server { return server }, nil)
	httpServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		mu.Lock()
		authorizations = append(authorizations, request.Header.Get("Authorization"))
		mu.Unlock()
		protocolHandler.ServeHTTP(response, request)
	}))
	defer httpServer.Close()

	calls := 0
	handler := testOAuthHandler{mu: &mu, calls: &calls}
	transport := &mcpsdk.StreamableClientTransport{
		Endpoint: httpServer.URL, HTTPClient: httpServer.Client(), DisableStandaloneSSE: true,
	}
	session, err := aimcp.Connect(t.Context(), transport, aimcp.WithOAuthHandler(handler))
	if err != nil {
		t.Fatal(err)
	}
	if transport.OAuthHandler != nil {
		t.Fatal("OAuth configuration mutated the caller's transport")
	}
	if err := session.Ping(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model)
	agent.AddToolset(aimcp.NewStreamableHTTPToolset[struct{}](
		httpServer.URL,
		aimcp.WithOAuthHandler(handler),
	))
	result, err := agent.Run(t.Context(), "hello", struct{}{})
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected OAuth toolset run: %+v err=%v", result, err)
	}

	mu.Lock()
	defer mu.Unlock()
	if calls == 0 || len(authorizations) == 0 {
		t.Fatalf("OAuth handler was not called: calls=%d headers=%v", calls, authorizations)
	}
	for _, authorization := range authorizations {
		if authorization != "Bearer oauth-token" {
			t.Fatalf("unexpected OAuth authorization %q", authorization)
		}
	}
}

func TestOAuthValidationAndTransportErrors(t *testing.T) {
	assertMCPPanic(t, "nil OAuth handler", "OAuth handler must not be nil", func() {
		aimcp.WithOAuthHandler(nil)
	})
	assertMCPPanic(t, "typed nil OAuth handler", "OAuth handler must not be nil", func() {
		var handler *testOAuthHandler
		aimcp.WithOAuthHandler(handler)
	})

	calls := 0
	var mu sync.Mutex
	handler := testOAuthHandler{mu: &mu, calls: &calls}
	clientTransport, _ := mcpsdk.NewInMemoryTransports()
	if _, err := aimcp.Connect(t.Context(), clientTransport, aimcp.WithOAuthHandler(handler)); err == nil ||
		!strings.Contains(err.Error(), "OAuth requires a Streamable HTTP transport") {
		t.Fatalf("unexpected Connect OAuth transport error: %v", err)
	}
	toolset := aimcp.NewToolset[struct{}](func(
		context.Context, *ai.RunContext[struct{}],
	) (mcpsdk.Transport, error) {
		transport, _ := mcpsdk.NewInMemoryTransports()
		return transport, nil
	}, aimcp.WithOAuthHandler(handler))
	if _, err := toolset.ForRun(t.Context(), nil); err == nil ||
		!strings.Contains(err.Error(), "OAuth requires a Streamable HTTP transport") {
		t.Fatalf("unexpected toolset OAuth transport error: %v", err)
	}

	var typedNil *mcpsdk.StreamableClientTransport
	typedNilToolset := aimcp.NewToolset[struct{}](func(
		context.Context, *ai.RunContext[struct{}],
	) (mcpsdk.Transport, error) {
		return typedNil, nil
	})
	if _, err := typedNilToolset.ForRun(t.Context(), nil); err == nil ||
		err.Error() != "ai/mcp: transport factory returned nil" {
		t.Fatalf("unexpected typed nil factory result: %v", err)
	}
}
