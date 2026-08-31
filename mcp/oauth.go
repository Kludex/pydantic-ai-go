package mcp

import (
	"fmt"
	"reflect"

	mcpauth "github.com/modelcontextprotocol/go-sdk/auth"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// OAuthHandler authorizes Streamable HTTP requests and refreshes their token source.
type OAuthHandler = mcpauth.OAuthHandler

// WithOAuthHandler enables OAuth for Streamable HTTP transports. The handler
// must coordinate concurrent authorization attempts and persist refreshed tokens.
func WithOAuthHandler(handler OAuthHandler) Option {
	if oauthHandlerIsNil(handler) {
		panic("ai/mcp: OAuth handler must not be nil")
	}
	return func(config *config) { config.oauthHandler = handler }
}

func applyOAuthHandler(transport mcpsdk.Transport, handler OAuthHandler) (mcpsdk.Transport, error) {
	if handler == nil {
		return transport, nil
	}
	streamable, ok := transport.(*mcpsdk.StreamableClientTransport)
	if !ok {
		return nil, fmt.Errorf("ai/mcp: OAuth requires a Streamable HTTP transport, got %T", transport)
	}
	cloned := *streamable
	cloned.OAuthHandler = handler
	return &cloned, nil
}

func oauthHandlerIsNil(handler OAuthHandler) bool {
	if handler == nil {
		return true
	}
	value := reflect.ValueOf(handler)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}
