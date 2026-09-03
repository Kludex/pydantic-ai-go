package mcp

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// HTTPToolsetConfig configures an inferred Streamable HTTP or legacy SSE MCP transport.
type HTTPToolsetConfig struct {
	// URL is the absolute MCP endpoint.
	URL string
	// Headers are attached only to same-origin requests.
	Headers map[string]string
	// Client supplies caller-owned HTTP behavior. The configuration is cloned.
	Client *http.Client
}

// NewHTTPToolset creates an MCP toolset from an HTTP URL. URLs ending in /sse
// use the legacy SSE transport; all other URLs use Streamable HTTP.
func NewHTTPToolset[Deps any](config HTTPToolsetConfig, opts ...Option) *Toolset[Deps] {
	endpoint, endpointErr := parseHTTPURL(config.URL)
	client := prepareHTTPClient(config.Client, config.Headers, endpoint)
	return NewToolset(func(context.Context, *ai.RunContext[Deps]) (mcpsdk.Transport, error) {
		if endpointErr != nil {
			return nil, endpointErr
		}
		if strings.HasSuffix(strings.TrimRight(endpoint.Path, "/"), "/sse") {
			return &mcpsdk.SSEClientTransport{Endpoint: config.URL, HTTPClient: client}, nil
		}
		return &mcpsdk.StreamableClientTransport{Endpoint: config.URL, HTTPClient: client}, nil
	}, opts...)
}

// HTTPServerCapabilityConfig configures one provider-hosted MCP server and its local HTTP fallback.
type HTTPServerCapabilityConfig struct {
	// Native describes the provider-hosted MCP server.
	Native ai.MCPServerTool
	// Client supplies HTTP behavior for the local fallback.
	Client *http.Client
}

// NewHTTPServerCapability creates provider-native MCP with a local HTTP client fallback.
// AuthorizationToken and Headers are applied to both paths.
func NewHTTPServerCapability[Deps any](
	config HTTPServerCapabilityConfig, opts ...Option,
) *ai.NativeOrLocalTool[Deps] {
	headers := cloneHeaders(config.Native.Headers)
	if config.Native.AuthorizationToken != "" {
		headers["Authorization"] = config.Native.AuthorizationToken
	}
	options := append([]Option(nil), opts...)
	options = append(options, WithID(config.Native.ID))
	local := NewHTTPToolset[Deps](HTTPToolsetConfig{
		URL: config.Native.URL, Headers: headers, Client: config.Client,
	}, options...)
	return ai.NewMCPServerCapability(ai.MCPServerCapabilityConfig[Deps]{
		Native: config.Native,
		Local:  local,
	})
}

func parseHTTPURL(value string) (*url.URL, error) {
	endpoint, err := url.ParseRequestURI(value)
	if err != nil || endpoint.Host == "" || (endpoint.Scheme != "http" && endpoint.Scheme != "https") {
		return nil, fmt.Errorf("ai/mcp: invalid HTTP URL %q", value)
	}
	return endpoint, nil
}

func prepareHTTPClient(client *http.Client, headers map[string]string, endpoint *url.URL) *http.Client {
	if client == nil {
		client = http.DefaultClient
	}
	cloned := *client
	base := cloned.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	origin := ""
	if endpoint != nil {
		origin = endpoint.Scheme + "://" + endpoint.Host
	}
	cloned.Transport = headerTransport{base: base, headers: cloneHeaders(headers), origin: origin}
	return &cloned
}
