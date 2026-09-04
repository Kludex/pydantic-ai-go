// Package webchat serves a minimal browser UI for typed agents.
package webchat

import (
	_ "embed"
	"fmt"
	"net"
	"net/http"
	"slices"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/mcp"
	"github.com/Kludex/pydantic-ai-go/ui/vercel"
)

//go:embed index.html
var indexHTML []byte

// Config controls browser serving, inbound trust, and MCP tools.
type Config struct {
	// AllowedHosts restricts HTTP Host values. Nil accepts every host.
	AllowedHosts []string
	// MCPConfigPath loads common mcpServers JSON when the handler is built.
	MCPConfigPath string
	// Sanitization controls browser-held history. The zero value is secure.
	Sanitization ai.MessageSanitizationOptions
	// MaxRequestBytes bounds chat request bodies. Zero defaults to 10 MiB.
	MaxRequestBytes int64
	// RunOptions are copied and reused across requests.
	RunOptions []ai.RunOption
}

// NewHandler builds a browser UI and /api/chat Vercel AI endpoint. Deps and
// run options must be safe for concurrent requests.
func NewHandler[Deps, Output any](
	agent *ai.Agent[Deps, Output], deps Deps, config Config,
) (http.Handler, error) {
	if agent == nil {
		return nil, fmt.Errorf("webchat: agent must not be nil")
	}
	if config.MaxRequestBytes < 0 {
		return nil, fmt.Errorf("webchat: maximum request bytes must not be negative")
	}
	options := append([]ai.RunOption(nil), config.RunOptions...)
	if config.MCPConfigPath != "" {
		toolsets, err := mcp.LoadToolsets[Deps](config.MCPConfigPath)
		if err != nil {
			return nil, fmt.Errorf("webchat: load MCP configuration: %w", err)
		}
		options = append(options, ai.WithRunToolsets(toolsets...))
	}
	allowedHosts := slices.Clone(config.AllowedHosts)
	for index := range allowedHosts {
		allowedHosts[index] = strings.ToLower(strings.TrimSpace(allowedHosts[index]))
	}
	chat := vercel.NewAdapter(agent, vercel.Config{
		Sanitization: config.Sanitization, MaxRequestBytes: config.MaxRequestBytes,
	}).Handler(deps, options...)
	return http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if !hostAllowed(request.Host, allowedHosts) {
			http.Error(response, "host not allowed", http.StatusForbidden)
			return
		}
		switch request.URL.Path {
		case "/":
			if request.Method != http.MethodGet {
				response.Header().Set("Allow", http.MethodGet)
				http.Error(response, "method not allowed", http.StatusMethodNotAllowed)
				return
			}
			response.Header().Set("Content-Type", "text/html; charset=utf-8")
			response.Header().Set("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
			_, _ = response.Write(indexHTML)
		case "/api/chat":
			chat.ServeHTTP(response, request)
		default:
			http.NotFound(response, request)
		}
	}), nil
}

func hostAllowed(value string, allowed []string) bool {
	if allowed == nil {
		return true
	}
	host := strings.ToLower(value)
	if parsed, _, err := net.SplitHostPort(value); err == nil {
		host = strings.ToLower(parsed)
	}
	return slices.Contains(allowed, host)
}
