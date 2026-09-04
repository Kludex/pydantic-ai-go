// Package webchat serves PydanticAI's official browser chat UI for typed agents.
package webchat

import (
	"fmt"
	"net/http"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/mcp"
	"github.com/Kludex/pydantic-ai-go/ai/ui/vercel"
)

// ModelOption exposes one model in the browser model selector.
type ModelOption struct {
	// ID is sent by the browser when this model is selected. Empty uses Model.Name().
	ID string
	// Name is the human-readable browser label. Empty uses ID.
	Name string
	// Model handles runs that select ID.
	Model ai.Model
}

// Config controls browser serving, model and tool selection, inbound trust, and MCP tools.
type Config struct {
	// AllowedHosts adds hostnames accepted alongside IP addresses and localhost.
	// Use "*.example.com" for subdomains or "*" only behind an authentication boundary.
	AllowedHosts []string
	// DefaultModelID overrides the browser ID for the agent's configured model.
	DefaultModelID string
	// DefaultModelName overrides the browser label for the agent's configured model.
	DefaultModelName string
	// Models exposes additional models in the browser selector.
	Models []ModelOption
	// NativeTools exposes configured provider-native tools in the browser selector.
	NativeTools []ai.NativeTool
	// HTMLSource is a local path or HTTP URL for the UI. Empty uses DefaultHTMLURL.
	HTMLSource string
	// CacheDir stores remotely fetched UI HTML. Empty uses the user cache directory.
	CacheDir string
	// HTTPClient fetches remote UI HTML. Nil uses http.DefaultClient.
	HTTPClient *http.Client
	// MCPConfigPath loads common mcpServers JSON when the handler is built.
	MCPConfigPath string
	// Sanitization controls browser-held history. The zero value is secure.
	Sanitization ai.MessageSanitizationOptions
	// MaxRequestBytes bounds chat request bodies. Zero defaults to 10 MiB.
	MaxRequestBytes int64
	// RunOptions are copied and reused across requests.
	RunOptions []ai.RunOption
}

// NewHandler builds the official web UI and its chat, configuration, and health endpoints.
// Deps, models, native tools, and run options must be safe for concurrent requests.
func NewHandler[Deps, Output any](
	agent *ai.Agent[Deps, Output], deps Deps, config Config,
) (http.Handler, error) {
	if agent == nil {
		return nil, fmt.Errorf("webchat: agent must not be nil")
	}
	if config.MaxRequestBytes < 0 {
		return nil, fmt.Errorf("webchat: maximum request bytes must not be negative")
	}
	allowedHosts, err := normalizeAllowedHosts(config.AllowedHosts)
	if err != nil {
		return nil, err
	}
	configuration, err := newFrontendConfiguration(
		agent.Model(), config.DefaultModelID, config.DefaultModelName, config.Models, config.NativeTools,
	)
	if err != nil {
		return nil, err
	}
	loader, err := newHTMLLoader(config)
	if err != nil {
		return nil, err
	}
	options := append([]ai.RunOption(nil), config.RunOptions...)
	if config.MCPConfigPath != "" {
		toolsets, err := mcp.LoadToolsets[Deps](config.MCPConfigPath)
		if err != nil {
			return nil, fmt.Errorf("webchat: load MCP configuration: %w", err)
		}
		options = append(options, ai.WithRunToolsets(toolsets...))
	}
	adapter := vercel.NewAdapter(agent, vercel.Config{
		SDKVersion: 7, Sanitization: config.Sanitization, MaxRequestBytes: config.MaxRequestBytes,
	})
	chat := newChatHandler(adapter, deps, options, configuration, config.MaxRequestBytes)
	return http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if !hostAllowed(request.Host, allowedHosts) {
			http.Error(response, "misdirected request", http.StatusMisdirectedRequest)
			return
		}
		switch request.URL.Path {
		case "/api/chat":
			chat.ServeHTTP(response, request)
		case "/api/configure":
			serveConfiguration(response, request, configuration)
		case "/api/health":
			serveHealth(response, request)
		default:
			if isUIPath(request.URL.Path) {
				loader.ServeHTTP(response, request)
				return
			}
			http.NotFound(response, request)
		}
	}), nil
}

func isUIPath(path string) bool {
	if path == "/" {
		return true
	}
	if len(path) < 2 || path[0] != '/' {
		return false
	}
	for _, character := range path[1:] {
		if character == '/' {
			return false
		}
	}
	return true
}
