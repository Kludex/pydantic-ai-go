package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

var environmentReference = regexp.MustCompile(`\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}`)

type fileConfig struct {
	Servers map[string]serverConfig `json:"mcpServers"`
}

type serverConfig struct {
	Command string            `json:"command"`
	Args    []string          `json:"args"`
	Env     map[string]string `json:"env"`
	CWD     string            `json:"cwd"`
	URL     string            `json:"url"`
	Headers map[string]string `json:"headers"`
}

// LoadToolsets reads the common mcpServers JSON format. Each returned toolset
// is prefixed with and identified by its server name.
func LoadToolsets[Deps any](path string, opts ...Option) ([]ai.Toolset[Deps], error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("ai/mcp: read config %q: %w", path, err)
	}
	var raw any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("ai/mcp: decode config %q: %w", path, err)
	}
	expanded, err := expandEnvironment(raw)
	if err != nil {
		return nil, fmt.Errorf("ai/mcp: config %q: %w", path, err)
	}
	object, ok := expanded.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("ai/mcp: expected JSON object at root of %q", path)
	}
	serversValue, exists := object["mcpServers"]
	if !exists {
		return nil, fmt.Errorf("ai/mcp: expected mcpServers object in %q", path)
	}
	serversObject, ok := serversValue.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("ai/mcp: expected mcpServers object in %q", path)
	}
	encoded, _ := json.Marshal(map[string]any{"mcpServers": serversObject})
	var config fileConfig
	if err := json.Unmarshal(encoded, &config); err != nil {
		return nil, fmt.Errorf("ai/mcp: decode server config %q: %w", path, err)
	}
	names := make([]string, 0, len(config.Servers))
	for name := range config.Servers {
		names = append(names, name)
	}
	sort.Strings(names)
	toolsets := make([]ai.Toolset[Deps], 0, len(names))
	for _, name := range names {
		configured, err := configuredToolset[Deps](name, config.Servers[name], opts)
		if err != nil {
			return nil, err
		}
		toolsets = append(toolsets, ai.PrefixToolset[Deps](configured, name))
	}
	return toolsets, nil
}

func configuredToolset[Deps any](name string, server serverConfig, opts []Option) (*Toolset[Deps], error) {
	if name == "" {
		return nil, fmt.Errorf("ai/mcp: server name must not be empty")
	}
	options := append([]Option(nil), opts...)
	options = append(options, WithID(name))
	switch {
	case server.Command != "" && server.URL != "":
		return nil, fmt.Errorf("ai/mcp: server %q must have either command or url, not both", name)
	case server.Command != "":
		return NewToolset(func(context.Context, *ai.RunContext[Deps]) (mcpsdk.Transport, error) {
			command := exec.Command(server.Command, server.Args...)
			command.Dir = server.CWD
			if len(server.Env) > 0 {
				command.Env = mergedEnvironment(server.Env)
			}
			return &mcpsdk.CommandTransport{Command: command}, nil
		}, options...), nil
	case server.URL != "":
		endpoint, err := url.ParseRequestURI(server.URL)
		if err != nil || (endpoint.Scheme != "http" && endpoint.Scheme != "https") {
			return nil, fmt.Errorf("ai/mcp: server %q has invalid HTTP URL %q", name, server.URL)
		}
		return NewToolset(func(context.Context, *ai.RunContext[Deps]) (mcpsdk.Transport, error) {
			client := &http.Client{Transport: headerTransport{
				base: http.DefaultTransport, headers: cloneHeaders(server.Headers),
			}}
			if strings.HasSuffix(strings.TrimRight(endpoint.Path, "/"), "/sse") {
				return &mcpsdk.SSEClientTransport{Endpoint: server.URL, HTTPClient: client}, nil
			}
			return &mcpsdk.StreamableClientTransport{Endpoint: server.URL, HTTPClient: client}, nil
		}, options...), nil
	default:
		return nil, fmt.Errorf("ai/mcp: server %q must have either command or url", name)
	}
}

func expandEnvironment(value any) (any, error) {
	switch value := value.(type) {
	case string:
		var expansionErr error
		expanded := environmentReference.ReplaceAllStringFunc(value, func(reference string) string {
			matches := environmentReference.FindStringSubmatch(reference)
			if replacement, exists := os.LookupEnv(matches[1]); exists {
				return replacement
			}
			if matches[2] != "" {
				return matches[3]
			}
			expansionErr = fmt.Errorf("environment variable ${%s} is not defined", matches[1])
			return reference
		})
		return expanded, expansionErr
	case []any:
		expanded := make([]any, len(value))
		for index, item := range value {
			var err error
			expanded[index], err = expandEnvironment(item)
			if err != nil {
				return nil, err
			}
		}
		return expanded, nil
	case map[string]any:
		expanded := make(map[string]any, len(value))
		for key, item := range value {
			var err error
			expanded[key], err = expandEnvironment(item)
			if err != nil {
				return nil, err
			}
		}
		return expanded, nil
	default:
		return value, nil
	}
}

func mergedEnvironment(overrides map[string]string) []string {
	environment := make(map[string]string)
	for _, entry := range os.Environ() {
		key, value, _ := strings.Cut(entry, "=")
		environment[key] = value
	}
	for key, value := range overrides {
		environment[key] = value
	}
	keys := make([]string, 0, len(environment))
	for key := range environment {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	values := make([]string, len(keys))
	for index, key := range keys {
		values[index] = key + "=" + environment[key]
	}
	return values
}

type headerTransport struct {
	base    http.RoundTripper
	headers map[string]string
}

func (t headerTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	cloned := request.Clone(request.Context())
	cloned.Header = request.Header.Clone()
	for key, value := range t.headers {
		cloned.Header.Set(key, value)
	}
	return t.base.RoundTrip(cloned)
}

func cloneHeaders(headers map[string]string) map[string]string {
	cloned := make(map[string]string, len(headers))
	for key, value := range headers {
		cloned[key] = value
	}
	return cloned
}
