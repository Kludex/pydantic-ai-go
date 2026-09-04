package ai

import (
	"context"
	"slices"
)

// MCPServerCapabilityConfig configures a provider-hosted MCP server and an optional local client toolset.
type MCPServerCapabilityConfig[Deps any] struct {
	// Native configures provider-hosted MCP access.
	Native MCPServerTool
	// Local is the lifecycle-aware client fallback.
	Local Toolset[Deps]
}

// NewMCPServerCapability creates native-first MCP access. A nil Local requires native support.
// Native AllowedTools also filters the local toolset so both paths expose the same names.
func NewMCPServerCapability[Deps any](
	config MCPServerCapabilityConfig[Deps],
) *NativeOrLocalTool[Deps] {
	local := filterMCPToolset(config.Local, config.Native.AllowedTools)
	var options []NativeOrLocalOption
	if toolsetIsNil(local) {
		options = nativeRequirementOption("no local MCP fallback was configured")
	}
	return NewNativeOrLocalToolset(config.Native, local, options...)
}

// MCPServerFunc resolves a provider-hosted MCP server before a model request.
// It may run concurrently and must not return shared mutable state.
type MCPServerFunc[Deps any] func(
	ctx context.Context, rc *RunContext[Deps],
) (MCPServerTool, error)

// DynamicMCPServerCapabilityConfig configures dependency-aware native MCP and a local fallback.
type DynamicMCPServerCapabilityConfig[Deps any] struct {
	// ID is the stable server identity every resolved definition must retain.
	ID string
	// Resolve returns detached provider-hosted settings for each request.
	Resolve MCPServerFunc[Deps]
	// Local is the lifecycle-aware client fallback.
	Local Toolset[Deps]
	// AllowedTools filters both native and local paths.
	AllowedTools []string
}

// NewDynamicMCPServerCapability creates dependency-aware native-first MCP access.
// ID is the stable MCP server ID every resolved definition must return.
func NewDynamicMCPServerCapability[Deps any](
	config DynamicMCPServerCapabilityConfig[Deps], options ...NativeOrLocalOption,
) *NativeOrLocalTool[Deps] {
	if config.ID == "" {
		panic("ai: dynamic MCP server ID must not be empty")
	}
	if config.Resolve == nil {
		panic("ai: dynamic MCP server resolver must not be nil")
	}
	allowedTools := slices.Clone(config.AllowedTools)
	local := filterMCPToolset(config.Local, allowedTools)
	options = nativeRequiredWithoutLocal(local, "no local MCP fallback was configured", options)
	return NewDynamicNativeOrLocalToolset(
		"mcp_server:"+config.ID,
		func(ctx context.Context, rc *RunContext[Deps]) (NativeTool, error) {
			tool, err := config.Resolve(ctx, rc)
			tool.AllowedTools = slices.Clone(allowedTools)
			return tool, err
		},
		local,
		options...,
	)
}

func filterMCPToolset[Deps any](local Toolset[Deps], allowedTools []string) Toolset[Deps] {
	if toolsetIsNil(local) || allowedTools == nil {
		return local
	}
	allowed := make(map[string]struct{}, len(allowedTools))
	for _, name := range allowedTools {
		allowed[name] = struct{}{}
	}
	return FilterToolset(local, func(
		_ context.Context, _ *RunContext[Deps], definition ToolDefinition,
	) (bool, error) {
		_, ok := allowed[definition.Name]
		return ok, nil
	})
}
