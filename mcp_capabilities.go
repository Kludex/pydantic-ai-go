package ai

import "context"

// MCPServerCapabilityConfig configures a provider-hosted MCP server and an optional local client toolset.
type MCPServerCapabilityConfig[Deps any] struct {
	Native MCPServerTool
	Local  Toolset[Deps]
}

// NewMCPServerCapability creates native-first MCP access. A nil Local requires native support.
// Native AllowedTools also filters the local toolset so both paths expose the same names.
func NewMCPServerCapability[Deps any](
	config MCPServerCapabilityConfig[Deps],
) *NativeOrLocalTool[Deps] {
	local := config.Local
	if !toolsetIsNil(local) && config.Native.AllowedTools != nil {
		allowed := make(map[string]struct{}, len(config.Native.AllowedTools))
		for _, name := range config.Native.AllowedTools {
			allowed[name] = struct{}{}
		}
		local = FilterToolset(local, func(
			_ context.Context, _ *RunContext[Deps], definition ToolDefinition,
		) (bool, error) {
			_, ok := allowed[definition.Name]
			return ok, nil
		})
	}
	var options []NativeOrLocalOption
	if toolsetIsNil(local) {
		options = nativeRequirementOption("no local MCP fallback was configured")
	}
	return NewNativeOrLocalToolset(config.Native, local, options...)
}
