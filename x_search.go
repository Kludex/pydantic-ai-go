package ai

import (
	"context"
	"fmt"
)

// XSearchCapabilityConfig configures native X search and an optional custom local fallback.
type XSearchCapabilityConfig[Deps any] struct {
	Native XSearchTool
	Local  Toolset[Deps]
}

// NewXSearchCapability creates native-first X search. A nil Local requires native support.
// Handle filters require native support because an arbitrary local toolset cannot enforce them.
func NewXSearchCapability[Deps any](config XSearchCapabilityConfig[Deps]) *NativeOrLocalTool[Deps] {
	reason := xSearchNativeRequirement(config.Native)
	if toolsetIsNil(config.Local) {
		reason = "no local X-search fallback was configured"
	}
	return NewNativeOrLocalToolset(config.Native, config.Local, nativeRequirementOption(reason)...)
}

// XSearchFunc resolves native X-search settings before a model request.
// It may run concurrently and must not return shared mutable state.
type XSearchFunc[Deps any] func(ctx context.Context, rc *RunContext[Deps]) (XSearchTool, error)

// NewDynamicXSearchCapability creates dependency-aware native-first X search.
func NewDynamicXSearchCapability[Deps any](
	resolve XSearchFunc[Deps], local Toolset[Deps], options ...NativeOrLocalOption,
) *NativeOrLocalTool[Deps] {
	if resolve == nil {
		panic("ai: dynamic X-search resolver must not be nil")
	}
	options = nativeRequiredWithoutLocal(local, "no local X-search fallback was configured", options)
	requiresNative := applyNativeOrLocalOptions(options).requiredReason != ""
	return NewDynamicNativeOrLocalToolset(
		"x_search",
		func(ctx context.Context, rc *RunContext[Deps]) (NativeTool, error) {
			tool, err := resolve(ctx, rc)
			if err == nil && !requiresNative {
				if reason := xSearchNativeRequirement(tool); reason != "" {
					return nil, fmt.Errorf(
						"ai: dynamic X search returned native-only constraint(s) %s with a local fallback; use WithNativeRequired",
						reason,
					)
				}
			}
			return tool, err
		},
		local,
		options...,
	)
}

func xSearchNativeRequirement(tool XSearchTool) string {
	switch {
	case tool.AllowedXHandles != nil:
		return "allowed X handles"
	case tool.ExcludedXHandles != nil:
		return "excluded X handles"
	default:
		return ""
	}
}
