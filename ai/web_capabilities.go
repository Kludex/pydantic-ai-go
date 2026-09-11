package ai

import (
	"context"
	"fmt"
	"strings"
)

// WebSearchCapabilityConfig configures native web search and an optional local fallback.
type WebSearchCapabilityConfig[Deps any] struct {
	// Native configures provider-hosted search.
	Native WebSearchTool
	// Local is the lifecycle-aware fallback toolset.
	Local Toolset[Deps]
}

// NewWebSearchCapability creates native-first web search. A nil Local requires native support.
func NewWebSearchCapability[Deps any](
	config WebSearchCapabilityConfig[Deps],
) *NativeOrLocalTool[Deps] {
	reason := webSearchNativeRequirement(config.Native)
	options := nativeRequirementOption(reason)
	if toolsetIsNil(config.Local) {
		options = nativeFallbackRequirementOption("no local web-search fallback was configured")
	}
	return NewNativeOrLocalToolset(config.Native, config.Local, options...)
}

// WebSearchFunc resolves native web-search settings before a model request.
// It may run concurrently and must not return shared mutable state.
type WebSearchFunc[Deps any] func(
	ctx context.Context, rc *RunContext[Deps],
) (WebSearchTool, error)

// NewDynamicWebSearchCapability creates dependency-aware native-first web search.
func NewDynamicWebSearchCapability[Deps any](
	resolve WebSearchFunc[Deps], local Toolset[Deps], options ...NativeOrLocalOption,
) *NativeOrLocalTool[Deps] {
	if resolve == nil {
		panic("ai: dynamic web-search resolver must not be nil")
	}
	options = nativeRequiredWithoutLocal(local, "no local web-search fallback was configured", options)
	requiresNative := applyNativeOrLocalOptions(options).requiredReason != ""
	return NewDynamicNativeOrLocalToolset(
		"web_search",
		func(ctx context.Context, rc *RunContext[Deps]) (NativeTool, error) {
			tool, err := resolve(ctx, rc)
			if err == nil && !requiresNative {
				if reason := webSearchNativeRequirement(tool); reason != "" {
					return nil, dynamicWebConstraintError("search", reason)
				}
			}
			return tool, err
		},
		local,
		options...,
	)
}

// NewWebSearchCapabilityWithDuckDuckGo adds the built-in DuckDuckGo local fallback.
func NewWebSearchCapabilityWithDuckDuckGo[Deps any](
	native WebSearchTool, local LocalWebSearchConfig,
) *NativeOrLocalTool[Deps] {
	tool := NewLocalWebSearchTool[Deps](local)
	return NewWebSearchCapability(WebSearchCapabilityConfig[Deps]{
		Native: native, Local: NewFunctionToolset(tool),
	})
}

// WebFetchCapabilityConfig configures native URL fetching and an optional local fallback.
type WebFetchCapabilityConfig[Deps any] struct {
	// Native configures provider-hosted URL retrieval.
	Native WebFetchTool
	// Local is the lifecycle-aware fallback toolset.
	Local Toolset[Deps]
}

// NewWebFetchCapability creates native-first URL fetching. A nil Local requires native support.
func NewWebFetchCapability[Deps any](
	config WebFetchCapabilityConfig[Deps],
) *NativeOrLocalTool[Deps] {
	reason := webFetchNativeRequirement(config.Native)
	options := nativeRequirementOption(reason)
	if toolsetIsNil(config.Local) {
		options = nativeFallbackRequirementOption("no local web-fetch fallback was configured")
	}
	return NewNativeOrLocalToolset(config.Native, config.Local, options...)
}

// WebFetchFunc resolves native web-fetch settings before a model request.
// It may run concurrently and must not return shared mutable state.
type WebFetchFunc[Deps any] func(
	ctx context.Context, rc *RunContext[Deps],
) (WebFetchTool, error)

// NewDynamicWebFetchCapability creates dependency-aware native-first URL fetching.
func NewDynamicWebFetchCapability[Deps any](
	resolve WebFetchFunc[Deps], local Toolset[Deps], options ...NativeOrLocalOption,
) *NativeOrLocalTool[Deps] {
	if resolve == nil {
		panic("ai: dynamic web-fetch resolver must not be nil")
	}
	options = nativeRequiredWithoutLocal(local, "no local web-fetch fallback was configured", options)
	requiresNative := applyNativeOrLocalOptions(options).requiredReason != ""
	return NewDynamicNativeOrLocalToolset(
		"web_fetch",
		func(ctx context.Context, rc *RunContext[Deps]) (NativeTool, error) {
			tool, err := resolve(ctx, rc)
			if err == nil && !requiresNative {
				if reason := webFetchNativeRequirement(tool); reason != "" {
					return nil, dynamicWebConstraintError("fetch", reason)
				}
			}
			return tool, err
		},
		local,
		options...,
	)
}

// NewWebFetchCapabilityWithLocal adds the built-in SSRF-protected local fallback.
// Native domain filters override matching local config fields so both paths enforce them.
func NewWebFetchCapabilityWithLocal[Deps any](
	native WebFetchTool, local LocalWebFetchConfig,
) *NativeOrLocalTool[Deps] {
	if native.AllowedDomains != nil {
		local.AllowedDomains = append([]string(nil), native.AllowedDomains...)
	}
	if native.BlockedDomains != nil {
		local.BlockedDomains = append([]string(nil), native.BlockedDomains...)
	}
	tool := NewLocalWebFetchTool[Deps](local)
	return NewWebFetchCapability(WebFetchCapabilityConfig[Deps]{
		Native: native, Local: NewFunctionToolset(tool),
	})
}

func webSearchNativeRequirement(tool WebSearchTool) string {
	requirements := make([]string, 0, 4)
	if tool.BlockedDomains != nil {
		requirements = append(requirements, "blocked domains")
	}
	if tool.AllowedDomains != nil {
		requirements = append(requirements, "allowed domains")
	}
	if tool.MaxUses != 0 {
		requirements = append(requirements, "maximum uses")
	}
	if tool.ExternalWebAccess != nil && !*tool.ExternalWebAccess {
		requirements = append(requirements, "disabled external web access")
	}
	return strings.Join(requirements, ", ")
}

func webFetchNativeRequirement(tool WebFetchTool) string {
	if tool.MaxUses != 0 {
		return "maximum uses"
	}
	return ""
}

func nativeRequirementOption(reason string) []NativeOrLocalOption {
	if reason == "" {
		return nil
	}
	return []NativeOrLocalOption{WithNativeRequired(reason)}
}

func nativeFallbackRequirementOption(reason string) []NativeOrLocalOption {
	return []NativeOrLocalOption{func(config *nativeOrLocalConfig) {
		config.requiredReason = reason
		config.requiredWithoutFallback = true
	}}
}

func dynamicWebConstraintError(kind string, reason string) error {
	return fmt.Errorf(
		"ai: dynamic web-%s returned native-only constraint(s) %s with a local fallback; use WithNativeRequired",
		kind, reason,
	)
}

func nativeRequiredWithoutLocal[Deps any](
	local Toolset[Deps], reason string, options []NativeOrLocalOption,
) []NativeOrLocalOption {
	if !toolsetIsNil(local) || applyNativeOrLocalOptions(options).requiredReason != "" {
		return options
	}
	cloned := append([]NativeOrLocalOption(nil), options...)
	return append(cloned, nativeFallbackRequirementOption(reason)...)
}
