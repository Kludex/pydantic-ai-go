package ai

import "strings"

// WebSearchCapabilityConfig configures native web search and an optional local fallback.
type WebSearchCapabilityConfig[Deps any] struct {
	Native WebSearchTool
	Local  Toolset[Deps]
}

// NewWebSearchCapability creates native-first web search. A nil Local requires native support.
func NewWebSearchCapability[Deps any](
	config WebSearchCapabilityConfig[Deps],
) *NativeOrLocalTool[Deps] {
	reason := webSearchNativeRequirement(config.Native)
	if toolsetIsNil(config.Local) {
		reason = "no local web-search fallback was configured"
	}
	options := nativeRequirementOption(reason)
	return NewNativeOrLocalToolset(config.Native, config.Local, options...)
}

// WebFetchCapabilityConfig configures native URL fetching and an optional local fallback.
type WebFetchCapabilityConfig[Deps any] struct {
	Native WebFetchTool
	Local  Toolset[Deps]
}

// NewWebFetchCapability creates native-first URL fetching. A nil Local requires native support.
func NewWebFetchCapability[Deps any](
	config WebFetchCapabilityConfig[Deps],
) *NativeOrLocalTool[Deps] {
	reason := webFetchNativeRequirement(config.Native)
	if toolsetIsNil(config.Local) {
		reason = "no local web-fetch fallback was configured"
	}
	options := nativeRequirementOption(reason)
	return NewNativeOrLocalToolset(config.Native, config.Local, options...)
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
