package ai

import (
	"context"
	"fmt"
)

func registerNativeOrLocal[Deps any](
	entries []nativeToolEntry[Deps], toolsets []Toolset[Deps], registrations []any,
) ([]nativeToolEntry[Deps], []Toolset[Deps], error) {
	candidateEntries := cloneNativeToolEntries(entries)
	candidateToolsets := append([]Toolset[Deps](nil), toolsets...)
	for _, erased := range registrations {
		registration, ok := erased.(nativeOrLocalRegistration[Deps])
		if !ok {
			return nil, nil, fmt.Errorf("ai: native-or-local dependencies do not match agent")
		}
		if toolsetIsNil(registration.local) {
			return nil, nil, fmt.Errorf("ai: native-or-local local toolset must not be nil")
		}
		entry := nativeToolEntry[Deps]{
			tool: registration.native, fn: registration.resolve,
			expectedID: registration.nativeID, requiredReason: registration.requiredReason,
		}
		if entry.fn == nil {
			if err := ValidateNativeTools([]NativeTool{entry.tool}); err != nil {
				return nil, nil, fmt.Errorf("ai: native-or-local tool: %w", err)
			}
			entry.expectedID = entry.tool.UniqueID()
			if entry.requiredReason != "" && entry.tool.IsOptional() {
				return nil, nil, nativeRequiredOptionalError(entry.expectedID, entry.requiredReason)
			}
		}
		candidateEntries = append(candidateEntries, entry)
		if entry.requiredReason == "" {
			candidateToolsets = append(candidateToolsets, nativeFallbackToolsetID(registration.local, entry.expectedID))
		}
	}
	if err := ValidateNativeTools(staticNativeTools(candidateEntries)); err != nil {
		return nil, nil, err
	}
	return candidateEntries, candidateToolsets, nil
}

func nativeFallbackToolsetID[Deps any](toolset Toolset[Deps], nativeID string) Toolset[Deps] {
	return PrepareToolset(toolset, func(
		_ context.Context, _ *RunContext[Deps], definitions []ToolDefinition,
	) ([]ToolDefinition, error) {
		for index := range definitions {
			definitions[index].NativeFallbackFor = nativeID
		}
		return definitions, nil
	})
}

func nativeRequiredOptionalError(nativeID string, reason string) error {
	return fmt.Errorf(
		"ai: native-or-local tool %q requires native support for %s, but the native definition is optional",
		nativeID, reason,
	)
}
