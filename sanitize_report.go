package ai

import "slices"

func (state *messageSanitizer) report() MessageSanitizationReport {
	return MessageSanitizationReport{
		StrippedSystemPrompts: state.strippedSystems, StrippedCompactionParts: state.strippedCompactions,
		DroppedFileURLSchemes: sortedSet(state.droppedSchemes), ResetFileDownloadModes: sortedModeSet(state.resetModes),
		DroppedUploadedFileProviders: sortedSet(state.droppedProviders),
		StrippedToolCalls:            slices.Clone(state.strippedToolCalls),
	}
}

func sortedSet(values map[string]struct{}) []string {
	result := make([]string, 0, len(values))
	for value := range values {
		result = append(result, value)
	}
	slices.Sort(result)
	return result
}

func sortedModeSet(values map[FileDownloadMode]struct{}) []FileDownloadMode {
	result := make([]FileDownloadMode, 0, len(values))
	for value := range values {
		result = append(result, value)
	}
	slices.Sort(result)
	return result
}
