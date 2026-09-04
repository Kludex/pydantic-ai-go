package vercel

import (
	"encoding/json"
	"fmt"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
)

const providerMetadataKey = "pydantic_ai"

type partMetadata struct {
	id              string
	signature       string
	providerName    string
	providerDetails map[string]any
	toolKind        ai.ToolPartKind
	vendorMetadata  map[string]any
	identifier      string
	fileID          string
	forceDownload   ai.FileDownloadMode
}

func loadPartMetadata(metadata map[string]any) partMetadata {
	wrapped, ok := metadata[providerMetadataKey].(map[string]any)
	if !ok {
		return partMetadata{}
	}
	result := partMetadata{}
	result.id, _ = wrapped["id"].(string)
	result.signature, _ = wrapped["signature"].(string)
	result.providerName, _ = wrapped["provider_name"].(string)
	result.providerDetails, _ = wrapped["provider_details"].(map[string]any)
	result.vendorMetadata, _ = wrapped["vendor_metadata"].(map[string]any)
	result.identifier, _ = wrapped["identifier"].(string)
	result.fileID, _ = wrapped["file_id"].(string)
	if value, ok := wrapped["force_download"].(string); ok {
		result.forceDownload = ai.FileDownloadMode(value)
	}
	if value, ok := wrapped["tool_kind"].(string); ok {
		result.toolKind = toolPartKind(value)
	}
	return result
}

func toolPartKind(value string) ai.ToolPartKind {
	switch kind := ai.ToolPartKind(value); kind {
	case ai.ToolPartKindToolSearch, ai.ToolPartKindCapabilityLoad, ai.ToolPartKindWebSearch,
		ai.ToolPartKindWebFetch, ai.ToolPartKindCodeExecution, ai.ToolPartKindImageGeneration,
		ai.ToolPartKindFileSearch, ai.ToolPartKindMCPServer, ai.ToolPartKindAdvisor:
		return kind
	default:
		return ""
	}
}

func dumpPartMetadata(
	id string,
	signature string,
	providerName string,
	providerDetails map[string]any,
	toolKind ai.ToolPartKind,
	vendorMetadata map[string]any,
) map[string]any {
	values := map[string]any{}
	if id != "" {
		values["id"] = id
	}
	if signature != "" {
		values["signature"] = signature
	}
	if providerName != "" {
		values["provider_name"] = providerName
	}
	if len(providerDetails) > 0 {
		values["provider_details"] = cloneMap(providerDetails)
	}
	if toolKind != "" {
		values["tool_kind"] = string(toolKind)
	}
	if len(vendorMetadata) > 0 {
		values["vendor_metadata"] = cloneMap(vendorMetadata)
	}
	if len(values) == 0 {
		return nil
	}
	return map[string]any{providerMetadataKey: values}
}

func loadMessageMetadata(metadata map[string]any) (map[string]any, time.Time) {
	application := cloneMap(metadata)
	delete(application, providerMetadataKey)
	if len(application) == 0 {
		application = nil
	}
	var timestamp time.Time
	if wrapped, ok := metadata[providerMetadataKey].(map[string]any); ok {
		if value, ok := wrapped["timestamp"].(string); ok {
			timestamp, _ = time.Parse(time.RFC3339Nano, value)
		}
	}
	return application, timestamp
}

func dumpMessageMetadata(metadata map[string]any, timestamp time.Time) map[string]any {
	result := cloneMap(metadata)
	if result == nil {
		result = map[string]any{}
	}
	if !timestamp.IsZero() {
		result[providerMetadataKey] = map[string]any{"timestamp": timestamp.Format(time.RFC3339Nano)}
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

func withExternalCallIDs(metadata map[string]any, callIDs []string) map[string]any {
	result := cloneMap(metadata)
	if result == nil {
		result = map[string]any{}
	}
	wrapped, _ := result[providerMetadataKey].(map[string]any)
	if wrapped == nil {
		wrapped = map[string]any{}
	}
	wrapped["external_tool_call_ids"] = append([]string(nil), callIDs...)
	result[providerMetadataKey] = wrapped
	return result
}

func externalCallIDs(metadata map[string]any) ([]string, error) {
	wrapped, ok := metadata[providerMetadataKey].(map[string]any)
	if !ok {
		return nil, nil
	}
	value, exists := wrapped["external_tool_call_ids"]
	if !exists {
		return nil, nil
	}
	var result []string
	switch values := value.(type) {
	case []string:
		result = append(result, values...)
	case []any:
		result = make([]string, 0, len(values))
		for _, value := range values {
			callID, ok := value.(string)
			if !ok {
				return nil, fmt.Errorf("vercel: external tool call IDs must be strings")
			}
			result = append(result, callID)
		}
	default:
		return nil, fmt.Errorf("vercel: external tool call IDs must be an array")
	}
	seen := make(map[string]struct{}, len(result))
	for _, callID := range result {
		if callID == "" {
			return nil, fmt.Errorf("vercel: external tool call ID must not be empty")
		}
		if _, ok := seen[callID]; ok {
			return nil, fmt.Errorf("vercel: duplicate external tool call ID %q", callID)
		}
		seen[callID] = struct{}{}
	}
	return result, nil
}

func cloneMap(value map[string]any) map[string]any {
	if value == nil {
		return nil
	}
	encoded, _ := json.Marshal(value)
	var cloned map[string]any
	_ = json.Unmarshal(encoded, &cloned)
	return cloned
}
