package vercel

import (
	"encoding/json"
	"fmt"
	"regexp"
	"slices"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
)

const toolResultChunksMetadataKey = "pydantic_ai_go_vercel_chunks"

var toolNamePattern = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// ToolResultMetadata creates detached metadata for data, source, or file chunks emitted after a tool result.
func ToolResultMetadata(chunks ...Chunk) (map[string]any, error) {
	values := make([]any, 0, len(chunks))
	for index, chunk := range chunks {
		if err := validateToolResultChunk(chunk); err != nil {
			return nil, fmt.Errorf("vercel: tool result chunk %d: %w", index, err)
		}
		encoded, err := json.Marshal(chunk)
		if err != nil {
			return nil, fmt.Errorf("vercel: encode tool result chunk %d: %w", index, err)
		}
		var value map[string]any
		_ = json.Unmarshal(encoded, &value)
		values = append(values, value)
	}
	return map[string]any{toolResultChunksMetadataKey: values}, nil
}

func toolResultChunks(metadata map[string]any) ([]Chunk, error) {
	value, ok := metadata[toolResultChunksMetadataKey]
	if !ok {
		return nil, nil
	}
	values, ok := value.([]any)
	if !ok {
		return nil, fmt.Errorf("vercel: tool result chunks metadata must be an array")
	}
	chunks := make([]Chunk, 0, len(values))
	for index, value := range values {
		encoded, err := json.Marshal(value)
		if err != nil {
			return nil, fmt.Errorf("vercel: encode tool result chunk %d: %w", index, err)
		}
		var chunk Chunk
		if err := json.Unmarshal(encoded, &chunk); err != nil {
			return nil, fmt.Errorf("vercel: decode tool result chunk %d: %w", index, err)
		}
		if err := validateToolResultChunk(chunk); err != nil {
			return nil, fmt.Errorf("vercel: tool result chunk %d: %w", index, err)
		}
		chunks = append(chunks, chunk)
	}
	return chunks, nil
}

func validateToolResultChunk(chunk Chunk) error {
	switch {
	case strings.HasPrefix(string(chunk.Type), "data-"):
		return nil
	case chunk.Type == ChunkSourceURL:
		if chunk.SourceID == "" || chunk.URL == "" {
			return fmt.Errorf("source-url requires sourceId and url")
		}
	case chunk.Type == ChunkSourceDocument:
		if chunk.SourceID == "" || chunk.MediaType == "" || chunk.Title == "" {
			return fmt.Errorf("source-document requires sourceId, mediaType, and title")
		}
	case chunk.Type == ChunkFile:
		if chunk.URL == "" || chunk.MediaType == "" {
			return fmt.Errorf("file requires url and mediaType")
		}
	default:
		return fmt.Errorf("unsupported data-carrying chunk type %q", chunk.Type)
	}
	return nil
}

func compactionPart(data map[string]any) (ai.CompactionPart, bool) {
	content, contentOK := data["content"].(string)
	id, idOK := data["id"].(string)
	providerName, providerOK := data["provider_name"].(string)
	providerDetails, detailsOK := data["provider_details"].(map[string]any)
	if (!contentOK && data["content"] != nil) || (!idOK && data["id"] != nil) ||
		(!providerOK && data["provider_name"] != nil) || (!detailsOK && data["provider_details"] != nil) {
		return ai.CompactionPart{}, false
	}
	if providerName == "" && (id != "" || len(providerDetails) > 0) {
		return ai.CompactionPart{}, false
	}
	if content == "" && id == "" && providerName == "" && len(providerDetails) == 0 {
		return ai.CompactionPart{}, false
	}
	return ai.CompactionPart{
		Content: content, ID: id, ProviderName: providerName, ProviderDetails: providerDetails,
	}, true
}

func compactionData(part ai.CompactionPart) map[string]any {
	data := map[string]any{}
	if part.Content != "" {
		data["content"] = part.Content
	}
	if part.ID != "" {
		data["id"] = part.ID
	}
	if part.ProviderName != "" {
		data["provider_name"] = part.ProviderName
	}
	if len(part.ProviderDetails) > 0 {
		data["provider_details"] = part.ProviderDetails
	}
	return data
}

func toolAvailabilityPart(data map[string]any) ai.ToolAvailabilityDeltaPart {
	part := ai.ToolAvailabilityDeltaPart{}
	if id, ok := data["tool_call_id"].(string); ok {
		part.ToolCallID = id
	}
	switch added := data["added"].(type) {
	case []string:
		part.ToolsAdded = slices.Clone(added)
	case []any:
		for _, value := range added {
			if name, ok := value.(string); ok && toolNamePattern.MatchString(name) {
				part.ToolsAdded = append(part.ToolsAdded, name)
			}
		}
	}
	return part
}

func toolAvailabilityData(part ai.ToolAvailabilityDeltaPart) map[string]any {
	return map[string]any{
		"added":        slices.Clone(part.ToolsAdded),
		"tool_call_id": part.ToolCallID,
	}
}
