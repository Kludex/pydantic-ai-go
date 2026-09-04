package vercel

import (
	"regexp"
	"slices"

	ai "github.com/Kludex/pydantic-ai-go"
)

var toolNamePattern = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

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
