package openrouter

import (
	"fmt"
	"slices"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func openRouterNativeTool(nativeTool ai.NativeTool) (openai.ChatNativeTool, bool, error) {
	switch tool := nativeTool.(type) {
	case ai.AdvisorTool:
		return openRouterAdvisorTool(tool), true, nil
	case ai.WebSearchTool:
		return openRouterWebSearchTool(tool), true, nil
	default:
		return openai.ChatNativeTool{}, false, nil
	}
}

func openRouterAdvisorTool(tool ai.AdvisorTool) openai.ChatNativeTool {
	parameters := map[string]any{"model": tool.Model, "forward_transcript": false}
	if tool.MaxTokens != nil {
		parameters["max_completion_tokens"] = *tool.MaxTokens
	}
	return openai.ChatNativeTool{Type: "openrouter:advisor", Parameters: parameters}
}

func openRouterWebSearchTool(tool ai.WebSearchTool) openai.ChatNativeTool {
	contextSize := tool.SearchContextSize
	if contextSize == "" {
		contextSize = ai.WebSearchContextMedium
	}
	parameters := map[string]any{"search_context_size": contextSize}
	if tool.UserLocation != nil {
		location := map[string]any{"type": "approximate"}
		for name, value := range map[string]string{
			"city": tool.UserLocation.City, "country": tool.UserLocation.Country,
			"region": tool.UserLocation.Region, "timezone": tool.UserLocation.Timezone,
		} {
			if value != "" {
				location[name] = value
			}
		}
		parameters["user_location"] = location
	}
	if len(tool.AllowedDomains) > 0 {
		parameters["allowed_domains"] = slices.Clone(tool.AllowedDomains)
	}
	if len(tool.BlockedDomains) > 0 {
		parameters["excluded_domains"] = slices.Clone(tool.BlockedDomains)
	}
	if tool.MaxUses != 0 {
		parameters["max_uses"] = tool.MaxUses
	}
	return openai.ChatNativeTool{Type: "openrouter:web_search", Parameters: parameters}
}

func validateOpenRouterTransform(transform Transform) error {
	if transform != TransformMiddleOut {
		return fmt.Errorf("openrouter: invalid transform %q", transform)
	}
	return nil
}
