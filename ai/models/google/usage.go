package google

import (
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func (response generateResponse) usage(modelName string) ai.Usage {
	usage := response.UsageMetadata.usage()
	queries := make(map[string]bool)
	returnedSource := false
	for _, candidate := range response.Candidates {
		values, _ := candidate.GroundingMetadata["webSearchQueries"].([]any)
		for _, value := range values {
			if query, ok := value.(string); ok && strings.TrimSpace(query) != "" {
				queries[query] = true
			}
		}
		for _, source := range googleWebSearchSources(candidate.GroundingMetadata) {
			if uri, _ := source["uri"].(string); uri != "" {
				returnedSource = true
			}
		}
	}
	if len(queries) > 0 {
		usage.Details["web_search_requests"] = len(queries)
		searches := len(queries)
		if response.ModelVersion != "" {
			modelName = response.ModelVersion
		}
		if strings.HasPrefix(modelName, "gemini-1") || strings.HasPrefix(modelName, "gemini-2") {
			searches = 0
			if returnedSource {
				searches = 1
			}
		}
		usage.Details["web_searches"] = searches
	}
	return usage
}

func googleWebSearchSources(metadata map[string]any) []map[string]any {
	var sources []map[string]any
	chunks, _ := metadata["groundingChunks"].([]any)
	for _, value := range chunks {
		chunk, _ := value.(map[string]any)
		if web, ok := chunk["web"].(map[string]any); ok {
			sources = append(sources, cloneGoogleMap(web))
		}
	}
	return sources
}
