package openai

import (
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
)

type chatWebSearchOptions struct {
	SearchContextSize ai.WebSearchContextSize    `json:"search_context_size"`
	UserLocation      *chatWebSearchUserLocation `json:"user_location,omitempty"`
}

type chatWebSearchUserLocation struct {
	Type        string                               `json:"type"`
	Approximate chatWebSearchUserLocationApproximate `json:"approximate"`
}

type chatWebSearchUserLocationApproximate struct {
	City     string `json:"city,omitempty"`
	Country  string `json:"country,omitempty"`
	Region   string `json:"region,omitempty"`
	Timezone string `json:"timezone,omitempty"`
}

func supportsChatWebSearch(modelName string, override *bool) bool {
	if override != nil {
		return *override
	}
	return strings.Contains(modelName, "-search-preview")
}
