package openai

import (
	"slices"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

type chatWebSearchOptions struct {
	SearchContextSize ai.WebSearchContextSize    `json:"search_context_size"`
	UserLocation      *chatWebSearchUserLocation `json:"user_location,omitempty"`
	Filters           *chatWebSearchFilters      `json:"filters,omitempty"`
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

type chatWebSearchFilters struct {
	AllowedDomains []string `json:"allowed_domains,omitempty"`
	BlockedDomains []string `json:"blocked_domains,omitempty"`
}

func supportsChatWebSearch(modelName string, override *bool) bool {
	if override != nil {
		return *override
	}
	return strings.Contains(modelName, "-search-preview")
}

// chatWebSearchOptionsFrom builds a chatWebSearchOptions from a portable
// WebSearchTool, forwarding allowed and blocked domains through the same
// `filters` shape OpenAI Responses uses.
func chatWebSearchOptionsFrom(webSearch ai.WebSearchTool, contextSize ai.WebSearchContextSize) chatWebSearchOptions {
	if contextSize == "" {
		contextSize = ai.WebSearchContextMedium
	}
	options := chatWebSearchOptions{SearchContextSize: contextSize}
	if webSearch.UserLocation != nil {
		options.UserLocation = &chatWebSearchUserLocation{
			Type: "approximate",
			Approximate: chatWebSearchUserLocationApproximate{
				City: webSearch.UserLocation.City, Country: webSearch.UserLocation.Country,
				Region: webSearch.UserLocation.Region, Timezone: webSearch.UserLocation.Timezone,
			},
		}
	}
	if len(webSearch.AllowedDomains) > 0 || len(webSearch.BlockedDomains) > 0 {
		options.Filters = &chatWebSearchFilters{
			AllowedDomains: slices.Clone(webSearch.AllowedDomains),
			BlockedDomains: slices.Clone(webSearch.BlockedDomains),
		}
	}
	return options
}
