package promptcache

import (
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// HasSettings reports whether any provider-local cache option is present, including false.
func HasSettings(settings ai.ModelSettings, keys ...string) bool {
	for _, key := range keys {
		if _, exists := settings.ExtraBody[key]; exists {
			return true
		}
	}
	return false
}

// BedrockRetentions returns the cache lifetimes AWS grants to a Claude model.
func BedrockRetentions(name string) []ai.CacheRetention {
	tiers := []ai.CacheRetention{ai.CacheRetention5Minutes}
	for _, prefix := range []string{
		"claude-fable-5", "claude-haiku-4-5", "claude-mythos-5", "claude-opus-4-5", "claude-opus-4-6",
		"claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-4-5", "claude-sonnet-4-6", "claude-sonnet-5",
	} {
		if strings.Contains(name, prefix) {
			return append(tiers, ai.CacheRetention1Hour)
		}
	}
	return tiers
}

// PreviousTail locates the prior request boundary when at least eighteen blocks follow it.
func PreviousTail(roles []string, counts []int) int {
	lastAssistant := -1
	for index := len(roles) - 1; index >= 0; index-- {
		if roles[index] == "assistant" {
			lastAssistant = index
			break
		}
	}
	if lastAssistant < 0 || lastAssistant == len(roles)-1 {
		return -1
	}
	blocks := 0
	for index := len(roles) - 1; index >= 0; index-- {
		if index < lastAssistant && roles[index] != "assistant" {
			if blocks >= 18 {
				return index
			}
			return -1
		}
		blocks += counts[index]
	}
	return -1
}
