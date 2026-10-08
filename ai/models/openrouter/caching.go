package openrouter

import (
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/promptcache"
)

func hasCacheSettings(settings ai.ModelSettings) bool {
	return promptcache.HasSettings(settings, cacheInstructionsKey, cacheMessagesKey, cacheToolsKey)
}

// CachingNotEnabled reports missing request-side configuration on Anthropic routes.
func (model *Model) CachingNotEnabled(settings ai.ModelSettings) bool {
	return strings.HasPrefix(strings.TrimPrefix(model.Name(), "~"), "anthropic/") &&
		settings.Cache == nil && !hasCacheSettings(settings)
}

func (model *Model) translateCache(settings ai.ModelSettings) (ai.ModelSettings, error) {
	cache, err := ai.ResolveCacheConfig(settings, model.ModelProfile())
	if err != nil || cache == nil || hasCacheSettings(settings) {
		return settings, err
	}
	settings = settings.Clone()
	if settings.ExtraBody == nil {
		settings.ExtraBody = map[string]any{}
	}
	ttl := CacheTTL(cache.Retention)
	if ttl == "" {
		ttl = CacheTTL5Minutes
	}
	settings.ExtraBody[cacheInstructionsKey] = ttl
	settings.ExtraBody[cacheToolsKey] = ttl
	if cache.Messages == nil || *cache.Messages {
		settings.ExtraBody[cacheMessagesKey] = ttl
	}
	return settings, nil
}
