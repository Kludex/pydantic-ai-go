package anthropic

import (
	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/promptcache"
)

func hasCacheSettings(settings ai.ModelSettings) bool {
	return promptcache.HasSettings(settings,
		cacheSetting, cacheInstructionsSetting, cacheMessagesSetting, cacheToolDefinitionsSetting,
	)
}

// CachingNotEnabled reports a request with no portable or Anthropic cache configuration.
func (m *Model) CachingNotEnabled(settings ai.ModelSettings) bool {
	return settings.Cache == nil && !hasCacheSettings(settings)
}

func (m *Model) translateCache(settings ai.ModelSettings) (ai.ModelSettings, error) {
	cache, err := ai.ResolveCacheConfig(settings, m.ModelProfile())
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
	messages := cache.Messages == nil || *cache.Messages
	if messages && m.ModelProfile().SupportsAutoCache {
		settings.ExtraBody[cacheSetting] = ttl
	} else {
		settings.ExtraBody[cacheInstructionsSetting] = ttl
		settings.ExtraBody[cacheToolDefinitionsSetting] = ttl
		if messages {
			settings.ExtraBody[cacheMessagesSetting] = ttl
		}
	}
	return settings, nil
}
