package bedrock

import (
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/promptcache"
)

// ModelProfile reports request-side cache support for Claude and Nova models.
func (model *Model) ModelProfile() ai.ModelProfile { return cacheProfile(model.name) }

func cacheProfile(name string) ai.ModelProfile {
	profile := ai.ModelProfile{DefaultOutputMode: ai.OutputModeTool}
	name = strings.ToLower(name)
	if strings.Contains(name, "claude") || strings.Contains(name, "amazon.nova") {
		profile.SupportsCache = true
		profile.DefaultCacheRetention = 5 * time.Minute
		profile.SupportedCacheRetentions = []ai.CacheRetention{ai.CacheRetention5Minutes}
		if strings.Contains(name, "claude") {
			profile.SupportedCacheRetentions = promptcache.BedrockRetentions(name)
		}
	}
	return profile
}

func hasCacheSettings(settings ai.ModelSettings) bool {
	return promptcache.HasSettings(settings, cacheInstructionsSetting, cacheMessagesSetting, cacheToolDefinitionsSetting)
}

// CachingNotEnabled reports missing request-side cache configuration on cacheable models.
func (model *Model) CachingNotEnabled(settings ai.ModelSettings) bool {
	return model.ModelProfile().SupportsCache && settings.Cache == nil && !hasCacheSettings(settings)
}

func translateCache(name string, settings ai.ModelSettings) (ai.ModelSettings, error) {
	cache, err := ai.ResolveCacheConfig(settings, cacheProfile(name))
	if err != nil || cache == nil || hasCacheSettings(settings) {
		return settings, err
	}
	settings = settings.Clone()
	if settings.ExtraBody == nil {
		settings.ExtraBody = map[string]any{}
	}
	ttl := CacheTTL(cache.Retention)
	if ttl == "" {
		ttl = "default"
	}
	settings.ExtraBody[cacheInstructionsSetting] = ttl
	if strings.Contains(name, "claude") {
		settings.ExtraBody[cacheToolDefinitionsSetting] = ttl
	}
	if cache.Messages == nil || *cache.Messages {
		settings.ExtraBody[cacheMessagesSetting] = ttl
	}
	return settings, nil
}
