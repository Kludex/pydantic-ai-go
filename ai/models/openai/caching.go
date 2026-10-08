package openai

import (
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/promptcache"
)

func translateCache(settings ai.ModelSettings, profile ai.ModelProfile) (ai.ModelSettings, error) {
	cache, err := ai.ResolveCacheConfig(settings, profile)
	if err != nil || cache == nil || promptcache.HasSettings(settings, promptCacheOptionsSetting, cacheInstructionsSetting) {
		return settings, err
	}
	settings = settings.Clone()
	if settings.ExtraBody == nil {
		settings.ExtraBody = map[string]any{}
	}
	mode := PromptCacheModeImplicit
	if cache.Messages != nil && !*cache.Messages {
		mode = PromptCacheModeExplicit
	}
	settings.ExtraBody[promptCacheOptionsSetting] = PromptCacheOptions{Mode: mode, TTL: PromptCacheTTL30Minutes}
	settings.ExtraBody[cacheInstructionsSetting] = true
	return settings, nil
}

func cacheRetention(settings ai.ModelSettings, profile ai.ModelProfile) (time.Duration, bool) {
	settings, err := translateCache(settings, profile)
	if err != nil {
		return 0, false
	}
	_, cache, err := extractPromptCacheSettings(settings)
	if err != nil {
		return 0, false
	}
	if profile.SupportsCache {
		if promptcache.HasSettings(settings, promptCacheOptionsSetting, cacheInstructionsSetting) &&
			(cache.Options == nil || cache.Options.Mode != PromptCacheModeExplicit || cache.Instructions) {
			return 30 * time.Minute, true
		}
		return 0, false
	}
	if profile.DefaultCacheRetention == 30*time.Minute || cache.Retention != PromptCacheRetention24Hours {
		return 0, false
	}
	return 24 * time.Hour, true
}
