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
	options := portablePromptCacheOptions{PromptCacheOptions: PromptCacheOptions{
		Mode: PromptCacheModeImplicit, TTL: PromptCacheTTL30Minutes,
	}}
	if cache.Messages != nil && !*cache.Messages {
		options.Mode = PromptCacheModeExplicit
		options.prefixOnly = true
	}
	settings.ExtraBody[promptCacheOptionsSetting] = options
	settings.ExtraBody[cacheInstructionsSetting] = true
	return settings, nil
}

func keepImplicitCacheWithoutBreakpoints(cache *promptCacheSettings, hasBreakpoint bool) {
	if cache.portablePrefixOnly && !hasBreakpoint {
		cache.Options = &PromptCacheOptions{Mode: PromptCacheModeImplicit, TTL: PromptCacheTTL30Minutes}
	}
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
