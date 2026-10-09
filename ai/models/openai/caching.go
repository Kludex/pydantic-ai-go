package openai

import (
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/promptcache"
)

// prefixOnlyCacheOptions marks prompt cache settings that came from the unified
// `cache` setting with `messages=False`. mode='explicit' disables OpenAI's implicit
// breakpoint and relies on the instruction breakpoint, which `openai_cache_instructions`
// does not place on every request; requests that end up with no breakpoint would cache
// nothing. keepImplicitCacheWithoutBreakpoints detects this exact object and downgrades
// it to implicit when the mapped request carries no breakpoint.
var prefixOnlyCacheOptions = &PromptCacheOptions{
	Mode: PromptCacheModeExplicit,
	TTL:  PromptCacheTTL30Minutes,
}

// implicitCacheOptions marks prompt cache settings that came from the unified `cache`
// setting with the default or messages=true.
var implicitCacheOptions = &PromptCacheOptions{
	Mode: PromptCacheModeImplicit,
	TTL:  PromptCacheTTL30Minutes,
}

// keepImplicitCacheWithoutBreakpoints downgrades mode from explicit to implicit when
// `prefixOnlyCacheOptions` was applied and the mapped request carries no breakpoint.
// Mutates options.
func keepImplicitCacheWithoutBreakpoints(options **PromptCacheOptions, hasBreakpoint bool) {
	if *options == prefixOnlyCacheOptions && !hasBreakpoint {
		*options = implicitCacheOptions
	}
}

// chatHasPromptCacheBreakpoint reports whether any chat message carries a
// prompt cache breakpoint part.
func chatHasPromptCacheBreakpoint(messages []chatMessage) bool {
	for _, msg := range messages {
		if parts, ok := msg.Content.([]contentPart); ok {
			for _, part := range parts {
				if part.PromptCacheBreakpoint != nil {
					return true
				}
			}
		}
	}
	return false
}

// responsesHasPromptCacheBreakpoint reports whether any Responses input item carries
// a prompt cache breakpoint part, inspecting function_call_output outputs.
func responsesHasPromptCacheBreakpoint(input []responsesInput) bool {
	for _, item := range input {
		body := item.Content
		if item.Type == "function_call_output" {
			body = item.Output
		}
		if parts, ok := body.([]responsesInputContent); ok {
			for _, part := range parts {
				if part.PromptCacheBreakpoint != nil {
					return true
				}
			}
		}
	}
	return false
}

func translateCache(settings ai.ModelSettings, profile ai.ModelProfile) (ai.ModelSettings, error) {
	cache, err := ai.ResolveCacheConfig(settings, profile)
	if err != nil || cache == nil || promptcache.HasSettings(settings, promptCacheOptionsSetting, cacheInstructionsSetting) {
		return settings, err
	}
	settings = settings.Clone()
	if settings.ExtraBody == nil {
		settings.ExtraBody = map[string]any{}
	}
	options := implicitCacheOptions
	if cache.Messages != nil && !*cache.Messages {
		options = prefixOnlyCacheOptions
	}
	settings.ExtraBody[promptCacheOptionsSetting] = options
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
