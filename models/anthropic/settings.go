package anthropic

import (
	"fmt"
	"maps"

	ai "github.com/Kludex/pydantic-ai-go"
)

// CacheTTL selects the lifetime of an Anthropic prompt-cache breakpoint.
type CacheTTL string

const (
	CacheTTL5Minutes CacheTTL = "5m"
	CacheTTL1Hour    CacheTTL = "1h"
)

// Settings combines portable settings with Anthropic prompt caching.
type Settings struct {
	Common               ai.ModelSettings
	Cache                CacheTTL
	CacheInstructions    CacheTTL
	CacheMessages        CacheTTL
	CacheToolDefinitions CacheTTL
}

const (
	cacheSetting                = "anthropic_cache"
	cacheInstructionsSetting    = "anthropic_cache_instructions"
	cacheMessagesSetting        = "anthropic_cache_messages"
	cacheToolDefinitionsSetting = "anthropic_cache_tool_definitions"
)

type cacheSettings struct {
	Automatic       CacheTTL
	Instructions    CacheTTL
	Messages        CacheTTL
	ToolDefinitions CacheTTL
}

// Build returns detached portable settings accepted by agents and direct requests.
func (settings Settings) Build() (ai.ModelSettings, error) {
	common := settings.Common.Clone()
	values := []struct {
		name string
		ttl  CacheTTL
	}{
		{name: cacheSetting, ttl: settings.Cache},
		{name: cacheInstructionsSetting, ttl: settings.CacheInstructions},
		{name: cacheMessagesSetting, ttl: settings.CacheMessages},
		{name: cacheToolDefinitionsSetting, ttl: settings.CacheToolDefinitions},
	}
	if settings.Cache != "" && settings.CacheMessages != "" {
		return ai.ModelSettings{}, fmt.Errorf("anthropic: automatic and explicit message caching are mutually exclusive")
	}
	extra := maps.Clone(common.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	for _, setting := range values {
		if _, exists := extra[setting.name]; exists {
			return ai.ModelSettings{}, fmt.Errorf("anthropic: setting field %q is reserved", setting.name)
		}
		if setting.ttl == "" {
			continue
		}
		if err := validateCacheTTL(setting.ttl); err != nil {
			return ai.ModelSettings{}, err
		}
		if setting.name == cacheSetting {
			if _, exists := extra["cache_control"]; exists {
				return ai.ModelSettings{}, fmt.Errorf(
					"anthropic: extra body field %q conflicts with typed settings", "cache_control",
				)
			}
		}
		extra[setting.name] = setting.ttl
	}
	if len(extra) == 0 {
		extra = nil
	}
	common.ExtraBody = extra
	return common, nil
}

func extractCacheSettings(settings ai.ModelSettings) (ai.ModelSettings, cacheSettings, error) {
	settings = settings.Clone()
	extra := maps.Clone(settings.ExtraBody)
	cache := cacheSettings{}
	values := []struct {
		name        string
		destination *CacheTTL
	}{
		{name: cacheSetting, destination: &cache.Automatic},
		{name: cacheInstructionsSetting, destination: &cache.Instructions},
		{name: cacheMessagesSetting, destination: &cache.Messages},
		{name: cacheToolDefinitionsSetting, destination: &cache.ToolDefinitions},
	}
	for _, setting := range values {
		value, exists := extra[setting.name]
		if !exists {
			continue
		}
		delete(extra, setting.name)
		ttl, ok := value.(CacheTTL)
		if !ok {
			return ai.ModelSettings{}, cacheSettings{}, fmt.Errorf(
				"anthropic: cache setting %q must use CacheTTL", setting.name,
			)
		}
		if err := validateCacheTTL(ttl); err != nil {
			return ai.ModelSettings{}, cacheSettings{}, err
		}
		*setting.destination = ttl
	}
	if cache.Automatic != "" && cache.Messages != "" {
		return ai.ModelSettings{}, cacheSettings{}, fmt.Errorf(
			"anthropic: automatic and explicit message caching are mutually exclusive",
		)
	}
	if len(extra) == 0 {
		extra = nil
	}
	settings.ExtraBody = extra
	return settings, cache, nil
}

func validateCacheTTL(ttl CacheTTL) error {
	switch ttl {
	case CacheTTL5Minutes, CacheTTL1Hour:
		return nil
	default:
		return fmt.Errorf("anthropic: invalid cache TTL %q", ttl)
	}
}

func promptCacheControl(ttl CacheTTL) *anthropicPromptCacheControl {
	if ttl == "" {
		return nil
	}
	return &anthropicPromptCacheControl{Type: "ephemeral", TTL: ai.CachePointTTL(ttl)}
}
