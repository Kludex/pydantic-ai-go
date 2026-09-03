package bedrock

import (
	"fmt"
	"maps"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
)

// CacheTTL selects the lifetime of a Bedrock prompt-cache boundary.
type CacheTTL string

const (
	// CacheTTL5Minutes requests five-minute retention.
	CacheTTL5Minutes CacheTTL = "5m"
	// CacheTTL1Hour requests one-hour retention.
	CacheTTL1Hour CacheTTL = "1h"
)

// Settings combines portable settings with Bedrock prompt caching.
type Settings struct {
	// Common contains portable model settings.
	Common ai.ModelSettings
	// CacheInstructions appends a cache point to system instructions.
	CacheInstructions CacheTTL
	// CacheMessages appends a cache point to the latest user message.
	CacheMessages CacheTTL
	// CacheToolDefinitions appends a cache point after function tools.
	CacheToolDefinitions CacheTTL
}

const (
	cacheInstructionsSetting    = "bedrock_cache_instructions"
	cacheMessagesSetting        = "bedrock_cache_messages"
	cacheToolDefinitionsSetting = "bedrock_cache_tool_definitions"
)

type cacheSettings struct {
	instructions    CacheTTL
	messages        CacheTTL
	toolDefinitions CacheTTL
}

// Build returns detached portable settings accepted by agents and direct requests.
func (settings Settings) Build() (ai.ModelSettings, error) {
	common := settings.Common.Clone()
	extra := maps.Clone(common.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	values := []struct {
		name string
		ttl  CacheTTL
	}{
		{name: cacheInstructionsSetting, ttl: settings.CacheInstructions},
		{name: cacheMessagesSetting, ttl: settings.CacheMessages},
		{name: cacheToolDefinitionsSetting, ttl: settings.CacheToolDefinitions},
	}
	for _, value := range values {
		if _, exists := extra[value.name]; exists {
			return ai.ModelSettings{}, fmt.Errorf("bedrock: setting field %q is reserved", value.name)
		}
		if value.ttl == "" {
			continue
		}
		if err := validateCacheTTL(value.ttl); err != nil {
			return ai.ModelSettings{}, err
		}
		extra[value.name] = value.ttl
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
		{name: cacheInstructionsSetting, destination: &cache.instructions},
		{name: cacheMessagesSetting, destination: &cache.messages},
		{name: cacheToolDefinitionsSetting, destination: &cache.toolDefinitions},
	}
	for _, value := range values {
		raw, exists := extra[value.name]
		if !exists {
			continue
		}
		delete(extra, value.name)
		ttl, ok := raw.(CacheTTL)
		if !ok {
			return ai.ModelSettings{}, cacheSettings{}, fmt.Errorf(
				"bedrock: cache setting %q must use CacheTTL", value.name,
			)
		}
		if err := validateCacheTTL(ttl); err != nil {
			return ai.ModelSettings{}, cacheSettings{}, err
		}
		*value.destination = ttl
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
		return fmt.Errorf("bedrock: invalid cache TTL %q", ttl)
	}
}

func (cache cacheSettings) retention() (time.Duration, bool) {
	for _, ttl := range []CacheTTL{cache.instructions, cache.messages, cache.toolDefinitions} {
		if ttl == CacheTTL1Hour {
			return time.Hour, true
		}
	}
	for _, ttl := range []CacheTTL{cache.instructions, cache.messages, cache.toolDefinitions} {
		if ttl == CacheTTL5Minutes {
			return 5 * time.Minute, true
		}
	}
	return 0, false
}
