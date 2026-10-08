package ai

import (
	"fmt"
	"slices"
	"time"
)

// CacheRetention selects a portable prompt-cache lifetime.
type CacheRetention string

const (
	// CacheRetentionDisabled disables library-managed caching, not explicit markers or implicit provider caching.
	CacheRetentionDisabled CacheRetention = "disabled"
	// CacheRetention5Minutes requests five-minute retention.
	CacheRetention5Minutes CacheRetention = "5m"
	// CacheRetention30Minutes requests thirty-minute retention.
	CacheRetention30Minutes CacheRetention = "30m"
	// CacheRetention1Hour requests one-hour retention.
	CacheRetention1Hour CacheRetention = "1h"
)

// CacheConfig enables prompt caching. The zero value caches the stable prefix and conversation with provider defaults.
// A nil ModelSettings.Cache leaves caching unconfigured.
type CacheConfig struct {
	// Retention defaults to the provider's shortest tier. Unsupported tiers snap down, or up to the shortest tier.
	Retention CacheRetention
	// Messages defaults to true. False caches only static instructions and tool definitions.
	Messages *bool
}

// Caching contributes portable prompt-cache settings. The zero value enables caching.
type Caching struct {
	// Retention selects the cache lifetime, or CacheRetentionDisabled.
	Retention CacheRetention
	// Messages defaults to true. False caches only the stable prefix.
	Messages *bool
	// ID overrides the stable capability identity, which defaults to "caching".
	ID string
}

// Setup registers detached cache settings.
func (c Caching) Setup(reg *CapabilityRegistry) error {
	cache := &CacheConfig{Retention: c.Retention, Messages: clonePointer(c.Messages)}
	if err := validateCacheConfig(cache); err != nil {
		return err
	}
	reg.AddModelSettings(ModelSettings{Cache: cache})
	return nil
}

// CapabilityID returns the caching configuration's identity.
func (c Caching) CapabilityID() string {
	if c.ID != "" {
		return c.ID
	}
	return "caching"
}

// CombineCapabilities keeps the last configuration in a registration layer.
func (Caching) CombineCapabilities(capabilities []Capability) (Capability, error) {
	if len(capabilities) == 0 {
		return nil, fmt.Errorf("ai: caching combination requires at least one capability")
	}
	return capabilities[len(capabilities)-1], nil
}

// ResolveCacheConfig validates and resolves portable settings for a provider profile.
// It returns nil when caching is unset, disabled, or unsupported. It never mutates settings.
func ResolveCacheConfig(settings ModelSettings, profile ModelProfile) (*CacheConfig, error) {
	if err := validateCacheConfig(settings.Cache); err != nil {
		return nil, err
	}
	if settings.Cache == nil || settings.Cache.Retention == CacheRetentionDisabled || !profile.SupportsCache {
		return nil, nil
	}
	cache := *settings.Cache
	cache.Messages = clonePointer(cache.Messages)
	if cache.Retention == "" || slices.Contains(profile.SupportedCacheRetentions, cache.Retention) {
		return &cache, nil
	}
	cache.Retention = ""
	requested, _ := time.ParseDuration(string(settings.Cache.Retention))
	var selected, shortest time.Duration
	for _, tier := range profile.SupportedCacheRetentions {
		duration, _ := time.ParseDuration(string(tier))
		if shortest == 0 || duration < shortest {
			shortest = duration
			if selected == 0 {
				cache.Retention = tier
			}
		}
		if duration <= requested && duration > selected {
			selected = duration
			cache.Retention = tier
		}
	}
	return &cache, nil
}

// CacheConfigurationModel reports missing request-side cache configuration for telemetry.
// Implicitly caching providers and composites should return false.
type CacheConfigurationModel interface {
	// CachingNotEnabled reports whether neither portable nor provider-local cache settings were supplied.
	CachingNotEnabled(settings ModelSettings) bool
}

func validateCacheConfig(cache *CacheConfig) error {
	if cache == nil {
		return nil
	}
	switch cache.Retention {
	case "", CacheRetentionDisabled, CacheRetention5Minutes, CacheRetention30Minutes, CacheRetention1Hour:
		return nil
	default:
		return fmt.Errorf("ai: invalid cache retention %q", cache.Retention)
	}
}
