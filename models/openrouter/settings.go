package openrouter

import (
	"fmt"
	"maps"
	"slices"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// Transform changes prompts that exceed a routed model's context window.
type Transform string

const (
	// TransformMiddleOut compresses content from the middle of the prompt.
	TransformMiddleOut Transform = "middle-out"
)

// DataCollection controls whether OpenRouter may select providers that retain prompts.
type DataCollection string

const (
	// DataCollectionAllow permits providers that retain request data.
	DataCollectionAllow DataCollection = "allow"
	// DataCollectionDeny selects only providers that do not retain request data.
	DataCollectionDeny DataCollection = "deny"
)

// ProviderSort selects how OpenRouter ranks eligible providers.
type ProviderSort string

const (
	// ProviderSortPrice prioritizes the lowest-cost provider.
	ProviderSortPrice ProviderSort = "price"
	// ProviderSortThroughput prioritizes generation throughput.
	ProviderSortThroughput ProviderSort = "throughput"
	// ProviderSortLatency prioritizes request latency.
	ProviderSortLatency ProviderSort = "latency"
)

// Quantization identifies an upstream model's numeric representation.
type Quantization string

const (
	// QuantizationInt4 selects 4-bit integer weights.
	QuantizationInt4 Quantization = "int4"
	// QuantizationInt8 selects 8-bit integer weights.
	QuantizationInt8 Quantization = "int8"
	// QuantizationFP4 selects 4-bit floating-point weights.
	QuantizationFP4 Quantization = "fp4"
	// QuantizationFP6 selects 6-bit floating-point weights.
	QuantizationFP6 Quantization = "fp6"
	// QuantizationFP8 selects 8-bit floating-point weights.
	QuantizationFP8 Quantization = "fp8"
	// QuantizationFP16 selects 16-bit floating-point weights.
	QuantizationFP16 Quantization = "fp16"
	// QuantizationBF16 selects bfloat16 weights.
	QuantizationBF16 Quantization = "bf16"
	// QuantizationFP32 selects 32-bit floating-point weights.
	QuantizationFP32 Quantization = "fp32"
	// QuantizationUnknown permits providers without reported quantization.
	QuantizationUnknown Quantization = "unknown"
)

// MaxPrice caps OpenRouter prices in US dollars per million units.
type MaxPrice struct {
	// Prompt caps prompt-token price.
	Prompt float64 `json:"prompt,omitempty"`
	// Completion caps completion-token price.
	Completion float64 `json:"completion,omitempty"`
	// Image caps image-unit price.
	Image float64 `json:"image,omitempty"`
	// Audio caps audio-unit price.
	Audio float64 `json:"audio,omitempty"`
	// Request caps per-request price.
	Request float64 `json:"request,omitempty"`
}

// ProviderRouting controls which upstream providers OpenRouter may use.
type ProviderRouting struct {
	// Order lists preferred providers in priority order.
	Order []string `json:"order,omitempty"`
	// AllowFallbacks permits routing beyond the preferred provider.
	AllowFallbacks *bool `json:"allow_fallbacks,omitempty"`
	// RequireParameters selects providers supporting every request parameter.
	RequireParameters *bool `json:"require_parameters,omitempty"`
	// DataCollection controls provider retention policy.
	DataCollection DataCollection `json:"data_collection,omitempty"`
	// ZeroDataRetention requires a zero-data-retention provider.
	ZeroDataRetention *bool `json:"zdr,omitempty"`
	// Only restricts routing to these providers.
	Only []string `json:"only,omitempty"`
	// Ignore excludes these providers.
	Ignore []string `json:"ignore,omitempty"`
	// Quantizations restricts acceptable model representations.
	Quantizations []Quantization `json:"quantizations,omitempty"`
	// Sort selects provider ranking behavior.
	Sort ProviderSort `json:"sort,omitempty"`
	// MaxPrice rejects providers above any configured price cap.
	MaxPrice *MaxPrice `json:"max_price,omitempty"`
}

// ReasoningEffort controls OpenRouter reasoning depth.
type ReasoningEffort string

const (
	// ReasoningEffortNone disables reasoning.
	ReasoningEffortNone ReasoningEffort = "none"
	// ReasoningEffortMinimal requests minimal reasoning.
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	// ReasoningEffortLow requests low reasoning effort.
	ReasoningEffortLow ReasoningEffort = "low"
	// ReasoningEffortMedium requests medium reasoning effort.
	ReasoningEffortMedium ReasoningEffort = "medium"
	// ReasoningEffortHigh requests high reasoning effort.
	ReasoningEffortHigh ReasoningEffort = "high"
	// ReasoningEffortXHigh requests the highest available reasoning effort.
	ReasoningEffortXHigh ReasoningEffort = "xhigh"
)

// Reasoning configures OpenRouter's cross-provider reasoning extension.
type Reasoning struct {
	// Effort selects a provider-portable reasoning depth.
	Effort ReasoningEffort `json:"effort,omitempty"`
	// MaxTokens limits reasoning tokens.
	MaxTokens int `json:"max_tokens,omitempty"`
	// Exclude omits reasoning content from the response.
	Exclude *bool `json:"exclude,omitempty"`
	// Enabled explicitly enables or disables reasoning.
	Enabled *bool `json:"enabled,omitempty"`
}

// IsEnabled reports whether this configuration requests reasoning.
func (reasoning Reasoning) IsEnabled() bool {
	return (reasoning.Enabled == nil || *reasoning.Enabled) && reasoning.Effort != ReasoningEffortNone &&
		(reasoning.Effort != "" || reasoning.MaxTokens > 0 || reasoning.Enabled != nil)
}

// CacheTTL selects the lifetime of one OpenRouter prompt-cache breakpoint.
type CacheTTL string

const (
	// CacheTTL5Minutes keeps a cache breakpoint for five minutes.
	CacheTTL5Minutes CacheTTL = "5m"
	// CacheTTL1Hour keeps a cache breakpoint for one hour.
	CacheTTL1Hour CacheTTL = "1h"
)

const (
	cacheInstructionsKey = "openrouter_cache_instructions"
	cacheMessagesKey     = "openrouter_cache_messages"
	cacheToolsKey        = "openrouter_cache_tool_definitions"
)

// UsageConfig requests OpenRouter's extended usage and cost fields.
type UsageConfig struct {
	// Include requests detailed token usage and provider cost.
	Include bool `json:"include"`
}

// Settings combines portable settings with OpenRouter routing extensions.
type Settings struct {
	// Common contains portable model settings.
	Common ai.ModelSettings
	// Models lists fallback model IDs in priority order.
	Models []string
	// Provider controls upstream routing.
	Provider *ProviderRouting
	// Preset applies a saved OpenRouter request preset.
	Preset string
	// Transforms modify prompts before provider dispatch.
	Transforms []Transform
	// Reasoning configures OpenRouter's reasoning extension.
	Reasoning *Reasoning
	// Usage requests detailed provider usage.
	Usage *UsageConfig
	// CacheInstructions caches the final stable instruction boundary.
	CacheInstructions CacheTTL
	// CacheMessages caches recent message boundaries.
	CacheMessages CacheTTL
	// CacheToolDefinitions caches the final function-tool definition.
	CacheToolDefinitions CacheTTL
}

// Build returns detached portable settings accepted by agents and direct requests.
func (settings Settings) Build() (ai.ModelSettings, error) {
	common := settings.Common.Clone()
	extra := maps.Clone(common.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	set := func(name string, value any) error {
		if _, exists := extra[name]; exists {
			return fmt.Errorf("openrouter: extra body field %q conflicts with typed settings", name)
		}
		extra[name] = value
		return nil
	}
	if len(settings.Models) > 0 {
		if err := set("models", slices.Clone(settings.Models)); err != nil {
			return ai.ModelSettings{}, err
		}
	}
	if settings.Provider != nil {
		provider := cloneProviderRouting(*settings.Provider)
		if err := validateProviderRouting(provider); err != nil {
			return ai.ModelSettings{}, err
		}
		if err := set("provider", provider); err != nil {
			return ai.ModelSettings{}, err
		}
	}
	if settings.Preset != "" {
		if err := set("preset", settings.Preset); err != nil {
			return ai.ModelSettings{}, err
		}
	}
	if len(settings.Transforms) > 0 {
		transforms := slices.Clone(settings.Transforms)
		for _, transform := range transforms {
			if err := validateOpenRouterTransform(transform); err != nil {
				return ai.ModelSettings{}, err
			}
		}
		if err := set("transforms", transforms); err != nil {
			return ai.ModelSettings{}, err
		}
	}
	if settings.Reasoning != nil {
		reasoning := *settings.Reasoning
		reasoning.Exclude = clonePointer(reasoning.Exclude)
		reasoning.Enabled = clonePointer(reasoning.Enabled)
		if reasoning.Effort != "" && reasoning.MaxTokens != 0 {
			return ai.ModelSettings{}, fmt.Errorf("openrouter: reasoning effort and max tokens are mutually exclusive")
		}
		if err := validateReasoningEffort(reasoning.Effort); err != nil {
			return ai.ModelSettings{}, err
		}
		if err := set("reasoning", reasoning); err != nil {
			return ai.ModelSettings{}, err
		}
	}
	if settings.Usage != nil {
		if err := set("usage", *settings.Usage); err != nil {
			return ai.ModelSettings{}, err
		}
	}
	cacheSettings := []struct {
		name string
		ttl  CacheTTL
	}{
		{name: cacheInstructionsKey, ttl: settings.CacheInstructions},
		{name: cacheMessagesKey, ttl: settings.CacheMessages},
		{name: cacheToolsKey, ttl: settings.CacheToolDefinitions},
	}
	for _, cache := range cacheSettings {
		if cache.ttl == "" {
			continue
		}
		if err := validateCacheTTL(cache.ttl); err != nil {
			return ai.ModelSettings{}, err
		}
		if err := set(cache.name, cache.ttl); err != nil {
			return ai.ModelSettings{}, err
		}
	}
	common.ExtraBody = extra
	return prepareSettings(common)
}

func prepareSettings(settings ai.ModelSettings) (ai.ModelSettings, error) {
	settings = settings.Clone()
	thinking := settings.Thinking
	settings.Thinking = nil
	if thinking == nil {
		return settings, nil
	}
	extra := maps.Clone(settings.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	if _, exists := extra["reasoning"]; !exists {
		reasoning := map[string]any{}
		if thinking.TokenBudget != nil {
			reasoning["max_tokens"] = *thinking.TokenBudget
		} else if thinking.Level != "" {
			effort, enabled, err := openRouterReasoningEffort(thinking.Level)
			if err != nil {
				return ai.ModelSettings{}, err
			}
			reasoning["effort"] = effort
			reasoning["enabled"] = enabled
		}
		if len(reasoning) > 0 {
			extra["reasoning"] = reasoning
		}
	}
	settings.ExtraBody = extra
	return settings, nil
}

func extractCacheSettings(settings ai.ModelSettings) (ai.ModelSettings, map[string]CacheTTL, error) {
	extra := maps.Clone(settings.ExtraBody)
	cache := map[string]CacheTTL{}
	for _, name := range []string{cacheInstructionsKey, cacheMessagesKey, cacheToolsKey} {
		value, exists := extra[name]
		if !exists {
			continue
		}
		delete(extra, name)
		var ttl CacheTTL
		switch value := value.(type) {
		case CacheTTL:
			ttl = value
		case string:
			ttl = CacheTTL(value)
		default:
			return ai.ModelSettings{}, nil, fmt.Errorf("openrouter: cache setting %q must be a TTL string", name)
		}
		if err := validateCacheTTL(ttl); err != nil {
			return ai.ModelSettings{}, nil, err
		}
		cache[name] = ttl
	}
	if len(extra) == 0 {
		extra = nil
	}
	settings.ExtraBody = extra
	return settings, cache, nil
}

func openRouterReasoningEffort(level ai.ThinkingLevel) (string, bool, error) {
	switch level {
	case ai.ThinkingLevelDisabled:
		return "none", false, nil
	case ai.ThinkingLevelEnabled, ai.ThinkingLevelMedium:
		return "medium", true, nil
	case ai.ThinkingLevelMinimal, ai.ThinkingLevelLow:
		return "low", true, nil
	case ai.ThinkingLevelHigh, ai.ThinkingLevelXHigh:
		return "high", true, nil
	default:
		return "", false, fmt.Errorf("openrouter: invalid thinking level %q", level)
	}
}

func validateProviderRouting(provider ProviderRouting) error {
	switch provider.DataCollection {
	case "", DataCollectionAllow, DataCollectionDeny:
	default:
		return fmt.Errorf("openrouter: invalid data collection policy %q", provider.DataCollection)
	}
	switch provider.Sort {
	case "", ProviderSortPrice, ProviderSortThroughput, ProviderSortLatency:
	default:
		return fmt.Errorf("openrouter: invalid provider sort %q", provider.Sort)
	}
	for _, quantization := range provider.Quantizations {
		switch quantization {
		case QuantizationInt4, QuantizationInt8, QuantizationFP4, QuantizationFP6, QuantizationFP8,
			QuantizationFP16, QuantizationBF16, QuantizationFP32, QuantizationUnknown:
		default:
			return fmt.Errorf("openrouter: invalid quantization %q", quantization)
		}
	}
	return nil
}

func validateCacheTTL(ttl CacheTTL) error {
	switch ttl {
	case CacheTTL5Minutes, CacheTTL1Hour:
		return nil
	default:
		return fmt.Errorf("openrouter: invalid cache TTL %q", ttl)
	}
}

func validateReasoningEffort(effort ReasoningEffort) error {
	switch effort {
	case "", ReasoningEffortNone, ReasoningEffortMinimal, ReasoningEffortLow,
		ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh:
		return nil
	default:
		return fmt.Errorf("openrouter: invalid reasoning effort %q", effort)
	}
}

func cloneProviderRouting(provider ProviderRouting) ProviderRouting {
	provider.Order = slices.Clone(provider.Order)
	provider.AllowFallbacks = clonePointer(provider.AllowFallbacks)
	provider.RequireParameters = clonePointer(provider.RequireParameters)
	provider.ZeroDataRetention = clonePointer(provider.ZeroDataRetention)
	provider.Only = slices.Clone(provider.Only)
	provider.Ignore = slices.Clone(provider.Ignore)
	provider.Quantizations = slices.Clone(provider.Quantizations)
	if provider.MaxPrice != nil {
		maximum := *provider.MaxPrice
		provider.MaxPrice = &maximum
	}
	return provider
}

func clonePointer[T any](value *T) *T {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}
