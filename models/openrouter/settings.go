package openrouter

import (
	"fmt"
	"maps"
	"slices"

	ai "github.com/Kludex/pydantic-ai-go"
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
	DataCollectionAllow DataCollection = "allow"
	DataCollectionDeny  DataCollection = "deny"
)

// ProviderSort selects how OpenRouter ranks eligible providers.
type ProviderSort string

const (
	ProviderSortPrice      ProviderSort = "price"
	ProviderSortThroughput ProviderSort = "throughput"
	ProviderSortLatency    ProviderSort = "latency"
)

// Quantization identifies an upstream model's numeric representation.
type Quantization string

const (
	QuantizationInt4    Quantization = "int4"
	QuantizationInt8    Quantization = "int8"
	QuantizationFP4     Quantization = "fp4"
	QuantizationFP6     Quantization = "fp6"
	QuantizationFP8     Quantization = "fp8"
	QuantizationFP16    Quantization = "fp16"
	QuantizationBF16    Quantization = "bf16"
	QuantizationFP32    Quantization = "fp32"
	QuantizationUnknown Quantization = "unknown"
)

// MaxPrice caps OpenRouter prices in US dollars per million units.
type MaxPrice struct {
	Prompt     float64 `json:"prompt,omitempty"`
	Completion float64 `json:"completion,omitempty"`
	Image      float64 `json:"image,omitempty"`
	Audio      float64 `json:"audio,omitempty"`
	Request    float64 `json:"request,omitempty"`
}

// ProviderRouting controls which upstream providers OpenRouter may use.
type ProviderRouting struct {
	Order             []string       `json:"order,omitempty"`
	AllowFallbacks    *bool          `json:"allow_fallbacks,omitempty"`
	RequireParameters *bool          `json:"require_parameters,omitempty"`
	DataCollection    DataCollection `json:"data_collection,omitempty"`
	ZeroDataRetention *bool          `json:"zdr,omitempty"`
	Only              []string       `json:"only,omitempty"`
	Ignore            []string       `json:"ignore,omitempty"`
	Quantizations     []Quantization `json:"quantizations,omitempty"`
	Sort              ProviderSort   `json:"sort,omitempty"`
	MaxPrice          *MaxPrice      `json:"max_price,omitempty"`
}

// ReasoningEffort controls OpenRouter reasoning depth.
type ReasoningEffort string

const (
	ReasoningEffortNone    ReasoningEffort = "none"
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
)

// Reasoning configures OpenRouter's cross-provider reasoning extension.
type Reasoning struct {
	Effort    ReasoningEffort `json:"effort,omitempty"`
	MaxTokens int             `json:"max_tokens,omitempty"`
	Exclude   *bool           `json:"exclude,omitempty"`
	Enabled   *bool           `json:"enabled,omitempty"`
}

// UsageConfig requests OpenRouter's extended usage and cost fields.
type UsageConfig struct {
	Include bool `json:"include"`
}

// Settings combines portable settings with OpenRouter routing extensions.
type Settings struct {
	Common     ai.ModelSettings
	Models     []string
	Provider   *ProviderRouting
	Preset     string
	Transforms []Transform
	Reasoning  *Reasoning
	Usage      *UsageConfig
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
