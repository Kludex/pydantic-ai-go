package openai

import (
	"fmt"
	"maps"

	ai "github.com/Kludex/pydantic-ai-go"
)

// PromptCacheMode selects implicit or caller-authored prompt cache boundaries.
type PromptCacheMode string

const (
	PromptCacheModeImplicit PromptCacheMode = "implicit"
	PromptCacheModeExplicit PromptCacheMode = "explicit"
)

// PromptCacheTTL selects the minimum lifetime of an OpenAI prompt cache entry.
type PromptCacheTTL string

const (
	PromptCacheTTL30Minutes PromptCacheTTL = "30m"
)

// PromptCacheRetention selects OpenAI's maximum prompt cache retention policy.
type PromptCacheRetention string

const (
	PromptCacheRetentionInMemory PromptCacheRetention = "in_memory"
	PromptCacheRetention24Hours  PromptCacheRetention = "24h"
)

// PromptCacheOptions configures request-wide caching for GPT-5.6 and later models.
type PromptCacheOptions struct {
	Mode PromptCacheMode `json:"mode,omitempty"`
	TTL  PromptCacheTTL  `json:"ttl,omitempty"`
}

// Settings combines portable settings with OpenAI-specific request options.
type Settings struct {
	Common     ai.ModelSettings
	Prediction *Prediction
	// IncludeRawAnnotations retains Responses text annotations such as citations.
	IncludeRawAnnotations *bool
	PromptCacheKey        string
	PromptCacheRetention  PromptCacheRetention
	PromptCacheOptions    *PromptCacheOptions
}

const (
	predictionSetting            = "openai_prediction"
	includeRawAnnotationsSetting = "openai_include_raw_annotations"
	promptCacheKeySetting        = "openai_prompt_cache_key"
	promptCacheRetentionSetting  = "openai_prompt_cache_retention"
	promptCacheOptionsSetting    = "openai_prompt_cache_options"
)

type promptCacheSettings struct {
	Key       string
	Retention PromptCacheRetention
	Options   *PromptCacheOptions
}

// Build returns detached portable settings accepted by agents and direct requests.
func (settings Settings) Build() (ai.ModelSettings, error) {
	common := settings.Common.Clone()
	if err := validatePromptCacheRetention(settings.PromptCacheRetention); err != nil {
		return ai.ModelSettings{}, err
	}
	options, err := clonePromptCacheOptions(settings.PromptCacheOptions)
	if err != nil {
		return ai.ModelSettings{}, err
	}
	prediction, err := marshalPrediction(settings.Prediction)
	if err != nil {
		return ai.ModelSettings{}, err
	}
	extra := maps.Clone(common.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	for _, name := range []string{
		predictionSetting, includeRawAnnotationsSetting,
		promptCacheKeySetting, promptCacheRetentionSetting, promptCacheOptionsSetting,
	} {
		if _, exists := extra[name]; exists {
			return ai.ModelSettings{}, fmt.Errorf("openai: setting field %q is reserved", name)
		}
	}
	if prediction != nil {
		if _, exists := extra["prediction"]; exists {
			return ai.ModelSettings{}, fmt.Errorf("openai: extra body field %q conflicts with typed settings", "prediction")
		}
		extra[predictionSetting] = prediction
	}
	if settings.IncludeRawAnnotations != nil {
		extra[includeRawAnnotationsSetting] = *settings.IncludeRawAnnotations
	}
	optionsValue := PromptCacheOptions{}
	if options != nil {
		optionsValue = *options
	}
	values := []struct {
		hidden string
		wire   string
		value  any
		set    bool
	}{
		{hidden: promptCacheKeySetting, wire: "prompt_cache_key", value: settings.PromptCacheKey,
			set: settings.PromptCacheKey != ""},
		{hidden: promptCacheRetentionSetting, wire: "prompt_cache_retention", value: settings.PromptCacheRetention,
			set: settings.PromptCacheRetention != ""},
		{hidden: promptCacheOptionsSetting, wire: "prompt_cache_options", value: optionsValue, set: options != nil},
	}
	for _, field := range values {
		if !field.set {
			continue
		}
		if _, exists := extra[field.wire]; exists {
			return ai.ModelSettings{}, fmt.Errorf(
				"openai: extra body field %q conflicts with typed settings", field.wire,
			)
		}
		extra[field.hidden] = field.value
	}
	if len(extra) == 0 {
		extra = nil
	}
	common.ExtraBody = extra
	return common, nil
}

func extractPromptCacheSettings(settings ai.ModelSettings) (ai.ModelSettings, promptCacheSettings, error) {
	settings = settings.Clone()
	extra := maps.Clone(settings.ExtraBody)
	cache := promptCacheSettings{}
	if value, exists := extra[promptCacheKeySetting]; exists {
		delete(extra, promptCacheKeySetting)
		key, ok := value.(string)
		if !ok {
			return ai.ModelSettings{}, promptCacheSettings{}, fmt.Errorf("openai: prompt cache key must be a string")
		}
		cache.Key = key
	}
	if value, exists := extra[promptCacheRetentionSetting]; exists {
		delete(extra, promptCacheRetentionSetting)
		retention, ok := value.(PromptCacheRetention)
		if !ok {
			return ai.ModelSettings{}, promptCacheSettings{}, fmt.Errorf(
				"openai: prompt cache retention must use PromptCacheRetention",
			)
		}
		if err := validatePromptCacheRetention(retention); err != nil {
			return ai.ModelSettings{}, promptCacheSettings{}, err
		}
		cache.Retention = retention
	}
	if value, exists := extra[promptCacheOptionsSetting]; exists {
		delete(extra, promptCacheOptionsSetting)
		options, ok := value.(PromptCacheOptions)
		if !ok {
			return ai.ModelSettings{}, promptCacheSettings{}, fmt.Errorf(
				"openai: prompt cache options must use PromptCacheOptions",
			)
		}
		cloned, err := clonePromptCacheOptions(&options)
		if err != nil {
			return ai.ModelSettings{}, promptCacheSettings{}, err
		}
		cache.Options = cloned
	}
	if len(extra) == 0 {
		extra = nil
	}
	settings.ExtraBody = extra
	return settings, cache, nil
}

func clonePromptCacheOptions(options *PromptCacheOptions) (*PromptCacheOptions, error) {
	if options == nil {
		return nil, nil
	}
	cloned := *options
	switch cloned.Mode {
	case "", PromptCacheModeImplicit, PromptCacheModeExplicit:
	default:
		return nil, fmt.Errorf("openai: invalid prompt cache mode %q", cloned.Mode)
	}
	switch cloned.TTL {
	case "", PromptCacheTTL30Minutes:
	default:
		return nil, fmt.Errorf("openai: invalid prompt cache TTL %q", cloned.TTL)
	}
	return &cloned, nil
}

func validatePromptCacheRetention(retention PromptCacheRetention) error {
	switch retention {
	case "", PromptCacheRetentionInMemory, PromptCacheRetention24Hours:
		return nil
	default:
		return fmt.Errorf("openai: invalid prompt cache retention %q", retention)
	}
}
