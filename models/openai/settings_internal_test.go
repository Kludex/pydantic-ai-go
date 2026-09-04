package openai

import (
	"reflect"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func TestPromptCacheSettingsBuildAndExtract(t *testing.T) {
	options := &PromptCacheOptions{Mode: PromptCacheModeExplicit, TTL: PromptCacheTTL30Minutes}
	settings, err := (Settings{
		Common: ai.ModelSettings{
			MaxTokens: 42,
			ExtraBody: map[string]any{"custom": map[string]any{"enabled": true}},
		},
		PromptCacheKey:       "conversation",
		PromptCacheRetention: PromptCacheRetention24Hours,
		PromptCacheOptions:   options,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	options.Mode = PromptCacheModeImplicit
	cleaned, cache, err := extractPromptCacheSettings(settings)
	if err != nil {
		t.Fatal(err)
	}
	if cleaned.MaxTokens != 42 || !reflect.DeepEqual(cleaned.ExtraBody, map[string]any{
		"custom": map[string]any{"enabled": true},
	}) || cache.Key != "conversation" || cache.Retention != PromptCacheRetention24Hours ||
		cache.Options == nil || cache.Options.Mode != PromptCacheModeExplicit ||
		cache.Options.TTL != PromptCacheTTL30Minutes {
		t.Fatalf("unexpected prompt cache settings: cleaned=%#v cache=%#v", cleaned, cache)
	}
	cache.Options.Mode = PromptCacheModeImplicit
	_, detached, err := extractPromptCacheSettings(settings)
	if err != nil || detached.Options.Mode != PromptCacheModeExplicit {
		t.Fatalf("prompt cache options were not detached: %#v, %v", detached, err)
	}

	empty, err := (Settings{}).Build()
	if err != nil || empty.ExtraBody != nil {
		t.Fatalf("unexpected empty settings: %#v, %v", empty, err)
	}
	cleaned, cache, err = extractPromptCacheSettings(empty)
	if err != nil || cleaned.ExtraBody != nil || cache.Key != "" || cache.Retention != "" || cache.Options != nil {
		t.Fatalf("unexpected empty extraction: %#v %#v %v", cleaned, cache, err)
	}
}

func TestPromptCacheSettingsValidation(t *testing.T) {
	tests := []struct {
		name     string
		settings Settings
		want     string
	}{
		{name: "retention", settings: Settings{PromptCacheRetention: "week"}, want: "invalid prompt cache retention"},
		{name: "mode", settings: Settings{PromptCacheOptions: &PromptCacheOptions{Mode: "manual"}}, want: "invalid prompt cache mode"},
		{name: "ttl", settings: Settings{PromptCacheOptions: &PromptCacheOptions{TTL: "1h"}}, want: "invalid prompt cache TTL"},
		{name: "wire conflict", settings: Settings{
			Common: ai.ModelSettings{ExtraBody: map[string]any{"prompt_cache_key": "raw"}}, PromptCacheKey: "typed",
		}, want: `extra body field "prompt_cache_key" conflicts`},
		{name: "reserved", settings: Settings{
			Common: ai.ModelSettings{ExtraBody: map[string]any{promptCacheKeySetting: "raw"}}, PromptCacheKey: "typed",
		}, want: `setting field "openai_prompt_cache_key" is reserved`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := test.settings.Build()
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected validation error: %v", err)
			}
		})
	}

	for name, value := range map[string]any{
		promptCacheKeySetting:       42,
		promptCacheRetentionSetting: "24h",
		promptCacheOptionsSetting:   map[string]any{"mode": "explicit"},
	} {
		t.Run("extract "+name, func(t *testing.T) {
			_, _, err := extractPromptCacheSettings(ai.ModelSettings{ExtraBody: map[string]any{name: value}})
			if err == nil || !strings.Contains(err.Error(), "prompt cache") {
				t.Fatalf("unexpected extraction error: %v", err)
			}
		})
	}
	for name, value := range map[string]any{
		promptCacheRetentionSetting: PromptCacheRetention("week"),
		promptCacheOptionsSetting:   PromptCacheOptions{Mode: "manual"},
	} {
		t.Run("extract invalid "+name, func(t *testing.T) {
			_, _, err := extractPromptCacheSettings(ai.ModelSettings{ExtraBody: map[string]any{name: value}})
			if err == nil || !strings.Contains(err.Error(), "invalid prompt cache") {
				t.Fatalf("unexpected extraction validation error: %v", err)
			}
		})
	}

	malformed := ai.ModelSettings{ExtraBody: map[string]any{promptCacheKeySetting: 42}}
	models := []ai.Model{NewModel("model"), NewResponsesModel("model")}
	for _, model := range models {
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: malformed})
		if err == nil || !strings.Contains(err.Error(), "prompt cache key") {
			t.Fatalf("unexpected malformed request error from %T: %v", model, err)
		}
	}
	if duration, ok := NewResponsesModel("model").PromptCacheRetention(malformed); ok || duration != 0 {
		t.Fatalf("malformed settings reported retention: %s %v", duration, ok)
	}
}
