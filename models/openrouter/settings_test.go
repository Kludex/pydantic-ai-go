package openrouter_test

import (
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openrouter"
)

func TestOpenRouterSettingsAreDetached(t *testing.T) {
	enabled := true
	budget := 512
	models := []string{"vendor/model"}
	order := []string{"vendor"}
	settings := openrouter.Settings{
		Common: ai.ModelSettings{Thinking: &ai.ThinkingSettings{
			Level: ai.ThinkingLevelHigh, TokenBudget: &budget,
		}},
		Models: models,
		Provider: &openrouter.ProviderRouting{
			Order: order, AllowFallbacks: &enabled, RequireParameters: &enabled,
			ZeroDataRetention: &enabled, Only: []string{"vendor"}, Ignore: []string{"other"},
			Quantizations: []openrouter.Quantization{openrouter.QuantizationFP8}, MaxPrice: &openrouter.MaxPrice{Completion: 2},
		},
	}
	built, err := settings.Build()
	if err != nil {
		t.Fatal(err)
	}
	models[0] = "changed"
	order[0] = "changed"
	*settings.Provider.AllowFallbacks = false
	settings.Provider.Only[0] = "changed"
	settings.Provider.Ignore[0] = "changed"
	settings.Provider.Quantizations[0] = "changed"
	settings.Provider.MaxPrice.Completion = 9
	if built.Thinking != nil || built.ExtraBody["models"].([]string)[0] != "vendor/model" {
		t.Fatalf("portable settings were not translated: %+v", built)
	}
	provider := built.ExtraBody["provider"].(openrouter.ProviderRouting)
	if provider.Order[0] != "vendor" || !*provider.AllowFallbacks || provider.Only[0] != "vendor" ||
		provider.Ignore[0] != "other" || provider.Quantizations[0] != "fp8" || provider.MaxPrice.Completion != 2 {
		t.Fatalf("provider settings share state: %+v", provider)
	}
	reasoning := built.ExtraBody["reasoning"].(map[string]any)
	if reasoning["max_tokens"] != 512 || reasoning["effort"] != nil {
		t.Fatalf("unexpected budget reasoning: %#v", reasoning)
	}
}

func TestOpenRouterThinkingLevels(t *testing.T) {
	for level, expected := range map[ai.ThinkingLevel]struct {
		effort  string
		enabled bool
	}{
		ai.ThinkingLevelDisabled: {effort: "none", enabled: false},
		ai.ThinkingLevelEnabled:  {effort: "medium", enabled: true},
		ai.ThinkingLevelMinimal:  {effort: "low", enabled: true},
		ai.ThinkingLevelLow:      {effort: "low", enabled: true},
		ai.ThinkingLevelMedium:   {effort: "medium", enabled: true},
		ai.ThinkingLevelHigh:     {effort: "high", enabled: true},
		ai.ThinkingLevelXHigh:    {effort: "high", enabled: true},
	} {
		t.Run(string(level), func(t *testing.T) {
			settings, err := (openrouter.Settings{Common: ai.ModelSettings{
				Thinking: &ai.ThinkingSettings{Level: level},
			}}).Build()
			if err != nil {
				t.Fatal(err)
			}
			reasoning := settings.ExtraBody["reasoning"].(map[string]any)
			if reasoning["effort"] != expected.effort || reasoning["enabled"] != expected.enabled {
				t.Fatalf("unexpected reasoning mapping: %#v", reasoning)
			}
		})
	}
	settings, err := (openrouter.Settings{Common: ai.ModelSettings{
		Thinking: &ai.ThinkingSettings{}, ExtraBody: map[string]any{"reasoning": map[string]any{"effort": "xhigh"}},
	}}).Build()
	if err != nil || settings.ExtraBody["reasoning"].(map[string]any)["effort"] != "xhigh" {
		t.Fatalf("explicit reasoning did not win: settings=%+v error=%v", settings, err)
	}
	_, err = (openrouter.Settings{Common: ai.ModelSettings{
		Thinking: &ai.ThinkingSettings{Level: "invalid"},
	}}).Build()
	if err == nil || !strings.Contains(err.Error(), "invalid thinking level") {
		t.Fatalf("unexpected thinking error: %v", err)
	}
}

func TestOpenRouterSettingsValidation(t *testing.T) {
	enabled := true
	for name, test := range map[string]struct {
		settings openrouter.Settings
		want     string
	}{
		"transform": {
			settings: openrouter.Settings{Transforms: []openrouter.Transform{"invalid"}},
			want:     "invalid transform",
		},
		"data collection": {
			settings: openrouter.Settings{Provider: &openrouter.ProviderRouting{DataCollection: "invalid"}},
			want:     "invalid data collection",
		},
		"provider sort": {
			settings: openrouter.Settings{Provider: &openrouter.ProviderRouting{Sort: "invalid"}},
			want:     "invalid provider sort",
		},
		"quantization": {
			settings: openrouter.Settings{Provider: &openrouter.ProviderRouting{
				Quantizations: []openrouter.Quantization{"invalid"},
			}},
			want: "invalid quantization",
		},
		"reasoning effort": {
			settings: openrouter.Settings{Reasoning: &openrouter.Reasoning{Effort: "invalid"}},
			want:     "invalid reasoning effort",
		},
		"reasoning conflict": {
			settings: openrouter.Settings{Reasoning: &openrouter.Reasoning{
				Effort: openrouter.ReasoningEffortHigh, MaxTokens: 100,
			}},
			want: "mutually exclusive",
		},
		"cache TTL": {
			settings: openrouter.Settings{CacheMessages: "1d"},
			want:     "invalid cache TTL",
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := test.settings.Build()
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected settings error: %v", err)
			}
		})
	}

	conflicts := []struct {
		name     string
		settings openrouter.Settings
	}{
		{name: "models", settings: openrouter.Settings{Models: []string{"vendor/model"}}},
		{name: "provider", settings: openrouter.Settings{Provider: &openrouter.ProviderRouting{}}},
		{name: "preset", settings: openrouter.Settings{Preset: "preset"}},
		{name: "transforms", settings: openrouter.Settings{Transforms: []openrouter.Transform{openrouter.TransformMiddleOut}}},
		{name: "reasoning", settings: openrouter.Settings{Reasoning: &openrouter.Reasoning{Enabled: &enabled}}},
		{name: "usage", settings: openrouter.Settings{Usage: &openrouter.UsageConfig{Include: true}}},
		{
			name:     "openrouter_cache_instructions",
			settings: openrouter.Settings{CacheInstructions: openrouter.CacheTTL5Minutes},
		},
		{
			name:     "openrouter_cache_messages",
			settings: openrouter.Settings{CacheMessages: openrouter.CacheTTL5Minutes},
		},
		{
			name:     "openrouter_cache_tool_definitions",
			settings: openrouter.Settings{CacheToolDefinitions: openrouter.CacheTTL5Minutes},
		},
	}
	for _, test := range conflicts {
		t.Run("conflict "+test.name, func(t *testing.T) {
			test.settings.Common.ExtraBody = map[string]any{test.name: true}
			_, err := test.settings.Build()
			if err == nil || !strings.Contains(err.Error(), `field "`+test.name+`" conflicts`) {
				t.Fatalf("unexpected conflict error: %v", err)
			}
		})
	}
}

func TestOpenRouterExplicitReasoningClone(t *testing.T) {
	exclude := true
	enabled := true
	settings := openrouter.Settings{Reasoning: &openrouter.Reasoning{
		Effort: openrouter.ReasoningEffortMinimal, Exclude: &exclude, Enabled: &enabled,
	}}
	built, err := settings.Build()
	if err != nil {
		t.Fatal(err)
	}
	*settings.Reasoning.Exclude = false
	*settings.Reasoning.Enabled = false
	reasoning := built.ExtraBody["reasoning"].(openrouter.Reasoning)
	if !*reasoning.Exclude || !*reasoning.Enabled {
		t.Fatalf("reasoning pointers share state: %+v", reasoning)
	}
	plain, err := (openrouter.Settings{}).Build()
	if err != nil || plain.ExtraBody == nil {
		t.Fatalf("zero settings failed: %+v error=%v", plain, err)
	}
}
