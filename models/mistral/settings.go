package mistral

import (
	"fmt"
	"maps"
	"slices"

	ai "github.com/Kludex/pydantic-ai-go"
)

const (
	promptCacheKeySetting = "mistral_prompt_cache_key"
	toolChoiceSetting     = "mistral_tool_choice"
	allowedToolsSetting   = "mistral_allowed_tools"
)

// ToolChoice controls whether Mistral may call a function tool.
type ToolChoice string

const (
	// ToolChoiceAuto lets Mistral decide whether to call a tool.
	ToolChoiceAuto ToolChoice = "auto"
	// ToolChoiceAny requires Mistral to call one available tool.
	ToolChoiceAny ToolChoice = "any"
	// ToolChoiceNone prevents tool declarations from being sent.
	ToolChoiceNone ToolChoice = "none"
)

// Settings combines portable settings with Mistral-specific request settings.
type Settings struct {
	// Common contains portable model settings.
	Common ai.ModelSettings
	// PromptCacheKey groups requests whose stable prefixes should share Mistral's prompt cache.
	PromptCacheKey string
	// ToolChoice controls function-tool selection.
	ToolChoice ToolChoice
	// AllowedTools restricts declarations to these names while preserving their original order.
	AllowedTools []string
}

// Build returns detached portable settings accepted by agents and direct requests.
func (settings Settings) Build() (ai.ModelSettings, error) {
	if err := validateToolChoice(settings.ToolChoice); err != nil {
		return ai.ModelSettings{}, err
	}
	if settings.AllowedTools != nil && len(settings.AllowedTools) == 0 {
		return ai.ModelSettings{}, fmt.Errorf("mistral: allowed tools must not be empty")
	}
	seen := map[string]bool{}
	for _, name := range settings.AllowedTools {
		if name == "" {
			return ai.ModelSettings{}, fmt.Errorf("mistral: allowed tool name must not be empty")
		}
		if seen[name] {
			return ai.ModelSettings{}, fmt.Errorf("mistral: duplicate allowed tool %q", name)
		}
		seen[name] = true
	}
	common := settings.Common.Clone()
	extra := maps.Clone(common.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	values := []struct {
		name  string
		value any
		set   bool
	}{
		{name: promptCacheKeySetting, value: settings.PromptCacheKey, set: settings.PromptCacheKey != ""},
		{name: toolChoiceSetting, value: settings.ToolChoice, set: settings.ToolChoice != ""},
		{name: allowedToolsSetting, value: slices.Clone(settings.AllowedTools), set: settings.AllowedTools != nil},
	}
	for _, value := range values {
		if _, exists := extra[value.name]; exists {
			return ai.ModelSettings{}, fmt.Errorf("mistral: setting field %q is reserved", value.name)
		}
		if value.set {
			extra[value.name] = value.value
		}
	}
	if len(extra) == 0 {
		extra = nil
	}
	common.ExtraBody = extra
	return common, nil
}

type providerSettings struct {
	promptCacheKey string
	toolChoice     ToolChoice
	allowedTools   []string
}

func extractSettings(settings ai.ModelSettings) (ai.ModelSettings, providerSettings, error) {
	settings = settings.Clone()
	extra := maps.Clone(settings.ExtraBody)
	provider := providerSettings{}
	if value, exists := extra[promptCacheKeySetting]; exists {
		delete(extra, promptCacheKeySetting)
		var ok bool
		provider.promptCacheKey, ok = value.(string)
		if !ok || provider.promptCacheKey == "" {
			return ai.ModelSettings{}, providerSettings{}, fmt.Errorf("mistral: prompt cache key must be a non-empty string")
		}
	}
	if value, exists := extra[toolChoiceSetting]; exists {
		delete(extra, toolChoiceSetting)
		var ok bool
		provider.toolChoice, ok = value.(ToolChoice)
		if !ok {
			return ai.ModelSettings{}, providerSettings{}, fmt.Errorf("mistral: tool choice must use ToolChoice")
		}
		if err := validateToolChoice(provider.toolChoice); err != nil {
			return ai.ModelSettings{}, providerSettings{}, err
		}
	}
	if value, exists := extra[allowedToolsSetting]; exists {
		delete(extra, allowedToolsSetting)
		var ok bool
		provider.allowedTools, ok = value.([]string)
		if !ok || len(provider.allowedTools) == 0 {
			return ai.ModelSettings{}, providerSettings{}, fmt.Errorf("mistral: allowed tools must be a non-empty string slice")
		}
		provider.allowedTools = slices.Clone(provider.allowedTools)
	}
	if len(extra) == 0 {
		extra = nil
	}
	settings.ExtraBody = extra
	return settings, provider, nil
}

func validateToolChoice(choice ToolChoice) error {
	switch choice {
	case "", ToolChoiceAuto, ToolChoiceAny, ToolChoiceNone:
		return nil
	default:
		return fmt.Errorf("mistral: invalid tool choice %q", choice)
	}
}
