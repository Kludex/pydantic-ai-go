package google

import (
	"encoding/json"
	"fmt"
	"maps"

	ai "github.com/Kludex/pydantic-ai-go"
)

const (
	cachedContentSetting = "google_cached_content"
	modelArmorSetting    = "google_model_armor_config"
)

// Settings combines portable settings with Google-specific request options.
type Settings struct {
	// Common contains portable model settings.
	Common ai.ModelSettings
	// CachedContent names a Gemini or Vertex cached-content resource.
	// The resource owns system instructions and tools, so those request fields are omitted.
	CachedContent string
	// ModelArmor configures Vertex AI prompt and response screening for non-streaming requests.
	ModelArmor *ModelArmorConfig
}

// ModelArmorConfig names Vertex AI Model Armor templates.
type ModelArmorConfig struct {
	// PromptTemplateName screens user prompts when non-empty.
	PromptTemplateName string `json:"promptTemplateName,omitempty"`
	// ResponseTemplateName screens generated responses when non-empty.
	ResponseTemplateName string `json:"responseTemplateName,omitempty"`
}

// Build returns detached portable settings accepted by agents and direct requests.
func (settings Settings) Build() (ai.ModelSettings, error) {
	common := settings.Common.Clone()
	if settings.CachedContent == "" && settings.ModelArmor == nil {
		return common, nil
	}
	extra := maps.Clone(common.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	set := func(name string, value any) error {
		if _, exists := extra[name]; exists {
			return fmt.Errorf("google: extra body field %q conflicts with typed settings", name)
		}
		extra[name] = value
		return nil
	}
	if settings.CachedContent != "" {
		if err := set(cachedContentSetting, settings.CachedContent); err != nil {
			return ai.ModelSettings{}, err
		}
	}
	if settings.ModelArmor != nil {
		config := *settings.ModelArmor
		if config.PromptTemplateName == "" && config.ResponseTemplateName == "" {
			return ai.ModelSettings{}, fmt.Errorf("google: Model Armor requires a prompt or response template")
		}
		if err := set(modelArmorSetting, config); err != nil {
			return ai.ModelSettings{}, err
		}
	}
	common.ExtraBody = extra
	return common, nil
}

func modelArmor(settings ai.ModelSettings) (*ModelArmorConfig, error) {
	value, exists := settings.ExtraBody[modelArmorSetting]
	if !exists {
		return nil, nil
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("google: encode Model Armor config: %w", err)
	}
	var config ModelArmorConfig
	if err := json.Unmarshal(encoded, &config); err != nil {
		return nil, fmt.Errorf("google: decode Model Armor config: %w", err)
	}
	if config.PromptTemplateName == "" && config.ResponseTemplateName == "" {
		return nil, fmt.Errorf("google: Model Armor requires a prompt or response template")
	}
	return &config, nil
}

func cachedContent(settings ai.ModelSettings) (string, error) {
	value, exists := settings.ExtraBody[cachedContentSetting]
	if !exists {
		return "", nil
	}
	name, ok := value.(string)
	if !ok || name == "" {
		return "", fmt.Errorf("google: cached content must be a non-empty string")
	}
	return name, nil
}
