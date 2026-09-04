package google

import (
	"fmt"
	"maps"

	ai "github.com/Kludex/pydantic-ai-go"
)

const cachedContentSetting = "google_cached_content"

// Settings combines portable settings with Google-specific request options.
type Settings struct {
	// Common contains portable model settings.
	Common ai.ModelSettings
	// CachedContent names a Gemini or Vertex cached-content resource.
	// The resource owns system instructions and tools, so those request fields are omitted.
	CachedContent string
}

// Build returns detached portable settings accepted by agents and direct requests.
func (settings Settings) Build() (ai.ModelSettings, error) {
	common := settings.Common.Clone()
	if settings.CachedContent == "" {
		return common, nil
	}
	extra := maps.Clone(common.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	if _, exists := extra[cachedContentSetting]; exists {
		return ai.ModelSettings{}, fmt.Errorf(
			"google: extra body field %q conflicts with typed settings", cachedContentSetting,
		)
	}
	extra[cachedContentSetting] = settings.CachedContent
	common.ExtraBody = extra
	return common, nil
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
