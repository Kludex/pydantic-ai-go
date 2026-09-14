package images

import (
	"fmt"
	"maps"
)

// Dimensions identifies exact output image dimensions in pixels.
type Dimensions struct {
	Width  int
	Height int
}

// AspectRatio identifies a portable output aspect ratio.
type AspectRatio string

const (
	AspectRatio1To1    AspectRatio = "1:1"
	AspectRatio1To2    AspectRatio = "1:2"
	AspectRatio1To4    AspectRatio = "1:4"
	AspectRatio1To8    AspectRatio = "1:8"
	AspectRatio2To1    AspectRatio = "2:1"
	AspectRatio2To3    AspectRatio = "2:3"
	AspectRatio3To2    AspectRatio = "3:2"
	AspectRatio3To4    AspectRatio = "3:4"
	AspectRatio4To1    AspectRatio = "4:1"
	AspectRatio4To3    AspectRatio = "4:3"
	AspectRatio4To5    AspectRatio = "4:5"
	AspectRatio5To4    AspectRatio = "5:4"
	AspectRatio8To1    AspectRatio = "8:1"
	AspectRatio9To16   AspectRatio = "9:16"
	AspectRatio9To19_5 AspectRatio = "9:19.5"
	AspectRatio9To20   AspectRatio = "9:20"
	AspectRatio16To9   AspectRatio = "16:9"
	AspectRatio19_5To9 AspectRatio = "19.5:9"
	AspectRatio20To9   AspectRatio = "20:9"
	AspectRatio21To9   AspectRatio = "21:9"
)

// Settings configures a direct image generation request.
type Settings struct {
	// Dimensions requests an exact width and height. It is mutually exclusive with AspectRatio.
	Dimensions *Dimensions
	// AspectRatio requests a provider-supported output shape.
	AspectRatio AspectRatio
	// ExtraHeaders contains per-request headers applied after provider defaults.
	ExtraHeaders map[string]string
	// ExtraBody contains provider-specific wire fields that do not conflict with typed settings.
	ExtraBody map[string]any
	// ProviderSettings contains typed adapter settings produced by provider packages.
	// Applications should use each provider's Settings.Build method instead of setting it directly.
	ProviderSettings map[string]any
}

// Validate checks provider-neutral image settings.
func (settings Settings) Validate() error {
	if settings.Dimensions == nil {
		return nil
	}
	if settings.AspectRatio != "" {
		return fmt.Errorf("images: dimensions and aspect ratio are mutually exclusive")
	}
	if settings.Dimensions.Width <= 0 || settings.Dimensions.Height <= 0 {
		return fmt.Errorf("images: dimensions must contain positive width and height values")
	}
	return nil
}

// Clone returns a detached settings value.
func (settings Settings) Clone() Settings {
	settings.Dimensions = clonePointer(settings.Dimensions)
	settings.ExtraHeaders = maps.Clone(settings.ExtraHeaders)
	settings.ExtraBody = cloneMap(settings.ExtraBody)
	settings.ProviderSettings = cloneMap(settings.ProviderSettings)
	return settings
}

// MergeSettings overlays non-zero override fields on base and returns a detached value.
func MergeSettings(base, override Settings) Settings {
	merged := base.Clone()
	if override.Dimensions != nil {
		merged.Dimensions = clonePointer(override.Dimensions)
	}
	if override.AspectRatio != "" {
		merged.AspectRatio = override.AspectRatio
	}
	if override.ExtraHeaders != nil {
		merged.ExtraHeaders = maps.Clone(override.ExtraHeaders)
	}
	if override.ExtraBody != nil {
		merged.ExtraBody = cloneMap(override.ExtraBody)
	}
	if override.ProviderSettings != nil {
		if merged.ProviderSettings == nil {
			merged.ProviderSettings = map[string]any{}
		}
		for name, value := range override.ProviderSettings {
			merged.ProviderSettings[name] = cloneMap(map[string]any{name: value})[name]
		}
	}
	return merged
}

func clonePointer[T any](value *T) *T {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}
