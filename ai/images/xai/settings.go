package xai

import (
	"fmt"

	"github.com/Kludex/pydantic-ai-go/ai/images"
)

// Resolution selects an xAI image resolution tier.
type Resolution string

const (
	Resolution1K Resolution = "1k"
	Resolution2K Resolution = "2k"
)

const (
	nKey           = "xai_n"
	userKey        = "xai_user"
	aspectRatioKey = "xai_aspect_ratio"
	resolutionKey  = "xai_resolution"
)

// Settings combines portable settings with xAI image controls.
type Settings struct {
	Common      images.Settings
	N           *int
	User        string
	AspectRatio images.AspectRatio
	Resolution  Resolution
}

// Build validates and returns settings accepted by images.Generator and Model.Generate.
func (settings Settings) Build() (images.Settings, error) {
	built := settings.Common.Clone()
	if err := built.Validate(); err != nil {
		return images.Settings{}, err
	}
	if settings.N != nil && *settings.N <= 0 {
		return images.Settings{}, fmt.Errorf("xai images: image count must be greater than zero")
	}
	if built.ProviderSettings == nil {
		built.ProviderSettings = map[string]any{}
	}
	for _, key := range []string{nKey, userKey, aspectRatioKey, resolutionKey} {
		if _, exists := built.ProviderSettings[key]; exists {
			return images.Settings{}, fmt.Errorf("xai images: setting field %q is already set", key)
		}
	}
	values := []struct {
		key   string
		value any
		set   bool
	}{
		{nKey, pointerValue(settings.N), settings.N != nil}, {userKey, settings.User, settings.User != ""},
		{aspectRatioKey, settings.AspectRatio, settings.AspectRatio != ""},
		{resolutionKey, settings.Resolution, settings.Resolution != ""},
	}
	for _, value := range values {
		if !value.set {
			continue
		}
		built.ProviderSettings[value.key] = value.value
	}
	if len(built.ProviderSettings) == 0 {
		built.ProviderSettings = nil
	}
	return built, nil
}

type providerSettings struct {
	n           *int
	user        string
	aspectRatio images.AspectRatio
	resolution  Resolution
}

func extractSettings(settings images.Settings) (providerSettings, error) {
	n, err := settingPointer[int](settings.ProviderSettings, nKey)
	if err != nil {
		return providerSettings{}, err
	}
	user, err := settingValue[string](settings.ProviderSettings, userKey)
	if err != nil {
		return providerSettings{}, err
	}
	aspectRatio, err := settingValue[images.AspectRatio](settings.ProviderSettings, aspectRatioKey)
	if err != nil {
		return providerSettings{}, err
	}
	resolution, err := settingValue[Resolution](settings.ProviderSettings, resolutionKey)
	if err != nil {
		return providerSettings{}, err
	}
	return providerSettings{n: n, user: user, aspectRatio: aspectRatio, resolution: resolution}, nil
}

func settingValue[T any](values map[string]any, key string) (T, error) {
	var zero T
	value, exists := values[key]
	if !exists {
		return zero, nil
	}
	typed, ok := value.(T)
	if !ok {
		return zero, fmt.Errorf("xai images: setting %q has type %T", key, value)
	}
	return typed, nil
}

func settingPointer[T any](values map[string]any, key string) (*T, error) {
	value, err := settingValue[T](values, key)
	if err != nil {
		return nil, err
	}
	if _, exists := values[key]; !exists {
		return nil, nil
	}
	return &value, nil
}

func pointerValue[T any](value *T) any {
	if value == nil {
		return nil
	}
	return *value
}
