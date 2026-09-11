package google

import (
	"fmt"

	"github.com/Kludex/pydantic-ai-go/ai/images"
)

// ImageSize selects a Google image resolution tier.
type ImageSize string

const (
	ImageSize512 ImageSize = "512"
	ImageSize1K  ImageSize = "1K"
	ImageSize2K  ImageSize = "2K"
	ImageSize4K  ImageSize = "4K"
)

const (
	aspectRatioKey        = "google_image_aspect_ratio"
	imageSizeKey          = "google_image_size"
	outputMIMETypeKey     = "google_image_output_mime_type"
	compressionQualityKey = "google_image_output_compression_quality"
)

// Settings combines portable settings with Google image configuration.
type Settings struct {
	Common                   images.Settings
	AspectRatio              string
	ImageSize                ImageSize
	OutputMIMEType           string
	OutputCompressionQuality *int
}

// Build validates and returns settings accepted by images.Generator and Model.Generate.
func (settings Settings) Build() (images.Settings, error) {
	built := settings.Common.Clone()
	if err := built.Validate(); err != nil {
		return images.Settings{}, err
	}
	if built.ProviderSettings == nil {
		built.ProviderSettings = map[string]any{}
	}
	for _, key := range []string{aspectRatioKey, imageSizeKey, outputMIMETypeKey, compressionQualityKey} {
		if _, exists := built.ProviderSettings[key]; exists {
			return images.Settings{}, fmt.Errorf("google images: setting field %q is already set", key)
		}
	}
	values := []struct {
		key   string
		value any
		set   bool
	}{
		{aspectRatioKey, settings.AspectRatio, settings.AspectRatio != ""},
		{imageSizeKey, settings.ImageSize, settings.ImageSize != ""},
		{outputMIMETypeKey, settings.OutputMIMEType, settings.OutputMIMEType != ""},
		{compressionQualityKey, pointerValue(settings.OutputCompressionQuality), settings.OutputCompressionQuality != nil},
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
	aspectRatio        string
	imageSize          ImageSize
	outputMIMEType     string
	compressionQuality *int
}

func extractSettings(settings images.Settings) (providerSettings, error) {
	values := settings.ProviderSettings
	aspectRatio, err := settingValue[string](values, aspectRatioKey)
	if err != nil {
		return providerSettings{}, err
	}
	imageSize, err := settingValue[ImageSize](values, imageSizeKey)
	if err != nil {
		return providerSettings{}, err
	}
	outputMIMEType, err := settingValue[string](values, outputMIMETypeKey)
	if err != nil {
		return providerSettings{}, err
	}
	compressionQuality, err := settingPointer[int](values, compressionQualityKey)
	if err != nil {
		return providerSettings{}, err
	}
	return providerSettings{
		aspectRatio: aspectRatio, imageSize: imageSize, outputMIMEType: outputMIMEType,
		compressionQuality: compressionQuality,
	}, nil
}

func settingValue[T any](values map[string]any, key string) (T, error) {
	var zero T
	value, exists := values[key]
	if !exists {
		return zero, nil
	}
	typed, ok := value.(T)
	if !ok {
		return zero, fmt.Errorf("google images: setting %q has type %T", key, value)
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
