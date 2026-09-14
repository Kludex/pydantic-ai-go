package openai

import (
	"fmt"

	"github.com/Kludex/pydantic-ai-go/ai/images"
)

type providerSettings struct {
	n                 *int
	outputFormat      OutputFormat
	size              *string
	quality           Quality
	background        Background
	inputFidelity     InputFidelity
	moderation        Moderation
	outputCompression *int
	user              *string
}

func extractSettings(settings images.Settings) (images.Settings, providerSettings, error) {
	settings = settings.Clone()
	values := settings.ProviderSettings
	provider := providerSettings{}
	var err error
	provider.n, err = typedPointer[int](values, nKey)
	if err != nil {
		return images.Settings{}, providerSettings{}, err
	}
	provider.size, err = typedPointer[string](values, sizeKey)
	if err != nil {
		return images.Settings{}, providerSettings{}, err
	}
	provider.outputCompression, err = typedPointer[int](values, outputCompressionKey)
	if err != nil {
		return images.Settings{}, providerSettings{}, err
	}
	provider.user, err = typedPointer[string](values, userKey)
	if err != nil {
		return images.Settings{}, providerSettings{}, err
	}
	provider.outputFormat, err = typedValue[OutputFormat](values, outputFormatKey)
	if err != nil {
		return images.Settings{}, providerSettings{}, err
	}
	provider.quality, err = typedValue[Quality](values, qualityKey)
	if err != nil {
		return images.Settings{}, providerSettings{}, err
	}
	provider.background, err = typedValue[Background](values, backgroundKey)
	if err != nil {
		return images.Settings{}, providerSettings{}, err
	}
	provider.inputFidelity, err = typedValue[InputFidelity](values, inputFidelityKey)
	if err != nil {
		return images.Settings{}, providerSettings{}, err
	}
	provider.moderation, err = typedValue[Moderation](values, moderationKey)
	if err != nil {
		return images.Settings{}, providerSettings{}, err
	}
	return settings, provider, nil
}

func typedPointer[T any](values map[string]any, key string) (*T, error) {
	value, exists := values[key]
	if !exists {
		return nil, nil
	}
	typed, ok := value.(T)
	if !ok {
		return nil, fmt.Errorf("openai images: setting %q has type %T", key, value)
	}
	return &typed, nil
}

func typedValue[T any](values map[string]any, key string) (T, error) {
	var zero T
	value, exists := values[key]
	if !exists {
		return zero, nil
	}
	typed, ok := value.(T)
	if !ok {
		return zero, fmt.Errorf("openai images: setting %q has type %T", key, value)
	}
	return typed, nil
}

func pointerValue[T any](value *T) any {
	if value == nil {
		return nil
	}
	return *value
}
