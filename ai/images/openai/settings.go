package openai

import (
	"fmt"

	"github.com/Kludex/pydantic-ai-go/ai/images"
)

// OutputFormat selects generated image encoding.
type OutputFormat string

const (
	OutputFormatPNG  OutputFormat = "png"
	OutputFormatWebP OutputFormat = "webp"
	OutputFormatJPEG OutputFormat = "jpeg"
)

// Quality selects GPT Image rendering quality.
type Quality string

const (
	QualityLow    Quality = "low"
	QualityMedium Quality = "medium"
	QualityHigh   Quality = "high"
	QualityAuto   Quality = "auto"
)

// Background selects transparent, opaque, or automatic output.
type Background string

const (
	BackgroundTransparent Background = "transparent"
	BackgroundOpaque      Background = "opaque"
	BackgroundAuto        Background = "auto"
)

// InputFidelity controls how closely edits preserve reference-image details.
type InputFidelity string

const (
	InputFidelityHigh InputFidelity = "high"
	InputFidelityLow  InputFidelity = "low"
)

// Moderation selects OpenAI image moderation strictness.
type Moderation string

const (
	ModerationAuto Moderation = "auto"
	ModerationLow  Moderation = "low"
)

const (
	nKey                 = "openai_n"
	outputFormatKey      = "openai_output_format"
	sizeKey              = "openai_size"
	qualityKey           = "openai_quality"
	backgroundKey        = "openai_background"
	inputFidelityKey     = "openai_input_fidelity"
	moderationKey        = "openai_moderation"
	outputCompressionKey = "openai_output_compression"
	userKey              = "openai_user"
)

// Settings combines portable settings with OpenAI image controls.
type Settings struct {
	Common            images.Settings
	N                 *int
	OutputFormat      OutputFormat
	Size              *string
	Quality           Quality
	Background        Background
	InputFidelity     InputFidelity
	Moderation        Moderation
	OutputCompression *int
	User              *string
}

// Build validates and returns settings accepted by images.Generator and Model.Generate.
func (settings Settings) Build() (images.Settings, error) {
	built := settings.Common.Clone()
	if err := built.Validate(); err != nil {
		return images.Settings{}, err
	}
	if settings.N != nil && *settings.N <= 0 {
		return images.Settings{}, fmt.Errorf("openai images: image count must be greater than zero")
	}
	values := []struct {
		key   string
		value any
		set   bool
	}{
		{nKey, pointerValue(settings.N), settings.N != nil},
		{outputFormatKey, settings.OutputFormat, settings.OutputFormat != ""},
		{sizeKey, pointerValue(settings.Size), settings.Size != nil},
		{qualityKey, settings.Quality, settings.Quality != ""},
		{backgroundKey, settings.Background, settings.Background != ""},
		{inputFidelityKey, settings.InputFidelity, settings.InputFidelity != ""},
		{moderationKey, settings.Moderation, settings.Moderation != ""},
		{outputCompressionKey, pointerValue(settings.OutputCompression), settings.OutputCompression != nil},
		{userKey, pointerValue(settings.User), settings.User != nil},
	}
	if built.ProviderSettings == nil {
		built.ProviderSettings = map[string]any{}
	}
	for _, key := range []string{
		nKey, outputFormatKey, sizeKey, qualityKey, backgroundKey, inputFidelityKey,
		moderationKey, outputCompressionKey, userKey,
	} {
		if _, exists := built.ProviderSettings[key]; exists {
			return images.Settings{}, fmt.Errorf("openai images: setting field %q is already set", key)
		}
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
