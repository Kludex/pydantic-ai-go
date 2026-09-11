// Package images provides provider-neutral direct image generation.
package images

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// Input is one reference image used for editing or transformation.
// Generate accepts ImageURL, BinaryContent, and UploadedFile values.
type Input = ai.UserContent

// Model generates images through a dedicated image endpoint.
type Model interface {
	// Generate creates one or more images from a prompt and optional references.
	Generate(ctx context.Context, prompt string, inputs []Input, settings Settings) (*Result, error)
	// Name returns the provider-local model name.
	Name() string
	// ProviderName returns the stable provider identity used for pricing and telemetry.
	ProviderName() string
	// ProviderURL returns the configured provider endpoint.
	ProviderURL() string
}

// DefaultSettingsModel exposes model-level defaults to generators and wrappers.
type DefaultSettingsModel interface {
	Model
	// DefaultSettings returns detached model-level defaults.
	DefaultSettings() Settings
}

// Wrapper delegates every image generation operation to Model.
type Wrapper struct {
	Model
}

// ModelUnwrapper exposes the model wrapped by a decorator.
type ModelUnwrapper interface {
	// UnwrapModel returns the directly wrapped model.
	UnwrapModel() Model
}

// WrapModel creates a transparent image generation model wrapper.
func WrapModel(model Model) *Wrapper {
	if modelIsNil(model) {
		panic("images: cannot wrap a nil model")
	}
	return &Wrapper{Model: model}
}

// UnwrapModel returns the directly wrapped model.
func (wrapper *Wrapper) UnwrapModel() Model { return wrapper.Model }

// DefaultSettings returns detached defaults when the wrapped model exposes them.
func (wrapper *Wrapper) DefaultSettings() Settings { return modelDefaultSettings(wrapper.Model) }

// PrepareRequest validates and detaches one direct image generation request.
// Provider implementations should call it before constructing their wire request.
func PrepareRequest(prompt string, inputs []Input, settings Settings) (string, []Input, Settings, error) {
	if strings.TrimSpace(prompt) == "" {
		return "", nil, Settings{}, fmt.Errorf("images: prompt must not be empty")
	}
	if err := settings.Validate(); err != nil {
		return "", nil, Settings{}, err
	}
	for _, input := range inputs {
		var mediaType string
		switch input := input.(type) {
		case ai.ImageURL:
			continue
		case ai.BinaryContent:
			mediaType = input.MediaType
		case ai.UploadedFile:
			mediaType = input.ResolvedMediaType()
		default:
			return "", nil, Settings{}, fmt.Errorf("images: unsupported reference image type %T", input)
		}
		if !strings.HasPrefix(strings.ToLower(mediaType), "image/") {
			return "", nil, Settings{}, fmt.Errorf("images: reference content must have an image media type, got %q", mediaType)
		}
	}
	return prompt, cloneInputs(inputs), settings.Clone(), nil
}

func modelDefaultSettings(model Model) Settings {
	if configured, ok := model.(DefaultSettingsModel); ok {
		return configured.DefaultSettings().Clone()
	}
	return Settings{}
}

func modelIsNil(model Model) bool {
	if model == nil {
		return true
	}
	value := reflect.ValueOf(model)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}
