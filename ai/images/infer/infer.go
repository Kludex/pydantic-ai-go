// Package infer resolves provider-prefixed direct image model names.
package infer

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/Kludex/pydantic-ai-go/ai/images"
	imagegoogle "github.com/Kludex/pydantic-ai-go/ai/images/google"
	"github.com/Kludex/pydantic-ai-go/ai/images/openai"
	"github.com/Kludex/pydantic-ai-go/ai/images/xai"
	modelgoogle "github.com/Kludex/pydantic-ai-go/ai/models/google"
)

// Resolver creates a model from a provider-local model name.
type Resolver func(modelName string) (images.Model, error)

// Option configures model-name inference.
type Option func(*config)

type config struct {
	resolvers    map[string]Resolver
	vertexConfig modelgoogle.VertexConfig
}

// WithProvider registers or replaces one provider resolver.
func WithProvider(name string, resolver Resolver) Option {
	if name == "" {
		panic("image inference: provider name must not be empty")
	}
	if resolver == nil {
		panic("image inference: provider resolver must not be nil")
	}
	return func(config *config) { config.resolvers[name] = resolver }
}

// WithVertexConfig configures inferred google-cloud models.
func WithVertexConfig(vertexConfig modelgoogle.VertexConfig) Option {
	return func(config *config) { config.vertexConfig = vertexConfig }
}

// Model resolves a provider-prefixed name such as openai:gpt-image-2.
func Model(name string, options ...Option) (images.Model, error) {
	providerName, modelName, found := strings.Cut(name, ":")
	if !found || providerName == "" || modelName == "" {
		return nil, fmt.Errorf("image inference: model name %q must include non-empty provider and model names", name)
	}
	configuration := config{resolvers: map[string]Resolver{
		"openai": func(name string) (images.Model, error) {
			if name == "dall-e-2" || name == "dall-e-3" {
				return nil, fmt.Errorf("image inference: OpenAI model %q is not supported; use a GPT Image model", name)
			}
			return openai.NewModel(name), nil
		},
		"google": func(name string) (images.Model, error) { return imagegoogle.NewModel(name), nil },
		"xai":    func(name string) (images.Model, error) { return xai.NewModel(name), nil },
	}}
	for _, option := range options {
		option(&configuration)
	}
	resolver, exists := configuration.resolvers[providerName]
	if !exists && providerName == "google-cloud" {
		return imagegoogle.NewVertexModel(modelName, configuration.vertexConfig)
	}
	if !exists {
		return nil, fmt.Errorf("image inference: unknown provider %q", providerName)
	}
	model, err := resolver(modelName)
	if err != nil {
		return nil, err
	}
	if modelIsNil(model) {
		return nil, fmt.Errorf("image inference: provider %q returned a nil model", providerName)
	}
	return model, nil
}

func modelIsNil(model images.Model) bool {
	if model == nil {
		return true
	}
	value := reflect.ValueOf(model)
	return value.Kind() == reflect.Pointer && value.IsNil()
}
