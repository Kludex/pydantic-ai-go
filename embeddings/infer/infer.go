// Package infer resolves provider-prefixed embedding model names.
package infer

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	"github.com/Kludex/pydantic-ai-go/embeddings/cohere"
	embeddinggoogle "github.com/Kludex/pydantic-ai-go/embeddings/google"
	"github.com/Kludex/pydantic-ai-go/embeddings/openai"
	"github.com/Kludex/pydantic-ai-go/embeddings/voyageai"
	modelazure "github.com/Kludex/pydantic-ai-go/models/azure"
	modelgoogle "github.com/Kludex/pydantic-ai-go/models/google"
)

// Resolver creates a model from the provider-local model name.
// It must be safe for concurrent calls.
type Resolver func(modelName string) (embeddings.Model, error)

// Option configures model-name inference.
type Option func(*config)

type config struct {
	resolvers    map[string]Resolver
	azureConfig  modelazure.Config
	vertexConfig modelgoogle.VertexConfig
}

// WithProvider registers or replaces one provider resolver.
func WithProvider(name string, resolver Resolver) Option {
	if name == "" {
		panic("embedding inference: provider name must not be empty")
	}
	if resolver == nil {
		panic("embedding inference: provider resolver must not be nil")
	}
	return func(config *config) { config.resolvers[name] = resolver }
}

// WithAzureConfig configures inferred azure models.
func WithAzureConfig(azureConfig modelazure.Config) Option {
	return func(config *config) { config.azureConfig = azureConfig }
}

// WithVertexConfig configures inferred google-cloud models.
func WithVertexConfig(vertexConfig modelgoogle.VertexConfig) Option {
	return func(config *config) { config.vertexConfig = vertexConfig }
}

// Model resolves a provider-prefixed name such as openai:text-embedding-3-small.
func Model(name string, options ...Option) (embeddings.Model, error) {
	providerName, modelName, found := strings.Cut(name, ":")
	if !found {
		return nil, fmt.Errorf("embedding inference: model name %q must include a provider prefix", name)
	}
	if providerName == "" || modelName == "" {
		return nil, fmt.Errorf("embedding inference: model name %q must include non-empty provider and model names", name)
	}
	configuration := config{resolvers: map[string]Resolver{}}
	configuration.resolvers["openai"] = func(modelName string) (embeddings.Model, error) {
		return openai.NewModel(modelName), nil
	}
	configuration.resolvers["azure"] = func(modelName string) (embeddings.Model, error) {
		provider, err := modelazure.NewProviderConfig(modelName, configuration.azureConfig)
		if err != nil {
			return nil, err
		}
		return openai.NewModel(modelName, openai.WithProvider(provider)), nil
	}
	configuration.resolvers["cohere"] = func(modelName string) (embeddings.Model, error) {
		return cohere.NewModel(modelName), nil
	}
	configuration.resolvers["google"] = func(modelName string) (embeddings.Model, error) {
		return embeddinggoogle.NewModel(modelName), nil
	}
	configuration.resolvers["voyageai"] = func(modelName string) (embeddings.Model, error) {
		return voyageai.NewModel(modelName), nil
	}
	for _, option := range options {
		option(&configuration)
	}
	if resolver, exists := configuration.resolvers[providerName]; exists {
		model, err := resolver(modelName)
		if err != nil {
			return nil, err
		}
		if modelIsNil(model) {
			return nil, fmt.Errorf("embedding inference: provider %q returned a nil model", providerName)
		}
		return model, nil
	}
	if providerName == "google-cloud" {
		return embeddinggoogle.NewVertexModel(modelName, configuration.vertexConfig)
	}
	return nil, fmt.Errorf("embedding inference: unknown provider %q", providerName)
}

func modelIsNil(model embeddings.Model) bool {
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
