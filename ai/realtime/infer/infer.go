// Package infer resolves provider-prefixed realtime model identifiers.
package infer

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	azurert "github.com/Kludex/pydantic-ai-go/ai/realtime/azure"
	googlert "github.com/Kludex/pydantic-ai-go/ai/realtime/google"
	openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
	xairt "github.com/Kludex/pydantic-ai-go/ai/realtime/xai"
)

// Resolver creates a realtime model from a provider-local name.
// It must be safe for concurrent calls.
type Resolver func(modelName string) (realtime.Model, error)

// Option configures realtime model inference.
type Option func(*config)

type config struct{ resolvers map[string]Resolver }

// WithProvider registers or replaces one provider resolver.
func WithProvider(name string, resolver Resolver) Option {
	if name == "" {
		panic("realtime inference: provider name must not be empty")
	}
	if resolver == nil {
		panic("realtime inference: provider resolver must not be nil")
	}
	return func(config *config) { config.resolvers[name] = resolver }
}

// Model resolves provider:model into a realtime provider adapter.
func Model(name string, options ...Option) (realtime.Model, error) {
	provider, modelName, found := strings.Cut(name, ":")
	if !found || provider == "" || modelName == "" {
		return nil, fmt.Errorf("realtime: model identifiers use provider:model, got %q", name)
	}
	configuration := config{resolvers: map[string]Resolver{
		"azure": func(name string) (realtime.Model, error) {
			return azurert.NewModel(name, azurert.Config{})
		},
		"openai": func(name string) (realtime.Model, error) { return openairt.NewModel(name), nil },
		"xai":    func(name string) (realtime.Model, error) { return xairt.NewModel(name), nil },
		"google": func(name string) (realtime.Model, error) { return googlert.NewModel(name), nil },
		"google-cloud": func(name string) (realtime.Model, error) {
			return googlert.NewModel(name, googlert.WithVertex("", "")), nil
		},
	}}
	for _, option := range options {
		option(&configuration)
	}
	resolver, exists := configuration.resolvers[provider]
	if !exists {
		return nil, fmt.Errorf("realtime: unknown provider %q", provider)
	}
	model, err := resolver(modelName)
	if err != nil {
		return nil, err
	}
	if modelIsNil(model) {
		return nil, fmt.Errorf("realtime: provider %q returned a nil model", provider)
	}
	return model, nil
}

func modelIsNil(model realtime.Model) bool {
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
