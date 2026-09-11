// Package infer resolves provider-prefixed generation model names.
package infer

import (
	"fmt"
	"reflect"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/anthropic"
	"github.com/Kludex/pydantic-ai-go/ai/models/bedrock"
	"github.com/Kludex/pydantic-ai-go/ai/models/bedrockmantle"
	"github.com/Kludex/pydantic-ai-go/ai/models/cerebras"
	"github.com/Kludex/pydantic-ai-go/ai/models/cohere"
	"github.com/Kludex/pydantic-ai-go/ai/models/crusoe"
	"github.com/Kludex/pydantic-ai-go/ai/models/githubcopilot"
	"github.com/Kludex/pydantic-ai-go/ai/models/google"
	"github.com/Kludex/pydantic-ai-go/ai/models/groq"
	"github.com/Kludex/pydantic-ai-go/ai/models/huggingface"
	"github.com/Kludex/pydantic-ai-go/ai/models/mistral"
	"github.com/Kludex/pydantic-ai-go/ai/models/ollama"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openrouter"
	"github.com/Kludex/pydantic-ai-go/ai/models/snowflake"
	"github.com/Kludex/pydantic-ai-go/ai/models/vllm"
	"github.com/Kludex/pydantic-ai-go/ai/models/xai"
	"github.com/Kludex/pydantic-ai-go/ai/models/zai"
)

// Resolver creates a model from a provider-local name. It must be safe for concurrent calls.
type Resolver func(modelName string) (ai.Model, error)

// Option configures model-name inference.
type Option func(*config)

type config struct{ resolvers map[string]Resolver }

// WithProvider registers or replaces one provider resolver.
func WithProvider(name string, resolver Resolver) Option {
	if name == "" {
		panic("model inference: provider name must not be empty")
	}
	if resolver == nil {
		panic("model inference: provider resolver must not be nil")
	}
	return func(config *config) { config.resolvers[name] = resolver }
}

// Model resolves a provider-prefixed name such as openai:gpt-5-mini.
func Model(name string, options ...Option) (ai.Model, error) {
	providerName, modelName, found := strings.Cut(name, ":")
	if !found {
		return nil, fmt.Errorf("model inference: model name %q must include a provider prefix", name)
	}
	if providerName == "" || modelName == "" {
		return nil, fmt.Errorf("model inference: model name %q must include non-empty provider and model names", name)
	}
	configuration := config{resolvers: map[string]Resolver{
		"openai":           func(name string) (ai.Model, error) { return openai.NewModel(name), nil },
		"openai-responses": func(name string) (ai.Model, error) { return openai.NewResponsesModel(name), nil },
		"anthropic":        func(name string) (ai.Model, error) { return anthropic.NewModel(name), nil },
		"google":           func(name string) (ai.Model, error) { return google.NewModel(name), nil },
		"github-copilot":   func(name string) (ai.Model, error) { return githubcopilot.NewModel(name) },
		"bedrock":          func(name string) (ai.Model, error) { return bedrock.NewModel(name), nil },
		"bedrock-mantle":   func(name string) (ai.Model, error) { return bedrockmantle.NewModel(name), nil },
		"cerebras":         func(name string) (ai.Model, error) { return cerebras.NewModel(name), nil },
		"cohere":           func(name string) (ai.Model, error) { return cohere.NewModel(name), nil },
		"crusoe":           func(name string) (ai.Model, error) { return crusoe.NewModel(name), nil },
		"groq":             func(name string) (ai.Model, error) { return groq.NewModel(name), nil },
		"huggingface":      func(name string) (ai.Model, error) { return huggingface.NewModel(name), nil },
		"mistral":          func(name string) (ai.Model, error) { return mistral.NewModel(name), nil },
		"ollama":           func(name string) (ai.Model, error) { return ollama.NewModel(name), nil },
		"openrouter":       func(name string) (ai.Model, error) { return openrouter.NewModel(name), nil },
		"snowflake":        func(name string) (ai.Model, error) { return snowflake.NewModel(name), nil },
		"vllm":             func(name string) (ai.Model, error) { return vllm.NewModel(name) },
		"xai":              func(name string) (ai.Model, error) { return xai.NewModel(name), nil },
		"zai":              func(name string) (ai.Model, error) { return zai.NewModel(name), nil },
	}}
	for _, option := range options {
		option(&configuration)
	}
	resolver, exists := configuration.resolvers[providerName]
	if !exists {
		return nil, fmt.Errorf("model inference: unknown provider %q", providerName)
	}
	model, err := resolver(modelName)
	if err != nil {
		return nil, err
	}
	if modelIsNil(model) {
		return nil, fmt.Errorf("model inference: provider %q returned a nil model", providerName)
	}
	return model, nil
}

func modelIsNil(model ai.Model) bool {
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
