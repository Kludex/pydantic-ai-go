// Package infer resolves provider-prefixed embedding model names.
package infer

import (
	"fmt"
	"os"
	"reflect"
	"strings"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
	"github.com/Kludex/pydantic-ai-go/ai/embeddings/bedrock"
	"github.com/Kludex/pydantic-ai-go/ai/embeddings/cohere"
	embeddinggoogle "github.com/Kludex/pydantic-ai-go/ai/embeddings/google"
	"github.com/Kludex/pydantic-ai-go/ai/embeddings/ollama"
	"github.com/Kludex/pydantic-ai-go/ai/embeddings/openai"
	"github.com/Kludex/pydantic-ai-go/ai/embeddings/voyageai"
	modelazure "github.com/Kludex/pydantic-ai-go/ai/models/azure"
	modelgithubcopilot "github.com/Kludex/pydantic-ai-go/ai/models/githubcopilot"
	modelgoogle "github.com/Kludex/pydantic-ai-go/ai/models/google"
	modelopenai "github.com/Kludex/pydantic-ai-go/ai/models/openai"
	modelopenrouter "github.com/Kludex/pydantic-ai-go/ai/models/openrouter"
	modelvllm "github.com/Kludex/pydantic-ai-go/ai/models/vllm"
	modelzai "github.com/Kludex/pydantic-ai-go/ai/models/zai"
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
	configuration.resolvers["ollama"] = func(modelName string) (embeddings.Model, error) {
		return ollama.NewModel(modelName), nil
	}
	configuration.resolvers["openrouter"] = func(modelName string) (embeddings.Model, error) {
		return openai.NewModel(modelName, openai.WithProvider(modelopenrouter.NewProviderConfig())), nil
	}
	configuration.resolvers["zai"] = func(modelName string) (embeddings.Model, error) {
		return openai.NewModel(modelName, openai.WithProvider(modelzai.NewProviderConfig())), nil
	}
	configuration.resolvers["vllm"] = func(modelName string) (embeddings.Model, error) {
		provider, err := modelvllm.NewProviderConfig()
		if err != nil {
			return nil, err
		}
		return openai.NewModel(modelName, openai.WithProvider(provider)), nil
	}
	configuration.resolvers["github-copilot"] = func(modelName string) (embeddings.Model, error) {
		provider, err := modelgithubcopilot.NewProviderConfig()
		if err != nil {
			return nil, err
		}
		return openai.NewModel(modelName, openai.WithProvider(provider)), nil
	}
	configuration.resolvers["bedrock"] = func(modelName string) (embeddings.Model, error) {
		return bedrock.NewModel(modelName)
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
	for providerName, provider := range compatibleProviders {
		configuration.resolvers[providerName] = func(modelName string) (embeddings.Model, error) {
			baseURL := provider.baseURL
			if configured := os.Getenv(provider.baseURLEnv); configured != "" {
				baseURL = configured
			}
			if baseURL == "" {
				return nil, fmt.Errorf("embedding inference: provider %q requires %s", provider.name, provider.baseURLEnv)
			}
			apiKey := ""
			for _, name := range provider.apiKeyEnvs {
				if apiKey = os.Getenv(name); apiKey != "" {
					break
				}
			}
			return openai.NewModel(modelName, openai.WithProvider(modelopenai.ProviderConfig{
				Name: provider.name, BaseURL: baseURL, APIKey: apiKey,
			})), nil
		}
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

type compatibleProvider struct {
	name       string
	baseURL    string
	baseURLEnv string
	apiKeyEnvs []string
}

var compatibleProviders = map[string]compatibleProvider{
	"alibaba": newCompatibleProvider(
		"alibaba", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "ALIBABA_BASE_URL",
		"ALIBABA_API_KEY", "DASHSCOPE_API_KEY",
	),
	"cerebras": newCompatibleProvider(
		"cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_BASE_URL", "CEREBRAS_API_KEY",
	),
	"crusoe": newCompatibleProvider(
		"crusoe", "https://api.inference.crusoecloud.com/v1", "CRUSOE_BASE_URL", "CRUSOE_API_KEY",
	),
	"deepseek": newCompatibleProvider(
		"deepseek", "https://api.deepseek.com", "DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY",
	),
	"fireworks": newCompatibleProvider(
		"fireworks", "https://api.fireworks.ai/inference/v1", "FIREWORKS_BASE_URL", "FIREWORKS_API_KEY",
	),
	"github": newCompatibleProvider(
		"github", "https://models.github.ai/inference", "GITHUB_MODELS_BASE_URL", "GITHUB_API_KEY",
	),
	"heroku": newCompatibleProvider(
		"heroku", "https://us.inference.heroku.com/v1", "HEROKU_INFERENCE_URL", "HEROKU_INFERENCE_KEY",
	),
	"litellm": newCompatibleProvider("litellm", "", "LITELLM_BASE_URL", "LITELLM_API_KEY"),
	"moonshotai": newCompatibleProvider(
		"moonshotai", "https://api.moonshot.ai/v1", "MOONSHOTAI_BASE_URL", "MOONSHOTAI_API_KEY",
	),
	"nebius": newCompatibleProvider(
		"nebius", "https://api.studio.nebius.com/v1", "NEBIUS_BASE_URL", "NEBIUS_API_KEY",
	),
	"ovhcloud": newCompatibleProvider(
		"ovhcloud", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1", "OVHCLOUD_BASE_URL", "OVHCLOUD_API_KEY",
	),
	"sambanova": newCompatibleProvider(
		"sambanova", "https://api.sambanova.ai/v1", "SAMBANOVA_BASE_URL", "SAMBANOVA_API_KEY",
	),
	"snowflake": newCompatibleProvider("snowflake", "", "SNOWFLAKE_BASE_URL", "SNOWFLAKE_TOKEN"),
	"together": newCompatibleProvider(
		"together", "https://api.together.xyz/v1", "TOGETHER_BASE_URL", "TOGETHER_API_KEY",
	),
	"vercel": newCompatibleProvider(
		"vercel", "https://ai-gateway.vercel.sh/v1", "VERCEL_AI_GATEWAY_BASE_URL",
		"VERCEL_AI_GATEWAY_API_KEY", "VERCEL_OIDC_TOKEN",
	),
}

func newCompatibleProvider(name string, baseURL string, baseURLEnv string, apiKeyEnvs ...string) compatibleProvider {
	return compatibleProvider{name: name, baseURL: baseURL, baseURLEnv: baseURLEnv, apiKeyEnvs: apiKeyEnvs}
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
