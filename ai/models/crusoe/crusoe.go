// Package crusoe implements ai.Model against Crusoe Serverless Inference.
package crusoe

import (
	"net/http"
	"os"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

const defaultBaseURL = "https://api.inference.crusoecloud.com/v1"

type config struct{ options []openai.Option }

// Option configures a Crusoe model.
type Option func(*config)

// WithAPIKey sets the API key. The default is CRUSOE_API_KEY.
func WithAPIKey(key string) Option {
	return func(config *config) { config.options = append(config.options, openai.WithAPIKey(key)) }
}

// WithBaseURL points the model at a Crusoe-compatible endpoint.
func WithBaseURL(baseURL string) Option {
	return func(config *config) { config.options = append(config.options, openai.WithBaseURL(baseURL)) }
}

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(config *config) { config.options = append(config.options, openai.WithHTTPClient(client)) }
}

// WithProvider configures a gateway while retaining Crusoe model semantics.
func WithProvider(provider openai.ProviderConfig) Option {
	option := openai.WithProvider(provider)
	return func(config *config) { config.options = append(config.options, option) }
}

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(config *config) {
		config.options = append(config.options, openai.WithDefaultSettings(settings))
	}
}

// NewProviderConfig returns reusable Crusoe endpoint and environment configuration.
func NewProviderConfig() openai.ProviderConfig {
	return openai.ProviderConfig{
		Name: "crusoe", BaseURL: defaultBaseURL, APIKey: os.Getenv("CRUSOE_API_KEY"),
	}
}

// NewModel creates a Crusoe model. The name must include its vendor prefix.
func NewModel(name string, options ...Option) *openai.Model {
	configuration := config{}
	for _, option := range options {
		option(&configuration)
	}
	vendor, _, _ := strings.Cut(strings.ToLower(name), "/")
	compatibility := openai.ChatCompatibility{Reasoning: true}
	if vendor == "deepseek-ai" {
		compatibility = openai.ChatCompatibility{ReasoningContent: true}
	}
	openAIOptions := []openai.Option{
		openai.WithProvider(NewProviderConfig()),
		openai.WithChatCompatibility(compatibility),
	}
	openAIOptions = append(openAIOptions, configuration.options...)
	return openai.NewModel(name, openAIOptions...)
}
