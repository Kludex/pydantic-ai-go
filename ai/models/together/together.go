// Package together implements ai.Model against Together AI's OpenAI-compatible API.
package together

import (
	"net/http"
	"os"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

const defaultBaseURL = "https://api.together.xyz/v1"

type config struct{ options []openai.Option }

// Option configures a Together model.
type Option func(*config)

// WithAPIKey sets the API key. The default is TOGETHER_API_KEY.
func WithAPIKey(apiKey string) Option {
	return func(config *config) { config.options = append(config.options, openai.WithAPIKey(apiKey)) }
}

// WithBaseURL points the model at a Together-compatible endpoint.
func WithBaseURL(baseURL string) Option {
	return func(config *config) { config.options = append(config.options, openai.WithBaseURL(baseURL)) }
}

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(config *config) { config.options = append(config.options, openai.WithHTTPClient(client)) }
}

// WithProvider configures a gateway while retaining Together model behavior.
func WithProvider(provider openai.ProviderConfig) Option {
	provider.Name = "together"
	option := openai.WithProvider(provider)
	return func(config *config) { config.options = append(config.options, option) }
}

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(config *config) { config.options = append(config.options, openai.WithDefaultSettings(settings)) }
}

// NewProviderConfig returns reusable Together endpoint and environment configuration.
func NewProviderConfig() openai.ProviderConfig {
	return openai.ProviderConfig{
		Name: "together", BaseURL: defaultBaseURL, APIKey: os.Getenv("TOGETHER_API_KEY"),
	}
}

// NewModel creates a Together Chat Completions model.
func NewModel(name string, options ...Option) *openai.Model {
	configuration := config{}
	for _, option := range options {
		option(&configuration)
	}
	lowerName := strings.ToLower(name)
	compatibility := openai.ChatCompatibility{ReasoningFallback: true}
	if strings.HasPrefix(lowerName, "deepseek-ai/deepseek-v4-") {
		compatibility.DisableRequiredToolChoice = true
	}
	openAIOptions := []openai.Option{
		openai.WithProvider(NewProviderConfig()),
		openai.WithChatCompatibility(compatibility),
		openai.WithDeferredToolSupport(false),
	}
	openAIOptions = append(openAIOptions, configuration.options...)
	return openai.NewModel(name, openAIOptions...)
}
