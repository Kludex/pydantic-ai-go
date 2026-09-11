// Package githubcopilot implements ai.Model against GitHub Copilot's Chat Completions API.
package githubcopilot

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"os"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

const defaultBaseURL = "https://api.githubcopilot.com"

// Model calls a model exposed by a GitHub Copilot subscription.
type Model struct {
	*ai.ModelWrapper
	model *openai.Model
	name  string
}

type config struct {
	apiKey          string
	baseURL         string
	httpClient      *http.Client
	provider        *openai.ProviderConfig
	defaultSettings ai.ModelSettings
}

// Option configures a GitHub Copilot model.
type Option func(*config)

// WithAPIKey sets the Copilot bearer token.
func WithAPIKey(apiKey string) Option { return func(config *config) { config.apiKey = apiKey } }

// WithBaseURL sets a Copilot Enterprise, GitHub Enterprise Server, or proxy endpoint.
func WithBaseURL(baseURL string) Option { return func(config *config) { config.baseURL = baseURL } }

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(config *config) { config.httpClient = client }
}

// WithProvider configures a gateway while retaining Copilot model behavior.
func WithProvider(provider openai.ProviderConfig) Option {
	provider.Headers = provider.Headers.Clone()
	return func(config *config) { config.provider = &provider }
}

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(config *config) { config.defaultSettings = settings }
}

// NewProviderConfig returns reusable GitHub Copilot endpoint configuration.
func NewProviderConfig() (openai.ProviderConfig, error) {
	apiKey := firstEnvironment("GITHUB_COPILOT_API_KEY", "GITHUB_COPILOT_API_TOKEN", "COPILOT_GITHUB_TOKEN")
	if apiKey == "" {
		return openai.ProviderConfig{}, fmt.Errorf(
			"githubcopilot: set GITHUB_COPILOT_API_KEY or use WithAPIKey",
		)
	}
	baseURL := firstEnvironment("GITHUB_COPILOT_BASE_URL", "COPILOT_API_URL", "GITHUB_COPILOT_API_BASE")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return openai.ProviderConfig{
		Name: "github-copilot", BaseURL: baseURL, APIKey: apiKey, Headers: copilotHeaders(),
	}, nil
}

// NewModel creates a GitHub Copilot Chat Completions model.
func NewModel(name string, options ...Option) (*Model, error) {
	configuration := config{
		apiKey:  firstEnvironment("GITHUB_COPILOT_API_KEY", "GITHUB_COPILOT_API_TOKEN", "COPILOT_GITHUB_TOKEN"),
		baseURL: firstEnvironment("GITHUB_COPILOT_BASE_URL", "COPILOT_API_URL", "GITHUB_COPILOT_API_BASE"),
	}
	if configuration.baseURL == "" {
		configuration.baseURL = defaultBaseURL
	}
	for _, option := range options {
		option(&configuration)
	}
	provider := openai.ProviderConfig{
		Name: "github-copilot", BaseURL: configuration.baseURL, APIKey: configuration.apiKey,
		HTTPClient: configuration.httpClient, Headers: copilotHeaders(),
	}
	if configuration.provider != nil {
		provider = *configuration.provider
		provider.Name = "github-copilot"
		headers := copilotHeaders()
		for name, values := range provider.Headers {
			headers[name] = append([]string(nil), values...)
		}
		provider.Headers = headers
	}
	if provider.APIKey == "" && provider.PrepareRequest == nil {
		return nil, fmt.Errorf("githubcopilot: set GITHUB_COPILOT_API_KEY or use WithAPIKey")
	}
	compatibility := openai.ChatCompatibility{DisableDocumentInput: true}
	bareName := strings.TrimPrefix(strings.ToLower(name), "copilot/")
	if strings.HasPrefix(bareName, "claude-") || strings.HasPrefix(bareName, "gemini-") {
		compatibility.ReasoningText = true
	}
	delegate := openai.NewModel(name,
		openai.WithProvider(provider),
		openai.WithChatCompatibility(compatibility),
		openai.WithDeferredToolSupport(false),
		openai.WithDefaultSettings(configuration.defaultSettings),
	)
	return &Model{ModelWrapper: ai.WrapModel(delegate), model: delegate, name: bareName}, nil
}

// Request sends one GitHub Copilot request.
func (model *Model) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	params = model.prepareParams(params)
	return model.model.Request(ctx, messages, params)
}

// StreamRequest streams one GitHub Copilot request.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	params = model.prepareParams(params)
	return model.model.StreamRequest(ctx, messages, params)
}

func (model *Model) prepareParams(params ai.ModelRequestParams) ai.ModelRequestParams {
	if copilotAnthropicDisallowsSampling(model.name) {
		params.Settings = params.Settings.Clone()
		params.Settings.Temperature = nil
		params.Settings.TopP = nil
	}
	return params
}

func copilotAnthropicDisallowsSampling(name string) bool {
	for _, prefix := range []string{
		"claude-opus-4.7", "claude-opus-4.8", "claude-opus-5", "claude-sonnet-5",
		"claude-fable-5", "claude-mythos-5",
	} {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

func copilotHeaders() http.Header {
	return http.Header{
		"Editor-Version":         {"vscode/1.95.0"},
		"Copilot-Integration-Id": {"vscode-chat"},
		"Editor-Plugin-Version":  {"copilot-chat/0.26.7"},
		"Openai-Intent":          {"conversation-panel"},
		"X-Github-Api-Version":   {"2025-04-01"},
	}
}

func firstEnvironment(names ...string) string {
	for _, name := range names {
		if value := os.Getenv(name); value != "" {
			return value
		}
	}
	return ""
}
