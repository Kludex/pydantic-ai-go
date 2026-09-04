package mistral

import (
	"net/http"
	"os"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
)

// Option configures a Mistral model.
type Option func(*Model)

// WithAPIKey sets the API key. The default is MISTRAL_API_KEY.
func WithAPIKey(key string) Option { return func(model *Model) { model.apiKey = key } }

// WithBaseURL points the model at a Mistral-compatible API root.
func WithBaseURL(baseURL string) Option {
	if baseURL == "" {
		panic("mistral: base URL must not be empty")
	}
	return func(model *Model) { model.baseURL = strings.TrimRight(baseURL, "/") }
}

// WithHTTPClient sets the caller-owned HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(model *Model) { model.httpClient = client }
}

// WithProvider configures a Mistral-compatible endpoint.
func WithProvider(provider ProviderConfig) Option {
	if provider.Name == "" {
		panic("mistral: provider name must not be empty")
	}
	if provider.BaseURL == "" {
		panic("mistral: provider base URL must not be empty")
	}
	provider = provider.Clone()
	return func(model *Model) {
		model.providerName = provider.Name
		model.baseURL = strings.TrimRight(provider.BaseURL, "/")
		model.apiKey = provider.APIKey
		if provider.HTTPClient != nil {
			model.httpClient = provider.HTTPClient
		}
		model.headers = provider.Headers.Clone()
		model.prepareRequest = provider.PrepareRequest
	}
}

// WithDefaultSettings sets request defaults overridden by agent and run settings.
func WithDefaultSettings(settings ai.ModelSettings) Option {
	settings = settings.Clone()
	return func(model *Model) { model.defaultSettings = settings.Clone() }
}

// NewProviderConfig returns reusable Mistral endpoint and environment configuration.
func NewProviderConfig() ProviderConfig {
	baseURL := os.Getenv("MISTRAL_BASE_URL")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return ProviderConfig{
		Name: "mistral", BaseURL: strings.TrimRight(baseURL, "/"), APIKey: os.Getenv("MISTRAL_API_KEY"),
	}
}

// NewModel creates a Mistral model, such as mistral-large-latest.
func NewModel(name string, options ...Option) *Model {
	provider := NewProviderConfig()
	model := &Model{
		name: name, providerName: provider.Name, baseURL: provider.BaseURL,
		apiKey: provider.APIKey, httpClient: http.DefaultClient,
	}
	for _, option := range options {
		option(model)
	}
	return model
}

// Name returns the configured model name.
func (model *Model) Name() string { return model.name }

// ProviderName returns the durable provider identity.
func (model *Model) ProviderName() string { return model.providerName }

// ProviderURL returns the configured Mistral API URL.
func (model *Model) ProviderURL() string { return model.baseURL }

// DefaultModelSettings returns a detached settings snapshot.
func (model *Model) DefaultModelSettings() ai.ModelSettings { return model.defaultSettings.Clone() }
