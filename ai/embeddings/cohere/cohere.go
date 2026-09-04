// Package cohere implements embeddings.Model against Cohere's embedding API.
package cohere

import (
	"context"
	"net/http"
	"os"
	"strings"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
	modelcohere "github.com/Kludex/pydantic-ai-go/ai/models/cohere"
)

const defaultBaseURL = "https://api.cohere.com"

// Model calls Cohere's v2 Embed and v1 Tokenize endpoints.
type Model struct {
	name           string
	providerName   string
	baseURL        string
	apiKey         string
	httpClient     *http.Client
	headers        http.Header
	prepareRequest modelcohere.RequestPreparationFunc
	settings       embeddings.Settings
}

// Option configures a Model.
type Option func(*Model)

// WithProvider configures a Cohere or compatible endpoint.
func WithProvider(provider modelcohere.ProviderConfig) Option {
	provider = provider.Clone()
	if provider.Name == "" {
		panic("cohere embeddings: provider name must not be empty")
	}
	if provider.BaseURL == "" {
		panic("cohere embeddings: provider base URL must not be empty")
	}
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

// WithAPIKey sets the API key. The default is CO_API_KEY.
func WithAPIKey(apiKey string) Option { return func(model *Model) { model.apiKey = apiKey } }

// WithBaseURL points the model at a Cohere-compatible endpoint.
func WithBaseURL(baseURL string) Option {
	return func(model *Model) { model.baseURL = strings.TrimRight(baseURL, "/") }
}

// WithHTTPClient sets the HTTP client used for requests.
func WithHTTPClient(client *http.Client) Option {
	return func(model *Model) { model.httpClient = client }
}

// WithDefaultSettings sets defaults overridden by each embedding request.
func WithDefaultSettings(settings embeddings.Settings) Option {
	settings = settings.Clone()
	return func(model *Model) { model.settings = settings.Clone() }
}

// NewModel creates a Cohere embedding model.
func NewModel(name string, options ...Option) *Model {
	baseURL := os.Getenv("CO_BASE_URL")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	model := &Model{
		name: name, providerName: "cohere", baseURL: strings.TrimRight(baseURL, "/"),
		apiKey: os.Getenv("CO_API_KEY"), httpClient: http.DefaultClient,
	}
	for _, option := range options {
		option(model)
	}
	return model
}

// Name returns the embedding model name.
func (model *Model) Name() string { return model.name }

// ProviderName returns the durable provider identity.
func (model *Model) ProviderName() string { return model.providerName }

// ProviderURL returns the configured provider endpoint.
func (model *Model) ProviderURL() string { return model.baseURL }

// MaxInputTokens returns Cohere's documented input limit when known.
func (model *Model) MaxInputTokens(context.Context) (int, bool, error) {
	limits := map[string]int{
		"embed-v4.0":                    128000,
		"embed-english-v3.0":            512,
		"embed-english-light-v3.0":      512,
		"embed-multilingual-v3.0":       512,
		"embed-multilingual-light-v3.0": 512,
	}
	limit, known := limits[model.name]
	return limit, known, nil
}
