// Package ollama implements embeddings.Model against Ollama's local embedding API.
package ollama

import (
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/Kludex/pydantic-ai-go/embeddings"
)

const defaultBaseURL = "http://localhost:11434"

// RequestPreparationFunc prepares a request after provider defaults and before per-request headers.
// The function must be safe for concurrent calls and must not retain the request.
type RequestPreparationFunc func(*http.Request) error

// ProviderConfig configures an Ollama endpoint. Headers are copied.
type ProviderConfig struct {
	BaseURL        string
	HTTPClient     *http.Client
	Headers        http.Header
	PrepareRequest RequestPreparationFunc
}

// APIError reports a non-successful Ollama HTTP response.
type APIError struct {
	StatusCode int
	Body       string
	Headers    http.Header
}

// Error implements error.
func (err *APIError) Error() string {
	return fmt.Sprintf("ollama API returned status %d: %s", err.StatusCode, err.Body)
}

// Model calls Ollama's native embedding endpoint.
type Model struct {
	name           string
	baseURL        string
	httpClient     *http.Client
	headers        http.Header
	prepareRequest RequestPreparationFunc
	settings       embeddings.Settings
}

// Option configures a Model.
type Option func(*Model)

// WithProvider configures the Ollama endpoint.
func WithProvider(provider ProviderConfig) Option {
	provider.Headers = provider.Headers.Clone()
	if provider.BaseURL == "" {
		panic("ollama embeddings: provider base URL must not be empty")
	}
	return func(model *Model) {
		model.baseURL = normalizeBaseURL(provider.BaseURL)
		if provider.HTTPClient != nil {
			model.httpClient = provider.HTTPClient
		}
		model.headers = provider.Headers.Clone()
		model.prepareRequest = provider.PrepareRequest
	}
}

// WithBaseURL points the model at an Ollama endpoint.
func WithBaseURL(baseURL string) Option {
	return func(model *Model) { model.baseURL = normalizeBaseURL(baseURL) }
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

// NewModel creates a local Ollama embedding model.
func NewModel(name string, options ...Option) *Model {
	baseURL := os.Getenv("OLLAMA_HOST")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	model := &Model{name: name, baseURL: normalizeBaseURL(baseURL), httpClient: http.DefaultClient}
	for _, option := range options {
		option(model)
	}
	return model
}

// Name returns the embedding model name.
func (model *Model) Name() string { return model.name }

// ProviderName returns the durable provider identity.
func (*Model) ProviderName() string { return "ollama" }

// ProviderURL returns the configured Ollama endpoint.
func (model *Model) ProviderURL() string { return model.baseURL }

func normalizeBaseURL(baseURL string) string {
	if !strings.Contains(baseURL, "://") {
		baseURL = "http://" + baseURL
	}
	return strings.TrimRight(baseURL, "/")
}
