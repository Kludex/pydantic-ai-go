// Package voyageai implements embeddings.Model against VoyageAI's embedding API.
package voyageai

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
)

const defaultBaseURL = "https://api.voyageai.com/v1"

// RequestPreparationFunc prepares a request after provider defaults and before per-request headers.
// The function must be safe for concurrent calls and must not retain the request.
type RequestPreparationFunc func(*http.Request) error

// ProviderConfig configures a VoyageAI or compatible endpoint. Headers are copied.
type ProviderConfig struct {
	// Name is persisted in results, usage, and telemetry.
	Name string
	// BaseURL is the VoyageAI-compatible API endpoint.
	BaseURL string
	// APIKey is sent as a bearer token.
	APIKey string
	// HTTPClient performs requests. Nil uses the shared default client.
	HTTPClient *http.Client
	// Headers contains detached provider-wide request headers.
	Headers http.Header
	// PrepareRequest adds dynamic authentication or routing data.
	PrepareRequest RequestPreparationFunc
}

// APIError reports a non-successful VoyageAI HTTP response.
type APIError struct {
	// StatusCode is the HTTP response status.
	StatusCode int
	// Body is the provider response body.
	Body string
	// Headers contains a detached response-header snapshot.
	Headers http.Header
	// ProviderName identifies the endpoint that returned the error.
	ProviderName string
}

// Error implements error.
func (err *APIError) Error() string {
	return fmt.Sprintf("%s API returned status %d: %s", err.ProviderName, err.StatusCode, err.Body)
}

// Model calls VoyageAI's embeddings endpoint.
type Model struct {
	name           string
	providerName   string
	baseURL        string
	apiKey         string
	httpClient     *http.Client
	headers        http.Header
	prepareRequest RequestPreparationFunc
	settings       embeddings.Settings
}

// Option configures a Model.
type Option func(*Model)

// WithProvider configures a VoyageAI or compatible endpoint.
func WithProvider(provider ProviderConfig) Option {
	provider.Headers = provider.Headers.Clone()
	if provider.Name == "" {
		panic("voyageai embeddings: provider name must not be empty")
	}
	if provider.BaseURL == "" {
		panic("voyageai embeddings: provider base URL must not be empty")
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

// WithAPIKey sets the API key. The default is VOYAGE_API_KEY.
func WithAPIKey(apiKey string) Option { return func(model *Model) { model.apiKey = apiKey } }

// WithBaseURL points the model at a VoyageAI-compatible endpoint.
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

// NewModel creates a VoyageAI embedding model.
func NewModel(name string, options ...Option) *Model {
	model := &Model{
		name: name, providerName: "voyageai", baseURL: defaultBaseURL,
		apiKey: os.Getenv("VOYAGE_API_KEY"), httpClient: http.DefaultClient,
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

// MaxInputTokens returns VoyageAI's documented input limit when known.
func (model *Model) MaxInputTokens(context.Context) (int, bool, error) {
	limits := map[string]int{
		"voyage-4-large": 32000, "voyage-4": 32000, "voyage-4-lite": 32000,
		"voyage-3-large": 32000, "voyage-3.5": 32000, "voyage-3.5-lite": 32000,
		"voyage-code-3": 32000, "voyage-finance-2": 32000,
		"voyage-law-2": 16000, "voyage-code-2": 16000,
	}
	limit, known := limits[model.name]
	return limit, known, nil
}
