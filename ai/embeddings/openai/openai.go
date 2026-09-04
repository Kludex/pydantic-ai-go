// Package openai implements embeddings.Model against OpenAI-compatible embedding APIs.
package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strings"
	"sync"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
	openaimodels "github.com/Kludex/pydantic-ai-go/ai/models/openai"
	"github.com/tiktoken-go/tokenizer"
)

const defaultBaseURL = "https://api.openai.com/v1"

// Model calls an OpenAI-compatible embeddings endpoint.
type Model struct {
	name           string
	providerName   string
	baseURL        string
	apiKey         string
	httpClient     *http.Client
	headers        http.Header
	query          url.Values
	prepareRequest openaimodels.RequestPreparationFunc
	settings       embeddings.Settings
	tokenizer      func() (tokenizer.Codec, error)
}

// Option configures a Model.
type Option func(*Model)

// WithProvider configures an OpenAI-compatible embeddings endpoint.
func WithProvider(provider openaimodels.ProviderConfig) Option {
	if provider.Name == "" {
		panic("openai embeddings: provider name must not be empty")
	}
	if provider.BaseURL == "" {
		panic("openai embeddings: provider base URL must not be empty")
	}
	headers := provider.Headers.Clone()
	query := cloneValues(provider.Query)
	return func(model *Model) {
		model.providerName = provider.Name
		model.baseURL = strings.TrimRight(provider.BaseURL, "/")
		model.apiKey = provider.APIKey
		if provider.HTTPClient != nil {
			model.httpClient = provider.HTTPClient
		}
		model.headers = headers.Clone()
		model.query = cloneValues(query)
		model.prepareRequest = provider.PrepareRequest
	}
}

// WithAPIKey sets the API key. The default is OPENAI_API_KEY.
func WithAPIKey(apiKey string) Option { return func(model *Model) { model.apiKey = apiKey } }

// WithBaseURL points the model at an OpenAI-compatible endpoint.
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

// NewModel creates an OpenAI embedding model.
func NewModel(name string, options ...Option) *Model {
	model := &Model{
		name: name, providerName: "openai", baseURL: envOr("OPENAI_BASE_URL", defaultBaseURL),
		apiKey: os.Getenv("OPENAI_API_KEY"), httpClient: http.DefaultClient,
	}
	for _, option := range options {
		option(model)
	}
	model.tokenizer = sync.OnceValues(func() (tokenizer.Codec, error) {
		if strings.HasPrefix(model.name, "text-embedding-3-") {
			return tokenizer.Get(tokenizer.Cl100kBase)
		}
		return tokenizer.ForModel(tokenizer.Model(model.name))
	})
	return model
}

// Name returns the embedding model name.
func (model *Model) Name() string { return model.name }

// ProviderName returns the durable provider identity.
func (model *Model) ProviderName() string { return model.providerName }

// ProviderURL returns the configured provider endpoint.
func (model *Model) ProviderURL() string { return model.baseURL }

// MaxInputTokens reports OpenAI's documented embedding-model input limit.
func (model *Model) MaxInputTokens(context.Context) (int, bool, error) {
	if model.providerName != "openai" {
		return 0, false, nil
	}
	return 8192, true, nil
}

// CountTokens counts input tokens with OpenAI's model-specific tokenizer.
func (model *Model) CountTokens(_ context.Context, text string) (int, error) {
	if model.providerName != "openai" {
		return 0, embeddings.ErrTokenCountingUnsupported
	}
	codec, err := model.tokenizer()
	if err != nil {
		return 0, fmt.Errorf("openai embeddings: tokenizer for model %q: %w", model.name, err)
	}
	return codec.Count(text)
}

func cloneValues(values url.Values) url.Values {
	if values == nil {
		return nil
	}
	cloned := make(url.Values, len(values))
	for key, value := range values {
		cloned[key] = slices.Clone(value)
	}
	return cloned
}

func envOr(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}
