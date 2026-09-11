// Package openai implements images.Model against OpenAI-compatible Images APIs.
package openai

import (
	"fmt"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strings"

	"github.com/Kludex/pydantic-ai-go/ai/images"
	modelopenai "github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

const defaultBaseURL = "https://api.openai.com/v1"

// Model calls OpenAI Images generation and editing endpoints.
type Model struct {
	name           string
	providerName   string
	baseURL        string
	apiKey         string
	httpClient     *http.Client
	headers        http.Header
	query          url.Values
	prepareRequest modelopenai.RequestPreparationFunc
	settings       images.Settings
}

// Option configures a Model.
type Option func(*Model)

// WithProvider configures an OpenAI-compatible image endpoint.
func WithProvider(provider modelopenai.ProviderConfig) Option {
	if provider.Name == "" {
		panic("openai images: provider name must not be empty")
	}
	if provider.BaseURL == "" {
		panic("openai images: provider base URL must not be empty")
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

// WithHTTPClient sets the caller-owned HTTP client used for requests.
func WithHTTPClient(client *http.Client) Option {
	return func(model *Model) { model.httpClient = client }
}

// WithDefaultSettings sets defaults overridden by each image request.
func WithDefaultSettings(settings images.Settings) Option {
	settings = settings.Clone()
	return func(model *Model) { model.settings = settings.Clone() }
}

// NewModel creates an OpenAI GPT Image model.
func NewModel(name string, options ...Option) *Model {
	if name == "dall-e-2" || name == "dall-e-3" {
		panic(fmt.Sprintf("openai images: model %q is not supported; use a GPT Image model", name))
	}
	model := &Model{
		name: name, providerName: "openai", baseURL: envOr("OPENAI_BASE_URL", defaultBaseURL),
		apiKey: os.Getenv("OPENAI_API_KEY"), httpClient: http.DefaultClient,
	}
	for _, option := range options {
		option(model)
	}
	return model
}

// Name returns the image model name.
func (model *Model) Name() string { return model.name }

// ProviderName returns the durable provider identity.
func (model *Model) ProviderName() string { return model.providerName }

// ProviderURL returns the configured provider endpoint.
func (model *Model) ProviderURL() string { return model.baseURL }

// DefaultSettings returns detached model defaults.
func (model *Model) DefaultSettings() images.Settings { return model.settings.Clone() }

func (model *Model) configureRequest(request *http.Request, extraHeaders map[string]string) error {
	if model.apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+model.apiKey)
	}
	for name, values := range model.headers {
		request.Header[name] = slices.Clone(values)
	}
	if model.prepareRequest != nil {
		if err := model.prepareRequest(request); err != nil {
			return fmt.Errorf("openai images: prepare request: %w", err)
		}
	}
	for name, value := range extraHeaders {
		request.Header.Set(name, value)
	}
	return nil
}

func (model *Model) endpoint(path string) string {
	endpoint := model.baseURL + path
	if query := model.query.Encode(); query != "" {
		endpoint += "?" + query
	}
	return endpoint
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
