// Package google implements images.Model for Gemini and Vertex AI image models.
package google

import (
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/Kludex/pydantic-ai-go/ai/images"
	modelgoogle "github.com/Kludex/pydantic-ai-go/ai/models/google"
)

const defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"

// Model calls the Google generateContent image endpoint.
type Model struct {
	name           string
	transport      modelgoogle.Transport
	providerName   string
	apiKey         string
	baseURL        string
	httpClient     *http.Client
	prepareRequest modelgoogle.RequestPreparationFunc
	settings       images.Settings
}

// Option configures a Model.
type Option func(*Model)

// WithProvider configures a Gemini API, Vertex AI, or compatible endpoint.
func WithProvider(provider modelgoogle.ProviderConfig) Option {
	if provider.Transport != "" && provider.Transport != modelgoogle.TransportGeminiAPI &&
		provider.Transport != modelgoogle.TransportVertexAI {
		panic(fmt.Sprintf("google images: invalid transport %q", provider.Transport))
	}
	return func(model *Model) {
		if provider.Transport != "" {
			model.transport = provider.Transport
		}
		if provider.Name != "" {
			model.providerName = provider.Name
		}
		if provider.BaseURL != "" {
			model.baseURL = strings.TrimRight(provider.BaseURL, "/")
		}
		model.apiKey = provider.APIKey
		model.prepareRequest = provider.PrepareRequest
		if provider.HTTPClient != nil {
			model.httpClient = provider.HTTPClient
		}
	}
}

// WithAPIKey sets the API key. The default is GOOGLE_API_KEY or GEMINI_API_KEY.
func WithAPIKey(apiKey string) Option { return func(model *Model) { model.apiKey = apiKey } }

// WithBaseURL points the model at a compatible endpoint.
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

// NewModel creates a Gemini API image model.
func NewModel(name string, options ...Option) *Model {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	model := &Model{
		name: name, transport: modelgoogle.TransportGeminiAPI, providerName: "google",
		baseURL: defaultBaseURL, apiKey: apiKey, httpClient: http.DefaultClient,
	}
	for _, option := range options {
		option(model)
	}
	return model
}

// NewVertexModel creates a Vertex AI image model.
func NewVertexModel(name string, config modelgoogle.VertexConfig, options ...Option) (*Model, error) {
	provider, err := modelgoogle.NewVertexProviderConfig(config)
	if err != nil {
		return nil, err
	}
	options = append(options, WithProvider(provider))
	return NewModel(name, options...), nil
}

// Name returns the image model name.
func (model *Model) Name() string { return model.name }

// ProviderName returns the durable provider identity.
func (model *Model) ProviderName() string { return model.providerName }

// ProviderURL returns the configured provider endpoint.
func (model *Model) ProviderURL() string { return model.baseURL }

// Transport returns the configured Gemini API or Vertex AI route.
func (model *Model) Transport() modelgoogle.Transport { return model.transport }

// DefaultSettings returns detached model defaults.
func (model *Model) DefaultSettings() images.Settings { return model.settings.Clone() }
