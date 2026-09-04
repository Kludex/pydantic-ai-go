package google

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"

	"golang.org/x/oauth2"
	oauthgoogle "golang.org/x/oauth2/google"
)

// Transport identifies the Google API used by a model.
type Transport string

const (
	// TransportGeminiAPI uses the Gemini Developer API.
	TransportGeminiAPI Transport = "gemini-api"
	// TransportVertexAI uses Google Cloud Vertex AI.
	TransportVertexAI Transport = "vertex-ai"
)

// RequestPreparationFunc prepares one Google request. It must be safe for
// concurrent calls. Model setting headers are applied after it returns.
type RequestPreparationFunc func(*http.Request) error

// ProviderConfig configures the transport used by a Google model.
type ProviderConfig struct {
	// Transport selects the Gemini Developer API or Vertex AI wire route.
	Transport Transport
	// Name is persisted in responses, usage, and telemetry.
	Name string
	// BaseURL is the transport-specific API endpoint.
	BaseURL string
	// APIKey authenticates Gemini API or Vertex Express Mode requests.
	APIKey string
	// HTTPClient performs requests. Nil uses the shared default client.
	HTTPClient *http.Client
	// PrepareRequest adds dynamic authentication or routing data.
	PrepareRequest RequestPreparationFunc
}

// WithProvider applies a detached provider configuration.
func WithProvider(provider ProviderConfig) Option {
	if provider.Transport != "" &&
		provider.Transport != TransportGeminiAPI && provider.Transport != TransportVertexAI {
		panic(fmt.Sprintf("google: invalid transport %q", provider.Transport))
	}
	provider.BaseURL = strings.TrimRight(provider.BaseURL, "/")
	return func(model *Model) {
		if provider.Transport != "" {
			model.transport = provider.Transport
		}
		if provider.Name != "" {
			model.providerName = provider.Name
		}
		if provider.BaseURL != "" {
			model.baseURL = provider.BaseURL
		}
		model.apiKey = provider.APIKey
		model.prepareRequest = provider.PrepareRequest
		if provider.HTTPClient != nil {
			model.httpClient = provider.HTTPClient
		}
	}
}

// TokenProvider returns one Google Cloud access token. It must be safe for
// concurrent calls.
type TokenProvider func(context.Context) (string, error)

// VertexConfig configures Google Cloud Vertex AI. Project defaults to
// GOOGLE_CLOUD_PROJECT. Location defaults to GOOGLE_CLOUD_LOCATION or global.
// APIKey enables Vertex AI Express Mode. Otherwise TokenProvider or Application
// Default Credentials provides OAuth authentication.
type VertexConfig struct {
	// Project is the Google Cloud project ID.
	Project string
	// Location is the Vertex region, multi-region, or global endpoint.
	Location string
	// APIKey enables Vertex AI Express Mode.
	APIKey string
	// TokenProvider supplies a fresh OAuth access token for each request.
	TokenProvider TokenProvider
	// Endpoint overrides the derived Vertex API endpoint.
	Endpoint string
	// HTTPClient performs requests. Nil uses an authenticated default client.
	HTTPClient *http.Client
}

// NewVertexModel creates a model routed through Google Cloud Vertex AI.
func NewVertexModel(name string, config VertexConfig, opts ...Option) (*Model, error) {
	provider, err := NewVertexProviderConfig(config)
	if err != nil {
		return nil, err
	}
	opts = append(opts, WithProvider(provider))
	return NewModel(name, opts...), nil
}

// NewVertexProviderConfig resolves a reusable Vertex AI provider configuration.
// The returned configuration can be used by chat and embedding models.
func NewVertexProviderConfig(config VertexConfig) (ProviderConfig, error) {
	if config.APIKey != "" && config.TokenProvider != nil {
		return ProviderConfig{}, fmt.Errorf("google: Vertex API key and token provider cannot both be set")
	}
	endpoint := strings.TrimRight(config.Endpoint, "/")
	if endpoint != "" {
		parsed, err := url.Parse(endpoint)
		if err != nil || parsed.Scheme == "" || parsed.Host == "" {
			return ProviderConfig{}, fmt.Errorf("google: Vertex endpoint must be an absolute URL")
		}
	}
	project := config.Project
	location := config.Location
	apiKey := config.APIKey
	explicitCloud := project != "" || location != "" || config.TokenProvider != nil
	if project == "" && config.APIKey == "" {
		project = os.Getenv("GOOGLE_CLOUD_PROJECT")
	}
	if location == "" && config.APIKey == "" {
		location = os.Getenv("GOOGLE_CLOUD_LOCATION")
	}
	if apiKey == "" && !explicitCloud && project == "" && location == "" &&
		os.Getenv("GOOGLE_APPLICATION_CREDENTIALS") == "" {
		apiKey = os.Getenv("GOOGLE_API_KEY")
		if apiKey == "" {
			apiKey = os.Getenv("GEMINI_API_KEY")
		}
	}
	provider := ProviderConfig{
		Transport: TransportVertexAI, Name: "google-cloud", APIKey: strings.TrimSpace(apiKey),
		HTTPClient: config.HTTPClient,
	}
	if location == "" {
		location = "global"
	}
	if url.PathEscape(location) != location {
		return ProviderConfig{}, fmt.Errorf("google: Vertex location %q is invalid", location)
	}
	if endpoint == "" {
		endpoint = defaultVertexEndpoint(location)
	}
	if provider.APIKey != "" {
		provider.BaseURL = endpoint + "/v1beta1/publishers/google"
		return provider, nil
	}
	if project == "" {
		return ProviderConfig{}, fmt.Errorf("google: Vertex project is required for OAuth authentication")
	}
	provider.BaseURL = fmt.Sprintf(
		"%s/v1beta1/projects/%s/locations/%s/publishers/google",
		endpoint, url.PathEscape(project), url.PathEscape(location),
	)
	tokenProvider := config.TokenProvider
	if tokenProvider == nil {
		tokenProvider = defaultGoogleTokenProvider()
	}
	provider.PrepareRequest = func(request *http.Request) error {
		token, err := tokenProvider(request.Context())
		if err != nil {
			return fmt.Errorf("get Google Cloud token: %w", err)
		}
		if token == "" {
			return fmt.Errorf("get Google Cloud token: empty token")
		}
		request.Header.Set("Authorization", "Bearer "+token)
		return nil
	}
	return provider, nil
}

func defaultVertexEndpoint(location string) string {
	switch location {
	case "global":
		return "https://aiplatform.googleapis.com"
	case "us", "eu":
		return "https://aiplatform." + location + ".rep.googleapis.com"
	default:
		return "https://" + location + "-aiplatform.googleapis.com"
	}
}

func defaultGoogleTokenProvider() TokenProvider {
	var once sync.Once
	var source oauth2.TokenSource
	var sourceErr error
	return func(context.Context) (string, error) {
		once.Do(func() {
			source, sourceErr = oauthgoogle.DefaultTokenSource(
				context.Background(), "https://www.googleapis.com/auth/cloud-platform",
			)
		})
		if sourceErr != nil {
			return "", sourceErr
		}
		token, err := source.Token()
		if err != nil {
			return "", err
		}
		return token.AccessToken, nil
	}
}
