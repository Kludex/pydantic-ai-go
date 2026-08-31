// Package azure configures OpenAI-compatible models for Azure OpenAI and Azure AI Foundry.
package azure

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/Kludex/pydantic-ai-go/models/openai"
)

// TokenProvider returns a Microsoft Entra ID token for one request. It must be
// safe for concurrent calls.
type TokenProvider func(context.Context) (string, error)

// Config configures an Azure OpenAI or Azure AI Foundry endpoint.
//
// Endpoint defaults to AZURE_OPENAI_ENDPOINT. APIKey defaults to
// AZURE_OPENAI_API_KEY. APIVersion defaults to OPENAI_API_VERSION for legacy
// Azure deployment endpoints. Endpoints ending in /v1 and Azure AI Foundry
// serverless endpoints use the current OpenAI-compatible API without an API
// version query parameter.
type Config struct {
	Endpoint      string
	APIKey        string
	APIVersion    string
	TokenProvider TokenProvider
	HTTPClient    *http.Client
}

// NewModel creates an OpenAI Chat Completions model for an Azure deployment.
// Additional OpenAI options tune model behavior. Config always owns transport,
// endpoint, authentication, provider identity, and deferred-tool support.
func NewModel(deployment string, config Config, opts ...openai.Option) (*openai.Model, error) {
	provider, err := providerConfig(deployment, config, false)
	if err != nil {
		return nil, err
	}
	opts = append(
		opts,
		openai.WithProvider(provider),
		openai.WithDeferredToolSupport(false),
		openai.WithChatDocumentInput(false),
	)
	return openai.NewModel(deployment, opts...), nil
}

// NewResponsesModel creates an OpenAI Responses model for an Azure deployment.
// Additional OpenAI options tune model behavior. Config always owns transport,
// endpoint, authentication, provider identity, and deferred-tool support.
func NewResponsesModel(
	deployment string, config Config, opts ...openai.Option,
) (*openai.ResponsesModel, error) {
	provider, err := providerConfig(deployment, config, true)
	if err != nil {
		return nil, err
	}
	opts = append(opts, openai.WithProvider(provider), openai.WithDeferredToolSupport(false))
	return openai.NewResponsesModel(deployment, opts...), nil
}

func providerConfig(deployment string, config Config, responses bool) (openai.ProviderConfig, error) {
	if deployment == "" {
		return openai.ProviderConfig{}, fmt.Errorf("azure: deployment must not be empty")
	}
	endpoint := config.Endpoint
	if endpoint == "" {
		endpoint = os.Getenv("AZURE_OPENAI_ENDPOINT")
	}
	endpoint = strings.TrimRight(endpoint, "/")
	parsed, err := url.Parse(endpoint)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return openai.ProviderConfig{}, fmt.Errorf("azure: endpoint must be an absolute URL")
	}
	apiKey := config.APIKey
	if apiKey == "" && config.TokenProvider == nil {
		apiKey = os.Getenv("AZURE_OPENAI_API_KEY")
	}
	if apiKey != "" && config.TokenProvider != nil {
		return openai.ProviderConfig{}, fmt.Errorf("azure: API key and token provider cannot both be set")
	}
	if apiKey == "" && config.TokenProvider == nil {
		return openai.ProviderConfig{}, fmt.Errorf("azure: API key or token provider is required")
	}

	apiVersion := config.APIVersion
	if apiVersion == "" {
		apiVersion = os.Getenv("OPENAI_API_VERSION")
	}
	provider := openai.ProviderConfig{Name: "azure", HTTPClient: config.HTTPClient}
	if baseURL, compatible := compatibleBaseURL(parsed); compatible {
		if apiVersion != "" {
			return openai.ProviderConfig{}, fmt.Errorf("azure: API version is not supported by OpenAI-compatible v1 endpoints")
		}
		provider.BaseURL = baseURL
		provider.APIKey = apiKey
	} else {
		if apiVersion == "" {
			return openai.ProviderConfig{}, fmt.Errorf("azure: API version is required for legacy deployment endpoints")
		}
		provider.BaseURL = endpoint + "/openai"
		if !responses {
			provider.BaseURL += "/deployments/" + url.PathEscape(deployment)
		}
		provider.Query = url.Values{"api-version": {apiVersion}}
		if apiKey != "" {
			provider.Headers = http.Header{"api-key": {apiKey}}
		}
	}
	if config.TokenProvider != nil {
		provider.PrepareRequest = func(request *http.Request) error {
			token, err := config.TokenProvider(request.Context())
			if err != nil {
				return fmt.Errorf("get Azure token: %w", err)
			}
			if token == "" {
				return fmt.Errorf("get Azure token: empty token")
			}
			request.Header.Set("Authorization", "Bearer "+token)
			return nil
		}
	}
	return provider, nil
}

func compatibleBaseURL(endpoint *url.URL) (string, bool) {
	baseURL := strings.TrimRight(endpoint.String(), "/")
	if strings.HasSuffix(endpoint.Path, "/v1") {
		return baseURL, true
	}
	if strings.HasSuffix(strings.ToLower(endpoint.Hostname()), ".models.ai.azure.com") {
		return baseURL + "/v1", true
	}
	return "", false
}
