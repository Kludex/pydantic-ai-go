// Package cohere provides shared Cohere provider configuration.
package cohere

import (
	"fmt"
	"net/http"
)

// RequestPreparationFunc prepares an HTTP request after provider defaults and before per-request headers.
// The function must be safe for concurrent calls and must not retain the request.
type RequestPreparationFunc func(*http.Request) error

// ProviderConfig configures Cohere API access. Headers are copied.
type ProviderConfig struct {
	// Name is persisted in responses, usage, and telemetry.
	Name string
	// BaseURL is the Cohere-compatible API endpoint.
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

// Clone returns a detached provider configuration.
func (config ProviderConfig) Clone() ProviderConfig {
	config.Headers = config.Headers.Clone()
	return config
}

// APIError reports a non-successful Cohere HTTP response.
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
