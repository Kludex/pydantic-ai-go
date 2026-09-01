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
	Name           string
	BaseURL        string
	APIKey         string
	HTTPClient     *http.Client
	Headers        http.Header
	PrepareRequest RequestPreparationFunc
}

// Clone returns a detached provider configuration.
func (config ProviderConfig) Clone() ProviderConfig {
	config.Headers = config.Headers.Clone()
	return config
}

// APIError reports a non-successful Cohere HTTP response.
type APIError struct {
	StatusCode   int
	Body         string
	Headers      http.Header
	ProviderName string
}

// Error implements error.
func (err *APIError) Error() string {
	return fmt.Sprintf("%s API returned status %d: %s", err.ProviderName, err.StatusCode, err.Body)
}
