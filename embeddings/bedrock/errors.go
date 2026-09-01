package bedrock

import (
	"fmt"
	"net/http"
)

// APIError describes a failed Bedrock Runtime HTTP response.
type APIError struct {
	StatusCode int
	Headers    http.Header
	Err        error
}

// Error returns the Bedrock HTTP failure.
func (err *APIError) Error() string {
	return fmt.Sprintf("bedrock embeddings: HTTP %d: %v", err.StatusCode, err.Err)
}

// Unwrap returns the underlying AWS SDK error.
func (err *APIError) Unwrap() error { return err.Err }
