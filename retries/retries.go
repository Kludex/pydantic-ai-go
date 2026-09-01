// Package retries provides configurable HTTP transport retries for model providers.
package retries

import (
	"errors"
	"fmt"
	"net/http"
	"time"
)

// ErrBodyNotReplayable reports a retry requested for a consumed request body.
var ErrBodyNotReplayable = errors.New("retries: request body is not replayable")

// RetryFunc decides whether an HTTP or response-validation error is retried.
// It may be called concurrently by an http.Client.
type RetryFunc func(err error) bool

// ResponseValidator converts a response into an error that can be retried.
// It may be called concurrently by an http.Client.
type ResponseValidator func(response *http.Response) error

// WaitFunc returns the delay before the next attempt. Attempt is one-based and
// describes the failed attempt that just completed.
type WaitFunc func(attempt Attempt) time.Duration

// Attempt describes one failed transport attempt.
type Attempt struct {
	Number   int
	Response *http.Response
	Err      error
}

// Config controls a retrying Transport.
type Config struct {
	MaxAttempts      int
	ShouldRetry      RetryFunc
	Wait             WaitFunc
	ValidateResponse ResponseValidator
	BeforeRetry      func(Attempt)
}

// ResponseError retains the response that failed validation.
type ResponseError struct {
	Response *http.Response
	Err      error
}

// Error implements error.
func (err *ResponseError) Error() string {
	status := "<nil>"
	if err.Response != nil {
		status = err.Response.Status
	}
	return fmt.Sprintf("retries: validate response %s: %v", status, err.Err)
}

// Unwrap returns the validation error.
func (err *ResponseError) Unwrap() error { return err.Err }

// IsModelAPIError marks exhausted response validation as eligible for model fallback.
func (*ResponseError) IsModelAPIError() bool { return true }
