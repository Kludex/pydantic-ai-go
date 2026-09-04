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
	// Number is the one-based failed attempt number.
	Number int
	// Response is present when response validation requested the retry.
	Response *http.Response
	// Err is the transport or response-validation failure.
	Err error
}

// Config controls a retrying Transport.
type Config struct {
	// MaxAttempts includes the initial request.
	MaxAttempts int
	// ShouldRetry classifies failures after the context and attempt limit checks.
	ShouldRetry RetryFunc
	// Wait calculates the context-aware delay after each retryable failure.
	Wait WaitFunc
	// ValidateResponse turns an HTTP response into a retryable failure.
	ValidateResponse ResponseValidator
	// BeforeRetry observes an accepted retry before waiting.
	BeforeRetry func(Attempt)
}

// ResponseError retains the response that failed validation.
type ResponseError struct {
	// Response is the response rejected by ValidateResponse.
	Response *http.Response
	// Err is the validator's failure.
	Err error
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
