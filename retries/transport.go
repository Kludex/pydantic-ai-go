package retries

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Transport retries requests around another http.RoundTripper.
// Request bodies must provide Request.GetBody when a retry is needed.
type Transport struct {
	wrapped http.RoundTripper
	config  Config
}

// NewTransport creates a retrying transport. wrapped defaults to
// http.DefaultTransport. MaxAttempts includes the initial attempt.
func NewTransport(config Config, wrapped http.RoundTripper) (*Transport, error) {
	if config.MaxAttempts < 1 {
		return nil, fmt.Errorf("retries: max attempts must be at least 1")
	}
	if wrapped == nil {
		wrapped = http.DefaultTransport
	}
	if config.ShouldRetry == nil {
		config.ShouldRetry = func(error) bool { return true }
	}
	if config.Wait == nil {
		config.Wait = func(Attempt) time.Duration { return 0 }
	}
	return &Transport{wrapped: wrapped, config: config}, nil
}

// RoundTrip implements http.RoundTripper.
func (transport *Transport) RoundTrip(request *http.Request) (*http.Response, error) {
	if transport == nil {
		return nil, fmt.Errorf("retries: transport must not be nil")
	}
	if request == nil {
		return nil, fmt.Errorf("retries: request must not be nil")
	}
	wrapped := transport.wrapped
	if wrapped == nil {
		wrapped = http.DefaultTransport
	}
	config := transport.config
	if config.MaxAttempts < 1 {
		config.MaxAttempts = 1
	}
	for number := 1; ; number++ {
		attemptRequest, err := retryRequest(request, number)
		if err != nil {
			return nil, err
		}
		response, err := wrapped.RoundTrip(attemptRequest)
		if err == nil && config.ValidateResponse != nil {
			if validationErr := config.ValidateResponse(response); validationErr != nil {
				err = &ResponseError{Response: response, Err: validationErr}
				closeResponse(response)
				response = nil
			}
		}
		if err == nil {
			return response, nil
		}
		if response != nil {
			closeResponse(response)
		}
		if contextErr := request.Context().Err(); contextErr != nil {
			return nil, contextErr
		}
		if number == config.MaxAttempts || !config.ShouldRetry(err) {
			return nil, err
		}
		attempt := Attempt{Number: number, Response: responseFromError(err), Err: err}
		if config.BeforeRetry != nil {
			config.BeforeRetry(attempt)
		}
		if err := wait(request.Context(), config.Wait(attempt)); err != nil {
			return nil, err
		}
	}
}

// CloseIdleConnections closes idle connections held by the wrapped transport when supported.
func (transport *Transport) CloseIdleConnections() {
	if transport == nil {
		return
	}
	wrapped := transport.wrapped
	if wrapped == nil {
		wrapped = http.DefaultTransport
	}
	if closer, ok := wrapped.(interface{ CloseIdleConnections() }); ok {
		closer.CloseIdleConnections()
	}
}

func retryRequest(request *http.Request, attempt int) (*http.Request, error) {
	cloned := request.Clone(request.Context())
	if attempt == 1 || request.Body == nil {
		cloned.Body = request.Body
		return cloned, nil
	}
	if request.GetBody == nil {
		return nil, ErrBodyNotReplayable
	}
	body, err := request.GetBody()
	if err != nil {
		return nil, fmt.Errorf("retries: replay request body: %w", err)
	}
	cloned.Body = body
	return cloned, nil
}

func responseFromError(err error) *http.Response {
	var responseErr *ResponseError
	if errors.As(err, &responseErr) {
		return responseErr.Response
	}
	return nil
}

func closeResponse(response *http.Response) {
	if response == nil || response.Body == nil {
		return
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 32<<10))
	_ = response.Body.Close()
}

func wait(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
