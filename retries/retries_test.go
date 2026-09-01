package retries_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
	"github.com/Kludex/pydantic-ai-go/retries"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

type trackedBody struct {
	io.Reader
	closed bool
}

type closableTransport struct {
	closed bool
}

func (*closableTransport) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, errors.New("unused")
}

func (transport *closableTransport) CloseIdleConnections() { transport.closed = true }

func (body *trackedBody) Close() error {
	body.closed = true
	return nil
}

func response(status int, body io.ReadCloser) *http.Response {
	return &http.Response{
		StatusCode: status,
		Status:     fmt.Sprintf("%d %s", status, http.StatusText(status)),
		Header:     make(http.Header),
		Body:       body,
	}
}

func TestTransportRetriesValidatedResponsesWithReplayableBodies(t *testing.T) {
	var mu sync.Mutex
	var bodies []string
	var closed []*trackedBody
	statuses := []int{http.StatusTooManyRequests, http.StatusInternalServerError, http.StatusOK}
	wrapped := roundTripFunc(func(request *http.Request) (*http.Response, error) {
		data, err := io.ReadAll(request.Body)
		if err != nil {
			t.Fatal(err)
		}
		body := &trackedBody{Reader: strings.NewReader("response")}
		mu.Lock()
		bodies = append(bodies, string(data))
		closed = append(closed, body)
		status := statuses[len(bodies)-1]
		mu.Unlock()
		result := response(status, body)
		if status == http.StatusTooManyRequests {
			result.Header.Set("Retry-After", "0")
		}
		return result, nil
	})
	var attempts []retries.Attempt
	transport, err := retries.NewTransport(retries.Config{
		MaxAttempts: 3, ValidateResponse: retries.ValidateStatus,
		ShouldRetry: retries.RetryTransient,
		Wait:        retries.WaitRetryAfter(func(retries.Attempt) time.Duration { return 0 }, time.Second),
		BeforeRetry: func(attempt retries.Attempt) { attempts = append(attempts, attempt) },
	}, wrapped)
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Transport: transport}
	request, err := http.NewRequestWithContext(
		t.Context(), http.MethodPost, "https://example.test", bytes.NewBufferString("input"),
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := client.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = result.Body.Close() }()
	if result.StatusCode != http.StatusOK || strings.Join(bodies, ",") != "input,input,input" {
		t.Fatalf("unexpected retry result: status=%d bodies=%v", result.StatusCode, bodies)
	}
	if len(attempts) != 2 || attempts[0].Number != 1 || attempts[0].Response.StatusCode != http.StatusTooManyRequests ||
		attempts[1].Number != 2 || attempts[1].Response.StatusCode != http.StatusInternalServerError {
		t.Fatalf("unexpected retry attempts: %#v", attempts)
	}
	if !closed[0].closed || !closed[1].closed || closed[2].closed {
		t.Fatalf("unexpected response closure: %#v", closed)
	}
}

func TestTransportIntegratesWithProviderClients(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		attempts++
		if request.Method != http.MethodPost || request.URL.Path != "/chat/completions" {
			t.Fatalf("unexpected request: %s %s", request.Method, request.URL.Path)
		}
		if attempts < 3 {
			response.WriteHeader(http.StatusServiceUnavailable)
			_, _ = response.Write([]byte(`{"error":"temporary"}`))
			return
		}
		response.Header().Set("Content-Type", "application/json")
		_, _ = response.Write([]byte(`{
			"id":"response","model":"gpt-5","created":1,
			"choices":[{"message":{"content":"done"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`))
	}))
	defer server.Close()
	transport, err := retries.NewTransport(retries.Config{
		MaxAttempts: 3, ValidateResponse: retries.ValidateStatus,
		ShouldRetry: retries.RetryTransient,
	}, server.Client().Transport)
	if err != nil {
		t.Fatal(err)
	}
	model := openai.NewModel(
		"gpt-5", openai.WithBaseURL(server.URL),
		openai.WithHTTPClient(&http.Client{Transport: transport}),
	)
	result, err := ai.RequestModel(t.Context(), model, nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text() != "done" || attempts != 3 {
		t.Fatalf("unexpected provider retry: response=%#v attempts=%d", result, attempts)
	}
}

func TestTransportRetryDecisionsAndDefaults(t *testing.T) {
	sentinel := errors.New("temporary")
	attempts := 0
	transport, err := retries.NewTransport(
		retries.Config{MaxAttempts: 2},
		roundTripFunc(func(*http.Request) (*http.Response, error) {
			attempts++
			if attempts == 1 {
				return nil, sentinel
			}
			return response(http.StatusOK, io.NopCloser(strings.NewReader("ok"))), nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	request, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, "https://example.test", nil)
	result, err := transport.RoundTrip(request)
	if err != nil || result.StatusCode != http.StatusOK || attempts != 2 {
		t.Fatalf("default retry failed: response=%#v err=%v attempts=%d", result, err, attempts)
	}
	_ = result.Body.Close()

	attempts = 0
	transport, err = retries.NewTransport(retries.Config{
		MaxAttempts: 3,
		ShouldRetry: func(err error) bool {
			return !errors.Is(err, sentinel)
		},
	}, roundTripFunc(func(*http.Request) (*http.Response, error) {
		attempts++
		return nil, sentinel
	}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := transport.RoundTrip(request); !errors.Is(err, sentinel) || attempts != 1 {
		t.Fatalf("retry predicate was ignored: err=%v attempts=%d", err, attempts)
	}
}

func TestTransportReturnsValidationAndReplayErrors(t *testing.T) {
	sentinel := errors.New("bad response")
	body := &trackedBody{Reader: strings.NewReader("bad")}
	transport, err := retries.NewTransport(retries.Config{
		MaxAttempts: 1,
		ValidateResponse: func(*http.Response) error {
			return sentinel
		},
	}, roundTripFunc(func(*http.Request) (*http.Response, error) {
		return response(http.StatusTeapot, body), nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	request, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, "https://example.test", nil)
	_, err = transport.RoundTrip(request)
	var responseErr *retries.ResponseError
	if !errors.As(err, &responseErr) || !errors.Is(err, sentinel) ||
		responseErr.Response.StatusCode != http.StatusTeapot || !body.closed {
		t.Fatalf("unexpected validation error: %T %v", err, err)
	}

	attempts := 0
	transport, err = retries.NewTransport(
		retries.Config{MaxAttempts: 2},
		roundTripFunc(func(*http.Request) (*http.Response, error) {
			attempts++
			return nil, sentinel
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	request, _ = http.NewRequestWithContext(
		t.Context(), http.MethodPost, "https://example.test", io.NopCloser(strings.NewReader("body")),
	)
	if _, err := transport.RoundTrip(request); !errors.Is(err, retries.ErrBodyNotReplayable) || attempts != 1 {
		t.Fatalf("unexpected replay error: err=%v attempts=%d", err, attempts)
	}

	replayErr := errors.New("cannot replay")
	request, _ = http.NewRequestWithContext(
		t.Context(), http.MethodPost, "https://example.test", strings.NewReader("body"),
	)
	request.GetBody = func() (io.ReadCloser, error) { return nil, replayErr }
	if _, err := transport.RoundTrip(request); !errors.Is(err, replayErr) {
		t.Fatalf("unexpected GetBody error: %v", err)
	}
}

func TestTransportClosesResponsesReturnedWithErrors(t *testing.T) {
	sentinel := errors.New("transport failed")
	transport, err := retries.NewTransport(retries.Config{MaxAttempts: 1}, roundTripFunc(
		func(*http.Request) (*http.Response, error) {
			return response(http.StatusBadGateway, nil), sentinel
		},
	))
	if err != nil {
		t.Fatal(err)
	}
	request, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, "https://example.test", nil)
	if _, err := transport.RoundTrip(request); !errors.Is(err, sentinel) {
		t.Fatalf("unexpected transport error: %v", err)
	}
}

func TestTransportStopsAtAttemptLimitAndCancellation(t *testing.T) {
	sentinel := errors.New("temporary")
	attempts := 0
	transport, err := retries.NewTransport(
		retries.Config{MaxAttempts: 2},
		roundTripFunc(func(*http.Request) (*http.Response, error) {
			attempts++
			return nil, sentinel
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	request, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, "https://example.test", nil)
	if _, err := transport.RoundTrip(request); !errors.Is(err, sentinel) || attempts != 2 {
		t.Fatalf("attempt limit failed: err=%v attempts=%d", err, attempts)
	}

	ctx, cancel := context.WithCancel(t.Context())
	beforeRetry := 0
	transport, err = retries.NewTransport(retries.Config{
		MaxAttempts: 3,
		BeforeRetry: func(retries.Attempt) { beforeRetry++ },
	}, roundTripFunc(func(*http.Request) (*http.Response, error) {
		cancel()
		return nil, sentinel
	}))
	if err != nil {
		t.Fatal(err)
	}
	request, _ = http.NewRequestWithContext(ctx, http.MethodGet, "https://example.test", nil)
	if _, err := transport.RoundTrip(request); !errors.Is(err, context.Canceled) || beforeRetry != 0 {
		t.Fatalf("cancellation was retried: err=%v before=%d", err, beforeRetry)
	}

	completedWaitAttempts := 0
	transport, err = retries.NewTransport(retries.Config{
		MaxAttempts: 2,
		Wait:        func(retries.Attempt) time.Duration { return time.Millisecond },
	}, roundTripFunc(func(*http.Request) (*http.Response, error) {
		completedWaitAttempts++
		if completedWaitAttempts == 1 {
			return nil, sentinel
		}
		return response(http.StatusOK, io.NopCloser(strings.NewReader("ok"))), nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	request, _ = http.NewRequestWithContext(t.Context(), http.MethodGet, "https://example.test", nil)
	result, err := transport.RoundTrip(request)
	if err != nil || completedWaitAttempts != 2 {
		t.Fatalf("positive retry wait failed: response=%#v err=%v", result, err)
	}
	_ = result.Body.Close()

	ctx, cancel = context.WithCancel(t.Context())
	transport, err = retries.NewTransport(retries.Config{
		MaxAttempts: 2,
		Wait:        func(retries.Attempt) time.Duration { return time.Hour },
		BeforeRetry: func(retries.Attempt) { cancel() },
	}, roundTripFunc(func(*http.Request) (*http.Response, error) { return nil, sentinel }))
	if err != nil {
		t.Fatal(err)
	}
	request, _ = http.NewRequestWithContext(ctx, http.MethodGet, "https://example.test", nil)
	if _, err := transport.RoundTrip(request); !errors.Is(err, context.Canceled) {
		t.Fatalf("retry wait ignored cancellation: %v", err)
	}
}

func TestTransportValidation(t *testing.T) {
	var nilTransport *retries.Transport
	request, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, "https://example.test", nil)
	if _, err := nilTransport.RoundTrip(request); err == nil ||
		!strings.Contains(err.Error(), "transport must not be nil") {
		t.Fatalf("unexpected nil transport error: %v", err)
	}
	if _, err := retries.NewTransport(retries.Config{}, nil); err == nil || !strings.Contains(err.Error(), "at least 1") {
		t.Fatalf("unexpected config error: %v", err)
	}
	transport, err := retries.NewTransport(retries.Config{MaxAttempts: 1}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := transport.RoundTrip(nil); err == nil || !strings.Contains(err.Error(), "must not be nil") {
		t.Fatalf("unexpected nil request error: %v", err)
	}
	request, _ = http.NewRequestWithContext(t.Context(), http.MethodGet, "unsupported://example", nil)
	if _, err := transport.RoundTrip(request); err == nil {
		t.Fatal("default transport accepted unsupported scheme")
	}
	var zero retries.Transport
	if _, err := zero.RoundTrip(request); err == nil {
		t.Fatal("zero transport accepted unsupported scheme")
	}
	zero.CloseIdleConnections()
	var nilCloser *retries.Transport
	nilCloser.CloseIdleConnections()
	wrapped := &closableTransport{}
	transport, err = retries.NewTransport(retries.Config{MaxAttempts: 1}, wrapped)
	if err != nil {
		t.Fatal(err)
	}
	transport.CloseIdleConnections()
	if !wrapped.closed {
		t.Fatal("idle connections were not closed")
	}
	transport, err = retries.NewTransport(retries.Config{MaxAttempts: 1}, roundTripFunc(
		func(*http.Request) (*http.Response, error) { return nil, errors.New("unused") },
	))
	if err != nil {
		t.Fatal(err)
	}
	transport.CloseIdleConnections()
}

func TestStatusHelpers(t *testing.T) {
	for status, valid := range map[int]bool{199: false, 200: true, 299: true, 300: false, 500: false} {
		err := retries.ValidateStatus(response(status, nil))
		if (err == nil) != valid {
			t.Fatalf("status %d validity was %v: %v", status, valid, err)
		}
	}
	if err := retries.ValidateStatus(nil); err == nil {
		t.Fatal("nil response was accepted")
	}

	network := errors.New("network")
	if !retries.RetryTransient(network) || retries.RetryTransient(context.Canceled) ||
		retries.RetryTransient(context.DeadlineExceeded) {
		t.Fatal("network retry classification failed")
	}
	for status, expected := range map[int]bool{400: false, 429: true, 500: true, 599: true, 600: false} {
		err := &retries.ResponseError{Response: response(status, nil), Err: errors.New("status")}
		if retries.RetryTransient(err) != expected {
			t.Fatalf("status %d retry decision was wrong", status)
		}
	}
	if retries.RetryTransient(&retries.ResponseError{Err: network}) {
		t.Fatal("response error without a response was retried")
	}
	responseErr := &retries.ResponseError{Err: network}
	if !strings.Contains(responseErr.Error(), "<nil>") || !errors.Is(responseErr, network) ||
		!responseErr.IsModelAPIError() {
		t.Fatalf("response error contract failed: %v", responseErr)
	}
	responseErr = &retries.ResponseError{Response: response(http.StatusBadGateway, nil), Err: network}
	if !strings.Contains(responseErr.Error(), "502 Bad Gateway") {
		t.Fatalf("response status was omitted: %v", responseErr)
	}
}

func TestWaitStrategies(t *testing.T) {
	backoff := retries.ExponentialBackoff(time.Second, 4*time.Second)
	expectedBackoffs := map[int]time.Duration{
		1: time.Second, 2: 2 * time.Second, 3: 4 * time.Second, 100: 4 * time.Second,
	}
	for attempt, expected := range expectedBackoffs {
		if delay := backoff(retries.Attempt{Number: attempt}); delay != expected {
			t.Fatalf("attempt %d delay was %s, expected %s", attempt, delay, expected)
		}
	}
	oddBackoff := retries.ExponentialBackoff(3*time.Second, 5*time.Second)
	if delay := oddBackoff(retries.Attempt{Number: 2}); delay != 5*time.Second {
		t.Fatalf("overflow-safe delay was %s", delay)
	}
	assertPanic(t, "exponential backoff", func() { retries.ExponentialBackoff(0, time.Second) })
	assertPanic(t, "exponential backoff", func() { retries.ExponentialBackoff(2*time.Second, time.Second) })
	assertPanic(t, "Retry-After maximum", func() { retries.WaitRetryAfter(nil, 0) })

	fallback := func(retries.Attempt) time.Duration { return time.Second }
	wait := retries.WaitRetryAfter(fallback, 2*time.Second)
	for value, expected := range map[string]time.Duration{
		"0": 0, "1": time.Second, "9223372036": 2 * time.Second, "999999999999999999999999999": 2 * time.Second,
		"invalid": time.Second, "-1": time.Second,
	} {
		result := response(http.StatusTooManyRequests, nil)
		result.Header.Set("Retry-After", value)
		if delay := wait(retries.Attempt{Number: 1, Response: result}); delay != expected {
			t.Fatalf("Retry-After %q delay was %s, expected %s", value, delay, expected)
		}
	}
	future := response(http.StatusTooManyRequests, nil)
	future.Header.Set("Retry-After", time.Now().Add(time.Hour).UTC().Format(http.TimeFormat))
	if delay := wait(retries.Attempt{Response: future}); delay != 2*time.Second {
		t.Fatalf("future Retry-After was not capped: %s", delay)
	}
	past := response(http.StatusTooManyRequests, nil)
	past.Header.Set("Retry-After", time.Now().Add(-time.Hour).UTC().Format(http.TimeFormat))
	if delay := wait(retries.Attempt{Response: past}); delay != time.Second {
		t.Fatalf("past Retry-After did not use fallback: %s", delay)
	}
	negative := retries.WaitRetryAfter(func(retries.Attempt) time.Duration { return -time.Second }, time.Second)
	if delay := negative(retries.Attempt{}); delay != 0 {
		t.Fatalf("negative fallback was not clamped: %s", delay)
	}
	defaultWait := retries.WaitRetryAfter(nil, 2*time.Minute)
	if delay := defaultWait(retries.Attempt{Number: 2}); delay != 2*time.Second {
		t.Fatalf("default fallback delay was %s", delay)
	}
}

func assertPanic(t *testing.T, contains string, function func()) {
	t.Helper()
	defer func() {
		value := recover()
		if value == nil || !strings.Contains(fmt.Sprint(value), contains) {
			t.Fatalf("unexpected panic: %v", value)
		}
	}()
	function()
}
