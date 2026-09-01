package cohere

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	modelcohere "github.com/Kludex/pydantic-ai-go/models/cohere"
)

func TestRequestErrors(t *testing.T) {
	invalidURL := NewModel("model", WithBaseURL("http://[::1"))
	if _, err := invalidURL.Embed(
		context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{},
	); err == nil || !strings.Contains(err.Error(), "build request") {
		t.Fatalf("unexpected build error: %v", err)
	}

	prepareErr := errors.New("credentials failed")
	prepareFailed := NewModel("model", WithProvider(modelcohere.ProviderConfig{
		Name: "cohere", BaseURL: "https://example.com",
		PrepareRequest: func(*http.Request) error { return prepareErr },
	}))
	if _, err := prepareFailed.Embed(
		context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{},
	); !errors.Is(err, prepareErr) {
		t.Fatalf("unexpected preparation error: %v", err)
	}

	transportErr := errors.New("offline")
	transportFailed := failureModel(func(*http.Request) (*http.Response, error) { return nil, transportErr })
	if _, err := transportFailed.Embed(
		context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{},
	); !errors.Is(err, transportErr) {
		t.Fatalf("unexpected transport error: %v", err)
	}
	if _, err := transportFailed.CountTokens(context.Background(), "x"); !errors.Is(err, transportErr) {
		t.Fatalf("unexpected token transport error: %v", err)
	}

	readErr := errors.New("read failed")
	readFailed := failureModel(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: errorBody{err: readErr}}, nil
	})
	if _, err := readFailed.Embed(
		context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{},
	); !errors.Is(err, readErr) {
		t.Fatalf("unexpected read error: %v", err)
	}

	responseHeaders := http.Header{}
	responseHeaders.Set("X-Request-ID", "request-id")
	apiFailed := failureModel(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusTooManyRequests, Header: responseHeaders,
			Body: io.NopCloser(strings.NewReader(`{"message":"slow down"}`)),
		}, nil
	})
	_, err := apiFailed.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{})
	responseHeaders.Set("X-Request-ID", "changed")
	var apiError *modelcohere.APIError
	if !errors.As(err, &apiError) || apiError.StatusCode != http.StatusTooManyRequests ||
		apiError.ProviderName != "cohere" || apiError.Headers.Get("X-Request-ID") != "request-id" {
		t.Fatalf("unexpected API error: %#v %v", apiError, err)
	}
}

func TestRequestBodyErrors(t *testing.T) {
	model := NewModel("model")
	if _, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{
		ExtraBody: map[string]any{"model": "other"},
	}); err == nil || !strings.Contains(err.Error(), "conflicts") {
		t.Fatalf("unexpected conflict error: %v", err)
	}
	if _, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{
		ExtraBody: map[string]any{"unsupported": make(chan int)},
	}); err == nil || !strings.Contains(err.Error(), "encode request") {
		t.Fatalf("unexpected encoding error: %v", err)
	}
	zero := 0
	if _, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{
		Dimensions: &zero,
	}); err == nil || !strings.Contains(err.Error(), "dimensions") {
		t.Fatalf("unexpected validation error: %v", err)
	}
}

func TestResponseErrors(t *testing.T) {
	tests := []struct {
		body  string
		match string
	}{
		{body: `{`, match: "decode response"},
		{body: `{}`, match: "omitted float embeddings"},
		{body: `{"embeddings":{"float":[]}}`, match: "0 vectors for 1 inputs"},
		{body: `{"embeddings":{"float":[null]}}`, match: "omitted vector at index 0"},
	}
	for _, test := range tests {
		model := responseBodyModel(test.body)
		if _, err := model.Embed(
			context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{},
		); err == nil || !strings.Contains(err.Error(), test.match) {
			t.Fatalf("unexpected error for %s: %v", test.body, err)
		}
	}
}

func TestCountTokenResponseErrors(t *testing.T) {
	for _, test := range []struct {
		body  string
		match string
	}{
		{body: `{`, match: "decode token count response"},
		{body: `{}`, match: "omitted tokens"},
	} {
		model := responseBodyModel(test.body)
		if _, err := model.CountTokens(context.Background(), "x"); err == nil ||
			!strings.Contains(err.Error(), test.match) {
			t.Fatalf("unexpected count error for %s: %v", test.body, err)
		}
	}
}

func TestOptionsEnvironmentAndLimits(t *testing.T) {
	t.Setenv("CO_API_KEY", "environment-key")
	t.Setenv("CO_BASE_URL", "https://environment.example/")
	model := NewModel("embed-v4.0", WithHTTPClient(keyCheckingClient(t, "environment-key")))
	if model.ProviderURL() != "https://environment.example" {
		t.Fatalf("unexpected environment URL: %q", model.ProviderURL())
	}
	if _, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{}); err != nil {
		t.Fatal(err)
	}

	explicit := NewModel("embed-v4.0", WithAPIKey("explicit"), WithBaseURL("https://explicit.example/"),
		WithHTTPClient(keyCheckingClient(t, "explicit")))
	if explicit.ProviderURL() != "https://explicit.example" {
		t.Fatalf("unexpected explicit URL: %q", explicit.ProviderURL())
	}
	if _, err := explicit.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{}); err != nil {
		t.Fatal(err)
	}

	if panicValue := capturePanic(func() { WithProvider(modelcohere.ProviderConfig{BaseURL: "https://example.com"}) }); panicValue != "cohere embeddings: provider name must not be empty" {
		t.Fatalf("unexpected provider-name panic: %v", panicValue)
	}
	if panicValue := capturePanic(func() { WithProvider(modelcohere.ProviderConfig{Name: "cohere"}) }); panicValue != "cohere embeddings: provider base URL must not be empty" {
		t.Fatalf("unexpected base-URL panic: %v", panicValue)
	}

	t.Setenv("CO_BASE_URL", "")
	if defaultModel := NewModel("unknown"); defaultModel.ProviderURL() != defaultBaseURL {
		t.Fatalf("unexpected default URL: %q", defaultModel.ProviderURL())
	}
	limits := map[string]int{
		"embed-v4.0":                    128000,
		"embed-english-v3.0":            512,
		"embed-english-light-v3.0":      512,
		"embed-multilingual-v3.0":       512,
		"embed-multilingual-light-v3.0": 512,
	}
	for name, expected := range limits {
		maximum, known, err := NewModel(name).MaxInputTokens(context.Background())
		if err != nil || !known || maximum != expected {
			t.Fatalf("unexpected limit for %s: %d %v %v", name, maximum, known, err)
		}
	}
	if maximum, known, err := NewModel("unknown").MaxInputTokens(context.Background()); err != nil || known || maximum != 0 {
		t.Fatalf("unexpected unknown limit: %d %v %v", maximum, known, err)
	}
}

func failureModel(roundTrip roundTripperFunc) *Model {
	return NewModel("model", WithBaseURL("https://example.com"), WithHTTPClient(&http.Client{Transport: roundTrip}))
}

func responseBodyModel(body string) *Model {
	return failureModel(func(*http.Request) (*http.Response, error) { return jsonResponse(body), nil })
}

func keyCheckingClient(t *testing.T, expected string) *http.Client {
	t.Helper()
	return &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
		if authorization := request.Header.Get("Authorization"); authorization != "Bearer "+expected {
			t.Errorf("unexpected authorization: %q", authorization)
		}
		return jsonResponse(`{"embeddings":{"float":[[1]]}}`), nil
	})}
}

type errorBody struct{ err error }

func (body errorBody) Read([]byte) (int, error) { return 0, body.err }
func (errorBody) Close() error                  { return nil }

func capturePanic(function func()) (value any) {
	defer func() { value = recover() }()
	function()
	return nil
}
