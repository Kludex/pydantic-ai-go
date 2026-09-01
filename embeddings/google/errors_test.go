package google

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	modelgoogle "github.com/Kludex/pydantic-ai-go/models/google"
)

func TestEmbeddingValidationErrors(t *testing.T) {
	model := NewModel("model")
	zero := 0
	if _, err := model.Embed(
		context.Background(), []string{"x"}, embeddings.InputTypeQuery,
		embeddings.Settings{Dimensions: &zero},
	); err == nil || !strings.Contains(err.Error(), "dimensions must be greater than zero") {
		t.Fatalf("unexpected dimensions error: %v", err)
	}
	for _, value := range []any{42, "invalid"} {
		if _, err := model.Embed(
			context.Background(), []string{"x"}, embeddings.InputTypeQuery,
			embeddings.Settings{ExtraBody: map[string]any{taskKey: value}},
		); err == nil {
			t.Fatalf("invalid task setting %#v unexpectedly succeeded", value)
		}
	}
}

func TestEmbeddingRequestBodyErrors(t *testing.T) {
	model := NewModel("model")
	if _, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{
		ExtraBody: map[string]any{"requests": []any{}},
	}); err == nil || !strings.Contains(err.Error(), "conflicts") {
		t.Fatalf("unexpected conflict error: %v", err)
	}
	if _, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{
		ExtraBody: map[string]any{"unsupported": make(chan int)},
	}); err == nil || !strings.Contains(err.Error(), "encode request") {
		t.Fatalf("unexpected encoding error: %v", err)
	}
}

func TestEmbeddingRequestErrors(t *testing.T) {
	invalidURL := NewModel("model", WithBaseURL("http://[::1"))
	if _, err := invalidURL.Embed(
		context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{},
	); err == nil || !strings.Contains(err.Error(), "build request") {
		t.Fatalf("unexpected build error: %v", err)
	}

	prepareErr := errors.New("credentials failed")
	prepareFailed := NewModel("model", WithProvider(modelgoogle.ProviderConfig{
		BaseURL: "https://example.com", PrepareRequest: func(*http.Request) error { return prepareErr },
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
		t.Fatalf("unexpected count transport error: %v", err)
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

	apiFailed := failureModel(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusBadRequest, Header: http.Header{},
			Body: io.NopCloser(strings.NewReader(`{"error":"bad"}`)),
		}, nil
	})
	if _, err := apiFailed.Embed(
		context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{},
	); err == nil || !strings.Contains(err.Error(), "API returned status 400") {
		t.Fatalf("unexpected API error: %v", err)
	}
}

func TestEmbeddingResponseErrors(t *testing.T) {
	tests := []struct {
		body  string
		match string
	}{
		{body: `{`, match: "decode response"},
		{body: `{"embeddings":[]}`, match: "0 vectors for 1 inputs"},
		{body: `{"embeddings":[{"values":null}]}`, match: "omits vector"},
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
		{body: `{}`, match: "omitted totalTokens"},
	} {
		model := responseBodyModel(test.body)
		if _, err := model.CountTokens(context.Background(), "x"); err == nil ||
			!strings.Contains(err.Error(), test.match) {
			t.Fatalf("unexpected count error for %s: %v", test.body, err)
		}
	}
}

func failureModel(roundTrip roundTripperFunc) *Model {
	return NewModel("model", WithBaseURL("https://example.com"), WithHTTPClient(&http.Client{Transport: roundTrip}))
}

func responseBodyModel(body string) *Model {
	return failureModel(func(*http.Request) (*http.Response, error) { return jsonResponse(body), nil })
}

type errorBody struct{ err error }

func (body errorBody) Read([]byte) (int, error) { return 0, body.err }
func (errorBody) Close() error                  { return nil }
