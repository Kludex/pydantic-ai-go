package voyageai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
)

func TestRequestErrors(t *testing.T) {
	invalidURL := NewModel("model", WithBaseURL("http://[::1"))
	if _, err := invalidURL.Embed(
		context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{},
	); err == nil || !strings.Contains(err.Error(), "build request") {
		t.Fatalf("unexpected build error: %v", err)
	}

	prepareErr := errors.New("credentials failed")
	prepareFailed := NewModel("model", WithProvider(ProviderConfig{
		Name: "voyageai", BaseURL: "https://example.com",
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
			StatusCode: http.StatusBadRequest, Header: responseHeaders,
			Body: io.NopCloser(strings.NewReader(`{"detail":"bad"}`)),
		}, nil
	})
	_, err := apiFailed.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{})
	responseHeaders.Set("X-Request-ID", "changed")
	var apiError *APIError
	if !errors.As(err, &apiError) || apiError.ProviderName != "voyageai" ||
		apiError.StatusCode != http.StatusBadRequest || apiError.Headers.Get("X-Request-ID") != "request-id" ||
		apiError.Error() != `voyageai API returned status 400: {"detail":"bad"}` {
		t.Fatalf("unexpected API error: %#v %v", apiError, err)
	}
}

func TestRequestBodyErrors(t *testing.T) {
	model := NewModel("model")
	if _, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{
		ExtraBody: map[string]any{"input": []string{"other"}},
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
		{body: `{"data":[]}`, match: "0 vectors for 1 inputs"},
		{body: `{"data":[{"embedding":"AAAAAA==","index":1}]}`, match: "invalid embedding index 1"},
		{body: `{"data":[{"embedding":"AAAAAA==","index":-1}]}`, match: "invalid embedding index -1"},
		{
			body:  `{"data":[{"embedding":"AAAAAA==","index":0},{"embedding":"AAAAAA==","index":0}]}`,
			match: "invalid embedding index 0",
		},
		{body: `{"data":[{"index":0}]}`, match: "response omitted embedding"},
		{body: `{"data":[{"embedding":"not-base64","index":0}]}`, match: "illegal base64"},
		{body: `{"data":[{"embedding":"AA==","index":0}]}`, match: "multiple of 4"},
		{body: `{"data":[{"embedding":{},"index":0}]}`, match: "cannot unmarshal"},
		{body: `{"data":[{"embedding":null,"index":0}]}`, match: "response omitted embedding"},
	}
	for _, test := range tests {
		inputs := []string{"x"}
		if strings.Contains(test.body, `},{`) {
			inputs = []string{"x", "y"}
		}
		model := responseBodyModel(test.body)
		if _, err := model.Embed(
			context.Background(), inputs, embeddings.InputTypeQuery, embeddings.Settings{},
		); err == nil || !strings.Contains(err.Error(), test.match) {
			t.Fatalf("unexpected error for %s: %v", test.body, err)
		}
	}
}

func TestOptionsAndLimits(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "environment-key")
	model := NewModel("voyage-4", WithBaseURL("https://example.com/v1/"),
		WithHTTPClient(keyCheckingClient(t, "environment-key")))
	if model.ProviderURL() != "https://example.com/v1" {
		t.Fatalf("unexpected URL: %q", model.ProviderURL())
	}
	if _, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{}); err != nil {
		t.Fatal(err)
	}

	explicit := NewModel("voyage-4", WithAPIKey("explicit"), WithHTTPClient(keyCheckingClient(t, "explicit")))
	if _, err := explicit.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{}); err != nil {
		t.Fatal(err)
	}
	if panicValue := capturePanic(func() { WithProvider(ProviderConfig{BaseURL: "https://example.com"}) }); panicValue != "voyageai embeddings: provider name must not be empty" {
		t.Fatalf("unexpected provider-name panic: %v", panicValue)
	}
	if panicValue := capturePanic(func() { WithProvider(ProviderConfig{Name: "voyageai"}) }); panicValue != "voyageai embeddings: provider base URL must not be empty" {
		t.Fatalf("unexpected base-URL panic: %v", panicValue)
	}

	limits := map[string]int{
		"voyage-4-large": 32000, "voyage-4": 32000, "voyage-4-lite": 32000,
		"voyage-3-large": 32000, "voyage-3.5": 32000, "voyage-3.5-lite": 32000,
		"voyage-code-3": 32000, "voyage-finance-2": 32000,
		"voyage-law-2": 16000, "voyage-code-2": 16000,
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
	return NewModel("model", WithBaseURL("https://example.com/v1"), WithHTTPClient(&http.Client{Transport: roundTrip}))
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
		return jsonResponse(fmtResponse([]responseVector{{index: 0, values: []float32{1}}}, 1)), nil
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
