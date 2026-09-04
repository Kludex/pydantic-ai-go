package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
	openaimodels "github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestEmbed(t *testing.T) {
	var received map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost || request.URL.Path != "/v1/embeddings" || request.URL.Query().Get("version") != "one" {
			t.Errorf("unexpected request target: %s %s", request.Method, request.URL.String())
		}
		if request.Header.Get("Authorization") != "Bearer secret" || request.Header.Get("X-Provider") != "provider" ||
			request.Header.Get("X-Dynamic") != "dynamic" || request.Header.Get("X-Order") != "extra" {
			t.Errorf("unexpected headers: %#v", request.Header)
		}
		if err := json.NewDecoder(request.Body).Decode(&received); err != nil {
			t.Error(err)
		}
		response.Header().Set("Content-Type", "application/json")
		response.WriteHeader(http.StatusCreated)
		_, _ = io.WriteString(response, `{
			"data":[{"index":1,"embedding":[3,4]},{"index":0,"embedding":[1,2]}],
			"model":"text-embedding-3-small-2025","usage":{"prompt_tokens":5,"total_tokens":5,"cached_tokens":2}
		}`)
	}))
	defer server.Close()
	headers := http.Header{"X-Provider": {"provider"}, "X-Order": {"provider"}}
	query := url.Values{"version": {"one"}}
	provider := openaimodels.ProviderConfig{
		Name: "compatible", BaseURL: server.URL + "/v1", APIKey: "secret", HTTPClient: server.Client(),
		Headers: headers, Query: query,
		PrepareRequest: func(request *http.Request) error {
			request.Header.Set("X-Dynamic", "dynamic")
			request.Header.Set("X-Order", "dynamic")
			return nil
		},
	}
	option := WithProvider(provider)
	headers.Set("X-Provider", "mutated")
	query.Set("version", "mutated")
	provider.Headers.Set("X-Order", "mutated")
	dimensions := 2
	defaultBody := map[string]any{"encoding_format": "float"}
	model := NewModel("text-embedding-3-small", option, WithDefaultSettings(embeddings.Settings{
		Dimensions: &dimensions, ExtraBody: defaultBody,
	}))
	defaultBody["encoding_format"] = "base64"
	inputs := []string{"first", "second"}
	result, err := model.Embed(context.Background(), inputs, embeddings.InputTypeDocument, embeddings.Settings{
		ExtraHeaders: map[string]string{"X-Order": "extra"},
	})
	if err != nil {
		t.Fatal(err)
	}
	inputs[0] = "mutated"
	if model.Name() != "text-embedding-3-small" || model.ProviderName() != "compatible" ||
		model.ProviderURL() != server.URL+"/v1" {
		t.Fatalf("unexpected identity: %q %q %q", model.Name(), model.ProviderName(), model.ProviderURL())
	}
	if received["model"] != "text-embedding-3-small" || received["dimensions"] != float64(2) ||
		received["encoding_format"] != "float" || !reflect.DeepEqual(received["input"], []any{"first", "second"}) {
		t.Fatalf("unexpected body: %#v", received)
	}
	if !reflect.DeepEqual(result.Embeddings, [][]float64{{1, 2}, {3, 4}}) ||
		!reflect.DeepEqual(result.Inputs, []string{"first", "second"}) || result.InputType != embeddings.InputTypeDocument ||
		result.ModelName != "text-embedding-3-small-2025" || result.ProviderName != "compatible" ||
		result.ProviderURL != server.URL+"/v1" || result.Usage.Requests != 1 || result.Usage.InputTokens != 5 ||
		result.Usage.Details["cached_tokens"] != 2 || result.Timestamp.IsZero() {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func TestOptionsAndEnvironment(t *testing.T) {
	t.Setenv("OPENAI_BASE_URL", "https://environment.example/v1/")
	t.Setenv("OPENAI_API_KEY", "environment-key")
	client := &http.Client{}
	model := NewModel("model", WithAPIKey("explicit"), WithBaseURL("https://explicit.example/v1/"), WithHTTPClient(client))
	if model.ProviderURL() != "https://explicit.example/v1" {
		t.Fatalf("unexpected explicit URL: %q", model.ProviderURL())
	}
	environment := NewModel("model")
	if environment.ProviderURL() != "https://environment.example/v1/" {
		t.Fatalf("unexpected environment URL: %q", environment.ProviderURL())
	}
	t.Setenv("OPENAI_BASE_URL", "")
	if fallback := NewModel("model"); fallback.ProviderURL() != "https://api.openai.com/v1" {
		t.Fatalf("unexpected default URL: %q", fallback.ProviderURL())
	}
	if panicValue := capturePanic(func() {
		WithProvider(openaimodels.ProviderConfig{BaseURL: "https://example.com"})
	}); panicValue != "openai embeddings: provider name must not be empty" {
		t.Fatalf("unexpected provider-name panic: %v", panicValue)
	}
	if panicValue := capturePanic(func() {
		WithProvider(openaimodels.ProviderConfig{Name: "provider"})
	}); panicValue != "openai embeddings: provider base URL must not be empty" {
		t.Fatalf("unexpected provider-URL panic: %v", panicValue)
	}
}

func TestConcurrentEmbeddingAndTokenCounting(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		_, _ = io.WriteString(response, `{
			"data":[{"index":0,"embedding":[1,2]}],"model":"text-embedding-3-small",
			"usage":{"prompt_tokens":2,"total_tokens":2}
		}`)
	}))
	defer server.Close()
	model := NewModel("text-embedding-3-small", WithBaseURL(server.URL), WithHTTPClient(server.Client()))
	var group sync.WaitGroup
	failures := make(chan error, 20)
	for range 10 {
		group.Go(func() {
			result, err := model.Embed(
				context.Background(), []string{"hello world"}, embeddings.InputTypeQuery, embeddings.Settings{},
			)
			if err == nil && !reflect.DeepEqual(result.Embeddings, [][]float64{{1, 2}}) {
				err = errors.New("unexpected embedding")
			}
			failures <- err
		})
		group.Go(func() {
			count, err := model.CountTokens(context.Background(), "hello world")
			if err == nil && count != 2 {
				err = errors.New("unexpected token count")
			}
			failures <- err
		})
	}
	group.Wait()
	close(failures)
	for err := range failures {
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestInputLimitsAndTokenCounting(t *testing.T) {
	model := NewModel("text-embedding-3-small")
	if maximum, known, err := model.MaxInputTokens(context.Background()); err != nil || !known || maximum != 8192 {
		t.Fatalf("unexpected maximum: %d %v %v", maximum, known, err)
	}
	count, err := model.CountTokens(context.Background(), "hello world")
	if err != nil || count != 2 {
		t.Fatalf("unexpected count: %d %v", count, err)
	}
	ada := NewModel("text-embedding-ada-002")
	if count, err = ada.CountTokens(context.Background(), "hello world"); err != nil || count != 2 {
		t.Fatalf("unexpected ada count: %d %v", count, err)
	}
	unknown := NewModel("unknown")
	if _, err := unknown.CountTokens(context.Background(), "hello"); err == nil ||
		!strings.Contains(err.Error(), `tokenizer for model "unknown"`) {
		t.Fatalf("unexpected tokenizer error: %v", err)
	}
	compatible := NewModel("model", WithProvider(openaimodels.ProviderConfig{
		Name: "compatible", BaseURL: "https://example.com", APIKey: "key",
	}))
	if maximum, known, err := compatible.MaxInputTokens(context.Background()); err != nil || known || maximum != 0 {
		t.Fatalf("unexpected compatible maximum: %d %v %v", maximum, known, err)
	}
	if _, err := compatible.CountTokens(context.Background(), "hello"); !errors.Is(
		err, embeddings.ErrTokenCountingUnsupported,
	) {
		t.Fatalf("unexpected compatible count error: %v", err)
	}
}

func TestRequestValidationErrors(t *testing.T) {
	model := NewModel("model")
	zero := 0
	if _, err := model.Embed(
		context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{Dimensions: &zero},
	); err == nil || !strings.Contains(err.Error(), "dimensions must be greater than zero") {
		t.Fatalf("unexpected dimensions error: %v", err)
	}
	for _, key := range []string{"model", "input", "dimensions"} {
		if _, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{
			ExtraBody: map[string]any{key: "conflict"},
		}); err == nil || !strings.Contains(err.Error(), `field "`+key+`" conflicts`) {
			t.Fatalf("unexpected %s conflict: %v", key, err)
		}
	}
	if _, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{
		ExtraBody: map[string]any{"invalid": make(chan int)},
	}); err == nil || !strings.Contains(err.Error(), "encode request") {
		t.Fatalf("unexpected encoding error: %v", err)
	}
	minimal := NewModel("model", WithBaseURL("https://example.com"), WithHTTPClient(&http.Client{
		Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
			body, err := io.ReadAll(request.Body)
			if err != nil {
				t.Error(err)
			}
			if strings.Contains(string(body), "dimensions") {
				t.Errorf("unexpected dimensions: %s", body)
			}
			return jsonResponse(`{"data":[{"index":0,"embedding":[1]}]}`, http.StatusOK), nil
		}),
	}))
	if _, err := minimal.Embed(
		context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{},
	); err != nil {
		t.Fatal(err)
	}
	invalidURL := NewModel("model", WithBaseURL("http://[::1"))
	if _, err := invalidURL.Embed(
		context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{},
	); err == nil || !strings.Contains(err.Error(), "build request") {
		t.Fatalf("unexpected endpoint error: %v", err)
	}
}

func TestRequestFailures(t *testing.T) {
	prepareErr := errors.New("credentials failed")
	prepared := NewModel("model", WithProvider(openaimodels.ProviderConfig{
		Name: "provider", BaseURL: "https://example.com", PrepareRequest: func(*http.Request) error {
			return prepareErr
		},
	}))
	if _, err := prepared.Embed(
		context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{},
	); !errors.Is(err, prepareErr) {
		t.Fatalf("unexpected prepare error: %v", err)
	}
	transportErr := errors.New("offline")
	failed := NewModel("model", WithBaseURL("https://example.com"), WithHTTPClient(&http.Client{
		Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) { return nil, transportErr }),
	}))
	if _, err := failed.Embed(
		context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{},
	); !errors.Is(err, transportErr) {
		t.Fatalf("unexpected transport error: %v", err)
	}
	readErr := errors.New("body failed")
	readFailed := NewModel("model", WithBaseURL("https://example.com"), WithHTTPClient(&http.Client{
		Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusOK, Body: errorBody{err: readErr}, Header: http.Header{}}, nil
		}),
	}))
	if _, err := readFailed.Embed(
		context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{},
	); !errors.Is(err, readErr) {
		t.Fatalf("unexpected read error: %v", err)
	}
	apiFailed := NewModel("model", WithBaseURL("https://example.com"), WithHTTPClient(&http.Client{
		Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
			return jsonResponse(`{"error":"bad"}`, http.StatusBadRequest), nil
		}),
	}))
	if _, err := apiFailed.Embed(
		context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{},
	); err == nil || !strings.Contains(err.Error(), "API returned status 400") {
		t.Fatalf("unexpected API error: %v", err)
	}
}

func TestParseResponseErrorsAndDefaults(t *testing.T) {
	tests := []struct {
		body   string
		inputs []string
		match  string
	}{
		{body: `{`, inputs: []string{"x"}, match: "decode response"},
		{body: `{"data":[]}`, inputs: []string{"x"}, match: "0 vectors for 1 inputs"},
		{body: `{"data":[{"index":-1,"embedding":[1]}]}`, inputs: []string{"x"}, match: "invalid embedding index"},
		{body: `{"data":[{"index":1,"embedding":[1]}]}`, inputs: []string{"x"}, match: "invalid embedding index"},
		{body: `{"data":[{"index":0,"embedding":null}]}`, inputs: []string{"x"}, match: "omits vector"},
		{
			body:   `{"data":[{"index":0,"embedding":[1]},{"index":0,"embedding":[2]}]}`,
			inputs: []string{"x", "y"}, match: "invalid embedding index",
		},
	}
	for _, test := range tests {
		model := responseModel("configured", test.body)
		if _, err := model.Embed(
			context.Background(), test.inputs, embeddings.InputTypeQuery, embeddings.Settings{},
		); err == nil || !strings.Contains(err.Error(), test.match) {
			t.Fatalf("unexpected error for %s: %v", test.body, err)
		}
	}
	model := responseModel("configured", `{
		"data":[{"index":0,"embedding":[]}],"usage":{"prompt_tokens":1,"total_tokens":1}
	}`)
	result, err := model.Embed(
		context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.ModelName != "configured" || result.Usage.Details != nil || len(result.Embeddings[0]) != 0 {
		t.Fatalf("unexpected defaults: %#v", result)
	}
}

func responseModel(name, body string) *Model {
	return NewModel(name, WithBaseURL("https://example.com"), WithHTTPClient(&http.Client{
		Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
			return jsonResponse(body, http.StatusOK), nil
		}),
	}))
}

func jsonResponse(body string, status int) *http.Response {
	return &http.Response{
		StatusCode: status, Body: io.NopCloser(strings.NewReader(body)), Header: http.Header{},
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (fn roundTripperFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}

type errorBody struct{ err error }

func (body errorBody) Read([]byte) (int, error) { return 0, body.err }
func (errorBody) Close() error                  { return nil }

func capturePanic(fn func()) (value any) {
	defer func() { value = recover() }()
	fn()
	return nil
}
