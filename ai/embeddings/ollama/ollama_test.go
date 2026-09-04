package ollama_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
	"github.com/Kludex/pydantic-ai-go/ai/embeddings/ollama"
)

func TestEmbed(t *testing.T) {
	var requestBody string
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Fatal(err)
		}
		requestBody = string(body)
		if request.URL.Path != "/api/embed" || request.Header.Get("X-Provider") != "provider" ||
			request.Header.Get("X-Prepared") != "prepared" || request.Header.Get("X-Extra") != "extra" {
			t.Fatalf("unexpected request: %s %#v", request.URL.Path, request.Header)
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{
			"model":"nomic-embed-text:latest",
			"embeddings":[[1,2,3],[4,5,6]],
			"total_duration":10,
			"load_duration":2,
			"prompt_eval_count":7
		}`)
	}))
	defer server.Close()

	dimensions := 2
	truncate := true
	provider := ollama.ProviderConfig{
		BaseURL: server.URL + "/", HTTPClient: server.Client(), Headers: http.Header{"X-Provider": {"provider"}},
		PrepareRequest: func(request *http.Request) error {
			request.Header.Set("X-Prepared", "prepared")
			request.Header.Set("X-Extra", "replaced")
			return nil
		},
	}
	model := ollama.NewModel(
		"nomic-embed-text", ollama.WithProvider(provider),
		ollama.WithDefaultSettings(embeddings.Settings{
			Dimensions: &dimensions, Truncate: &truncate,
			ExtraHeaders: map[string]string{"X-Extra": "extra"},
			ExtraBody:    map[string]any{"keep_alive": "1m"},
		}),
	)
	provider.Headers.Set("X-Provider", "mutated")
	result, err := model.Embed(
		context.Background(), []string{"first", "second"}, embeddings.InputTypeDocument, embeddings.Settings{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if model.Name() != "nomic-embed-text" || model.ProviderName() != "ollama" || model.ProviderURL() != server.URL {
		t.Fatalf("unexpected model identity: %q %q %q", model.Name(), model.ProviderName(), model.ProviderURL())
	}
	if requestBody != `{"dimensions":2,"input":["first","second"],"keep_alive":"1m","model":"nomic-embed-text","truncate":true}` {
		t.Fatalf("unexpected request body: %s", requestBody)
	}
	if result.ModelName != "nomic-embed-text:latest" || result.ProviderName != "ollama" ||
		result.ProviderURL != server.URL || result.InputType != embeddings.InputTypeDocument ||
		result.Usage.Requests != 1 || result.Usage.InputTokens != 7 || len(result.Embeddings) != 2 ||
		result.ProviderDetails["total_duration"] != int64(10) || result.ProviderDetails["load_duration"] != int64(2) {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func TestDefaultsAndOverrides(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "ollama.example:11434/")
	model := ollama.NewModel("model")
	if model.ProviderURL() != "http://ollama.example:11434" {
		t.Fatalf("unexpected environment endpoint: %q", model.ProviderURL())
	}

	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		_, _ = io.WriteString(writer, `{"embeddings":[],"prompt_eval_count":0}`)
	}))
	defer server.Close()
	model = ollama.NewModel("model", ollama.WithBaseURL(server.URL+"/"), ollama.WithHTTPClient(server.Client()))
	result, err := model.Embed(context.Background(), nil, embeddings.InputTypeQuery, embeddings.Settings{})
	if err != nil {
		t.Fatal(err)
	}
	if result.ModelName != "model" || result.ProviderDetails != nil {
		t.Fatalf("unexpected minimal result: %#v", result)
	}
}

func TestRequestFailures(t *testing.T) {
	prepareErr := errors.New("prepare failed")
	requestErr := errors.New("request failed")
	readErr := errors.New("read failed")
	tests := []struct {
		name     string
		model    *ollama.Model
		settings embeddings.Settings
		match    string
	}{
		{name: "settings", model: ollama.NewModel("model"), settings: dimensions(0), match: "dimensions"},
		{name: "conflict", model: ollama.NewModel("model"), settings: embeddings.Settings{
			ExtraBody: map[string]any{"model": "other"},
		}, match: `field "model" conflicts`},
		{name: "encode", model: ollama.NewModel("model"), settings: embeddings.Settings{
			ExtraBody: map[string]any{"invalid": func() {}},
		}, match: "encode request"},
		{name: "build", model: ollama.NewModel("model", ollama.WithBaseURL("%")), match: "build request"},
		{name: "prepare", model: ollama.NewModel("model", ollama.WithProvider(ollama.ProviderConfig{
			BaseURL: "http://example.com", PrepareRequest: func(*http.Request) error { return prepareErr },
		})), match: "prepare failed"},
		{name: "request", model: ollama.NewModel("model", ollama.WithHTTPClient(&http.Client{
			Transport: roundTripFunc(func(*http.Request) (*http.Response, error) { return nil, requestErr }),
		})), match: "request failed"},
		{name: "read", model: ollama.NewModel("model", ollama.WithHTTPClient(&http.Client{
			Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusOK, Body: errorBody{err: readErr}, Header: http.Header{}}, nil
			}),
		})), match: "read failed"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := test.model.Embed(context.Background(), []string{"input"}, embeddings.InputTypeQuery, test.settings)
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestResponseFailures(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
		match  string
	}{
		{name: "api", status: http.StatusBadRequest, body: "bad input", match: "status 400: bad input"},
		{name: "decode", status: http.StatusOK, body: "{", match: "decode response"},
		{name: "omitted embeddings", status: http.StatusOK, body: `{}`, match: "omits embeddings"},
		{name: "wrong count", status: http.StatusOK, body: `{"embeddings":[]}`, match: "0 vectors for 1 inputs"},
		{name: "null vector", status: http.StatusOK, body: `{"embeddings":[null]}`, match: "omits vector at index 0"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				writer.Header().Set("X-Error", "detail")
				writer.WriteHeader(test.status)
				_, _ = io.WriteString(writer, test.body)
			}))
			defer server.Close()
			model := ollama.NewModel("model", ollama.WithBaseURL(server.URL), ollama.WithHTTPClient(server.Client()))
			_, err := model.Embed(
				context.Background(), []string{"input"}, embeddings.InputTypeQuery, embeddings.Settings{},
			)
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
			if test.name == "api" {
				var apiErr *ollama.APIError
				if !errors.As(err, &apiErr) || apiErr.Headers.Get("X-Error") != "detail" {
					t.Fatalf("unexpected API error: %#v", err)
				}
			}
		})
	}
}

func TestInvalidProvider(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("empty provider endpoint did not panic")
		}
	}()
	ollama.WithProvider(ollama.ProviderConfig{})
}

func dimensions(value int) embeddings.Settings { return embeddings.Settings{Dimensions: &value} }

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

type errorBody struct{ err error }

func (body errorBody) Read([]byte) (int, error) { return 0, body.err }
func (errorBody) Close() error                  { return nil }
