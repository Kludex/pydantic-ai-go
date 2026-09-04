package cohere

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/embeddings"
	modelcohere "github.com/Kludex/pydantic-ai-go/models/cohere"
)

func TestEmbedding(t *testing.T) {
	maxTokens := 256
	defaults, err := (Settings{
		Common: embeddings.Settings{
			ExtraHeaders: map[string]string{"X-Order": "default"},
			ExtraBody:    map[string]any{"custom": map[string]any{"value": "original"}},
		},
		InputType: InputTypeClassification, MaxTokens: &maxTokens, Truncate: TruncationStart,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.String() != "https://example.com/v2/embed" {
			t.Errorf("unexpected endpoint: %s", request.URL)
		}
		if request.Header.Get("Authorization") != "Bearer secret" || request.Header.Get("X-Static") != "static" ||
			request.Header.Get("X-Dynamic") != "dynamic" || request.Header.Get("X-Order") != "call" ||
			request.Header.Get("Content-Type") != "application/json" {
			t.Errorf("unexpected headers: %#v", request.Header)
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["model"] != "embed-v4.0" || !reflect.DeepEqual(body["texts"], []any{"one", "two"}) ||
			body["input_type"] != "classification" || body["max_tokens"] != float64(256) ||
			body["truncate"] != "END" || body["output_dimension"] != float64(2) ||
			!reflect.DeepEqual(body["embedding_types"], []any{"float"}) {
			t.Errorf("unexpected request body: %#v", body)
		}
		if _, exists := body["custom"]; exists || body["per_call"] != true {
			t.Errorf("unexpected extra body: %#v", body)
		}
		return jsonResponse(`{
			"id":"response-id",
			"embeddings":{"float":[[1,2],[3,4]]},
			"meta":{"billed_units":{"input_tokens":7,"search_units":2,"zero":0}}
		}`), nil
	})}
	provider := modelcohere.ProviderConfig{
		Name: "custom-cohere", BaseURL: "https://example.com/", APIKey: "secret", HTTPClient: client,
		Headers: http.Header{"X-Static": {"static"}, "X-Order": {"static"}},
		PrepareRequest: func(request *http.Request) error {
			request.Header.Set("X-Dynamic", "dynamic")
			request.Header.Set("X-Order", "dynamic")
			return nil
		},
	}
	model := NewModel("embed-v4.0", WithProvider(provider), WithDefaultSettings(defaults))
	provider.Headers.Set("X-Static", "mutated")
	maxTokens = 999
	defaults.ExtraBody["custom"].(map[string]any)["value"] = "mutated"
	dimensions := 2
	perCall, err := (Settings{
		Common: embeddings.Settings{
			Dimensions: &dimensions, ExtraHeaders: map[string]string{"X-Order": "call"},
			ExtraBody: map[string]any{"per_call": true},
		},
		Truncate: TruncationEnd,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	inputs := []string{"one", "two"}
	result, err := model.Embed(context.Background(), inputs, embeddings.InputTypeQuery, perCall)
	if err != nil {
		t.Fatal(err)
	}
	inputs[0] = "mutated"
	if model.Name() != "embed-v4.0" || model.ProviderName() != "custom-cohere" ||
		model.ProviderURL() != "https://example.com" {
		t.Fatalf("unexpected identity: %q %q %q", model.Name(), model.ProviderName(), model.ProviderURL())
	}
	if !reflect.DeepEqual(result.Embeddings, [][]float64{{1, 2}, {3, 4}}) ||
		!reflect.DeepEqual(result.Inputs, []string{"one", "two"}) || result.InputType != embeddings.InputTypeQuery ||
		result.ModelName != "embed-v4.0" || result.ProviderName != "custom-cohere" ||
		result.ProviderResponseID != "response-id" || result.Timestamp.IsZero() ||
		!reflect.DeepEqual(result.Usage, ai.Usage{Requests: 1, InputTokens: 7, Details: map[string]int{"search_units": 2}}) {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func TestDefaultSettingsAreDetached(t *testing.T) {
	defaults := embeddings.Settings{ExtraBody: map[string]any{"custom": map[string]any{"value": "original"}}}
	model := inspectingModel(t, func(body map[string]any) {
		if body["custom"].(map[string]any)["value"] != "original" {
			t.Errorf("unexpected defaults: %#v", body)
		}
	})
	WithDefaultSettings(defaults)(model)
	defaults.ExtraBody["custom"].(map[string]any)["value"] = "changed"
	if _, err := model.Embed(
		context.Background(), []string{"text"}, embeddings.InputTypeQuery, embeddings.Settings{},
	); err != nil {
		t.Fatal(err)
	}
}

func TestInputDefaultsAndTruncation(t *testing.T) {
	tests := []struct {
		name      string
		inputType embeddings.InputType
		settings  embeddings.Settings
		wantInput string
		wantTrunc string
	}{
		{name: "query", inputType: embeddings.InputTypeQuery, wantInput: "search_query", wantTrunc: "NONE"},
		{name: "document", inputType: embeddings.InputTypeDocument, wantInput: "search_document", wantTrunc: "NONE"},
		{
			name: "portable truncate", inputType: embeddings.InputTypeQuery,
			settings: embeddings.Settings{Truncate: boolPointer(true)}, wantInput: "search_query", wantTrunc: "END",
		},
		{
			name: "portable false", inputType: embeddings.InputTypeQuery,
			settings: embeddings.Settings{Truncate: boolPointer(false)}, wantInput: "search_query", wantTrunc: "NONE",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := inspectingModel(t, func(body map[string]any) {
				if body["input_type"] != test.wantInput || body["truncate"] != test.wantTrunc ||
					body["output_dimension"] != nil || body["max_tokens"] != nil {
					t.Errorf("unexpected body: %#v", body)
				}
			})
			if _, err := model.Embed(context.Background(), []string{"text"}, test.inputType, test.settings); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestCountTokens(t *testing.T) {
	model := NewModel("embed-v4.0", WithAPIKey("key"), WithBaseURL("https://example.com/"),
		WithHTTPClient(&http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
			if request.URL.String() != "https://example.com/v1/tokenize" ||
				request.Header.Get("Authorization") != "Bearer key" {
				t.Errorf("unexpected token request: %s %#v", request.URL, request.Header)
			}
			var body map[string]any
			if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			if body["model"] != "embed-v4.0" || body["text"] != "Hello, world!" {
				t.Errorf("unexpected token body: %#v", body)
			}
			return jsonResponse(`{"tokens":[9707,11,1879,0]}`), nil
		})}),
	)
	count, err := model.CountTokens(context.Background(), "Hello, world!")
	if err != nil || count != 4 {
		t.Fatalf("unexpected count: %d %v", count, err)
	}
}

func TestConcurrentUse(t *testing.T) {
	model := NewModel("embed-v4.0", WithBaseURL("https://example.com"), WithHTTPClient(&http.Client{
		Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
			if strings.HasSuffix(request.URL.Path, "/tokenize") {
				return jsonResponse(`{"tokens":[1]}`), nil
			}
			return jsonResponse(`{"embeddings":{"float":[[1]]}}`), nil
		}),
	}))
	var group sync.WaitGroup
	failures := make(chan string, 20)
	for range 10 {
		group.Go(func() {
			result, err := model.Embed(
				context.Background(), []string{"text"}, embeddings.InputTypeQuery, embeddings.Settings{},
			)
			if err != nil || !reflect.DeepEqual(result.Embeddings, [][]float64{{1}}) {
				failures <- "embedding"
				return
			}
			failures <- ""
		})
		group.Go(func() {
			count, err := model.CountTokens(context.Background(), "text")
			if err != nil || count != 1 {
				failures <- "token count"
				return
			}
			failures <- ""
		})
	}
	group.Wait()
	close(failures)
	for failure := range failures {
		if failure != "" {
			t.Fatal(failure)
		}
	}
}

func inspectingModel(t *testing.T, inspect func(map[string]any)) *Model {
	t.Helper()
	return NewModel("embed-v4.0", WithBaseURL("https://example.com"), WithHTTPClient(&http.Client{
		Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
			var body map[string]any
			if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			inspect(body)
			return jsonResponse(`{"embeddings":{"float":[[1]]}}`), nil
		}),
	}))
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (function roundTripperFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

func jsonResponse(body string) *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(body)),
	}
}

func boolPointer(value bool) *bool { return &value }
