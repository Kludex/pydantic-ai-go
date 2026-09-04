package voyageai

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/embeddings"
)

func TestEmbedding(t *testing.T) {
	defaults, err := (Settings{
		Common:    embeddings.Settings{ExtraBody: map[string]any{"default": "replaced"}},
		InputType: InputTypeNone,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.String() != "https://example.com/v1/embeddings" {
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
		if body["model"] != "voyage-4" || !reflect.DeepEqual(body["input"], []any{"one", "two"}) ||
			body["input_type"] != nil || body["truncation"] != true || body["output_dimension"] != float64(2) ||
			body["output_dtype"] != nil || body["encoding_format"] != "base64" || body["per_call"] != true {
			t.Errorf("unexpected request body: %#v", body)
		}
		if _, exists := body["default"]; exists {
			t.Errorf("per-call extra body did not replace defaults: %#v", body)
		}
		return jsonResponse(fmtResponse([]responseVector{
			{index: 1, values: []float32{3, 4}}, {index: 0, values: []float32{1, 2}},
		}, 5)), nil
	})}
	provider := ProviderConfig{
		Name: "custom-voyage", BaseURL: "https://example.com/v1/", APIKey: "secret", HTTPClient: client,
		Headers: http.Header{"X-Static": {"static"}, "X-Order": {"static"}},
		PrepareRequest: func(request *http.Request) error {
			request.Header.Set("X-Dynamic", "dynamic")
			request.Header.Set("X-Order", "dynamic")
			return nil
		},
	}
	model := NewModel("voyage-4", WithProvider(provider), WithDefaultSettings(defaults))
	provider.Headers.Set("X-Static", "mutated")
	defaults.ExtraBody[inputTypeKey] = InputTypeQuery
	dimensions := 2
	inputs := []string{"one", "two"}
	result, err := model.Embed(context.Background(), inputs, embeddings.InputTypeQuery, embeddings.Settings{
		Dimensions: &dimensions, Truncate: boolPointer(true),
		ExtraHeaders: map[string]string{"X-Order": "call"}, ExtraBody: map[string]any{"per_call": true},
	})
	if err != nil {
		t.Fatal(err)
	}
	inputs[0] = "mutated"
	if model.Name() != "voyage-4" || model.ProviderName() != "custom-voyage" ||
		model.ProviderURL() != "https://example.com/v1" {
		t.Fatalf("unexpected identity: %q %q %q", model.Name(), model.ProviderName(), model.ProviderURL())
	}
	if !reflect.DeepEqual(result.Embeddings, [][]float64{{1, 2}, {3, 4}}) ||
		!reflect.DeepEqual(result.Inputs, []string{"one", "two"}) || result.InputType != embeddings.InputTypeQuery ||
		result.ModelName != "voyage-4" || result.ProviderName != "custom-voyage" || result.Timestamp.IsZero() ||
		!reflect.DeepEqual(result.Usage, ai.Usage{Requests: 1, InputTokens: 5}) {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func TestInputTypeDefaults(t *testing.T) {
	for _, test := range []struct {
		name      string
		inputType embeddings.InputType
		wire      any
	}{
		{name: "query", inputType: embeddings.InputTypeQuery, wire: "query"},
		{name: "document", inputType: embeddings.InputTypeDocument, wire: "document"},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := inspectingModel(t, func(body map[string]any) {
				if body["input_type"] != test.wire || body["truncation"] != false ||
					body["output_dimension"] != nil {
					t.Errorf("unexpected body: %#v", body)
				}
			})
			if _, err := model.Embed(
				context.Background(), []string{"text"}, test.inputType, embeddings.Settings{},
			); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestFloatEmbeddingCompatibility(t *testing.T) {
	model := responseBodyModel(`{
		"data":[{"embedding":[1.5,2.5],"index":0}],"usage":{"total_tokens":2}
	}`)
	result, err := model.Embed(
		context.Background(), []string{"text"}, embeddings.InputTypeQuery, embeddings.Settings{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Embeddings, [][]float64{{1.5, 2.5}}) {
		t.Fatalf("unexpected float response: %#v", result.Embeddings)
	}
}

func TestConcurrentUse(t *testing.T) {
	model := responseBodyModel(fmtResponse([]responseVector{{index: 0, values: []float32{1}}}, 1))
	var group sync.WaitGroup
	failures := make(chan error, 20)
	for range 20 {
		group.Go(func() {
			_, err := model.Embed(
				context.Background(), []string{"text"}, embeddings.InputTypeQuery, embeddings.Settings{},
			)
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

func inspectingModel(t *testing.T, inspect func(map[string]any)) *Model {
	t.Helper()
	return NewModel("voyage-4", WithBaseURL("https://example.com/v1"), WithHTTPClient(&http.Client{
		Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
			var body map[string]any
			if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			inspect(body)
			return jsonResponse(fmtResponse([]responseVector{{index: 0, values: []float32{1}}}, 1)), nil
		}),
	}))
}

type responseVector struct {
	index  int
	values []float32
}

func fmtResponse(vectors []responseVector, tokens int) string {
	data := make([]map[string]any, len(vectors))
	for position, vector := range vectors {
		encoded := make([]byte, len(vector.values)*4)
		for index, value := range vector.values {
			binary.LittleEndian.PutUint32(encoded[index*4:index*4+4], math.Float32bits(value))
		}
		data[position] = map[string]any{"embedding": base64.StdEncoding.EncodeToString(encoded), "index": vector.index}
	}
	body, _ := json.Marshal(map[string]any{"data": data, "usage": map[string]any{"total_tokens": tokens}})
	return string(body)
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
