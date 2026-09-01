package google

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	modelgoogle "github.com/Kludex/pydantic-ai-go/models/google"
)

func TestGeminiEmbedding(t *testing.T) {
	dimensions := 2
	defaults, err := (Settings{
		Common: embeddings.Settings{
			Dimensions: &dimensions, ExtraHeaders: map[string]string{"X-Order": "default"},
			ExtraBody: map[string]any{"nested": []any{"original"}},
		},
		TaskType: "CLASSIFICATION", Title: "Greeting",
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.String() != "https://example.com/v1beta/models/gemini-embedding-001:batchEmbedContents" {
			t.Errorf("unexpected endpoint: %s", request.URL)
		}
		if request.Header.Get("x-goog-api-key") != "secret" || request.Header.Get("X-Dynamic") != "dynamic" ||
			request.Header.Get("X-Order") != "call" || request.Header.Get("Content-Type") != "application/json" {
			t.Errorf("unexpected headers: %#v", request.Header)
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		requests := body["requests"].([]any)
		if len(requests) != 2 {
			t.Fatalf("unexpected requests: %#v", requests)
		}
		first := requests[0].(map[string]any)
		if first["model"] != "models/gemini-embedding-001" || first["taskType"] != "CLASSIFICATION" ||
			first["title"] != "Greeting" || first["outputDimensionality"] != float64(2) ||
			first["content"].(map[string]any)["parts"].([]any)[0].(map[string]any)["text"] != "one" {
			t.Errorf("unexpected request: %#v", first)
		}
		return jsonResponse(`{
			"embeddings":[
				{"values":[1,2],"statistics":{"tokenCount":3}},
				{"values":[3,4],"statistics":{"token_count":4}}
			]
		}`), nil
	})}
	provider := modelgoogle.ProviderConfig{
		Transport: modelgoogle.TransportGeminiAPI, Name: "custom-google", BaseURL: "https://example.com/v1beta/",
		APIKey: "secret", HTTPClient: client, PrepareRequest: func(request *http.Request) error {
			request.Header.Set("X-Dynamic", "dynamic")
			request.Header.Set("X-Order", "dynamic")
			return nil
		},
	}
	model := NewModel("gemini-embedding-001", WithProvider(provider), WithDefaultSettings(defaults))
	*defaults.Dimensions = 99
	defaults.ExtraHeaders["X-Order"] = "mutated"
	defaults.ExtraBody["nested"].([]any)[0] = "mutated"
	inputs := []string{"one", "two"}
	result, err := model.Embed(context.Background(), inputs, embeddings.InputTypeQuery, embeddings.Settings{
		ExtraHeaders: map[string]string{"X-Order": "call"},
	})
	if err != nil {
		t.Fatal(err)
	}
	inputs[0] = "mutated"
	if model.Name() != "gemini-embedding-001" || model.ProviderName() != "custom-google" ||
		model.ProviderURL() != "https://example.com/v1beta" || model.Transport() != modelgoogle.TransportGeminiAPI {
		t.Fatalf("unexpected identity: %q %q %q %q", model.Name(), model.ProviderName(), model.ProviderURL(), model.Transport())
	}
	if !reflect.DeepEqual(result.Embeddings, [][]float64{{1, 2}, {3, 4}}) ||
		!reflect.DeepEqual(result.Inputs, []string{"one", "two"}) || result.InputType != embeddings.InputTypeQuery ||
		result.ModelName != "gemini-embedding-001" || result.ProviderName != "custom-google" ||
		result.Usage.Requests != 1 || result.Usage.InputTokens != 7 || result.Timestamp.IsZero() {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func TestGeminiEmbeddingTaskPrefixes(t *testing.T) {
	tests := []struct {
		name      string
		inputType embeddings.InputType
		settings  Settings
		expected  string
		warning   string
	}{
		{name: "default query", inputType: embeddings.InputTypeQuery, expected: "task: search result | query: text"},
		{
			name: "question query", inputType: embeddings.InputTypeQuery,
			settings: Settings{Task: TaskQuestionAnswering}, expected: "task: question answering | query: text",
		},
		{name: "document", inputType: embeddings.InputTypeDocument, expected: "title: none | text: text"},
		{
			name: "titled document", inputType: embeddings.InputTypeDocument,
			settings: Settings{Task: TaskFactChecking, Title: "Evidence"},
			expected: "title: Evidence | text: text",
		},
		{
			name: "symmetric document", inputType: embeddings.InputTypeDocument,
			settings: Settings{Task: TaskClassification}, expected: "task: classification | query: text",
		},
		{
			name: "raw", inputType: embeddings.InputTypeDocument,
			settings: Settings{Task: TaskRaw, TaskType: "IGNORED"}, expected: "text",
			warning: "google embeddings: TaskType is not supported by gemini-embedding-2 and was ignored",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			built, err := test.settings.Build()
			if err != nil {
				t.Fatal(err)
			}
			model := responseModel(t, "gemini-embedding-2", modelgoogle.TransportGeminiAPI, func(body map[string]any) {
				request := body["requests"].([]any)[0].(map[string]any)
				text := request["content"].(map[string]any)["parts"].([]any)[0].(map[string]any)["text"]
				if text != test.expected {
					t.Errorf("got %q, want %q", text, test.expected)
				}
				if _, exists := request["taskType"]; exists {
					t.Errorf("gemini-embedding-2 unexpectedly received taskType: %#v", request)
				}
				if _, exists := request["title"]; exists {
					t.Errorf("gemini-embedding-2 unexpectedly received title: %#v", request)
				}
			})
			result, err := model.Embed(context.Background(), []string{"text"}, test.inputType, built)
			if err != nil {
				t.Fatal(err)
			}
			if test.warning == "" && result.Warnings != nil ||
				test.warning != "" && !reflect.DeepEqual(result.Warnings, []string{test.warning}) {
				t.Fatalf("unexpected warnings: %#v", result.Warnings)
			}
		})
	}
}

func TestLegacyDocumentTaskDefaults(t *testing.T) {
	model := responseModel(t, "gemini-embedding-001", modelgoogle.TransportGeminiAPI, func(body map[string]any) {
		request := body["requests"].([]any)[0].(map[string]any)
		if request["taskType"] != "RETRIEVAL_DOCUMENT" {
			t.Errorf("unexpected task type: %#v", request)
		}
	})
	settings, err := (Settings{Task: TaskCodeRetrieval}).Build()
	if err != nil {
		t.Fatal(err)
	}
	result, err := model.Embed(context.Background(), []string{"text"}, embeddings.InputTypeDocument, settings)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(
		result.Warnings,
		[]string{"google embeddings: Task is only supported by gemini-embedding-2 and was ignored"},
	) {
		t.Fatalf("unexpected warning: %#v", result.Warnings)
	}
}

func TestVertexEmbedding(t *testing.T) {
	dimensions := 3
	client := &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.String() != "https://vertex.example/v1beta1/publishers/google/models/gemini-embedding-001:predict" {
			t.Errorf("unexpected endpoint: %s", request.URL)
		}
		if request.Header.Get("x-goog-api-key") != "vertex-key" {
			t.Errorf("unexpected API key: %#v", request.Header)
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		instance := body["instances"].([]any)[0].(map[string]any)
		if instance["content"] != "hello" || instance["task_type"] != "RETRIEVAL_QUERY" {
			t.Errorf("unexpected instance: %#v", instance)
		}
		if body["parameters"].(map[string]any)["outputDimensionality"] != float64(3) {
			t.Errorf("unexpected parameters: %#v", body)
		}
		return jsonResponse(`{
			"metadata":{"billableCharacterCount":5},
			"predictions":[{"embeddings":{"values":[1,2,3],"statistics":{"token_count":4}}}]
		}`), nil
	})}
	model, err := NewVertexModel("gemini-embedding-001", modelgoogle.VertexConfig{
		APIKey: "vertex-key", Endpoint: "https://vertex.example", HTTPClient: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := model.Embed(context.Background(), []string{"hello"}, embeddings.InputTypeQuery, embeddings.Settings{
		Dimensions: &dimensions,
	})
	if err != nil {
		t.Fatal(err)
	}
	if model.ProviderName() != "google-cloud" || model.Transport() != modelgoogle.TransportVertexAI ||
		!reflect.DeepEqual(result.Embeddings, [][]float64{{1, 2, 3}}) || result.Usage.InputTokens != 4 {
		t.Fatalf("unexpected Vertex result: %#v %#v", model, result)
	}

	_, err = NewVertexModel("model", modelgoogle.VertexConfig{
		APIKey: "key", TokenProvider: func(context.Context) (string, error) { return "token", nil },
	})
	if err == nil || !strings.Contains(err.Error(), "cannot both be set") {
		t.Fatalf("unexpected Vertex configuration error: %v", err)
	}
}

func TestVertexOAuthEmbedding(t *testing.T) {
	client := &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
		if request.Header.Get("Authorization") != "Bearer cloud-token" || request.Header.Get("X-Order") != "call" {
			t.Errorf("unexpected headers: %#v", request.Header)
		}
		return jsonResponse(`{"predictions":[{"embeddings":{"values":[1]}}]}`), nil
	})}
	model, err := NewVertexModel("gemini-embedding-001", modelgoogle.VertexConfig{
		Project: "project", Location: "global", Endpoint: "https://vertex.example", HTTPClient: client,
		TokenProvider: func(context.Context) (string, error) { return "cloud-token", nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Embed(context.Background(), []string{"hello"}, embeddings.InputTypeQuery, embeddings.Settings{
		ExtraHeaders: map[string]string{"X-Order": "call"},
	}); err != nil {
		t.Fatal(err)
	}
}

func TestConcurrentEmbeddingAndTokenCounting(t *testing.T) {
	model := NewModel("gemini-embedding-2-preview", WithBaseURL("https://example.com"), WithHTTPClient(&http.Client{
		Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
			if strings.HasSuffix(request.URL.Path, ":countTokens") {
				return jsonResponse(`{"totalTokens":2}`), nil
			}
			return jsonResponse(`{"embeddings":[{"values":[1,2]}]}`), nil
		}),
	}))
	var group sync.WaitGroup
	failures := make(chan string, 20)
	for range 10 {
		group.Go(func() {
			result, err := model.Embed(
				context.Background(), []string{"hello world"}, embeddings.InputTypeQuery, embeddings.Settings{},
			)
			if err != nil || !reflect.DeepEqual(result.Embeddings, [][]float64{{1, 2}}) {
				failures <- "embedding"
				return
			}
			failures <- ""
		})
		group.Go(func() {
			count, err := model.CountTokens(context.Background(), "hello world")
			if err != nil || count != 2 {
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

func TestCountTokens(t *testing.T) {
	model := NewModel("gemini-embedding-2-preview", WithProvider(modelgoogle.ProviderConfig{
		BaseURL: "https://example.com/v1beta", APIKey: "secret",
		HTTPClient: &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
			if request.URL.Path != "/v1beta/models/gemini-embedding-2-preview:countTokens" {
				t.Errorf("unexpected count path: %s", request.URL.Path)
			}
			var body map[string]any
			if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			content := body["contents"].([]any)[0].(map[string]any)
			if content["role"] != "user" || content["parts"].([]any)[0].(map[string]any)["text"] != "hello" {
				t.Errorf("unexpected count body: %#v", body)
			}
			return jsonResponse(`{"totalTokens":5}`), nil
		})},
	}))
	count, err := model.CountTokens(context.Background(), "hello")
	if err != nil || count != 5 {
		t.Fatalf("unexpected token count: %d %v", count, err)
	}
}

func TestSettings(t *testing.T) {
	dimensions := 4
	common := embeddings.Settings{
		Dimensions: &dimensions, ExtraHeaders: map[string]string{"X": "one"},
		ExtraBody: map[string]any{"nested": []any{"original"}},
	}
	built, err := (Settings{
		Common: common, Task: TaskClustering, TaskType: "CUSTOM", Title: "Title",
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	*common.Dimensions = 9
	common.ExtraHeaders["X"] = "changed"
	common.ExtraBody["nested"].([]any)[0] = "changed"
	if *built.Dimensions != 4 || built.ExtraHeaders["X"] != "one" ||
		built.ExtraBody["nested"].([]any)[0] != "original" {
		t.Fatalf("settings were not detached: %#v", built)
	}
	if empty, err := (Settings{}).Build(); err != nil || empty.ExtraBody != nil {
		t.Fatalf("unexpected empty settings: %#v %v", empty, err)
	}
	for _, task := range []Task{
		TaskSearchResult, TaskQuestionAnswering, TaskFactChecking, TaskCodeRetrieval,
		TaskClassification, TaskClustering, TaskSentenceSimilarity, TaskRaw,
	} {
		if _, err := (Settings{Task: task}).Build(); err != nil {
			t.Fatalf("valid task %q failed: %v", task, err)
		}
	}
	if _, err := (Settings{Task: Task("bad")}).Build(); err == nil || !strings.Contains(err.Error(), "invalid task") {
		t.Fatalf("unexpected task error: %v", err)
	}
	conflicts := []struct {
		key      string
		settings Settings
	}{
		{key: taskKey, settings: Settings{Task: TaskRaw}},
		{key: taskTypeKey, settings: Settings{TaskType: "CUSTOM"}},
		{key: titleKey, settings: Settings{Title: "Title"}},
	}
	for _, conflict := range conflicts {
		conflict.settings.Common.ExtraBody = map[string]any{conflict.key: "existing"}
		if _, err := conflict.settings.Build(); err == nil || !strings.Contains(err.Error(), "conflicts") {
			t.Fatalf("unexpected conflict for %s: %v", conflict.key, err)
		}
	}
}

func TestOptionsEnvironmentAndLimits(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "google-key")
	t.Setenv("GEMINI_API_KEY", "gemini-key")
	model := NewModel("model", WithBaseURL("https://example.com"), WithHTTPClient(keyCheckingClient(t, "google-key")))
	if model.ProviderURL() != "https://example.com" {
		t.Fatalf("unexpected environment configuration: %#v", model)
	}
	if _, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{}); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOOGLE_API_KEY", "")
	fallback := NewModel("model", WithBaseURL("https://example.com"), WithHTTPClient(keyCheckingClient(t, "gemini-key")))
	if _, err := fallback.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{}); err != nil {
		t.Fatal(err)
	}
	custom := NewModel(
		"model", WithAPIKey("explicit"), WithBaseURL("https://custom.example/"),
		WithHTTPClient(keyCheckingClient(t, "explicit")),
	)
	if custom.ProviderURL() != "https://custom.example" {
		t.Fatalf("unexpected explicit URL: %q", custom.ProviderURL())
	}
	if _, err := custom.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{}); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GEMINI_API_KEY", "")
	if defaultModel := NewModel("model"); defaultModel.ProviderURL() != "https://generativelanguage.googleapis.com/v1beta" {
		t.Fatalf("unexpected default URL: %q", defaultModel.ProviderURL())
	}
	if panicValue := capturePanic(func() {
		WithProvider(modelgoogle.ProviderConfig{Transport: modelgoogle.Transport("bad")})
	}); panicValue != `google embeddings: invalid transport "bad"` {
		t.Fatalf("unexpected panic: %v", panicValue)
	}
	limits := map[string]int{
		"gemini-embedding-001": 2048, "gemini-embedding-2-preview": 8192, "gemini-embedding-2": 8192,
		"text-embedding-005": 2048, "text-multilingual-embedding-002": 2048,
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

func keyCheckingClient(t *testing.T, expected string) *http.Client {
	t.Helper()
	return &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
		if key := request.Header.Get("x-goog-api-key"); key != expected {
			t.Errorf("got API key %q, want %q", key, expected)
		}
		return jsonResponse(`{"embeddings":[{"values":[1]}]}`), nil
	})}
}

func responseModel(
	t *testing.T, name string, transport modelgoogle.Transport, inspect func(map[string]any),
) *Model {
	t.Helper()
	return NewModel(name, WithProvider(modelgoogle.ProviderConfig{
		Transport: transport, BaseURL: "https://example.com/v1beta",
		HTTPClient: &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
			var body map[string]any
			if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			inspect(body)
			if transport == modelgoogle.TransportVertexAI {
				return jsonResponse(`{"predictions":[{"embeddings":{"values":[1]}}]}`), nil
			}
			return jsonResponse(`{"embeddings":[{"values":[1]}]}`), nil
		})},
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

func capturePanic(function func()) (value any) {
	defer func() { value = recover() }()
	function()
	return nil
}
