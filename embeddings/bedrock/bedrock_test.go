package bedrock_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"

	"github.com/Kludex/pydantic-ai-go/embeddings"
	"github.com/Kludex/pydantic-ai-go/embeddings/bedrock"
)

func TestTitanEmbeddings(t *testing.T) {
	normalize := false
	dimensions := 256
	concurrency := 2
	settings, err := (bedrock.Settings{
		Common: embeddings.Settings{
			Dimensions: &dimensions, ExtraBody: map[string]any{"custom": "value"},
			ExtraHeaders: map[string]string{"X-Test": "value"},
		},
		TitanNormalize: &normalize, InferenceProfile: "profile-arn", MaxConcurrency: &concurrency,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	client := &fakeClient{delay: time.Millisecond}
	model, err := bedrock.NewModel(
		"us.amazon.titan-embed-text-v2:0", bedrock.WithClient(client),
		bedrock.WithProviderURL("https://bedrock.example/"), bedrock.WithSettings(settings),
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := embeddings.New(model).EmbedDocuments(context.Background(), []string{"a", "bb", "ccc"})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(result.Embeddings[0], []float64{1}) || !slices.Equal(result.Embeddings[2], []float64{3}) ||
		result.ModelName != "us.amazon.titan-embed-text-v2:0" || result.ProviderName != "bedrock" ||
		result.ProviderURL != "https://bedrock.example" || result.Usage.Requests != 3 || result.Usage.InputTokens != 6 ||
		result.InputType != embeddings.InputTypeDocument || len(result.Inputs) != 3 {
		t.Fatalf("unexpected result: %#v", result)
	}
	requests, maxActive := client.snapshot()
	if len(requests) != 3 || maxActive != 2 {
		t.Fatalf("unexpected concurrency: %d requests, %d active", len(requests), maxActive)
	}
	for _, request := range requests {
		var body map[string]any
		if err := json.Unmarshal(request.Body, &body); err != nil {
			t.Fatal(err)
		}
		if request.ModelID != "profile-arn" || body["dimensions"] != float64(256) || body["normalize"] != false ||
			body["custom"] != "value" || request.Headers["X-Test"] != "value" {
			t.Fatalf("unexpected Titan request: %#v %#v", request, body)
		}
	}
	limit, known, err := model.MaxInputTokens(context.Background())
	if err != nil || !known || limit != 8192 || model.Name() != "us.amazon.titan-embed-text-v2:0" ||
		model.ProviderName() != "bedrock" || model.ProviderURL() != "https://bedrock.example" {
		t.Fatalf("unexpected model identity: %d %t %v", limit, known, err)
	}

	v1, err := bedrock.NewModel("amazon.titan-embed-text-v1", bedrock.WithClient(client))
	if err != nil {
		t.Fatal(err)
	}
	_, err = v1.Embed(context.Background(), []string{"v1"}, embeddings.InputTypeQuery, settings)
	if err != nil {
		t.Fatal(err)
	}
	requests, _ = client.snapshot()
	var v1Body map[string]any
	if err := json.Unmarshal(requests[len(requests)-1].Body, &v1Body); err != nil {
		t.Fatal(err)
	}
	if _, exists := v1Body["dimensions"]; exists {
		t.Fatalf("Titan v1 received dimensions: %#v", v1Body)
	}
	if _, exists := v1Body["normalize"]; exists {
		t.Fatalf("Titan v1 received normalization: %#v", v1Body)
	}

	_, err = model.Embed(context.Background(), []string{"override"}, embeddings.InputTypeQuery, embeddings.Settings{
		ExtraBody: map[string]any{"per_call": true, "bedrock_titan_normalize": true},
	})
	if err != nil {
		t.Fatal(err)
	}
	requests, _ = client.snapshot()
	var overrideBody map[string]any
	if err := json.Unmarshal(requests[len(requests)-1].Body, &overrideBody); err != nil {
		t.Fatal(err)
	}
	if overrideBody["normalize"] != true || overrideBody["per_call"] != true || requests[len(requests)-1].ModelID != "profile-arn" {
		t.Fatalf("provider defaults were not merged fieldwise: %#v", overrideBody)
	}
}

func TestCohereEmbeddings(t *testing.T) {
	dimensions := 512
	maxTokens := 1_000
	settings, err := (bedrock.Settings{
		Common: embeddings.Settings{Dimensions: &dimensions}, CohereMaxTokens: &maxTokens,
		CohereInputType: bedrock.CohereInputClustering, CohereTruncate: bedrock.TruncationStart,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	client := &fakeClient{cohereByType: true}
	model, err := bedrock.NewModel("cohere.embed-v4:0", bedrock.WithClient(client))
	if err != nil {
		t.Fatal(err)
	}
	result, err := model.Embed(context.Background(), []string{"one", "three"}, embeddings.InputTypeQuery, settings)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 2 || result.ProviderResponseID != "cohere-id" || result.Usage.Requests != 1 ||
		result.Usage.InputTokens != 2 {
		t.Fatalf("unexpected Cohere result: %#v", result)
	}
	requests, _ := client.snapshot()
	var body map[string]any
	if err := json.Unmarshal(requests[0].Body, &body); err != nil {
		t.Fatal(err)
	}
	if body["input_type"] != "clustering" || body["truncate"] != "START" ||
		body["max_tokens"] != float64(1_000) || body["output_dimension"] != float64(512) {
		t.Fatalf("unexpected Cohere v4 body: %#v", body)
	}

	truncate := true
	v3, err := bedrock.NewModel("cohere.embed-english-v3", bedrock.WithClient(client))
	if err != nil {
		t.Fatal(err)
	}
	_, err = v3.Embed(context.Background(), []string{"doc"}, embeddings.InputTypeDocument, embeddings.Settings{
		Dimensions: &dimensions, Truncate: &truncate, ExtraBody: map[string]any{"bedrock_cohere_max_tokens": 42},
	})
	if err != nil {
		t.Fatal(err)
	}
	requests, _ = client.snapshot()
	body = nil
	if err := json.Unmarshal(requests[len(requests)-1].Body, &body); err != nil {
		t.Fatal(err)
	}
	if body["input_type"] != "search_document" || body["truncate"] != "END" {
		t.Fatalf("unexpected Cohere defaults: %#v", body)
	}
	if _, exists := body["max_tokens"]; exists {
		t.Fatalf("Cohere v3 received max tokens: %#v", body)
	}
}

func TestNovaEmbeddings(t *testing.T) {
	dimensions := 384
	settings, err := (bedrock.Settings{
		Common: embeddings.Settings{Dimensions: &dimensions}, NovaTruncate: bedrock.TruncationStart,
		NovaPurpose: bedrock.NovaPurposeClassification,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	client := &fakeClient{}
	model, err := bedrock.NewModel("amazon.nova-2-multimodal-embeddings-v1:0", bedrock.WithClient(client))
	if err != nil {
		t.Fatal(err)
	}
	result, err := model.Embed(context.Background(), []string{"nova"}, embeddings.InputTypeQuery, settings)
	if err != nil || !slices.Equal(result.Embeddings[0], []float64{4}) {
		t.Fatalf("unexpected Nova result: %#v %v", result, err)
	}
	requests, _ := client.snapshot()
	var body struct {
		TaskType              string `json:"taskType"`
		SingleEmbeddingParams struct {
			EmbeddingPurpose   string `json:"embeddingPurpose"`
			EmbeddingDimension int    `json:"embeddingDimension"`
			Text               struct {
				Value          string `json:"value"`
				TruncationMode string `json:"truncationMode"`
			} `json:"text"`
		} `json:"singleEmbeddingParams"`
	}
	if err := json.Unmarshal(requests[0].Body, &body); err != nil {
		t.Fatal(err)
	}
	if body.TaskType != "SINGLE_EMBEDDING" || body.SingleEmbeddingParams.EmbeddingPurpose != "CLASSIFICATION" ||
		body.SingleEmbeddingParams.EmbeddingDimension != 384 || body.SingleEmbeddingParams.Text.Value != "nova" ||
		body.SingleEmbeddingParams.Text.TruncationMode != "START" {
		t.Fatalf("unexpected Nova body: %#v", body)
	}

	truncate := true
	_, err = model.Embed(context.Background(), []string{"doc"}, embeddings.InputTypeDocument, embeddings.Settings{Truncate: &truncate})
	if err != nil {
		t.Fatal(err)
	}
	requests, _ = client.snapshot()
	if err := json.Unmarshal(requests[len(requests)-1].Body, &body); err != nil {
		t.Fatal(err)
	}
	if body.SingleEmbeddingParams.EmbeddingPurpose != "GENERIC_INDEX" ||
		body.SingleEmbeddingParams.Text.TruncationMode != "END" {
		t.Fatalf("unexpected Nova defaults: %#v", body)
	}
}

func TestSettingsValidation(t *testing.T) {
	zero, negative := 0, -1
	tests := []bedrock.Settings{
		{CohereMaxTokens: &zero}, {MaxConcurrency: &zero},
		{CohereInputType: "invalid"}, {CohereTruncate: "invalid"},
		{NovaTruncate: "invalid"}, {NovaPurpose: "invalid"},
		{Common: embeddings.Settings{Dimensions: &negative}},
		{Common: embeddings.Settings{ExtraBody: map[string]any{"bedrock_titan_normalize": true}}, TitanNormalize: boolPointer(true)},
	}
	for _, settings := range tests {
		if _, err := settings.Build(); err == nil {
			t.Fatalf("invalid settings succeeded: %#v", settings)
		}
	}
	built, err := (bedrock.Settings{}).Build()
	if err != nil || built.ExtraBody != nil {
		t.Fatalf("unexpected empty settings: %#v %v", built, err)
	}
	built, err = (bedrock.Settings{CohereMaxTokens: &negative}).Build()
	if err == nil || built.ExtraBody != nil {
		t.Fatalf("unexpected negative settings: %#v %v", built, err)
	}
}

func TestAWSClient(t *testing.T) {
	var status int
	var receivedHeader string
	var tokenHeader = "7"
	server := httptest.NewTLSServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if status != 0 {
			writer.Header().Set("Retry-After", "1")
			http.Error(writer, "failed", status)
			return
		}
		receivedHeader = request.Header.Get("X-Test")
		writer.Header().Set("Content-Type", "application/json")
		writer.Header().Set("x-amzn-bedrock-input-token-count", tokenHeader)
		_, _ = writer.Write([]byte(`{"embedding":[1,2]}`))
	}))
	defer server.Close()
	baseEndpoint := server.URL
	config := aws.Config{
		Region: "us-east-1", BaseEndpoint: &baseEndpoint, HTTPClient: server.Client(),
		Credentials: credentials.NewStaticCredentialsProvider("key", "secret", ""),
	}
	configOption := bedrock.WithAWSConfig(config)
	baseEndpoint = "://mutated"
	model, err := bedrock.NewModel("amazon.titan-embed-text-v2:0", configOption)
	if err != nil {
		t.Fatal(err)
	}
	result, err := model.Embed(context.Background(), []string{"text"}, embeddings.InputTypeQuery, embeddings.Settings{
		ExtraHeaders: map[string]string{"X-Test": "request-value"},
	})
	if err != nil || result.Usage.InputTokens != 7 || model.ProviderURL() != server.URL || receivedHeader != "request-value" {
		t.Fatalf("unexpected AWS result: %#v %v", result, err)
	}
	tokenHeader = "invalid"
	result, err = model.Embed(context.Background(), []string{"text"}, embeddings.InputTypeQuery, embeddings.Settings{})
	if err != nil || result.Usage.InputTokens != 0 {
		t.Fatalf("unexpected invalid token header result: %#v %v", result, err)
	}
	status = http.StatusTooManyRequests
	_, err = model.Embed(context.Background(), []string{"text"}, embeddings.InputTypeQuery, embeddings.Settings{})
	var apiError *bedrock.APIError
	if !errors.As(err, &apiError) || apiError.StatusCode != http.StatusTooManyRequests ||
		apiError.Headers.Get("Retry-After") != "1" || !errors.Is(apiError, apiError.Err) ||
		!strings.Contains(err.Error(), "HTTP 429") {
		t.Fatalf("unexpected AWS HTTP error: %v", err)
	}

	regionModel, err := bedrock.NewModel("amazon.titan-embed-text-v2:0", bedrock.WithAWSConfig(aws.Config{Region: "eu-west-1"}))
	if err != nil || regionModel.ProviderURL() != "https://bedrock-runtime.eu-west-1.amazonaws.com" {
		t.Fatalf("unexpected region provider URL: %q %v", regionModel.ProviderURL(), err)
	}
	chinaModel, err := bedrock.NewModel("amazon.titan-embed-text-v2:0", bedrock.WithAWSConfig(aws.Config{Region: "cn-north-1"}))
	if err != nil || chinaModel.ProviderURL() != "https://bedrock-runtime.cn-north-1.amazonaws.com.cn" {
		t.Fatalf("unexpected China provider URL: %q %v", chinaModel.ProviderURL(), err)
	}
	emptyModel, err := bedrock.NewModel("amazon.titan-embed-text-v2:0", bedrock.WithAWSConfig(aws.Config{}))
	if err != nil || emptyModel.ProviderURL() != "" {
		t.Fatalf("unexpected empty provider URL: %q %v", emptyModel.ProviderURL(), err)
	}
	invalidModel, err := bedrock.NewModel("amazon.titan-embed-text-v2:0", bedrock.WithAWSConfig(aws.Config{
		Region: "us-east-1", BaseEndpoint: aws.String("://invalid"),
		Credentials: credentials.NewStaticCredentialsProvider("key", "secret", ""),
	}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := invalidModel.Embed(
		context.Background(), []string{"text"}, embeddings.InputTypeQuery, embeddings.Settings{},
	); err == nil || strings.Contains(err.Error(), "HTTP ") {
		t.Fatalf("unexpected non-HTTP AWS error: %v", err)
	}
}

func TestDefaultAWSConfig(t *testing.T) {
	server := httptest.NewTLSServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"embedding":[1]}`))
	}))
	defer server.Close()
	model, err := bedrock.NewModel("amazon.titan-embed-text-v2:0", bedrock.WithAWSLoadOptions(
		awsconfig.WithRegion("us-east-1"),
		awsconfig.WithBaseEndpoint(server.URL),
		awsconfig.WithHTTPClient(server.Client()),
		awsconfig.WithCredentialsProvider(credentials.NewStaticCredentialsProvider("key", "secret", "")),
	))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Embed(context.Background(), []string{"text"}, embeddings.InputTypeQuery, embeddings.Settings{}); err != nil {
		t.Fatal(err)
	}
	if model.ProviderURL() != server.URL {
		t.Fatalf("unexpected default provider URL: %q", model.ProviderURL())
	}

	t.Setenv("AWS_PROFILE", "profile-that-does-not-exist")
	t.Setenv("AWS_CONFIG_FILE", t.TempDir()+"/missing")
	failed, err := bedrock.NewModel("amazon.titan-embed-text-v2:0")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := failed.Embed(context.Background(), []string{"text"}, embeddings.InputTypeQuery, embeddings.Settings{}); err == nil {
		t.Fatal("invalid default AWS configuration succeeded")
	}
}

func TestRequestAndResponseErrors(t *testing.T) {
	if _, err := bedrock.NewModel("unsupported"); err == nil {
		t.Fatal("unsupported model succeeded")
	}
	var typedNil *fakeClient
	for _, client := range []bedrock.Client{nil, typedNil} {
		if panicValue := capturePanic(func() { bedrock.WithClient(client) }); panicValue == nil {
			t.Fatal("nil client did not panic")
		}
	}
	if _, err := bedrock.NewModel("amazon.titan-embed-text-v2:0", bedrock.WithClient(valueClient{})); err != nil {
		t.Fatal(err)
	}
	client := &fakeClient{}
	model, err := bedrock.NewModel("amazon.titan-embed-text-v2:0", bedrock.WithClient(client))
	if err != nil {
		t.Fatal(err)
	}
	invalidProviderSettings := []map[string]any{
		{"bedrock_cohere_max_tokens": "invalid"}, {"bedrock_cohere_max_tokens": 0},
		{"bedrock_cohere_input_type": 1}, {"bedrock_cohere_input_type": "invalid"},
		{"bedrock_cohere_truncate": 1}, {"bedrock_cohere_truncate": "invalid"},
		{"bedrock_nova_truncate": 1}, {"bedrock_nova_truncate": "invalid"},
		{"bedrock_nova_embedding_purpose": 1}, {"bedrock_nova_embedding_purpose": "invalid"},
		{"bedrock_inference_profile": 1},
	}
	for _, extraBody := range invalidProviderSettings {
		if _, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery,
			embeddings.Settings{ExtraBody: extraBody}); err == nil {
			t.Fatalf("invalid provider setting succeeded: %#v", extraBody)
		}
	}
	for _, call := range []func() error{
		func() error {
			_, err := model.Embed(context.Background(), nil, embeddings.InputTypeQuery, embeddings.Settings{})
			return err
		},
		func() error {
			_, err := model.Embed(context.Background(), []string{"x"}, "invalid", embeddings.Settings{})
			return err
		},
		func() error {
			zero := 0
			_, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{Dimensions: &zero})
			return err
		},
		func() error {
			_, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery,
				embeddings.Settings{ExtraBody: map[string]any{"bedrock_max_concurrency": 0}})
			return err
		},
		func() error {
			_, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery,
				embeddings.Settings{ExtraBody: map[string]any{"bedrock_titan_normalize": "invalid"}})
			return err
		},
		func() error {
			_, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery,
				embeddings.Settings{ExtraBody: map[string]any{"inputText": "conflict"}})
			return err
		},
		func() error {
			_, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery,
				embeddings.Settings{ExtraBody: map[string]any{"invalid": make(chan int)}})
			return err
		},
	} {
		if err := call(); err == nil {
			t.Fatal("invalid request succeeded")
		}
	}

	for _, response := range []string{"not json", `{}`, `{"embedding":null}`} {
		client.response = response
		if _, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{}); err == nil {
			t.Fatalf("invalid response succeeded: %s", response)
		}
	}
	cohere, err := bedrock.NewModel("cohere.embed-v4:0", bedrock.WithClient(client))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cohere.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{
		ExtraBody: map[string]any{"bedrock_cohere_input_type": "classification", "texts": []string{"conflict"}},
	}); err == nil {
		t.Fatal("conflicting Cohere request succeeded")
	}
	for _, response := range []string{"not json", `{}`, `{"embeddings":42}`, `{"embeddings":{}}`} {
		client.response = response
		if _, err := cohere.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{}); err == nil {
			t.Fatalf("invalid Cohere response succeeded: %s", response)
		}
	}
	client.response = `{"embeddings":[[1]]}`
	if _, err := cohere.Embed(context.Background(), []string{"x", "y"}, embeddings.InputTypeQuery, embeddings.Settings{}); err == nil {
		t.Fatal("mismatched Cohere response succeeded")
	}
	client.response = `{"embeddings":[[1]]}`
	client.err = errors.New("invoke failed")
	if _, err := cohere.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{}); !errors.Is(err, client.err) {
		t.Fatalf("Cohere invoke error was not preserved: %v", err)
	}

	client.response = `{"embedding":[1],"extra":true}`
	if _, err := model.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{}); !errors.Is(err, client.err) {
		t.Fatalf("invoke error was not preserved: %v", err)
	}

	client.err = nil
	unknown, err := bedrock.NewModel("amazon.titan-embed-future", bedrock.WithClient(client))
	if err != nil {
		t.Fatal(err)
	}
	limit, known, err := unknown.MaxInputTokens(context.Background())
	if err != nil || known || limit != 0 {
		t.Fatalf("unexpected unknown limit: %d %t %v", limit, known, err)
	}
	if _, err := unknown.Embed(context.Background(), []string{"future"}, embeddings.InputTypeQuery, embeddings.Settings{}); err != nil {
		t.Fatal(err)
	}

	nova, err := bedrock.NewModel("amazon.nova-2-multimodal-embeddings-v1:0", bedrock.WithClient(client))
	if err != nil {
		t.Fatal(err)
	}
	for _, response := range []string{"not json", `{}`, `{"embeddings":[{}]}`} {
		client.response = response
		if _, err := nova.Embed(context.Background(), []string{"x"}, embeddings.InputTypeQuery, embeddings.Settings{}); err == nil {
			t.Fatalf("invalid Nova response succeeded: %s", response)
		}
	}

	client.response = ""
	client.err = errors.New("stop")
	maxConcurrency := 1
	cancelSettings, err := (bedrock.Settings{MaxConcurrency: &maxConcurrency}).Build()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Embed(context.Background(), []string{"a", "b", "c"}, embeddings.InputTypeQuery, cancelSettings); err == nil {
		t.Fatal("concurrent invocation error was ignored")
	}
}

func boolPointer(value bool) *bool { return &value }

func capturePanic(function func()) (value any) {
	defer func() { value = recover() }()
	function()
	return nil
}

type fakeClient struct {
	mutex        sync.Mutex
	requests     []bedrock.Request
	active       int
	maxActive    int
	delay        time.Duration
	cohereByType bool
	response     string
	err          error
}

func (client *fakeClient) InvokeModel(ctx context.Context, request bedrock.Request) (bedrock.Response, error) {
	client.mutex.Lock()
	client.requests = append(client.requests, bedrock.Request{
		ModelID: request.ModelID, Body: slices.Clone(request.Body), Headers: maps.Clone(request.Headers),
	})
	client.active++
	client.maxActive = max(client.maxActive, client.active)
	delay, response, invokeErr, byType := client.delay, client.response, client.err, client.cohereByType
	client.mutex.Unlock()
	defer func() {
		client.mutex.Lock()
		client.active--
		client.mutex.Unlock()
	}()
	if delay > 0 {
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return bedrock.Response{}, ctx.Err()
		}
	}
	if invokeErr != nil {
		return bedrock.Response{}, invokeErr
	}
	if response != "" {
		return bedrock.Response{Body: []byte(response), InputTokens: 2}, nil
	}
	var body map[string]any
	if err := json.Unmarshal(request.Body, &body); err != nil {
		return bedrock.Response{}, err
	}
	switch {
	case body["inputText"] != nil:
		text := body["inputText"].(string)
		response = fmt.Sprintf(`{"embedding":[%d]}`, len(text))
	case body["texts"] != nil:
		texts := body["texts"].([]any)
		vectors := make([][]float64, len(texts))
		for index, text := range texts {
			vectors[index] = []float64{float64(len(text.(string)))}
		}
		encoded, _ := json.Marshal(vectors)
		if byType {
			response = fmt.Sprintf(`{"embeddings":{"float":%s},"id":"cohere-id"}`, encoded)
		} else {
			response = fmt.Sprintf(`{"embeddings":%s,"id":"cohere-id"}`, encoded)
		}
	case body["taskType"] == "SINGLE_EMBEDDING":
		params := body["singleEmbeddingParams"].(map[string]any)
		text := params["text"].(map[string]any)["value"].(string)
		response = fmt.Sprintf(`{"embeddings":[{"embeddingType":"TEXT","embedding":[%d]}]}`, len(text))
	default:
		return bedrock.Response{}, fmt.Errorf("unexpected body: %#v", body)
	}
	return bedrock.Response{Body: []byte(response), InputTokens: 2}, nil
}

func (client *fakeClient) snapshot() ([]bedrock.Request, int) {
	client.mutex.Lock()
	defer client.mutex.Unlock()
	return slices.Clone(client.requests), client.maxActive
}

type valueClient struct{}

func (valueClient) InvokeModel(context.Context, bedrock.Request) (bedrock.Response, error) {
	return bedrock.Response{}, nil
}

var _ bedrock.Client = (*fakeClient)(nil)
var _ bedrock.Client = valueClient{}
