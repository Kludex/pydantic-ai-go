package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	smithyhttp "github.com/aws/smithy-go/transport/http"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/anthropic"
)

type valueLegacyBedrockClient struct{}

func (valueLegacyBedrockClient) InvokeModel(
	context.Context, *bedrockruntime.InvokeModelInput, ...func(*bedrockruntime.Options),
) (*bedrockruntime.InvokeModelOutput, error) {
	return &bedrockruntime.InvokeModelOutput{Body: []byte(`{"model":"model","content":[],"usage":{}}`)}, nil
}

func (valueLegacyBedrockClient) CountTokens(
	context.Context, *bedrockruntime.CountTokensInput, ...func(*bedrockruntime.Options),
) (*bedrockruntime.CountTokensOutput, error) {
	return &bedrockruntime.CountTokensOutput{InputTokens: aws.Int32(1)}, nil
}

type legacyBedrockClient struct {
	invoke func(*bedrockruntime.InvokeModelInput) (*bedrockruntime.InvokeModelOutput, error)
	count  func(*bedrockruntime.CountTokensInput) (*bedrockruntime.CountTokensOutput, error)
}

func (client *legacyBedrockClient) InvokeModel(
	_ context.Context, input *bedrockruntime.InvokeModelInput, options ...func(*bedrockruntime.Options),
) (*bedrockruntime.InvokeModelOutput, error) {
	for _, option := range options {
		option(&bedrockruntime.Options{})
	}
	return client.invoke(input)
}

func (client *legacyBedrockClient) CountTokens(
	_ context.Context, input *bedrockruntime.CountTokensInput, options ...func(*bedrockruntime.Options),
) (*bedrockruntime.CountTokensOutput, error) {
	for _, option := range options {
		option(&bedrockruntime.Options{})
	}
	return client.count(input)
}

func TestLegacyBedrockRequestAndTokenCount(t *testing.T) {
	var requestBody map[string]any
	client := &legacyBedrockClient{}
	client.invoke = func(input *bedrockruntime.InvokeModelInput) (*bedrockruntime.InvokeModelOutput, error) {
		if *input.ModelId != "us.anthropic.claude-sonnet-4-v1:0" || *input.ContentType != "application/json" ||
			*input.Accept != "application/json" {
			t.Fatalf("unexpected InvokeModel input: %#v", input)
		}
		if err := json.Unmarshal(input.Body, &requestBody); err != nil {
			t.Fatal(err)
		}
		return &bedrockruntime.InvokeModelOutput{Body: []byte(`{
			"id":"message-1","model":"claude-sonnet-4","stop_reason":"end_turn",
			"content":[{"type":"text","text":"hello"}],
			"usage":{"input_tokens":3,"output_tokens":2}
		}`)}, nil
	}
	client.count = func(input *bedrockruntime.CountTokensInput) (*bedrockruntime.CountTokensOutput, error) {
		counted, ok := input.Input.(*types.CountTokensInputMemberInvokeModel)
		if !ok || *input.ModelId != "us.anthropic.claude-sonnet-4-v1:0" {
			t.Fatalf("unexpected CountTokens input: %#v", input)
		}
		var body map[string]any
		if err := json.Unmarshal(counted.Value.Body, &body); err != nil {
			t.Fatal(err)
		}
		if body["anthropic_version"] != "bedrock-2023-05-31" || body["model"] != nil {
			t.Fatalf("unexpected token-count body: %#v", body)
		}
		return &bedrockruntime.CountTokensOutput{InputTokens: aws.Int32(17)}, nil
	}
	model := anthropic.NewLegacyBedrockModel(
		"us.anthropic.claude-sonnet-4-v1:0",
		anthropic.LegacyBedrockConfig{Client: client, ProviderURL: "https://bedrock.example"},
	)
	if model.ProviderName() != "anthropic" || model.ProviderURL() != "https://bedrock.example" {
		t.Fatalf("unexpected identity: %q %q", model.ProviderName(), model.ProviderURL())
	}
	cacheSettings, err := (anthropic.Settings{CacheInstructions: anthropic.CacheTTL1Hour}).Build()
	if err != nil {
		t.Fatal(err)
	}
	response, err := model.Request(context.Background(), []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}}},
	}, ai.ModelRequestParams{Instructions: "system", Settings: ai.ModelSettings{
		ExtraBody: cacheSettings.ExtraBody, ExtraHeaders: map[string]string{"x-test": "value"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if response.Text() != "hello" || response.Usage.InputTokens != 3 || response.ProviderURL != "https://bedrock.example" {
		t.Fatalf("unexpected response: %#v", response)
	}
	if requestBody["model"] != nil || requestBody["anthropic_version"] != "bedrock-2023-05-31" ||
		requestBody["max_tokens"] != float64(16384) {
		t.Fatalf("unexpected request body: %#v", requestBody)
	}
	usage, err := model.CountTokens(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil || usage.InputTokens != 17 {
		t.Fatalf("unexpected usage: %#v err=%v", usage, err)
	}

	sequence, err := model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for event, eventErr := range sequence {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		finish := event.(ai.FinishEvent)
		if len(finish.Parts) != 1 || finish.ProviderURL != "https://bedrock.example" {
			t.Fatalf("unexpected fallback stream event: %#v", finish)
		}
	}
}

func TestLegacyBedrockToolSearchProfile(t *testing.T) {
	var body map[string]any
	client := &legacyBedrockClient{invoke: func(input *bedrockruntime.InvokeModelInput) (*bedrockruntime.InvokeModelOutput, error) {
		if err := json.Unmarshal(input.Body, &body); err != nil {
			t.Fatal(err)
		}
		return &bedrockruntime.InvokeModelOutput{Body: []byte(`{"model":"model","content":[],"usage":{}}`)}, nil
	}}
	model := anthropic.NewLegacyBedrockModel("claude-sonnet-4-6", anthropic.LegacyBedrockConfig{Client: client})
	if model.SupportsToolSearchStrategy(ai.ToolSearchStrategyBM25) ||
		!model.SupportsToolSearchStrategy(ai.ToolSearchStrategyRegex) ||
		model.SupportsToolSearchStrategy(ai.ToolSearchStrategyKeywords) {
		t.Fatal("unexpected legacy Bedrock tool-search support")
	}
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	params := ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{
			Name: ai.ToolSearchName, Schema: schema, ToolKind: ai.ToolPartKindToolSearch,
			ToolSearchStrategy: ai.ToolSearchStrategyAuto,
		}},
		DeferredTools: []ai.ToolDefinition{{Name: "hidden", Schema: schema, DeferLoading: true}},
	}
	if _, err := model.Request(context.Background(), nil, params); err != nil {
		t.Fatal(err)
	}
	tools := body["tools"].([]any)
	if len(tools) != 2 || tools[1].(map[string]any)["type"] != "tool_search_tool_regex_20251119" {
		t.Fatalf("legacy Bedrock did not default to regex: %#v", tools)
	}
	params.Tools[0].ToolSearchStrategy = ai.ToolSearchStrategyBM25
	_, err := model.Request(context.Background(), nil, params)
	if err == nil || !strings.Contains(err.Error(), `strategy "bm25" is not supported by legacy Bedrock`) {
		t.Fatalf("unexpected BM25 error: %v", err)
	}
}

func TestLegacyBedrockStructuredOutput(t *testing.T) {
	strict := true
	schema := map[string]any{"type": "object"}
	params := ai.ModelRequestParams{
		Tools:      []ai.ToolDefinition{{Name: "add", Schema: schema, Strict: &strict}},
		OutputMode: ai.OutputModeNative, OutputSchema: schema,
	}
	for name, served := range map[string]bool{
		"global.anthropic.claude-sonnet-4-6": true, "us.anthropic.claude-opus-4-7": false,
		"global.anthropic.claude-sonnet-5": false, "anthropic.claude-3-5-haiku-20241022-v1:0": false,
	} {
		var body map[string]any
		client := &legacyBedrockClient{invoke: func(input *bedrockruntime.InvokeModelInput) (*bedrockruntime.InvokeModelOutput, error) {
			if err := json.Unmarshal(input.Body, &body); err != nil {
				t.Fatal(err)
			}
			return &bedrockruntime.InvokeModelOutput{Body: []byte(`{"model":"model","content":[],"usage":{}}`)}, nil
		}}
		model := anthropic.NewLegacyBedrockModel(name, anthropic.LegacyBedrockConfig{Client: client})
		_, err := model.Request(context.Background(), nil, params)
		if served != (err == nil) {
			t.Fatalf("%s: unexpected native output error: %v", name, err)
		}
		textOnly := params
		textOnly.OutputSchema = nil
		if _, err := model.Request(context.Background(), nil, textOnly); err != nil {
			t.Fatal(err)
		}
		if sent := body["tools"].([]any)[0].(map[string]any)["strict"] == true; sent != served {
			t.Fatalf("%s: unexpected strict tools %#v", name, body["tools"])
		}
	}
	overridden := anthropic.NewLegacyBedrockModel("anthropic.claude-sonnet-5", anthropic.LegacyBedrockConfig{
		Client: &legacyBedrockClient{invoke: func(input *bedrockruntime.InvokeModelInput) (*bedrockruntime.InvokeModelOutput, error) {
			if !strings.Contains(string(input.Body), `"strict":true`) {
				t.Fatalf("WithStrictToolSupport did not override the Bedrock default: %s", input.Body)
			}
			return &bedrockruntime.InvokeModelOutput{Body: []byte(`{"model":"model","content":[],"usage":{}}`)}, nil
		}},
	}, anthropic.WithStrictToolSupport(true))
	if _, err := overridden.Request(context.Background(), nil, ai.ModelRequestParams{Tools: params.Tools}); err != nil {
		t.Fatal(err)
	}
}

func TestLegacyBedrockAWSClientHeaders(t *testing.T) {
	server := httptest.NewTLSServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("x-custom") != "present" || request.Header.Get("anthropic-beta") != "" {
			t.Errorf("unexpected headers: %v", request.Header)
		}
		if strings.HasSuffix(request.URL.Path, "/converse-stream") || strings.HasSuffix(request.URL.Path, "/invoke-with-response-stream") {
			response.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
			return
		}
		response.Header().Set("Content-Type", "application/json")
		if strings.HasSuffix(request.URL.Path, "/count-tokens") {
			_, _ = response.Write([]byte(`{"inputTokens":5}`))
			return
		}
		_, _ = response.Write([]byte(`{"model":"model","content":[],"usage":{}}`))
	}))
	defer server.Close()
	client := bedrockruntime.NewFromConfig(aws.Config{
		Region: "us-east-1", BaseEndpoint: aws.String(server.URL), HTTPClient: server.Client(),
		Credentials: credentials.NewStaticCredentialsProvider("key", "secret", ""),
	})
	model := anthropic.NewLegacyBedrockModel("model", anthropic.LegacyBedrockConfig{
		Client: client, ProviderURL: server.URL,
	})
	params := ai.ModelRequestParams{Settings: ai.ModelSettings{ExtraHeaders: map[string]string{
		"x-custom": "present", "anthropic-beta": "custom-beta",
	}}}
	_, err := model.Request(context.Background(), nil, params)
	if err != nil {
		t.Fatal(err)
	}
	usage, err := model.CountTokens(context.Background(), nil, params)
	if err != nil || usage.InputTokens != 5 {
		t.Fatalf("unexpected AWS token count: %#v err=%v", usage, err)
	}
	sequence, err := model.StreamRequest(context.Background(), nil, params)
	if err != nil {
		t.Fatal(err)
	}
	var streamErr error
	for _, eventErr := range sequence {
		streamErr = eventErr
	}
	if streamErr == nil || !strings.Contains(streamErr.Error(), "without message_stop") {
		t.Fatalf("unexpected empty AWS stream error: %v", streamErr)
	}

	failureServer := httptest.NewTLSServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.WriteHeader(http.StatusInternalServerError)
	}))
	defer failureServer.Close()
	failureClient := bedrockruntime.NewFromConfig(aws.Config{
		Region: "us-east-1", BaseEndpoint: aws.String(failureServer.URL), HTTPClient: failureServer.Client(),
		Credentials: credentials.NewStaticCredentialsProvider("key", "secret", ""),
	})
	failureModel := anthropic.NewLegacyBedrockModel("model", anthropic.LegacyBedrockConfig{Client: failureClient})
	if _, err := failureModel.StreamRequest(context.Background(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected AWS stream-open error")
	}

	var bodies []map[string]any
	staleServer := httptest.NewTLSServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		response.Header().Set("Content-Type", "application/json")
		if len(bodies) == 1 {
			response.Header().Set("X-Amzn-ErrorType", "ValidationException")
			response.WriteHeader(http.StatusBadRequest)
			_, _ = response.Write([]byte(`{"message":"messages.1.content.0: The block is bound to a different conversation"}`))
			return
		}
		_, _ = response.Write([]byte(`{"content":[],"usage":{}}`))
	}))
	defer staleServer.Close()
	staleModel := anthropic.NewLegacyBedrockModel("us.anthropic.claude-fable-5-1-v1:0", anthropic.LegacyBedrockConfig{
		Client: bedrockruntime.NewFromConfig(aws.Config{
			Region: "us-east-1", BaseEndpoint: aws.String(staleServer.URL), HTTPClient: staleServer.Client(),
			Credentials: credentials.NewStaticCredentialsProvider("key", "secret", ""),
		}),
	})
	if _, err := staleModel.Request(context.Background(), nil, params); err != nil {
		t.Fatal(err)
	}
	if len(bodies) != 2 || bodies[1]["thinking"] == nil ||
		!reflect.DeepEqual(bodies[1]["anthropic_beta"], []any{"custom-beta", "thinking-binding-controls-2026-08-01"}) {
		t.Fatalf("unexpected AWS stale-thinking recovery: %#v", bodies)
	}
}

func TestLegacyBedrockErrors(t *testing.T) {
	assertPanic(t, func() {
		anthropic.NewLegacyBedrockModel("model", anthropic.LegacyBedrockConfig{})
	})
	var typedNil *legacyBedrockClient
	assertPanic(t, func() {
		anthropic.NewLegacyBedrockModel("model", anthropic.LegacyBedrockConfig{Client: typedNil})
	})
	if model := anthropic.NewLegacyBedrockModel(
		"model", anthropic.LegacyBedrockConfig{Client: valueLegacyBedrockClient{}},
	); model.Name() != "model" {
		t.Fatal("value client was not accepted")
	}

	tests := []struct {
		name   string
		output *bedrockruntime.InvokeModelOutput
		err    error
		match  string
	}{
		{name: "nil response", match: "response is nil"},
		{name: "transport", err: errors.New("offline"), match: "offline"},
		{name: "api", err: &smithyhttp.ResponseError{
			Response: &smithyhttp.Response{Response: &http.Response{StatusCode: http.StatusTooManyRequests}},
			Err:      errors.New("throttled"),
		}, match: "status 429"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := &legacyBedrockClient{invoke: func(*bedrockruntime.InvokeModelInput) (*bedrockruntime.InvokeModelOutput, error) {
				return test.output, test.err
			}}
			model := anthropic.NewLegacyBedrockModel("model", anthropic.LegacyBedrockConfig{Client: client})
			_, err := model.Request(context.Background(), nil, ai.ModelRequestParams{})
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}

	countTests := []struct {
		name   string
		output *bedrockruntime.CountTokensOutput
		err    error
		match  string
	}{
		{name: "nil", match: "omitted inputTokens"},
		{name: "missing", output: &bedrockruntime.CountTokensOutput{}, match: "omitted inputTokens"},
		{name: "transport", err: errors.New("count offline"), match: "count offline"},
	}
	for _, test := range countTests {
		t.Run("count "+test.name, func(t *testing.T) {
			client := &legacyBedrockClient{count: func(*bedrockruntime.CountTokensInput) (*bedrockruntime.CountTokensOutput, error) {
				return test.output, test.err
			}}
			model := anthropic.NewLegacyBedrockModel("model", anthropic.LegacyBedrockConfig{Client: client})
			_, err := model.CountTokens(context.Background(), nil, ai.ModelRequestParams{})
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}

	invalidBody := ai.ModelRequestParams{Settings: ai.ModelSettings{ExtraBody: map[string]any{"bad": make(chan int)}}}
	client := &legacyBedrockClient{invoke: func(*bedrockruntime.InvokeModelInput) (*bedrockruntime.InvokeModelOutput, error) {
		return nil, errors.New("fallback failed")
	}}
	model := anthropic.NewLegacyBedrockModel("model", anthropic.LegacyBedrockConfig{Client: client})
	if _, err := model.Request(context.Background(), nil, invalidBody); err == nil ||
		!strings.Contains(err.Error(), "marshal legacy Bedrock request") {
		t.Fatalf("unexpected request marshal error: %v", err)
	}
	if _, err := model.CountTokens(context.Background(), nil, invalidBody); err == nil ||
		!strings.Contains(err.Error(), "marshal legacy Bedrock request") {
		t.Fatalf("unexpected count marshal error: %v", err)
	}
	if _, err := model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected static fallback stream error")
	}
}

func assertPanic(t *testing.T, function func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	function()
}

func TestLegacyBedrockStaleThinkingRecovery(t *testing.T) {
	stale := &smithyhttp.ResponseError{
		Response: &smithyhttp.Response{Response: &http.Response{StatusCode: http.StatusBadRequest}},
		Err:      errors.New("ValidationException: The block is bound to a different conversation"),
	}
	params := ai.ModelRequestParams{Settings: ai.ModelSettings{ExtraHeaders: map[string]string{
		"Anthropic-Beta": "custom-beta",
	}}}
	for _, test := range []struct {
		model   string
		retries bool
	}{
		{model: "us.anthropic.claude-fable-5-1-v1:0", retries: true},
		{model: "anthropic.claude-opus-5-5", retries: true},
		{model: "us.anthropic.claude-sonnet-4-6"},
	} {
		t.Run(test.model, func(t *testing.T) {
			var bodies []map[string]any
			fail := func(raw []byte) error {
				var body map[string]any
				if err := json.Unmarshal(raw, &body); err != nil {
					t.Fatal(err)
				}
				bodies = append(bodies, body)
				if test.retries && len(bodies)%2 == 0 {
					return nil
				}
				return stale
			}
			client := &streamingLegacyBedrockClient{legacyBedrockClient: &legacyBedrockClient{
				invoke: func(input *bedrockruntime.InvokeModelInput) (*bedrockruntime.InvokeModelOutput, error) {
					if err := fail(input.Body); err != nil {
						return nil, err
					}
					return &bedrockruntime.InvokeModelOutput{Body: []byte(`{"content":[],"usage":{}}`)}, nil
				},
				count: func(input *bedrockruntime.CountTokensInput) (*bedrockruntime.CountTokensOutput, error) {
					if err := fail(input.Input.(*types.CountTokensInputMemberInvokeModel).Value.Body); err != nil {
						return nil, err
					}
					return &bedrockruntime.CountTokensOutput{InputTokens: aws.Int32(1)}, nil
				},
			}, stream: func(input *bedrockruntime.InvokeModelWithResponseStreamInput) (anthropic.LegacyBedrockEventStream, error) {
				if err := fail(input.Body); err != nil {
					return nil, err
				}
				return newLegacyEventStream(legacyChunk(`{"type":"message_stop"}`)), nil
			}}
			model := anthropic.NewLegacyBedrockModel(test.model, anthropic.LegacyBedrockConfig{Client: client})
			_, requestErr := model.Request(t.Context(), nil, params)
			_, countErr := model.CountTokens(t.Context(), nil, params)
			_, streamErr := model.StreamRequest(t.Context(), nil, params)
			for _, err := range []error{requestErr, countErr, streamErr} {
				var apiErr *anthropic.APIError
				if test.retries != (err == nil) || !test.retries && (!errors.As(err, &apiErr) || apiErr.StatusCode != 400) {
					t.Fatalf("unexpected stale-thinking result: %v", err)
				}
			}
			if !test.retries {
				if len(bodies) != 3 || bodies[0]["thinking"] != nil ||
					!reflect.DeepEqual(bodies[0]["anthropic_beta"], []any{"custom-beta"}) {
					t.Fatalf("unexpected requests without a binding model: %#v", bodies)
				}
				return
			}
			for index, body := range bodies {
				thinking, _ := body["thinking"].(map[string]any)
				betas := []any{"custom-beta"}
				if index%2 == 1 {
					betas = append(betas, "thinking-binding-controls-2026-08-01")
					if thinking["block_binding"].(map[string]any)["prefix_mismatch_behavior"] != "drop_block" {
						t.Fatalf("retry %d did not drop stale thinking: %#v", index, body)
					}
				}
				if len(bodies) != 6 || !reflect.DeepEqual(body["anthropic_beta"], betas) {
					t.Fatalf("unexpected legacy Bedrock body %d: %#v", index, body)
				}
			}
		})
	}
}
