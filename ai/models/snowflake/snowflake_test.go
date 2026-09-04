package snowflake_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
	"github.com/Kludex/pydantic-ai-go/ai/models/snowflake"
)

func TestClaudeRequestAndStream(t *testing.T) {
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		if request.Header.Get("Authorization") != "Bearer token" {
			t.Errorf("unexpected authorization: %q", request.Header.Get("Authorization"))
		}
		if request.Header.Get("Accept") == "text/event-stream" {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(response,
				"data: "+`{"choices":[{"delta":{"reasoning_details":[{"id":"reason","type":"reasoning.summary","index":0,"summary":"think"}],"tool_calls":[{"index":0,"id":"call","function":{"name":"work","arguments":"{}"}}]}}]}`+"\n\n"+
					"data: "+`{"choices":[{"delta":{},"finish_reason":""}],"usage":{"prompt_tokens":2,"completion_tokens":3}}`+"\n\n"+
					"data: [DONE]\n\n")
			return
		}
		_, _ = io.WriteString(response, `{
			"choices":[{"message":{"reasoning_details":[{"id":"reason","type":"reasoning.text",
			"index":0,"text":"consider","signature":"signed"}],"content":"",
			"tool_calls":[{"id":"call","type":"function","function":{"name":"work","arguments":"{}"}}]},
			"finish_reason":""}],"usage":{"prompt_tokens":4,"completion_tokens":5}}`)
	}))
	defer server.Close()
	settings, err := (snowflake.Settings{Common: ai.ModelSettings{
		Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelHigh},
	}}).Build()
	if err != nil {
		t.Fatal(err)
	}
	model := snowflake.NewModel(
		"claude-sonnet-4-6", snowflake.WithBaseURL(server.URL), snowflake.WithToken("token"),
		snowflake.WithHTTPClient(server.Client()), snowflake.WithDefaultSettings(ai.ModelSettings{MaxTokens: 10}),
	)
	params := ai.ModelRequestParams{
		Tools:     []ai.ToolDefinition{{Name: "work", Schema: map[string]any{"type": "object"}}},
		AllowText: true, Settings: settings,
	}
	result, err := model.Request(t.Context(), nil, params)
	if err != nil {
		t.Fatal(err)
	}
	if result.ModelName != "claude-sonnet-4-6" || result.FinishReason != ai.FinishReasonToolCall ||
		result.ProviderName != "snowflake" || len(result.Parts) != 2 ||
		result.Parts[0].(ai.ThinkingPart).Signature != "signed" {
		t.Fatalf("unexpected result: %#v", result)
	}
	stream, err := model.StreamRequest(t.Context(), nil, params)
	if err != nil {
		t.Fatal(err)
	}
	var finish ai.FinishEvent
	for event, eventErr := range stream {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		if value, ok := event.(ai.FinishEvent); ok {
			finish = value
		}
	}
	if finish.FinishReason != ai.FinishReasonToolCall || finish.ModelName != "claude-sonnet-4-6" ||
		finish.ProviderDetails["finish_reason"] != "tool_calls" {
		t.Fatalf("unexpected finish: %#v", finish)
	}
	for _, body := range bodies {
		reasoning := body["reasoning"].(map[string]any)
		if reasoning["effort"] != "high" || body["temperature"] != float64(1) || body["reasoning_effort"] != nil {
			t.Fatalf("unexpected Claude settings: %#v", body)
		}
		tools := body["tools"].([]any)
		function := tools[0].(map[string]any)["function"].(map[string]any)
		if _, exists := function["strict"]; exists {
			t.Fatalf("strict was sent: %#v", function)
		}
	}
	if model.Name() != "claude-sonnet-4-6" || model.ProviderName() != "snowflake" ||
		model.ProviderURL() != server.URL || model.DefaultModelSettings().MaxTokens != 10 {
		t.Fatal("unexpected model identity")
	}
}

func TestExplicitReasoningAndGateway(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("X-Gateway") != "yes" {
			t.Error("gateway preparation was omitted")
		}
		_ = json.NewDecoder(request.Body).Decode(&body)
		_, _ = io.WriteString(response, `{"model":"claude","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()
	reasoning := snowflake.Reasoning{MaxTokens: 200}
	settings, err := (snowflake.Settings{
		Common: ai.ModelSettings{Temperature: floatPointer(0.5)}, Reasoning: &reasoning,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	model := snowflake.NewModel("claude-opus-4-6", snowflake.WithProvider(openai.ProviderConfig{
		Name: "snowflake-gateway", BaseURL: server.URL, HTTPClient: server.Client(),
		PrepareRequest: func(request *http.Request) error {
			request.Header.Set("X-Gateway", "yes")
			return nil
		},
	}))
	result, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: settings})
	if err != nil || result.ProviderName != "snowflake-gateway" || body["temperature"] != 0.5 ||
		body["reasoning"].(map[string]any)["max_tokens"] != float64(200) {
		t.Fatalf("unexpected explicit request: %#v %#v %v", body, result, err)
	}
}

func TestProviderConfiguration(t *testing.T) {
	t.Setenv("SNOWFLAKE_ACCOUNT", "https://org-account.snowflakecomputing.com/")
	t.Setenv("SNOWFLAKE_TOKEN", "token")
	provider, err := snowflake.NewProviderConfig()
	if err != nil || provider.Name != "snowflake" || provider.APIKey != "token" ||
		provider.BaseURL != "https://org-account.snowflakecomputing.com/api/v2/cortex/v1" {
		t.Fatalf("unexpected provider: %#v %v", provider, err)
	}
	t.Setenv("SNOWFLAKE_BASE_URL", "https://private.example/cortex/")
	provider, err = snowflake.NewProviderConfig()
	if err != nil || provider.BaseURL != "https://private.example/cortex" {
		t.Fatalf("unexpected custom provider: %#v %v", provider, err)
	}
	model := snowflake.NewModel("openai-gpt-5", snowflake.WithAccount("org.account"), snowflake.WithToken("token"))
	if !strings.Contains(model.ProviderURL(), "org.account.snowflakecomputing.com") ||
		model.ModelProfile().DefaultOutputMode != ai.OutputModeTool {
		t.Fatalf("unexpected account model: %q %#v", model.ProviderURL(), model.ModelProfile())
	}
	other := snowflake.NewModel("mistral-large", snowflake.WithProvider(openai.ProviderConfig{
		Name: "snowflake", BaseURL: "https://example.com/v1",
	}))
	if other.ModelProfile().DefaultOutputMode != ai.OutputModePrompted {
		t.Fatalf("unexpected other-family profile: %#v", other.ModelProfile())
	}
}

func floatPointer(value float64) *float64 { return &value }

var _ ai.Model = snowflake.NewModel("openai-gpt-5")
var _ ai.StreamingModel = snowflake.NewModel("openai-gpt-5")
var _ ai.ModelProviderIdentity = snowflake.NewModel("openai-gpt-5")
