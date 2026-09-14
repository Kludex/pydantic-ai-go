package deepseek_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/deepseek"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestDeepSeekChatThinkingAndToolChoice(t *testing.T) {
	for name, test := range map[string]struct {
		model         string
		thinking      *ai.ThinkingSettings
		wantChoice    string
		wantReasoning string
	}{
		"V4 default thinking":  {model: "deepseek-v4-flash", wantChoice: "auto"},
		"V4 disabled thinking": {model: "deepseek-v4-pro", thinking: disabledThinking(), wantChoice: "required", wantReasoning: "none"},
		"reasoner":             {model: "deepseek-reasoner", wantChoice: "auto"},
		"reasoner cannot disable": {
			model: "deepseek-reasoner", thinking: disabledThinking(), wantChoice: "auto",
		},
		"chat ignores thinking": {
			model: "deepseek-chat", thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelHigh}, wantChoice: "required",
		},
	} {
		t.Run(name, func(t *testing.T) {
			var body map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
				if request.URL.Path != "/chat/completions" || request.Header.Get("Authorization") != "Bearer secret" {
					t.Errorf("unexpected request: %s headers=%v", request.URL.Path, request.Header)
				}
				if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				_, _ = io.WriteString(response, `{"id":"response","model":"`+test.model+`","choices":[{"message":{"reasoning_content":"think","content":"done"},"finish_reason":"stop"}],"usage":{}}`)
			}))
			defer server.Close()

			client := server.Client()
			client.Timeout = time.Second
			model := deepseek.NewModel(test.model, deepseek.WithAPIKey("secret"), deepseek.WithBaseURL(server.URL),
				deepseek.WithHTTPClient(client))
			response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
				OutputTool: &ai.ToolDefinition{Name: "final", Schema: map[string]any{"type": "object"}},
				Settings:   ai.ModelSettings{Thinking: test.thinking},
			})
			if err != nil {
				t.Fatal(err)
			}
			if body["tool_choice"] != test.wantChoice || body["reasoning_effort"] != valueOrNil(test.wantReasoning) {
				t.Fatalf("unexpected request body: %#v", body)
			}
			if client.Timeout != time.Second || response.ProviderName != "deepseek" {
				t.Fatalf("caller client or provider identity changed: timeout=%v response=%#v", client.Timeout, response)
			}
		})
	}
}

func TestDeepSeekChatReasoningRoundTripAndStream(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		calls++
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if calls == 2 {
			assistant := body["messages"].([]any)[0].(map[string]any)
			if assistant["reasoning_content"] != "think" {
				t.Errorf("reasoning was not replayed: %#v", assistant)
			}
		}
		if body["stream"] == true {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(response, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"stream think\",\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
			return
		}
		_, _ = io.WriteString(response, `{"model":"deepseek-v4-flash","choices":[{"message":{"reasoning_content":"think","content":"done"},"finish_reason":"stop"}],"usage":{}}`)
	}))
	defer server.Close()
	model := deepseek.NewModel("deepseek-v4-flash", deepseek.WithProvider(openai.ProviderConfig{
		Name: "gateway", BaseURL: server.URL, APIKey: "token", HTTPClient: server.Client(),
	}), deepseek.WithDefaultSettings(ai.ModelSettings{MaxTokens: 20}))
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil || len(response.Parts) != 2 || response.Parts[0].(ai.ThinkingPart).Content != "think" {
		t.Fatalf("unexpected response: %#v %v", response, err)
	}
	if _, err = model.Request(t.Context(), []ai.ModelMessage{*response}, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var sawThinking bool
	for event, streamErr := range stream {
		if streamErr != nil {
			t.Fatal(streamErr)
		}
		if delta, ok := event.(ai.ThinkingDeltaEvent); ok && delta.Delta == "stream think" {
			sawThinking = true
		}
	}
	if !sawThinking || model.DefaultModelSettings().MaxTokens != 20 ||
		model.ModelProfile().DefaultOutputMode != ai.OutputModeTool {
		t.Fatal("DeepSeek chat configuration was not retained")
	}
}

func TestDeepSeekChatRejectsNativeOutput(t *testing.T) {
	model := deepseek.NewModel("deepseek-chat", deepseek.WithBaseURL("https://example.invalid"))
	params := ai.ModelRequestParams{OutputMode: ai.OutputModeNative, OutputSchema: map[string]any{"type": "object"}}
	if _, err := model.Request(t.Context(), nil, params); err == nil {
		t.Fatal("expected native output error")
	}
	if _, err := model.StreamRequest(t.Context(), nil, params); err == nil {
		t.Fatal("expected streamed native output error")
	}
}

func TestDeepSeekResponsesNativeOutputAndToolChoice(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		calls++
		if request.URL.Path != "/responses" {
			t.Errorf("unexpected path %q", request.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		switch calls {
		case 1:
			format := body["text"].(map[string]any)["format"].(map[string]any)
			if format["type"] != "json_schema" || format["name"] != "final_result" {
				t.Errorf("native output missing: %#v", body)
			}
		case 2:
			if body["tool_choice"] != "auto" {
				t.Errorf("thinking request forced a tool: %#v", body)
			}
		default:
			if body["tool_choice"] != "required" || body["reasoning"].(map[string]any)["effort"] != "none" {
				t.Errorf("disabled thinking did not force a tool: %#v", body)
			}
		}
		_, _ = io.WriteString(response, `{"id":"response","model":"deepseek-v4-pro","status":"completed","output":[{"type":"message","id":"message","role":"assistant","content":[{"type":"output_text","text":"{\"value\":\"ok\"}"}]}],"usage":{}}`)
	}))
	defer server.Close()
	model := deepseek.NewResponsesModel("deepseek-v4-pro", deepseek.WithBaseURL(server.URL),
		deepseek.WithHTTPClient(server.Client()))
	_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		OutputMode: ai.OutputModeNative,
		OutputSchema: map[string]any{
			"type": "object", "properties": map[string]any{"value": map[string]any{"type": "string"}},
			"required": []any{"value"}, "additionalProperties": false,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, thinking := range []*ai.ThinkingSettings{nil, disabledThinking()} {
		_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{
			OutputTool: &ai.ToolDefinition{Name: "final", Schema: map[string]any{"type": "object"}},
			Settings:   ai.ModelSettings{Thinking: thinking},
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if model.SupportsNativeTool(ai.WebSearchTool{}) || model.ModelProfile().SupportsImageOutput {
		t.Fatal("DeepSeek Responses advertised unsupported native output")
	}
}

func TestDeepSeekResponsesGroupsInterleavedFunctionCalls(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		_, _ = io.WriteString(response, `{"id":"response","model":"deepseek-v4-flash","status":"completed","output":[],"usage":{}}`)
	}))
	defer server.Close()
	first := ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.ThinkingPart{Content: "first", ID: "reasoning-a", ProviderName: "deepseek"},
		ai.ToolCallPart{ToolName: "read", ToolCallID: "call-a", Args: []byte(`{"path":"a"}`)},
		ai.ThinkingPart{Content: "second", ID: "reasoning-b", ProviderName: "deepseek"},
		ai.ToolCallPart{ToolName: "view", ToolCallID: "call-b", Args: []byte(`{"path":"b"}`)},
	}}
	history := []ai.ModelMessage{first, ai.ModelRequest{Parts: []ai.RequestPart{
		ai.ToolReturnPart{ToolName: "read", ToolCallID: "call-a", Content: "a"},
		ai.ToolReturnPart{ToolName: "view", ToolCallID: "call-b", Content: "b"},
	}}}
	model := deepseek.NewResponsesModel("deepseek-v4-flash", deepseek.WithBaseURL(server.URL),
		deepseek.WithHTTPClient(server.Client()))
	if _, err := model.Request(t.Context(), history, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	input := body["input"].([]any)
	firstReasoning := input[0].(map[string]any)
	if firstReasoning["type"] != "reasoning" ||
		firstReasoning["content"].([]any)[0].(map[string]any)["text"] != "first" ||
		input[1].(map[string]any)["type"] != "reasoning" || input[2].(map[string]any)["type"] != "function_call" ||
		input[3].(map[string]any)["type"] != "function_call" ||
		input[4].(map[string]any)["type"] != "function_call_output" {
		t.Fatalf("function calls were not grouped: %#v", input)
	}
	if len(first.Parts) != 4 || first.Parts[1].(ai.ToolCallPart).ToolName != "read" {
		t.Fatal("history was mutated")
	}
}

func TestDeepSeekResponsesGroupingBoundaries(t *testing.T) {
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		bodies = append(bodies, body)
		_, _ = io.WriteString(response, `{"id":"response","model":"deepseek-v4-flash","status":"completed","output":[],"usage":{}}`)
	}))
	defer server.Close()
	model := deepseek.NewResponsesModel("deepseek-v4-flash", deepseek.WithBaseURL(server.URL),
		deepseek.WithHTTPClient(server.Client()))
	call := ai.ToolCallPart{ToolName: "read", ToolCallID: "call", Args: []byte(`{}`)}
	for _, history := range [][]ai.ModelMessage{
		{ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}},
		{ai.ModelResponse{Parts: []ai.ResponsePart{call, ai.CompactionPart{ProviderName: "other"}}}, ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "read", ToolCallID: "call", Content: "done"},
		}}},
		{ai.ModelResponse{Parts: []ai.ResponsePart{call, ai.TextPart{Content: "after"}}}, ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.TextPart{Content: "next"},
		}}},
		{ai.ModelResponse{Parts: []ai.ResponsePart{call, ai.TextPart{Content: "after"}}}, ai.ModelRequest{Parts: []ai.RequestPart{
			ai.RetryPromptPart{Content: "retry"},
		}}},
		{ai.ModelResponse{Parts: []ai.ResponsePart{call, ai.TextPart{Content: "after"}}}, ai.ModelRequest{Parts: []ai.RequestPart{
			ai.RetryPromptPart{ToolName: "read", ToolCallID: "call", Content: "retry"},
		}}},
		{ai.ModelResponse{Parts: []ai.ResponsePart{call, call, ai.TextPart{Content: "after"}}}, ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "read", ToolCallID: "call", Content: "one"},
			ai.ToolReturnPart{ToolName: "read", ToolCallID: "call", Content: "two"},
		}}},
	} {
		if _, err := model.Request(t.Context(), history, ai.ModelRequestParams{}); err != nil {
			t.Fatal(err)
		}
	}
	if len(bodies) != 6 {
		t.Fatalf("unexpected requests: %d", len(bodies))
	}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{
		ai.ModelResponse{Parts: []ai.ResponsePart{call}}, nil,
	}, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected invalid message error")
	}
}

func TestDeepSeekResponsesStreamAndConfiguration(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "environment-key")
	provider := deepseek.NewProviderConfig()
	if provider.BaseURL != "https://api.deepseek.com" || provider.APIKey != "environment-key" {
		t.Fatalf("unexpected provider config: %#v", provider)
	}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(response, "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"item\",\"delta\":\"done\"}\n\n"+
			"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"response\",\"model\":\"deepseek-chat\",\"status\":\"completed\",\"output\":[],\"usage\":{}}}\n\n")
	}))
	defer server.Close()
	model := deepseek.NewResponsesModel("deepseek-chat", deepseek.WithProvider(openai.ProviderConfig{
		BaseURL: server.URL, HTTPClient: server.Client(), PrepareRequest: func(request *http.Request) error {
			request.Header.Set("X-Test", "value")
			return nil
		},
	}))
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{
		Settings: ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelHigh}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, streamErr := range stream {
		if streamErr != nil {
			t.Fatal(streamErr)
		}
	}
	if model.ProviderName() != "deepseek" || model.ProviderURL() != server.URL {
		t.Fatalf("unexpected identity: %q %q", model.ProviderName(), model.ProviderURL())
	}
}

func disabledThinking() *ai.ThinkingSettings {
	return &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled}
}

func valueOrNil(value string) any {
	if value == "" {
		return nil
	}
	return value
}
