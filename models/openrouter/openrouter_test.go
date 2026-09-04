package openrouter_test

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/openai"
	"github.com/Kludex/pydantic-ai-go/models/openrouter"
)

type unsupportedNativeTool struct {
	optional bool
}

func TestProviderConfig(t *testing.T) {
	t.Setenv("OPENROUTER_API_KEY", "key")
	t.Setenv("OPENROUTER_APP_URL", "https://app.example")
	t.Setenv("OPENROUTER_APP_TITLE", "App")
	provider := openrouter.NewProviderConfig()
	if provider.Name != "openrouter" || provider.BaseURL != "https://openrouter.ai/api/v1" ||
		provider.APIKey != "key" || provider.Headers.Get("HTTP-Referer") != "https://app.example" ||
		provider.Headers.Get("X-Title") != "App" {
		t.Fatalf("unexpected provider: %#v", provider)
	}
	provider.Headers.Set("X-Title", "mutated")
	if openrouter.NewProviderConfig().Headers.Get("X-Title") != "App" {
		t.Fatal("provider headers were shared")
	}
}

func (unsupportedNativeTool) Kind() string                        { return "unsupported" }
func (unsupportedNativeTool) UniqueID() string                    { return "unsupported" }
func (tool unsupportedNativeTool) IsOptional() bool               { return tool.optional }
func (tool unsupportedNativeTool) CloneNativeTool() ai.NativeTool { return tool }

func TestInlineSystemPromptsUseUserFallback(t *testing.T) {
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		bodies = append(bodies, body)
		if request.Header.Get("Accept") == "text/event-stream" {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(response, "data: [DONE]\n\n")
			return
		}
		_, _ = io.WriteString(response, `{"id":"response","model":"anthropic/claude","choices":[{"message":{"content":"done"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()
	model := openrouter.NewModel("anthropic/claude", openrouter.WithBaseURL(server.URL),
		openrouter.WithAPIKey("token"), openrouter.WithHTTPClient(server.Client()))
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "standing"}, ai.UserPromptPart{Content: "hello"},
			ai.SystemPromptPart{Content: "inline"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "answer"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.SystemPromptPart{Content: "later"}}},
	}
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	stream, err := model.StreamRequest(t.Context(), messages, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for range stream {
	}
	if _, ok := messages[0].(ai.ModelRequest).Parts[2].(ai.SystemPromptPart); !ok {
		t.Fatal("request preparation mutated caller history")
	}
	for _, body := range bodies {
		wire := body["messages"].([]any)
		roles := []string{"system", "user", "user", "assistant", "user"}
		for index, role := range roles {
			if wire[index].(map[string]any)["role"] != role {
				t.Fatalf("unexpected messages: %#v", wire)
			}
		}
		if wire[2].(map[string]any)["content"] != "<system>inline</system>" ||
			wire[4].(map[string]any)["content"] != "<system>later</system>" {
			t.Fatalf("unexpected system fallback: %#v", wire)
		}
	}
}

func TestOpenRouterRequest(t *testing.T) {
	t.Setenv("OPENROUTER_APP_URL", "https://environment.example")
	t.Setenv("OPENROUTER_APP_TITLE", "Environment")
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/chat/completions" || request.Header.Get("Authorization") != "Bearer token" ||
			request.Header.Get("HTTP-Referer") != "https://app.example" || request.Header.Get("X-Title") != "My App" {
			t.Errorf("unexpected OpenRouter request: %s headers=%v", request.URL.Path, request.Header)
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		if request.Header.Get("Accept") == "text/event-stream" {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(response, "data: "+`{"id":"stream","model":"anthropic/claude",`+
				`"provider":"Anthropic","choices":[{"delta":{"reasoning_details":[`+
				`{"id":"reason","type":"reasoning.summary","index":0,"summary":"think"}],"content":"done"},`+
				`"finish_reason":"stop","native_finish_reason":"end_turn"}],`+
				`"usage":{"prompt_tokens":3,"completion_tokens":2,"cost":0.002}}`+"\n\n")
			_, _ = io.WriteString(response, "data: [DONE]\n\n")
			return
		}
		_, _ = io.WriteString(response, `{
			"id":"response","model":"anthropic/claude","provider":"Anthropic",
			"choices":[{"message":{"reasoning_details":[{"id":"reason","type":"reasoning.text",
			"format":"anthropic-claude-v1","index":0,"text":"consider","signature":"signed"}],"content":"answer"},
			"finish_reason":"stop","native_finish_reason":"end_turn"}],
			"usage":{"prompt_tokens":7,"completion_tokens":4,"cost":0.004}
		}`)
	}))
	defer server.Close()

	enabled := true
	maxTokens := 2048
	settings, err := (openrouter.Settings{
		Common: ai.ModelSettings{MaxTokens: 100},
		Models: []string{"anthropic/claude", "openai/gpt"},
		Provider: &openrouter.ProviderRouting{
			Order: []string{"anthropic"}, AllowFallbacks: &enabled,
			DataCollection: openrouter.DataCollectionDeny, Only: []string{"anthropic"},
			Sort: openrouter.ProviderSortLatency, MaxPrice: &openrouter.MaxPrice{Prompt: 1},
		},
		Preset: "careful", Transforms: []openrouter.Transform{openrouter.TransformMiddleOut},
		Reasoning: &openrouter.Reasoning{
			Effort: openrouter.ReasoningEffortHigh, Enabled: &enabled,
		},
		Usage: &openrouter.UsageConfig{Include: true},
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	model := openrouter.NewModel(
		"anthropic/claude-sonnet-4.6", openrouter.WithAPIKey("token"),
		openrouter.WithBaseURL(server.URL), openrouter.WithHTTPClient(server.Client()),
		openrouter.WithAppAttribution("https://app.example", "My App"), openrouter.WithDefaultSettings(settings),
	)
	if model.Name() != "anthropic/claude-sonnet-4.6" || model.ProviderName() != "openrouter" ||
		model.ProviderURL() != server.URL || model.DefaultModelSettings().MaxTokens != 100 {
		t.Fatalf("unexpected OpenRouter identity: name=%q provider=%q URL=%q", model.Name(), model.ProviderName(), model.ProviderURL())
	}
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{Name: "local", Schema: map[string]any{"type": "object"}}},
		NativeTools: []ai.NativeTool{
			&ai.AdvisorTool{Model: "anthropic/claude-opus-4.8", MaxTokens: &maxTokens},
			&ai.WebSearchTool{
				SearchContextSize: ai.WebSearchContextHigh,
				UserLocation: &ai.WebSearchUserLocation{
					City: "London", Country: "GB", Region: "England", Timezone: "Europe/London",
				},
				AllowedDomains: []string{"pydantic.dev"}, BlockedDomains: []string{"example.com"}, MaxUses: 2,
			},
		},
		Settings: settings,
	})
	if err != nil {
		t.Fatal(err)
	}
	body := bodies[0]
	tools := body["tools"].([]any)
	if body["max_tokens"] != float64(100) || body["max_completion_tokens"] != nil || len(tools) != 3 ||
		tools[0].(map[string]any)["type"] != "function" ||
		tools[1].(map[string]any)["type"] != "openrouter:advisor" ||
		tools[1].(map[string]any)["parameters"].(map[string]any)["max_completion_tokens"] != float64(2048) ||
		tools[2].(map[string]any)["type"] != "openrouter:web_search" {
		t.Fatalf("unexpected OpenRouter body: %#v", body)
	}
	search := tools[2].(map[string]any)["parameters"].(map[string]any)
	if search["search_context_size"] != "high" || search["max_uses"] != float64(2) ||
		search["excluded_domains"].([]any)[0] != "example.com" ||
		search["user_location"].(map[string]any)["timezone"] != "Europe/London" {
		t.Fatalf("unexpected search tool: %#v", search)
	}
	if body["models"].([]any)[0] != "anthropic/claude" || body["preset"] != "careful" ||
		body["reasoning"].(map[string]any)["effort"] != "high" ||
		body["provider"].(map[string]any)["data_collection"] != "deny" ||
		body["usage"].(map[string]any)["include"] != true {
		t.Fatalf("unexpected OpenRouter settings: %#v", body)
	}
	if len(response.Parts) != 2 || response.Parts[0].(ai.ThinkingPart).Content != "consider" ||
		response.Parts[0].(ai.ThinkingPart).Signature != "signed" ||
		response.Parts[0].(ai.ThinkingPart).ProviderDetails["type"] != "reasoning.text" ||
		response.ProviderDetails["downstream_provider"] != "Anthropic" || response.Usage.CostUSD == nil ||
		*response.Usage.CostUSD != 0.004 {
		t.Fatalf("unexpected OpenRouter response: %+v", response)
	}

	stream, err := model.StreamRequest(t.Context(), []ai.ModelMessage{*response}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var thinking, text bool
	for event, streamErr := range stream {
		if streamErr != nil {
			t.Fatal(streamErr)
		}
		switch event := event.(type) {
		case ai.ThinkingDeltaEvent:
			thinking = event.Delta == "think"
		case ai.TextDeltaEvent:
			text = event.Delta == "done"
		case ai.FinishEvent:
			if event.ProviderDetails["downstream_provider"] != "Anthropic" || event.Usage.CostUSD == nil {
				t.Fatalf("unexpected streamed OpenRouter response: %+v", event)
			}
		}
	}
	if !thinking || !text {
		t.Fatalf("missing OpenRouter events: thinking=%v text=%v", thinking, text)
	}
	replayed := bodies[1]["messages"].([]any)[0].(map[string]any)["reasoning_details"].([]any)
	if replayed[0].(map[string]any)["signature"] != "signed" || replayed[0].(map[string]any)["text"] != "consider" {
		t.Fatalf("unexpected OpenRouter reasoning replay: %#v", replayed)
	}
}

func TestOpenRouterNativeToolDefaultsAndCompatibility(t *testing.T) {
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		_, _ = response.Write([]byte(`{"model":"routed","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer server.Close()
	model := openrouter.NewModel("~openai/gpt", openrouter.WithProvider(openai.ProviderConfig{
		Name: "custom-router", BaseURL: server.URL, HTTPClient: server.Client(),
	}))
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{
			ai.AdvisorTool{Model: "anthropic/claude"}, ai.WebSearchTool{}, unsupportedNativeTool{optional: true},
		},
		Settings: ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelEnabled}},
	}); err != nil {
		t.Fatal(err)
	}
	tools := bodies[0]["tools"].([]any)
	if len(tools) != 2 || tools[0].(map[string]any)["parameters"].(map[string]any)["forward_transcript"] != false ||
		tools[1].(map[string]any)["parameters"].(map[string]any)["search_context_size"] != "medium" ||
		bodies[0]["reasoning"].(map[string]any)["effort"] != "medium" {
		t.Fatalf("unexpected default native tools: %#v", tools)
	}
	_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{unsupportedNativeTool{}}})
	if err == nil || !strings.Contains(err.Error(), `custom-router: Chat Completions does not support native tool "unsupported"`) {
		t.Fatalf("unexpected required native tool error: %v", err)
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		Thinking: &ai.ThinkingSettings{Level: "invalid"},
	}})
	if err == nil || !strings.Contains(err.Error(), "invalid thinking level") {
		t.Fatalf("unexpected request settings error: %v", err)
	}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		ExtraBody: map[string]any{"openrouter_cache_messages": "5m"},
	}}); err != nil {
		t.Fatal(err)
	}
	for _, value := range []any{true, "1d"} {
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
			ExtraBody: map[string]any{"openrouter_cache_messages": value},
		}})
		if err == nil || !strings.Contains(err.Error(), "cache") {
			t.Fatalf("unexpected cache setting error for %#v: %v", value, err)
		}
	}
}

func TestOpenRouterDownstreamSchemaAndToolChoice(t *testing.T) {
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		_, _ = response.Write([]byte(`{"model":"routed","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer server.Close()
	newModel := func(name string) *openrouter.Model {
		return openrouter.NewModel(name, openrouter.WithBaseURL(server.URL), openrouter.WithHTTPClient(server.Client()))
	}
	schema := map[string]any{
		"$schema": "draft", "title": "Input", "type": "object",
		"properties": map[string]any{
			"status": map[string]any{"const": "active"},
			"email":  map[string]any{"type": "string", "format": "email"},
		},
	}
	params := ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{Name: "get_weather", Schema: schema}},
		OutputTool: &ai.ToolDefinition{Name: "final_result", Schema: map[string]any{
			"type": "object", "properties": map[string]any{},
		}},
		Settings: ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelLow}},
	}
	if _, err := newModel("anthropic/claude-sonnet-4.6").Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	if _, err := newModel("google/gemini-2.5-flash").Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	anthropicBody := bodies[0]
	googleBody := bodies[1]
	if anthropicBody["tool_choice"] != "auto" || googleBody["tool_choice"] != "required" ||
		anthropicBody["reasoning"].(map[string]any)["effort"] != "low" {
		t.Fatalf("unexpected downstream tool choices: anthropic=%#v google=%#v", anthropicBody, googleBody)
	}
	googleSchema := googleBody["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)["parameters"].(map[string]any)
	properties := googleSchema["properties"].(map[string]any)
	if googleSchema["$schema"] != nil || googleSchema["title"] != nil ||
		properties["status"].(map[string]any)["enum"].([]any)[0] != "active" ||
		properties["email"].(map[string]any)["description"] != "Format: email" || schema["title"] != "Input" {
		t.Fatalf("Google schema profile was not applied defensively: %#v", googleSchema)
	}

	disabledSettings, err := (openrouter.Settings{Reasoning: &openrouter.Reasoning{Enabled: new(bool)}}).Build()
	if err != nil {
		t.Fatal(err)
	}
	params.Settings = disabledSettings
	if _, err := newModel("anthropic/claude-sonnet-4.6").Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	if bodies[2]["tool_choice"] != "required" {
		t.Fatalf("disabled reasoning changed forced tool choice: %#v", bodies[2])
	}
	for _, reasoning := range []any{
		&openrouter.Reasoning{Effort: openrouter.ReasoningEffortLow},
		true,
	} {
		params.Settings = ai.ModelSettings{ExtraBody: map[string]any{"reasoning": reasoning}}
		if _, err := newModel("anthropic/claude-sonnet-4.6").Request(t.Context(), nil, params); err != nil {
			t.Fatal(err)
		}
		if bodies[len(bodies)-1]["tool_choice"] != "auto" {
			t.Fatalf("reasoning form did not relax inferred forcing: %#v", bodies[len(bodies)-1])
		}
	}

	for name, choice := range map[string]any{"required": "required", "list": []any{"final_result"}} {
		t.Run(name, func(t *testing.T) {
			params.Settings = ai.ModelSettings{
				Thinking:  &ai.ThinkingSettings{Level: ai.ThinkingLevelLow},
				ExtraBody: map[string]any{"tool_choice": choice},
			}
			before := len(bodies)
			_, err := newModel("anthropic/claude-sonnet-4.6").Request(t.Context(), nil, params)
			if err == nil || !strings.Contains(err.Error(), "cannot be forced") &&
				!strings.Contains(err.Error(), "specific tools cannot be forced") {
				t.Fatalf("unexpected explicit forcing error: %v", err)
			}
			if len(bodies) != before {
				t.Fatal("invalid forced tool choice reached transport")
			}
		})
	}
}

func TestOpenRouterModelNameValidation(t *testing.T) {
	for _, name := range []string{"model", "/model", "provider/", "~/model"} {
		model := openrouter.NewModel(name)
		_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
		if err == nil || !strings.Contains(err.Error(), "provider/model form") {
			t.Fatalf("unexpected error for %q: %v", name, err)
		}
		if _, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
			t.Fatalf("stream accepted invalid model %q", name)
		}
	}
}

func TestOpenRouterPromptCaching(t *testing.T) {
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		_, _ = response.Write([]byte(`{"model":"routed","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer server.Close()
	settings, err := (openrouter.Settings{
		CacheInstructions:    openrouter.CacheTTL1Hour,
		CacheMessages:        openrouter.CacheTTL5Minutes,
		CacheToolDefinitions: openrouter.CacheTTL1Hour,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	request := func(modelName string) {
		t.Helper()
		model := openrouter.NewModel(
			modelName, openrouter.WithBaseURL(server.URL), openrouter.WithHTTPClient(server.Client()),
		)
		_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.UserPromptPart{Contents: []ai.UserContent{
				ai.TextContent{Text: "old"}, ai.CachePoint{TTL: ai.CachePointTTL1Hour},
				ai.TextContent{Text: "middle"}, ai.CachePoint{TTL: ai.CachePointTTL1Hour},
				ai.TextContent{Text: "new"}, ai.CachePoint{},
			}},
		}}}, ai.ModelRequestParams{
			Instructions: "static\n\ndynamic",
			InstructionParts: []ai.InstructionPart{
				{Content: "static"}, {Content: "dynamic", Dynamic: true},
			},
			Tools:       []ai.ToolDefinition{{Name: "local", Schema: map[string]any{"type": "object"}}},
			NativeTools: []ai.NativeTool{ai.WebSearchTool{}}, Settings: settings,
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	request("anthropic/claude-sonnet-4.6")
	anthropicMessages := bodies[0]["messages"].([]any)
	instruction := anthropicMessages[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	userContent := anthropicMessages[2].(map[string]any)["content"].([]any)
	tools := bodies[0]["tools"].([]any)
	if instruction["cache_control"].(map[string]any)["ttl"] != "1h" ||
		userContent[0].(map[string]any)["cache_control"] != nil ||
		userContent[1].(map[string]any)["cache_control"].(map[string]any)["ttl"] != "1h" ||
		userContent[2].(map[string]any)["cache_control"].(map[string]any)["ttl"] != "5m" ||
		tools[0].(map[string]any)["cache_control"].(map[string]any)["ttl"] != "1h" ||
		tools[1].(map[string]any)["cache_control"] != nil {
		t.Fatalf("unexpected Anthropic cache boundaries: %#v", bodies[0])
	}

	request("google/gemini-3-flash")
	googleMessages := bodies[1]["messages"].([]any)
	if _, cached := googleMessages[0].(map[string]any)["content"].([]any); cached {
		t.Fatalf("dynamic Google instructions were cached: %#v", googleMessages)
	}
	googleContent := googleMessages[2].(map[string]any)["content"].([]any)
	for _, item := range googleContent {
		cache := item.(map[string]any)["cache_control"].(map[string]any)
		if cache["type"] != "ephemeral" || cache["ttl"] != nil {
			t.Fatalf("unexpected Google message cache boundary: %#v", googleContent)
		}
	}
	if bodies[1]["tools"].([]any)[0].(map[string]any)["cache_control"] != nil {
		t.Fatalf("Google tool definition was cached: %#v", bodies[1]["tools"])
	}

	request("openai/gpt-5")
	encodedBytes, err := json.Marshal(bodies[2])
	if err != nil {
		t.Fatal(err)
	}
	encoded := string(encodedBytes)
	if strings.Contains(encoded, "cache_control") || strings.Contains(encoded, "openrouter_cache_") {
		t.Fatalf("unsupported cache settings leaked to OpenRouter: %s", encoded)
	}

	request("openai/gpt-5.6")
	openAIMessages := bodies[3]["messages"].([]any)
	openAIContent := openAIMessages[len(openAIMessages)-1].(map[string]any)["content"].([]any)
	for _, item := range openAIContent {
		if item.(map[string]any)["prompt_cache_breakpoint"].(map[string]any)["mode"] != "explicit" {
			t.Fatalf("unexpected routed OpenAI cache breakpoint: %#v", openAIContent)
		}
	}
}

func TestOpenRouterResponseVariants(t *testing.T) {
	var staticRequests int
	var streamRequests int
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Accept") == "text/event-stream" {
			response.Header().Set("Content-Type", "text/event-stream")
			streamRequests++
			if streamRequests == 1 {
				_, _ = io.WriteString(response, "data: "+`{"error":{"code":503,"message":"upstream unavailable"}}`+"\n\n")
			} else {
				_, _ = io.WriteString(response, "data: "+`{"model":"vendor/model","choices":null,"error":null}`+"\n\n")
			}
			return
		}
		staticRequests++
		switch staticRequests {
		case 1:
			_, _ = response.Write([]byte(`{"created":10,"provider":{"id":"nested","model":"vendor/model",
				"provider":null,"choices":[{"message":{"content":"nested"},"finish_reason":"stop"}],
				"usage":{"prompt_tokens":1,"completion_tokens":1}}}`))
		case 2:
			_, _ = response.Write([]byte(`{"model":"vendor/model","choices":null,
				"error":{"code":429,"message":"route limited"}}`))
		default:
			_, _ = response.Write([]byte(`{"model":"vendor/model","provider":null,"choices":null,"error":null}`))
		}
	}))
	defer server.Close()
	model := openrouter.NewModel(
		"vendor/model", openrouter.WithBaseURL(server.URL), openrouter.WithHTTPClient(server.Client()),
	)
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if response.ProviderResponseID != "nested" || response.Parts[0].(ai.TextPart).Content != "nested" ||
		response.ProviderDetails["downstream_provider"] != "unknown" || response.Timestamp.Unix() != 10 {
		t.Fatalf("unexpected nested response: %+v", response)
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{})
	var apiError *openai.APIError
	if !errors.As(err, &apiError) || apiError.StatusCode != 429 || apiError.ProviderName != "openrouter" ||
		apiError.Body != "route limited" {
		t.Fatalf("unexpected embedded error: %v", err)
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{})
	var modelAPIError ai.ModelAPIError
	if !errors.As(err, &modelAPIError) || !strings.Contains(err.Error(), "null choices") {
		t.Fatalf("unexpected no-completion error: %v", err)
	}

	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for _, streamErr := range stream {
		if !errors.As(streamErr, &apiError) || apiError.StatusCode != 503 {
			t.Fatalf("unexpected streamed provider error: %v", streamErr)
		}
		break
	}
	stream, err = model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for _, streamErr := range stream {
		if !errors.As(streamErr, &modelAPIError) || !strings.Contains(streamErr.Error(), "null choices") {
			t.Fatalf("unexpected streamed no-completion error: %v", streamErr)
		}
		break
	}
}

func TestOpenRouterFileInput(t *testing.T) {
	videoServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "video/webm")
		_, _ = response.Write([]byte("video-bytes"))
	}))
	defer videoServer.Close()
	var body map[string]any
	apiServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = io.WriteString(response, `{
			"id":"response","model":"google/gemini-3","choices":[{"message":{"content":"done"},"finish_reason":"stop"}]
		}`)
	}))
	defer apiServer.Close()
	model := openrouter.NewModel(
		"google/gemini-3-pro", openrouter.WithBaseURL(apiServer.URL), openrouter.WithHTTPClient(apiServer.Client()),
	)
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
		ai.TextContent{Text: "Describe these videos."},
		ai.VideoURL{URL: "https://example.com/video.mp4"},
		ai.BinaryContent{Data: []byte("inline"), MediaType: "video/mp4"},
		ai.VideoURL{URL: videoServer.URL + "/clip.webm", ForceDownload: ai.FileDownloadAllowLocal},
		ai.DocumentURL{URL: "https://example.com/report.pdf"},
		ai.BinaryContent{Data: []byte("audio"), MediaType: "audio/mpeg"},
	}}}}}
	response, err := model.Request(t.Context(), messages, ai.ModelRequestParams{})
	if err != nil || response.Text() != "done" {
		t.Fatalf("unexpected video response=%+v err=%v", response, err)
	}
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(content) != 6 || content[0].(map[string]any)["type"] != "text" ||
		content[1].(map[string]any)["video_url"].(map[string]any)["url"] != "https://example.com/video.mp4" ||
		content[2].(map[string]any)["video_url"].(map[string]any)["url"] != "data:video/mp4;base64,aW5saW5l" ||
		content[3].(map[string]any)["video_url"].(map[string]any)["url"] !=
			"data:video/webm;base64,dmlkZW8tYnl0ZXM=" ||
		content[4].(map[string]any)["file"].(map[string]any)["file_data"] !=
			"https://example.com/report.pdf" ||
		content[5].(map[string]any)["input_audio"].(map[string]any)["data"] != "YXVkaW8=" {
		t.Fatalf("unexpected OpenRouter video content: %#v", content)
	}

	for name, video := range map[string]ai.VideoURL{
		"blocked local": {URL: videoServer.URL + "/clip.webm", ForceDownload: ai.FileDownloadSafe},
		"YouTube":       {URL: "https://youtu.be/example", ForceDownload: ai.FileDownloadSafe},
		"invalid mode":  {URL: "https://example.com/video.mp4", ForceDownload: "invalid"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Contents: []ai.UserContent{video}},
			}}}, ai.ModelRequestParams{})
			if err == nil {
				t.Fatal("invalid video request succeeded")
			}
		})
	}
}

func TestOpenRouterProviderError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.WriteHeader(http.StatusBadRequest)
		_, _ = response.Write([]byte(`{"error":"bad route"}`))
	}))
	defer server.Close()
	model := openrouter.NewModel(
		"openai/gpt", openrouter.WithBaseURL(server.URL), openrouter.WithHTTPClient(server.Client()),
		openrouter.WithAppAttribution("", ""),
	)
	_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	var apiError *openai.APIError
	if !errors.As(err, &apiError) || apiError.StatusCode != http.StatusBadRequest ||
		apiError.ProviderName != "openrouter" || !strings.HasPrefix(err.Error(), "openrouter: API returned status 400") {
		t.Fatalf("unexpected provider error: %v", err)
	}
}
