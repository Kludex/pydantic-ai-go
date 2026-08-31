package openrouter_test

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
	"github.com/Kludex/pydantic-ai-go/models/openrouter"
)

type unsupportedNativeTool struct {
	optional bool
}

func (unsupportedNativeTool) Kind() string                        { return "unsupported" }
func (unsupportedNativeTool) UniqueID() string                    { return "unsupported" }
func (tool unsupportedNativeTool) IsOptional() bool               { return tool.optional }
func (tool unsupportedNativeTool) CloneNativeTool() ai.NativeTool { return tool }

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
			ai.UserPromptPart{Content: "question"},
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
	lastMessage := anthropicMessages[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	tools := bodies[0]["tools"].([]any)
	if instruction["cache_control"].(map[string]any)["ttl"] != "1h" ||
		lastMessage["cache_control"].(map[string]any)["ttl"] != "5m" ||
		tools[0].(map[string]any)["cache_control"].(map[string]any)["ttl"] != "1h" ||
		tools[1].(map[string]any)["cache_control"] != nil {
		t.Fatalf("unexpected Anthropic cache boundaries: %#v", bodies[0])
	}

	request("google/gemini-3-flash")
	googleMessages := bodies[1]["messages"].([]any)
	if _, cached := googleMessages[0].(map[string]any)["content"].([]any); cached {
		t.Fatalf("dynamic Google instructions were cached: %#v", googleMessages)
	}
	googleLast := googleMessages[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	if cache := googleLast["cache_control"].(map[string]any); cache["type"] != "ephemeral" || cache["ttl"] != nil {
		t.Fatalf("unexpected Google message cache boundary: %#v", googleLast)
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
