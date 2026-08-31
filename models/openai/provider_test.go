package openai_test

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func TestOpenAICompatibleProvider(t *testing.T) {
	headers := http.Header{"X-Provider": {"configured"}}
	query := url.Values{"region": {"west"}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.URL.Query().Get("region") != "west" {
			t.Errorf("unexpected request URL: %s", r.URL.String())
		}
		if authorization := r.Header.Get("Authorization"); authorization != "" {
			t.Errorf("unexpected authorization: %q", authorization)
		}
		if value := r.Header.Get("X-Provider"); value != "per-request" {
			t.Errorf("unexpected provider header: %q", value)
		}
		if value := r.Header.Get("X-Dynamic"); value != "token" {
			t.Errorf("unexpected dynamic header: %q", value)
		}
		_, _ = w.Write([]byte(`{
			"id":"response-id","model":"compatible-model","created":1,
			"choices":[{"message":{"content":"hello"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`))
	}))
	defer server.Close()

	model := openai.NewModel("compatible-model", openai.WithProvider(openai.ProviderConfig{
		Name: "local", BaseURL: server.URL + "/v1/", HTTPClient: server.Client(),
		Headers: headers, Query: query,
		PrepareRequest: func(request *http.Request) error {
			request.Header.Set("X-Dynamic", "token")
			return nil
		},
	}))
	headers.Set("X-Provider", "mutated")
	query.Set("region", "mutated")

	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		Settings: ai.ModelSettings{ExtraHeaders: map[string]string{"X-Provider": "per-request"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if response.ProviderName != "local" || response.ProviderURL != server.URL+"/v1" {
		t.Fatalf("unexpected provider identity: %+v", response)
	}
}

func TestOpenAIPromptCacheRequestSettings(t *testing.T) {
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		switch request.URL.Path {
		case "/chat/completions":
			_, _ = io.WriteString(response, `{
				"model":"gpt-5.6","choices":[{"message":{"content":"done"},"finish_reason":"stop"}],
				"usage":{"prompt_tokens":10,"completion_tokens":1,
				"prompt_tokens_details":{"cached_tokens":4,"cache_write_tokens":6}}
			}`)
		case "/responses":
			_, _ = io.WriteString(response, `{
				"id":"response","model":"gpt-5.6","status":"completed",
				"output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}],
				"usage":{"input_tokens":11,"output_tokens":1,
				"input_tokens_details":{"cached_tokens":4,"cache_write_tokens":7}}
			}`)
		case "/responses/input_tokens":
			_, _ = io.WriteString(response, `{"input_tokens":12}`)
		default:
			t.Errorf("unexpected request path: %s", request.URL.Path)
		}
	}))
	defer server.Close()

	settings, err := (openai.Settings{
		Common:               ai.ModelSettings{ExtraBody: map[string]any{"custom": true}},
		PromptCacheKey:       "conversation",
		PromptCacheRetention: openai.PromptCacheRetention24Hours,
		PromptCacheOptions: &openai.PromptCacheOptions{
			Mode: openai.PromptCacheModeExplicit, TTL: openai.PromptCacheTTL30Minutes,
		},
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	provider := openai.ProviderConfig{Name: "openai", BaseURL: server.URL, HTTPClient: server.Client()}
	chat := openai.NewModel("gpt-5.6", openai.WithProvider(provider), openai.WithDefaultSettings(settings))
	responses := openai.NewResponsesModel(
		"gpt-5.6", openai.WithProvider(provider), openai.WithDefaultSettings(settings),
	)
	prompt := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Content: "hello"},
	}}}
	chatResponse, err := chat.Request(t.Context(), prompt, ai.ModelRequestParams{Settings: settings})
	if err != nil {
		t.Fatal(err)
	}
	responsesResponse, err := responses.Request(t.Context(), prompt, ai.ModelRequestParams{Settings: settings})
	if err != nil {
		t.Fatal(err)
	}
	counted, err := responses.CountTokens(t.Context(), prompt, ai.ModelRequestParams{Settings: settings})
	if err != nil {
		t.Fatal(err)
	}
	legacy := openai.NewModel("gpt-4o", openai.WithProvider(provider))
	_, err = legacy.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{
			ai.TextContent{Text: "stable"}, ai.CachePoint{}, ai.TextContent{Text: "question"},
		}},
	}}}, ai.ModelRequestParams{Settings: settings})
	if err != nil {
		t.Fatal(err)
	}
	if chatResponse.Usage.CacheWriteTokens != 6 || responsesResponse.Usage.CacheWriteTokens != 7 ||
		counted.InputTokens != 12 {
		t.Fatalf("unexpected cache usage: chat=%+v responses=%+v counted=%+v", chatResponse.Usage, responsesResponse.Usage, counted)
	}
	for index, body := range bodies[:2] {
		options := body["prompt_cache_options"].(map[string]any)
		if body["prompt_cache_key"] != "conversation" || body["prompt_cache_retention"] != "24h" ||
			options["mode"] != "explicit" || options["ttl"] != "30m" || body["custom"] != true {
			t.Fatalf("unexpected prompt cache request %d: %#v", index, body)
		}
	}
	for _, field := range []string{"prompt_cache_key", "prompt_cache_retention", "prompt_cache_options"} {
		if _, exists := bodies[2][field]; exists {
			t.Fatalf("token count request included %q: %#v", field, bodies[2])
		}
	}
	if bodies[2]["custom"] != true {
		t.Fatalf("token count request dropped extra body: %#v", bodies[2])
	}
	legacyOptions := bodies[3]["prompt_cache_options"].(map[string]any)
	legacyContent := bodies[3]["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if legacyOptions["mode"] != "explicit" || legacyContent[0].(map[string]any)["prompt_cache_breakpoint"] != nil {
		t.Fatalf("legacy cache options or marker gate changed: %#v", bodies[3])
	}
	if duration, ok := ai.ResolvePromptCacheRetention(chat, nil); !ok || duration != 24*time.Hour {
		t.Fatalf("unexpected chat retention: %s %v", duration, ok)
	}
	if duration, ok := ai.ResolvePromptCacheRetention(responses, nil); !ok || duration != 24*time.Hour {
		t.Fatalf("unexpected Responses retention: %s %v", duration, ok)
	}
	memory, err := (openai.Settings{PromptCacheRetention: openai.PromptCacheRetentionInMemory}).Build()
	if err != nil {
		t.Fatal(err)
	}
	if duration, ok := ai.ResolvePromptCacheRetention(chat, &memory); ok || duration != 0 {
		t.Fatalf("unexpected in-memory retention: %s %v", duration, ok)
	}
}

func TestOpenAIToolReturnSchemaDescriptionFallback(t *testing.T) {
	var descriptions []string
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		tools := body["tools"].([]any)
		tool := tools[0].(map[string]any)
		description, _ := tool["description"].(string)
		if function, ok := tool["function"].(map[string]any); ok {
			description, _ = function["description"].(string)
		}
		descriptions = append(descriptions, description)
		if strings.HasSuffix(request.URL.Path, "/responses") {
			_, _ = io.WriteString(response, `{
				"id":"response","model":"model","status":"completed",
				"output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}]
			}`)
			return
		}
		_, _ = io.WriteString(response, `{
			"model":"model","choices":[{"message":{"content":"done"},"finish_reason":"stop"}]
		}`)
	}))
	defer server.Close()
	included := true
	definition := ai.ToolDefinition{
		Name: "lookup", Description: "Lookup.", Schema: map[string]any{"type": "object"},
		ReturnSchema: map[string]any{"type": "string"}, IncludeReturnSchema: &included,
	}
	provider := openai.ProviderConfig{Name: "compatible", BaseURL: server.URL, HTTPClient: server.Client()}
	models := []ai.Model{
		openai.NewModel("model", openai.WithProvider(provider)),
		openai.NewResponsesModel("model", openai.WithProvider(provider)),
	}
	for _, model := range models {
		if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Tools: []ai.ToolDefinition{definition}}); err != nil {
			t.Fatal(err)
		}
	}
	for _, description := range descriptions {
		if description != "Lookup.\n\nReturn schema:\n\n{\n  \"type\": \"string\"\n}" {
			t.Fatalf("unexpected return schema description: %q", description)
		}
	}
	invalid := definition
	invalid.ReturnSchema = map[string]any{"bad": make(chan struct{})}
	for _, model := range models {
		if _, err := model.Request(
			t.Context(), nil, ai.ModelRequestParams{Tools: []ai.ToolDefinition{invalid}},
		); err == nil || !strings.Contains(err.Error(), "marshal return schema") {
			t.Fatalf("unexpected return schema error: %v", err)
		}
	}
}

func TestOpenAIChatCompatibility(t *testing.T) {
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		if request.Header.Get("Accept") == "text/event-stream" {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(response, "data: "+`{"id":"stream","model":"compatible",`+
				`"choices":[{"delta":{"reasoning_content":"think","tool_calls":[{"index":0,`+
				`"id":"call","function":{"name":"tool","arguments":"{}"}}]},`+
				`"finish_reason":"interrupted"}]}`+"\n\n")
			_, _ = io.WriteString(response, "data: [DONE]\n\n")
			return
		}
		_, _ = io.WriteString(response, `{
			"id":"static","model":"compatible",
			"choices":[{"message":{"reasoning_content":"reason","content":"answer",
			"tool_calls":[{"id":"call","type":"function","function":{"name":"tool","arguments":"{}"}}]},
			"finish_reason":"interrupted"}]
		}`)
	}))
	defer server.Close()
	finishReasons := map[string]ai.FinishReason{"interrupted": ai.FinishReasonError}
	model := openai.NewModel("compatible", openai.WithProvider(openai.ProviderConfig{
		Name: "provider", BaseURL: server.URL, HTTPClient: server.Client(),
	}), openai.WithChatCompatibility(openai.ChatCompatibility{
		ReasoningContent: true, FinishReasons: finishReasons,
	}))
	finishReasons["interrupted"] = ai.FinishReasonStop
	history := ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.ThinkingPart{Content: "kept", ProviderName: "provider"},
		ai.ThinkingPart{Content: "dropped", ProviderName: "other"},
		ai.TextPart{Content: "previous"},
	}}
	static, err := model.Request(t.Context(), []ai.ModelMessage{history}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if static.FinishReason != ai.FinishReasonError || len(static.Parts) != 3 {
		t.Fatalf("unexpected compatible response: %+v", static)
	}
	for _, part := range static.Parts {
		switch part := part.(type) {
		case ai.ThinkingPart:
			if part.Content != "reason" || part.ProviderName != "provider" {
				t.Fatalf("unexpected reasoning part: %+v", part)
			}
		case ai.TextPart:
			if part.ProviderName != "provider" {
				t.Fatalf("unexpected text provider: %+v", part)
			}
		case ai.ToolCallPart:
			if part.ProviderName != "provider" {
				t.Fatalf("unexpected tool provider: %+v", part)
			}
		}
	}
	messages := bodies[0]["messages"].([]any)
	assistant := messages[0].(map[string]any)
	if assistant["reasoning_content"] != "kept" {
		t.Fatalf("unexpected reasoning history: %v", assistant)
	}

	events, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var sawThinking, sawTool bool
	for event, streamErr := range events {
		if streamErr != nil {
			t.Fatal(streamErr)
		}
		switch event := event.(type) {
		case ai.ThinkingDeltaEvent:
			sawThinking = event.Delta == "think" && event.ProviderName == "provider"
		case ai.ToolCallStartEvent:
			sawTool = event.ProviderName == "provider"
		case ai.FinishEvent:
			if event.FinishReason != ai.FinishReasonError {
				t.Fatalf("unexpected streamed finish reason: %+v", event)
			}
		}
	}
	if !sawThinking || !sawTool {
		t.Fatalf("missing compatibility events: thinking=%v tool=%v", sawThinking, sawTool)
	}

	abandoned, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for event, streamErr := range abandoned {
		if streamErr != nil {
			t.Fatal(streamErr)
		}
		if _, ok := event.(ai.ThinkingDeltaEvent); ok {
			break
		}
	}
}

func TestOpenAIExtendedChatCompatibility(t *testing.T) {
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		if request.Header.Get("Accept") == "text/event-stream" {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(response, "data: "+`{"id":"stream","model":"routed","provider":"vendor",`+
				`"choices":[{"delta":{"reasoning":"think","annotations":[{"type":"url_citation"}]},`+
				`"finish_reason":"stop","native_finish_reason":"end_turn"}],`+
				`"usage":{"prompt_tokens":4,"completion_tokens":2,"cost":0.02,`+
				`"prompt_tokens_details":{"cache_write_tokens":1},"is_byok":false,`+
				`"server_tool_use_details":{"web_search_requests":1}}}`+"\n\n")
			_, _ = io.WriteString(response, "data: [DONE]\n\n")
			return
		}
		_, _ = io.WriteString(response, `{
			"id":"static","model":"routed","provider":"vendor",
			"choices":[{"message":{"reasoning":"reason","content":"answer","annotations":[{"type":"file"}]},
			"finish_reason":"stop","native_finish_reason":"end_turn"}],
			"usage":{"prompt_tokens":5,"completion_tokens":3,"cost":0.03,"is_byok":true,
			"prompt_tokens_details":{"cache_write_tokens":2},
			"cost_details":{"upstream_inference_cost":0.01,"upstream_inference_prompt_cost":0.004,
			"upstream_inference_completions_cost":0.006},
			"server_tool_use_details":{"tool_calls_requested":2,"tool_calls_executed":1}}
		}`)
	}))
	defer server.Close()
	mapper := func(tool ai.NativeTool) (openai.ChatNativeTool, bool, error) {
		if _, ok := tool.(ai.WebSearchTool); ok {
			return openai.ChatNativeTool{Type: "provider:web", Parameters: map[string]any{"context": "high"}}, true, nil
		}
		return openai.ChatNativeTool{}, false, nil
	}
	model := openai.NewModel("routed", openai.WithProvider(openai.ProviderConfig{
		Name: "gateway", BaseURL: server.URL, HTTPClient: server.Client(),
	}), openai.WithChatCompatibility(openai.ChatCompatibility{
		Reasoning: true, LegacyMaxTokens: true, ExtendedMetadata: true, NativeToolFunc: mapper,
	}))
	history := ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.ThinkingPart{Content: "kept", ProviderName: "gateway"},
		ai.ThinkingPart{Content: "dropped", ProviderName: "other"},
	}}
	static, err := model.Request(t.Context(), []ai.ModelMessage{history}, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{Name: "local", Schema: map[string]any{"type": "object"}}},
		NativeTools: []ai.NativeTool{
			ai.WebSearchTool{}, ai.CodeExecutionTool{Optional: true},
		},
		Settings: ai.ModelSettings{MaxTokens: 20},
	})
	if err != nil {
		t.Fatal(err)
	}
	tools := bodies[0]["tools"].([]any)
	if bodies[0]["max_tokens"] != float64(20) || bodies[0]["max_completion_tokens"] != nil ||
		len(tools) != 2 || tools[0].(map[string]any)["type"] != "function" ||
		tools[1].(map[string]any)["type"] != "provider:web" ||
		bodies[0]["messages"].([]any)[0].(map[string]any)["reasoning"] != "kept" {
		t.Fatalf("unexpected extended request: %#v", bodies[0])
	}
	if len(static.Parts) != 2 || static.Parts[0].(ai.ThinkingPart).Content != "reason" ||
		static.Usage.CacheWriteTokens != 2 || static.Usage.CostUSD == nil || *static.Usage.CostUSD != 0.03 ||
		static.ProviderDetails["downstream_provider"] != "vendor" ||
		static.ProviderDetails["finish_reason"] != "end_turn" || static.ProviderDetails["cost"] != 0.03 ||
		static.ProviderDetails["upstream_inference_cost"] != 0.01 ||
		static.ProviderDetails["upstream_inference_prompt_cost"] != 0.004 ||
		static.ProviderDetails["upstream_inference_completions_cost"] != 0.006 ||
		static.ProviderDetails["is_byok"] != true || len(static.ProviderDetails["annotations"].([]map[string]any)) != 1 ||
		static.ProviderDetails["server_tool_use"].(map[string]int)["tool_calls_requested"] != 2 {
		t.Fatalf("unexpected extended response: %+v", static)
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
		switch event := event.(type) {
		case ai.ThinkingDeltaEvent:
			sawThinking = event.Delta == "think"
		case ai.FinishEvent:
			if event.ProviderDetails["downstream_provider"] != "vendor" ||
				event.ProviderDetails["finish_reason"] != "end_turn" ||
				len(event.ProviderDetails["annotations"].([]map[string]any)) != 1 ||
				event.ProviderDetails["is_byok"] != false || event.Usage.CacheWriteTokens != 1 ||
				event.ProviderDetails["server_tool_use"].(map[string]int)["web_search_requests"] != 1 {
				t.Fatalf("unexpected extended stream finish: %+v", event)
			}
		}
	}
	if !sawThinking {
		t.Fatal("missing extended reasoning delta")
	}
	abandoned, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for event, streamErr := range abandoned {
		if streamErr != nil {
			t.Fatal(streamErr)
		}
		if _, ok := event.(ai.ThinkingDeltaEvent); ok {
			break
		}
	}

	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.CodeExecutionTool{}}})
	if err == nil || !strings.Contains(err.Error(), `gateway: Chat Completions does not support native tool "code_execution"`) {
		t.Fatalf("unexpected unsupported native tool error: %v", err)
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		ai.WebSearchTool{SearchContextSize: "huge"},
	}})
	if err == nil || !strings.Contains(err.Error(), "invalid web search context") {
		t.Fatalf("unexpected native tool validation error: %v", err)
	}
	errorModel := openai.NewModel("routed", openai.WithChatCompatibility(openai.ChatCompatibility{
		NativeToolFunc: func(ai.NativeTool) (openai.ChatNativeTool, bool, error) {
			return openai.ChatNativeTool{}, false, errors.New("render failed")
		},
	}))
	_, err = errorModel.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.WebSearchTool{}}})
	if err == nil || err.Error() != "render failed" {
		t.Fatalf("unexpected native mapper error: %v", err)
	}
	emptyModel := openai.NewModel("routed", openai.WithChatCompatibility(openai.ChatCompatibility{
		NativeToolFunc: func(ai.NativeTool) (openai.ChatNativeTool, bool, error) {
			return openai.ChatNativeTool{}, true, nil
		},
	}))
	_, err = emptyModel.Request(t.Context(), nil, ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.WebSearchTool{}}})
	if err == nil || !strings.Contains(err.Error(), "rendered an empty type") {
		t.Fatalf("unexpected empty native type error: %v", err)
	}
}

func TestOpenAIReasoningDetailsCompatibility(t *testing.T) {
	var bodies []map[string]any
	var staticRequests int
	var streamRequests int
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		if request.Header.Get("Accept") == "text/event-stream" {
			response.Header().Set("Content-Type", "text/event-stream")
			streamRequests++
			if streamRequests == 2 {
				_, _ = io.WriteString(response, "data: "+`{"model":"routed","choices":[{"delta":{`+
					`"reasoning_details":[{"type":"unknown"}]}}]}`+"\n\n")
				return
			}
			_, _ = io.WriteString(response, "data: "+`{"model":"routed","choices":[{"delta":{`+
				`"reasoning":"ignored","reasoning_details":[`+
				`{"id":"encrypted","type":"reasoning.encrypted","format":"openai-responses-v1","index":2,"data":"opaque"},`+
				`{"id":"summary","type":"reasoning.summary","summary":"brief"}]}}]}`+"\n\n")
			_, _ = io.WriteString(response, "data: [DONE]\n\n")
			return
		}
		staticRequests++
		if staticRequests == 2 {
			_, _ = response.Write([]byte(`{"model":"routed","choices":[{"message":{
				"reasoning_details":[{"type":"unknown"}]},"finish_reason":"stop"}],"usage":{}}`))
			return
		}
		_, _ = response.Write([]byte(`{"model":"routed","choices":[{"message":{
			"reasoning":"ignored","reasoning_details":[
				{"id":"text","type":"reasoning.text","format":"anthropic-claude-v1","index":0,
				 "text":"detail","signature":"signed"},
				{"id":"summary","type":"reasoning.summary","summary":"brief"},
				{"id":"encrypted","type":"reasoning.encrypted","data":"opaque"}
			],"content":"answer"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer server.Close()
	model := openai.NewModel("routed", openai.WithProvider(openai.ProviderConfig{
		Name: "gateway", BaseURL: server.URL, HTTPClient: server.Client(),
	}), openai.WithChatCompatibility(openai.ChatCompatibility{Reasoning: true, ReasoningDetails: true}))
	history := ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.ThinkingPart{
			ID: "text", Content: "prior", Signature: "signature", ProviderName: "gateway",
			ProviderDetails: map[string]any{
				"type": "reasoning.text", "format": "anthropic-claude-v1", "index": float64(1),
			},
		},
		ai.ThinkingPart{
			ID: "summary", Content: "summary", ProviderName: "gateway",
			ProviderDetails: map[string]any{"type": "reasoning.summary", "index": 2},
		},
		ai.ThinkingPart{
			ID: "encrypted", Signature: "encrypted", ProviderName: "gateway",
			ProviderDetails: map[string]any{"type": "reasoning.encrypted"},
		},
		ai.ThinkingPart{Content: "fallback", ProviderName: "gateway"},
		ai.ThinkingPart{
			Content: "invalid index", ProviderName: "gateway",
			ProviderDetails: map[string]any{"type": "reasoning.summary", "index": 1.5},
		},
		ai.ThinkingPart{
			ProviderName: "gateway", ProviderDetails: map[string]any{"type": "reasoning.encrypted"},
		},
		ai.ThinkingPart{
			Content: "unknown", ProviderName: "gateway", ProviderDetails: map[string]any{"type": "unknown"},
		},
	}}
	response, err := model.Request(t.Context(), []ai.ModelMessage{history}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	assistant := bodies[0]["messages"].([]any)[0].(map[string]any)
	details := assistant["reasoning_details"].([]any)
	if len(details) != 4 || details[0].(map[string]any)["text"] != "prior" ||
		details[0].(map[string]any)["index"] != float64(1) ||
		details[1].(map[string]any)["summary"] != "summary" ||
		details[2].(map[string]any)["data"] != "encrypted" ||
		details[3].(map[string]any)["id"] != nil || details[3].(map[string]any)["format"] != nil ||
		details[3].(map[string]any)["index"] != nil || assistant["reasoning"] != "fallbackunknown" {
		t.Fatalf("unexpected reasoning-detail replay: %#v", assistant)
	}
	if len(response.Parts) != 4 {
		t.Fatalf("unexpected reasoning-detail response: %#v", response.Parts)
	}
	text := response.Parts[0].(ai.ThinkingPart)
	summary := response.Parts[1].(ai.ThinkingPart)
	encrypted := response.Parts[2].(ai.ThinkingPart)
	if text.ID != "text" || text.Content != "detail" || text.Signature != "signed" ||
		text.ProviderDetails["format"] != "anthropic-claude-v1" || text.ProviderDetails["index"] != 0 ||
		summary.Content != "brief" || encrypted.Signature != "opaque" || encrypted.Content != "" {
		t.Fatalf("unexpected normalized reasoning details: %#v", response.Parts)
	}

	stream, err := model.StreamRequest(t.Context(), []ai.ModelMessage{*response}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var events []ai.ThinkingDeltaEvent
	for event, streamErr := range stream {
		if streamErr != nil {
			t.Fatal(streamErr)
		}
		if event, ok := event.(ai.ThinkingDeltaEvent); ok {
			events = append(events, event)
		}
	}
	if len(events) != 2 || events[0].PartID != "reasoning_detail_reasoning.encrypted_2" ||
		events[0].ID != "encrypted" || events[0].SignatureDelta != "opaque" ||
		events[1].PartID != "reasoning_detail_reasoning.summary_1" || events[1].Delta != "brief" {
		t.Fatalf("unexpected streamed reasoning details: %#v", events)
	}

	unknownStream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for _, streamErr := range unknownStream {
		if streamErr == nil || !strings.Contains(streamErr.Error(), `unknown reasoning detail type "unknown"`) {
			t.Fatalf("unexpected streamed reasoning detail error: %v", streamErr)
		}
		break
	}
	abandoned, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for event, streamErr := range abandoned {
		if streamErr != nil {
			t.Fatal(streamErr)
		}
		if _, ok := event.(ai.ThinkingDeltaEvent); ok {
			break
		}
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), `unknown reasoning detail type "unknown"`) {
		t.Fatalf("unexpected unknown reasoning detail error: %v", err)
	}
}

func TestOpenAIExtendedResponseVariants(t *testing.T) {
	var staticRequests int
	var streamRequests int
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Accept") == "text/event-stream" {
			response.Header().Set("Content-Type", "text/event-stream")
			streamRequests++
			if streamRequests == 1 {
				_, _ = io.WriteString(response, "data: "+`{"error":{"code":503,"message":"unavailable"}}`+"\n\n")
			} else {
				_, _ = io.WriteString(response, "data: "+`{"choices":null,"error":null}`+"\n\n")
			}
			return
		}
		staticRequests++
		switch staticRequests {
		case 1:
			_, _ = response.Write([]byte(`{"created":10,"provider":{"id":"nested","model":"routed",
				"provider":null,"choices":[{"message":{"content":"nested"},"finish_reason":"stop"}],"usage":{}}}`))
		case 2:
			_, _ = response.Write([]byte(`{"choices":null,"error":{"code":429,"message":"limited"}}`))
		case 3:
			_, _ = response.Write([]byte(`{"choices":null,"error":null}`))
		case 4:
			_, _ = response.Write([]byte(`{"choices":[],"error":null}`))
		default:
			_, _ = response.Write([]byte(`{"provider":`))
		}
	}))
	defer server.Close()
	model := openai.NewModel("routed", openai.WithProvider(openai.ProviderConfig{
		Name: "gateway", BaseURL: server.URL, HTTPClient: server.Client(),
	}), openai.WithChatCompatibility(openai.ChatCompatibility{ExtendedMetadata: true}))
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if response.ProviderResponseID != "nested" || response.ProviderDetails["downstream_provider"] != "unknown" ||
		response.Timestamp.Unix() != 10 {
		t.Fatalf("unexpected nested response: %+v", response)
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{})
	var apiError *openai.APIError
	if !errors.As(err, &apiError) || apiError.StatusCode != 429 ||
		apiError.Error() != "gateway: API returned status 429: limited" {
		t.Fatalf("unexpected embedded error: %v", err)
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if !errors.As(err, &apiError) || apiError.StatusCode != 0 ||
		apiError.Error() != "gateway: returned a response with null choices and no error for model routed" {
		t.Fatalf("unexpected no-completion error: %v", err)
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err == nil || err.Error() != "openai: response has no choices" {
		t.Fatalf("unexpected empty-choice error: %v", err)
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), "parse response") {
		t.Fatalf("unexpected malformed response error: %v", err)
	}

	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for _, streamErr := range stream {
		if !errors.As(streamErr, &apiError) || apiError.StatusCode != 503 {
			t.Fatalf("unexpected streamed embedded error: %v", streamErr)
		}
		break
	}
	stream, err = model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for _, streamErr := range stream {
		if !errors.As(streamErr, &apiError) || !strings.Contains(streamErr.Error(), "model routed") {
			t.Fatalf("unexpected streamed no-completion error: %v", streamErr)
		}
		break
	}
}

func TestOpenAIChatPromptCache(t *testing.T) {
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
	model := openai.NewModel("routed", openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client()))
	ctx := openai.WithChatPromptCache(t.Context(), openai.ChatPromptCache{
		InstructionsTTL: "1h", MessagesTTL: "5m", ToolsTTL: "1h",
		IncludeTTL: true, SupportsDynamicInstructions: true, ExplicitMarkerStyle: openai.ChatPromptCacheMarkerControl, MaxPoints: 3,
	})
	_, err := model.Request(ctx, []ai.ModelMessage{
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "earlier"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.TextContent{Text: "look"}, ai.CachePoint{TTL: ai.CachePointTTL1Hour},
			ai.TextContent{Text: "uncached"},
			ai.ImageURL{URL: "https://example.com/image.png"}, ai.CachePoint{},
		}}}},
	}, ai.ModelRequestParams{
		Instructions: "static one\n\nstatic two\n\ndynamic",
		InstructionParts: []ai.InstructionPart{
			{Content: "static one"}, {Content: "static two"}, {Content: "dynamic", Dynamic: true},
		},
		Tools: []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	messages := bodies[0]["messages"].([]any)
	static := messages[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	if static["text"] != "static two" || static["cache_control"].(map[string]any)["ttl"] != "1h" ||
		messages[2].(map[string]any)["content"] != "dynamic" {
		t.Fatalf("unexpected instruction cache boundary: %#v", messages)
	}
	userContent := messages[4].(map[string]any)["content"].([]any)
	if userContent[0].(map[string]any)["cache_control"] != nil ||
		userContent[1].(map[string]any)["cache_control"] != nil ||
		userContent[2].(map[string]any)["cache_control"].(map[string]any)["ttl"] != "5m" {
		t.Fatalf("unexpected message cache boundary: %#v", userContent)
	}
	tool := bodies[0]["tools"].([]any)[0].(map[string]any)
	if tool["cache_control"].(map[string]any)["ttl"] != "1h" {
		t.Fatalf("unexpected tool cache boundary: %#v", tool)
	}

	ctx = openai.WithChatPromptCache(t.Context(), openai.ChatPromptCache{
		InstructionsTTL: "5m", IncludeTTL: false,
	})
	_, err = model.Request(ctx, nil, ai.ModelRequestParams{
		Instructions: "static\n\ndynamic",
		InstructionParts: []ai.InstructionPart{
			{Content: "static"}, {Content: "dynamic", Dynamic: true},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, message := range bodies[1]["messages"].([]any) {
		if _, ok := message.(map[string]any)["content"].([]any); ok {
			t.Fatalf("dynamic-incompatible instructions were cached: %#v", bodies[1])
		}
	}

	ctx = openai.WithChatPromptCache(t.Context(), openai.ChatPromptCache{
		InstructionsTTL: "5m", SupportsDynamicInstructions: true,
	})
	_, err = model.Request(ctx, nil, ai.ModelRequestParams{
		Instructions: "dynamic", InstructionParts: []ai.InstructionPart{{Content: "dynamic", Dynamic: true}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if bodies[2]["messages"].([]any)[0].(map[string]any)["content"] != "dynamic" {
		t.Fatalf("all-dynamic instructions were cached: %#v", bodies[2])
	}

	ctx = openai.WithChatPromptCache(t.Context(), openai.ChatPromptCache{InstructionsTTL: "5m"})
	_, err = model.Request(ctx, nil, ai.ModelRequestParams{Instructions: "aggregate"})
	if err != nil {
		t.Fatal(err)
	}
	aggregate := bodies[3]["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	if aggregate["cache_control"].(map[string]any)["type"] != "ephemeral" ||
		aggregate["cache_control"].(map[string]any)["ttl"] != nil {
		t.Fatalf("unexpected aggregate cache boundary: %#v", aggregate)
	}

	ctx = openai.WithChatPromptCache(t.Context(), openai.ChatPromptCache{MessagesTTL: "5m"})
	_, err = model.Request(ctx, []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: ""}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "tool", ToolCallID: "call", Args: []byte(`{}`)}}},
	}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Request(ctx, []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Content: ""},
	}}}, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}

	explicit := openai.WithChatPromptCache(t.Context(), openai.ChatPromptCache{
		ExplicitMarkerStyle: openai.ChatPromptCacheMarkerControl, IncludeTTL: true,
	})
	for name, contents := range map[string][]ai.UserContent{
		"first":   {ai.CachePoint{}, ai.TextContent{Text: "later"}},
		"invalid": {ai.TextContent{Text: "first"}, ai.CachePoint{TTL: "1d"}},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := model.Request(explicit, []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Contents: contents},
			}}}, ai.ModelRequestParams{})
			if err == nil || !strings.Contains(err.Error(), "cache point") {
				t.Fatalf("unexpected cache-point error: %v", err)
			}
		})
	}
	limited := openai.WithChatPromptCache(t.Context(), openai.ChatPromptCache{
		InstructionsTTL: "5m", ToolsTTL: "5m", MaxPoints: 1,
	})
	_, err = model.Request(limited, nil, ai.ModelRequestParams{
		Instructions: "system", Tools: []ai.ToolDefinition{{Name: "tool", Schema: map[string]any{"type": "object"}}},
	})
	if err == nil || !strings.Contains(err.Error(), "exceeding the maximum") {
		t.Fatalf("unexpected reserved cache-point error: %v", err)
	}
	invalidStyle := openai.WithChatPromptCache(t.Context(), openai.ChatPromptCache{
		ExplicitMarkerStyle: "invalid",
	})
	_, err = model.Request(invalidStyle, []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{ai.TextContent{Text: "first"}, ai.CachePoint{}}},
	}}}, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), "invalid chat prompt cache marker style") {
		t.Fatalf("unexpected marker-style error: %v", err)
	}

	_, err = model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{ai.TextContent{Text: "kept"}, ai.CachePoint{}}},
	}}}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	unsupported := bodies[len(bodies)-1]["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if len(unsupported) != 1 || unsupported[0].(map[string]any)["cache_control"] != nil {
		t.Fatalf("unsupported explicit cache point leaked: %#v", unsupported)
	}
	gpt := openai.NewModel("gpt-5.6", openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client()))
	_, err = gpt.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{
			ai.TextContent{Text: "cache me"}, ai.CachePoint{TTL: ai.CachePointTTL1Hour},
		}},
	}}}, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	openAIPart := bodies[len(bodies)-1]["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	if openAIPart["prompt_cache_breakpoint"].(map[string]any)["mode"] != "explicit" ||
		openAIPart["cache_control"] != nil {
		t.Fatalf("unexpected OpenAI cache breakpoint: %#v", openAIPart)
	}
}

func TestOpenAIChatCompatibilityValidation(t *testing.T) {
	for name, compatibility := range map[string]openai.ChatCompatibility{
		"empty reason":              {FinishReasons: map[string]ai.FinishReason{"": ai.FinishReasonStop}},
		"invalid normalized reason": {FinishReasons: map[string]ai.FinishReason{"custom": "invalid"}},
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			_ = openai.WithChatCompatibility(compatibility)
		})
	}
}

func TestOpenAICompatibleResponsesProviderIdentity(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"response-id","model":"compatible-model","created_at":1,"status":"completed",
			"output":[
				{"type":"reasoning","id":"reasoning-id","encrypted_content":"signature"},
				{"type":"message","id":"message-id","content":[{"type":"output_text","text":"hello"}]},
				{"type":"function_call","id":"item-id","call_id":"call-id","name":"tool","arguments":"{}"},
				{"type":"compaction","id":"compaction-id","encrypted_content":"opaque"},
				{"type":"tool_search_call","id":"search-call","call_id":"search","execution":"server","arguments":{}},
				{"type":"tool_search_output","id":"search-output","call_id":"search","execution":"server","tools":[]}
			],
			"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
		}`))
	}))
	defer server.Close()

	model := openai.NewResponsesModel("compatible-model", openai.WithProvider(openai.ProviderConfig{
		Name: "compatible", BaseURL: server.URL, HTTPClient: server.Client(),
	}))
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if response.ProviderName != "compatible" || model.NativeToolSearchProvider() != "compatible" {
		t.Fatalf("unexpected provider identity: %+v", response)
	}
	for _, part := range response.Parts {
		switch part := part.(type) {
		case ai.TextPart:
			if part.ProviderName != "compatible" {
				t.Fatalf("unexpected text provider: %+v", part)
			}
		case ai.ThinkingPart:
			if part.ProviderName != "compatible" {
				t.Fatalf("unexpected thinking provider: %+v", part)
			}
		case ai.ToolCallPart:
			if part.ProviderName != "compatible" {
				t.Fatalf("unexpected tool provider: %+v", part)
			}
		case ai.NativeToolCallPart:
			if part.ProviderName != "compatible" {
				t.Fatalf("unexpected native call provider: %+v", part)
			}
		case ai.NativeToolReturnPart:
			if part.ProviderName != "compatible" {
				t.Fatalf("unexpected native return provider: %+v", part)
			}
		case ai.CompactionPart:
			if part.ProviderName != "compatible" {
				t.Fatalf("unexpected compaction provider: %+v", part)
			}
		}
	}
}

func TestOpenAICompatibleResponsesStreamingIdentity(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"item_id\":\"message\",\"delta\":\"hi\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"response\",\"model\":\"model\",\"status\":\"completed\",\"output\":[{\"id\":\"message\",\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"hi\"}]}],\"usage\":{}}}\n\n"))
	}))
	defer server.Close()

	model := openai.NewResponsesModel("model", openai.WithProvider(openai.ProviderConfig{
		Name: "compatible", BaseURL: server.URL, HTTPClient: server.Client(),
	}))
	events, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var text ai.TextDeltaEvent
	var finish ai.FinishEvent
	for event, streamErr := range events {
		if streamErr != nil {
			t.Fatal(streamErr)
		}
		switch event := event.(type) {
		case ai.TextDeltaEvent:
			text = event
		case ai.FinishEvent:
			finish = event
		}
	}
	if text.ProviderName != "compatible" || finish.ProviderName != "compatible" {
		t.Fatalf("unexpected streaming provider identity: text=%+v finish=%+v", text, finish)
	}
	if len(finish.Parts) != 1 || finish.Parts[0].(ai.TextPart).ProviderName != "compatible" {
		t.Fatalf("unexpected snapshot provider identity: %+v", finish.Parts)
	}
}

func TestOpenAICompatibleResponsesCanDisableNativeHistory(t *testing.T) {
	model := openai.NewResponsesModel("model", openai.WithDeferredToolSupport(false))
	if provider := model.NativeToolSearchProvider(); provider != "" {
		t.Fatalf("unexpected native history provider: %q", provider)
	}
}

func TestOpenAIProviderPreparationError(t *testing.T) {
	provider := openai.ProviderConfig{
		Name: "local", BaseURL: "http://localhost", PrepareRequest: func(*http.Request) error {
			return errors.New("credentials unavailable")
		},
	}
	chat := openai.NewModel("model", openai.WithProvider(provider))
	responses := openai.NewResponsesModel("model", openai.WithProvider(provider))
	suspended := ai.ModelResponse{
		ProviderName: "local", ProviderResponseID: "response",
		ProviderDetails: map[string]any{"background": true}, State: ai.ModelResponseStateSuspended,
	}
	streamSuspended := suspended
	streamSuspended.ProviderDetails = map[string]any{"background": true, "sequence_number": 1}
	tests := map[string]func() error{
		"chat": func() error {
			_, err := chat.Request(t.Context(), nil, ai.ModelRequestParams{})
			return err
		},
		"chat stream": func() error {
			_, err := chat.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			return err
		},
		"responses": func() error {
			_, err := responses.Request(t.Context(), nil, ai.ModelRequestParams{})
			return err
		},
		"responses stream": func() error {
			_, err := responses.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
			return err
		},
		"compaction": func() error {
			_, err := responses.CompactMessages(t.Context(), nil, ai.ModelRequestParams{})
			return err
		},
		"retrieve": func() error {
			_, err := responses.Request(t.Context(), []ai.ModelMessage{suspended}, ai.ModelRequestParams{})
			return err
		},
		"retrieve stream": func() error {
			_, err := responses.StreamRequest(
				t.Context(), []ai.ModelMessage{streamSuspended}, ai.ModelRequestParams{},
			)
			return err
		},
		"cancel": func() error { return responses.CancelSuspendedResponse(t.Context(), suspended) },
	}
	for name, operation := range tests {
		t.Run(name, func(t *testing.T) {
			err := operation()
			if err == nil || !strings.Contains(err.Error(), "openai: prepare request: credentials unavailable") {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestOpenAIProviderValidation(t *testing.T) {
	tests := map[string]openai.ProviderConfig{
		"name":     {BaseURL: "https://example.com/v1"},
		"base URL": {Name: "compatible"},
	}
	for name, provider := range tests {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			_ = openai.WithProvider(provider)
		})
	}
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	_ = openai.WithProviderName("")
}

func TestOpenAIBaseURLFromEnvironment(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{
			"model":"model","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]
		}`))
	}))
	defer server.Close()
	t.Setenv("OPENAI_BASE_URL", server.URL+"/v1/")
	t.Setenv("OPENAI_API_KEY", "")

	model := openai.NewModel("model", openai.WithHTTPClient(server.Client()), openai.WithProviderName("proxy"))
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if response.ProviderName != "proxy" {
		t.Fatalf("unexpected provider name: %q", response.ProviderName)
	}
}
