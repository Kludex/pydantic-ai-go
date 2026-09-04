package openai_test

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func TestResponsesXAITools(t *testing.T) {
	requestCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		requestCount++
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		tools, _ := body["tools"].([]any)
		if requestCount == 1 {
			if len(tools) != 3 || tools[0].(map[string]any)["type"] != "x_search" ||
				tools[1].(map[string]any)["type"] != "collections_search" ||
				tools[2].(map[string]any)["excluded_domains"].([]any)[0] != "spam.example" ||
				tools[2].(map[string]any)["allowed_domains"].([]any)[0] != "go.dev" {
				t.Errorf("unexpected xAI tools: %#v", tools)
			}
		}
		_, _ = io.WriteString(response, `{"id":"response","model":"grok-4.3","status":"completed","citations":["https://x.com/citation"],"usage":{"server_side_tools_used":["x_search","x_search"]},"output":[{"id":"x","type":"x_search_call","status":"completed","output":"result","results":[{"url":"https://x.com/post"}]},{"id":"collection","type":"collections_search_call","status":"completed","queries":["docs"],"results":[{"text":"result"}]},{"id":"attachment","type":"attachment_search_call","status":"completed","action":{"query":"file"},"output":"attachment result"}]}`)
	}))
	defer server.Close()
	model := openai.NewResponsesModel("grok-4.3", openai.WithProvider(openai.ProviderConfig{
		Name: "xai", BaseURL: server.URL, HTTPClient: server.Client(),
	}))
	from := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	to := time.Date(2026, 1, 2, 0, 0, 0, 0, time.UTC)
	maximum := 2
	external := true
	settings, err := (openai.Settings{ResponsesInclude: []string{"x_search_call.outputs"}}).Build()
	if err != nil {
		t.Fatal(err)
	}
	response, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: settings, NativeTools: []ai.NativeTool{
		&ai.XSearchTool{FromDate: &from, ToDate: &to, ExcludedXHandles: []string{"spam"}, IncludeOutput: true},
		&ai.FileSearchTool{FileStoreIDs: []string{"collection"}, MaxNumResults: &maximum},
		ai.WebSearchTool{
			AllowedDomains: []string{"go.dev"}, BlockedDomains: []string{"spam.example"}, ExternalWebAccess: &external,
		},
	}})
	if err != nil || len(response.Parts) != 6 || response.Parts[0].(ai.NativeToolCallPart).ToolKind != ai.ToolPartKindXSearch {
		t.Fatalf("unexpected xAI response: %#v %v", response, err)
	}
	xResult := response.Parts[1].(ai.NativeToolReturnPart).Content.(map[string]any)
	if xResult["output"] != "result" || xResult["citations"].([]string)[0] != "https://x.com/citation" ||
		response.Parts[3].(ai.NativeToolReturnPart).Content.(map[string]any)["results"] == nil ||
		response.Parts[4].(ai.NativeToolCallPart).ToolName != "attachment_search" ||
		response.Usage.Details["server_side_tools_x_search"] != 2 {
		t.Fatalf("unexpected xAI tool results: %#v", response)
	}
	history := []ai.ModelMessage{*response}
	if _, err := model.Request(t.Context(), history, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	if requestCount != 2 || !model.SupportsNativeTool(ai.XSearchTool{}) ||
		model.SupportsNativeTool(ai.ImageGenerationTool{}) {
		t.Fatalf("unexpected xAI Responses support")
	}
}

func TestResponsesXAIToolRejectionAndIncludes(t *testing.T) {
	chatServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(response, `{"model":"model","choices":[{"message":{"content":"done"},"finish_reason":"stop"}]}`)
	}))
	defer chatServer.Close()
	chat := openai.NewModel("model", openai.WithBaseURL(chatServer.URL), openai.WithHTTPClient(chatServer.Client()))
	if _, err := chat.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.CodeExecutionTool{Optional: true}},
	}); err != nil {
		t.Fatal(err)
	}

	model := openai.NewResponsesModel("gpt-5", openai.WithProvider(openai.ProviderConfig{
		Name: "openai", BaseURL: "https://example.invalid/v1",
	}))
	if model.SupportsNativeTool(ai.XSearchTool{}) {
		t.Fatal("OpenAI unexpectedly supports X search")
	}
	_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.XSearchTool{}},
	})
	if err == nil || !strings.Contains(err.Error(), "does not support native tool") {
		t.Fatalf("unexpected required X search error: %v", err)
	}
	_, err = model.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.XSearchTool{Optional: true}},
	})
	if err == nil || !strings.Contains(err.Error(), "example.invalid") {
		t.Fatalf("optional tool should reach transport: %v", err)
	}
	if _, err := (openai.Settings{ResponsesInclude: []string{""}}).Build(); err == nil {
		t.Fatal("empty include value was accepted")
	}
	settings, err := (openai.Settings{ResponsesInclude: []string{"x_search_call.outputs"}}).Build()
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(request.Body).Decode(&body)
		if body["include"].([]any)[0] != "x_search_call.outputs" {
			t.Errorf("unexpected includes: %#v", body)
		}
		_, _ = io.WriteString(response, `{"id":"id","model":"model","status":"completed","output":[]}`)
	}))
	defer server.Close()
	includedModel := openai.NewResponsesModel("model", openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client()))
	if _, err := includedModel.Request(t.Context(), nil, ai.ModelRequestParams{Settings: settings}); err != nil {
		t.Fatal(err)
	}
	_, err = includedModel.Request(t.Context(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		ExtraBody: map[string]any{"openai_responses_include": []any{"bad"}},
	}})
	if err == nil || !strings.Contains(err.Error(), "must be strings") {
		t.Fatalf("unexpected malformed include error: %v", err)
	}
}

func TestResponsesXAIToolStream(t *testing.T) {
	events := []string{
		`{"type":"response.created","response":{"id":"response","model":"grok","created_at":100,"status":"in_progress"}}`,
		`{"type":"response.output_item.added","item":{"id":"x","type":"x_search_call","status":"in_progress"}}`,
		`{"type":"response.output_item.done","item":{"id":"x","type":"x_search_call","status":"completed","action":{"query":"go"},"output":"result"}}`,
		`{"type":"response.output_item.added","item":{"id":"collection","type":"collections_search_call","status":"in_progress"}}`,
		`{"type":"response.output_item.done","item":{"id":"collection","type":"collections_search_call","status":"completed","queries":["docs"]}}`,
		`{"type":"response.output_item.added","item":{"id":"attachment","type":"attachment_search_call","status":"in_progress"}}`,
		`{"type":"response.output_item.done","item":{"id":"attachment","type":"attachment_search_call","status":"completed","action":{"query":"file"},"output":"result"}}`,
		`{"type":"response.completed","response":{"id":"response","model":"grok","created_at":100,"status":"completed","output":[{"id":"x","type":"x_search_call","status":"completed","action":{"query":"go"}},{"id":"collection","type":"collections_search_call","status":"completed","queries":["docs"]},{"id":"attachment","type":"attachment_search_call","status":"completed","action":{"query":"file"},"output":"result"}]}}`,
	}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("Content-Type", "text/event-stream")
		for _, event := range events {
			_, _ = fmt.Fprintf(response, "data: %s\n\n", event)
		}
		_, _ = io.WriteString(response, "data: [DONE]\n\n")
	}))
	defer server.Close()
	model := openai.NewResponsesModel("grok", openai.WithProvider(openai.ProviderConfig{
		Name: "xai", BaseURL: server.URL, HTTPClient: server.Client(),
	}))
	stream, err := model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	starts, returns := 0, 0
	for event, eventErr := range stream {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		switch event.(type) {
		case ai.ToolCallStartEvent:
			starts++
		case ai.NativeToolReturnEvent:
			returns++
		}
	}
	if starts != 3 || returns != 3 {
		t.Fatalf("unexpected lifecycle: %d starts, %d returns", starts, returns)
	}
	stream, err = model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	stream(func(event ai.ModelStreamEvent, eventErr error) bool {
		_, start := event.(ai.ToolCallStartEvent)
		return eventErr == nil && !start
	})
	stream, err = model.StreamRequest(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	stream(func(event ai.ModelStreamEvent, eventErr error) bool {
		_, delta := event.(ai.ToolCallDeltaEvent)
		return eventErr == nil && !delta
	})
}
