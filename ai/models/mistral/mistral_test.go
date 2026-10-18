package mistral_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/mistral"
)

func TestRequest(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/chat/completions" || request.Header.Get("Authorization") != "Bearer key" ||
			request.Header.Get("X-Run") != "request" || request.Header.Get("X-Dynamic") != "yes" {
			t.Errorf("unexpected request: %s %#v", request.URL, request.Header)
		}
		if err := json.NewDecoder(request.Body).Decode(&captured); err != nil {
			t.Error(err)
		}
		_, _ = io.WriteString(response, `{
			"id":"response-id","model":"mistral-resolved","created":1704067200,
			"choices":[{"finish_reason":"model_length","message":{"content":[
				{"type":"thinking","thinking":[{"type":"text","text":"one"},{"type":"text","text":"two"}]},
				{"type":"text","text":"hello "},{"type":"reference"},{"type":"text","text":"world"}],
				"tool_calls":[
					{"id":"call-1","type":"function","function":{"name":"keep","arguments":"{\"x\":1}"}},
					{"type":"function","function":{"name":"keep","arguments":{"x":2}}}]}}],
			"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18,
				"prompt_tokens_details":{"cached_tokens":5},"completion_tokens_details":{"reasoning_tokens":2}}}`)
	}))
	defer server.Close()
	settings, err := (mistral.Settings{
		Common: ai.ModelSettings{
			MaxTokens: 80, Temperature: floatPointer(0.2), TopP: floatPointer(0.9), Seed: intPointer(4),
			PresencePenalty: floatPointer(0.3), FrequencyPenalty: floatPointer(0.1),
			StopSequences: []string{"STOP"}, ParallelToolCalls: boolPointer(false),
			ExtraBody: map[string]any{"safe_prompt": true}, ExtraHeaders: map[string]string{"X-Run": "request"},
			Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelHigh},
		},
		PromptCacheKey: "conversation", ToolChoice: mistral.ToolChoiceAuto, AllowedTools: []string{"keep"},
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	model := mistral.NewModel(
		"mistral-small-latest",
		mistral.WithProvider(mistral.ProviderConfig{
			Name: "mistral-gateway", BaseURL: server.URL, APIKey: "key", HTTPClient: server.Client(),
			Headers: http.Header{"X-Run": []string{"provider"}},
			PrepareRequest: func(request *http.Request) error {
				request.Header.Set("X-Dynamic", "yes")
				return nil
			},
		}),
		mistral.WithDefaultSettings(ai.ModelSettings{MaxTokens: 10}),
	)
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "legacy"},
			ai.UserPromptPart{Contents: []ai.UserContent{
				ai.TextContent{Text: "look"},
				ai.ImageURL{URL: "https://example.com/image.png", VendorMetadata: map[string]any{"detail": "high"}},
				ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"},
				ai.BinaryContent{Data: []byte("document"), MediaType: "application/pdf"},
				ai.BinaryContent{Data: []byte("text"), MediaType: "text/plain", Identifier: "notes"},
				ai.CachePoint{},
			}},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ThinkingPart{Content: "prior reasoning"}, ai.TextPart{Content: "prior"},
			ai.ToolCallPart{ToolName: "keep", Args: json.RawMessage(`{"old":true}`), ToolCallID: "old-call"},
		}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "keep", ToolCallID: "old-call", Content: ai.ToolReturn{
				ReturnValue: map[string]any{"ok": true},
				Content:     []ai.UserContent{ai.BinaryContent{Data: []byte("tool-image"), MediaType: "image/png"}},
			}},
		}},
	}
	response, err := model.Request(t.Context(), messages, ai.ModelRequestParams{
		InstructionParts: []ai.InstructionPart{{Content: "instruction"}},
		Tools: []ai.ToolDefinition{
			{Name: "keep", Description: "Keep it", Schema: map[string]any{"type": "object"}},
			{Name: "drop", Schema: map[string]any{"type": "object"}},
		},
		AllowText: true, Settings: settings,
	})
	if err != nil {
		t.Fatal(err)
	}
	if response.ModelName != "mistral-resolved" || response.ProviderName != "mistral-gateway" ||
		response.FinishReason != ai.FinishReasonLength || response.ProviderResponseID != "response-id" ||
		response.Usage.InputTokens != 11 || response.Usage.OutputTokens != 7 || response.Usage.CacheReadTokens != 5 ||
		response.Usage.ReasoningTokens != 2 || !response.Timestamp.Equal(time.Unix(1704067200, 0)) {
		t.Fatalf("unexpected response: %#v", response)
	}
	if len(response.Parts) != 5 || response.Parts[0].(ai.ThinkingPart).Content != "one" ||
		response.Parts[1].(ai.ThinkingPart).Content != "two" || response.Parts[2].(ai.TextPart).Content != "hello world" ||
		string(response.Parts[3].(ai.ToolCallPart).Args) != `{"x":1}` ||
		response.Parts[4].(ai.ToolCallPart).ToolCallID == "" {
		t.Fatalf("unexpected parts: %#v", response.Parts)
	}
	if model.Name() != "mistral-small-latest" || model.ProviderName() != "mistral-gateway" ||
		model.ProviderURL() != server.URL || model.DefaultModelSettings().MaxTokens != 10 {
		t.Fatalf("unexpected model metadata")
	}
	assertRequest(t, captured)
}

func assertRequest(t *testing.T, captured map[string]any) {
	t.Helper()
	if captured["model"] != "mistral-small-latest" || captured["reasoning_effort"] != "high" ||
		captured["prompt_cache_key"] != "conversation" || captured["tool_choice"] != "auto" ||
		captured["random_seed"] != float64(4) || captured["safe_prompt"] != true {
		t.Fatalf("unexpected payload: %#v", captured)
	}
	tools := captured["tools"].([]any)
	if len(tools) != 1 || tools[0].(map[string]any)["function"].(map[string]any)["name"] != "keep" {
		t.Fatalf("unexpected tools: %#v", tools)
	}
	messages := captured["messages"].([]any)
	roles := make([]string, len(messages))
	for index, message := range messages {
		roles[index] = message.(map[string]any)["role"].(string)
	}
	if !reflect.DeepEqual(roles, []string{"system", "system", "user", "assistant", "tool", "assistant", "user"}) {
		t.Fatalf("unexpected message roles: %#v", roles)
	}
	userContent := messages[2].(map[string]any)["content"].([]any)
	if len(userContent) != 5 || userContent[1].(map[string]any)["type"] != "image_url" ||
		userContent[3].(map[string]any)["type"] != "document_url" ||
		!strings.Contains(userContent[4].(map[string]any)["text"].(string), `id="notes"`) {
		t.Fatalf("unexpected user content: %#v", userContent)
	}
	toolContent := messages[4].(map[string]any)["content"].(string)
	if !strings.Contains(toolContent, `{"ok":true}`) || !strings.Contains(toolContent, "See file") {
		t.Fatalf("unexpected tool content: %q", toolContent)
	}
}

func TestProviderConfiguration(t *testing.T) {
	t.Setenv("MISTRAL_API_KEY", "environment-key")
	t.Setenv("MISTRAL_BASE_URL", "https://mistral.example/v1/")
	provider := mistral.NewProviderConfig()
	provider.Headers = http.Header{"X-Test": []string{"value"}}
	clone := provider.Clone()
	provider.Headers.Set("X-Test", "changed")
	if clone.Name != "mistral" || clone.BaseURL != "https://mistral.example/v1" ||
		clone.APIKey != "environment-key" || clone.Headers.Get("X-Test") != "value" {
		t.Fatalf("unexpected provider: %#v", clone)
	}
}

func floatPointer(value float64) *float64 { return &value }
func intPointer(value int) *int           { return &value }
func boolPointer(value bool) *bool        { return &value }

var _ ai.Model = mistral.NewModel("model")
var _ ai.StreamingModel = mistral.NewModel("model")
var _ ai.ModelProviderIdentity = mistral.NewModel("model")
var _ ai.ModelDefaultSettings = mistral.NewModel("model")
var _ ai.ModelAPIError = (*mistral.APIError)(nil)

func TestSingleForcedToolKeepsDeclarations(t *testing.T) {
	forceKeep, err := (mistral.Settings{ToolChoice: mistral.ToolChoiceAny, AllowedTools: []string{"keep"}}).Build()
	if err != nil {
		t.Fatal(err)
	}
	disableFunctions, err := (mistral.Settings{ToolChoice: mistral.ToolChoiceNone}).Build()
	if err != nil {
		t.Fatal(err)
	}
	tools := []ai.ToolDefinition{{Name: "keep"}, {Name: "other"}}
	output := &ai.ToolDefinition{Name: "final_result"}
	tests := []struct {
		name   string
		params ai.ModelRequestParams
		tools  []string
		choice any
	}{
		{
			name:   "allowed single",
			params: ai.ModelRequestParams{Tools: tools, AllowText: true, Settings: forceKeep},
			tools:  []string{"keep", "other"},
			choice: map[string]any{"type": "function", "function": map[string]any{"name": "keep"}},
		},
		{
			name:   "none with output",
			params: ai.ModelRequestParams{Tools: tools, OutputTool: output, Settings: disableFunctions},
			tools:  []string{"keep", "other", "final_result"},
			choice: map[string]any{"type": "function", "function": map[string]any{"name": "final_result"}},
		},
		{
			name:   "only output",
			params: ai.ModelRequestParams{OutputTool: output, Settings: disableFunctions},
			tools:  []string{"final_result"},
			choice: "any",
		},
		{
			name:   "allowed single with output",
			params: ai.ModelRequestParams{Tools: tools, OutputTool: output, AllowText: true, Settings: forceKeep},
			tools:  []string{"keep", "final_result"},
			choice: "any",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var captured map[string]any
			server := responseServer(t, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`, &captured)
			defer server.Close()
			model := mistral.NewModel("mistral-small-latest", mistral.WithBaseURL(server.URL), mistral.WithHTTPClient(server.Client()))
			if _, err := model.Request(t.Context(), nil, test.params); err != nil {
				t.Fatal(err)
			}
			var names []string
			for _, tool := range captured["tools"].([]any) {
				names = append(names, tool.(map[string]any)["function"].(map[string]any)["name"].(string))
			}
			if !reflect.DeepEqual(names, test.tools) || !reflect.DeepEqual(captured["tool_choice"], test.choice) {
				t.Fatalf("unexpected tools %v and choice %#v", names, captured["tool_choice"])
			}
		})
	}
}

func TestLarge4AdjustableReasoning(t *testing.T) {
	for _, name := range []string{"mistral-large-4", "mistral-large-4-0", "mistral-large-4-preview"} {
		for _, level := range []ai.ThinkingLevel{ai.ThinkingLevelHigh, ai.ThinkingLevelDisabled} {
			var captured map[string]any
			server := responseServer(t, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`, &captured)
			model := mistral.NewModel(name, mistral.WithBaseURL(server.URL), mistral.WithHTTPClient(server.Client()))
			_, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
				Settings: ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: level}},
			})
			server.Close()
			if err != nil {
				t.Fatal(err)
			}
			want := map[ai.ThinkingLevel]any{ai.ThinkingLevelHigh: "high", ai.ThinkingLevelDisabled: "none"}[level]
			if name == "mistral-large-4-preview" {
				want = nil
			}
			if captured["reasoning_effort"] != want {
				t.Fatalf("%s %s: unexpected reasoning effort %#v", name, level, captured["reasoning_effort"])
			}
		}
	}
}

func TestForeignToolCallIDsAreRewritten(t *testing.T) {
	var bodies []string
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		body, _ := io.ReadAll(request.Body)
		bodies = append(bodies, string(body))
		_, _ = io.WriteString(response, `{"model":"voxtral-small-latest","choices":[
			{"finish_reason":"stop","message":{"content":"Sunny, rainy and cloudy."}}],"usage":{}}`)
	}))
	defer server.Close()
	model := mistral.NewModel("voxtral-small-latest", mistral.WithProvider(mistral.ProviderConfig{
		Name: "mistral", BaseURL: server.URL, HTTPClient: server.Client(),
	}))
	agent := ai.NewAgent[struct{}, string](model)
	call := func(city, id string) ai.ToolCallPart {
		return ai.ToolCallPart{ToolName: "get_weather", ToolCallID: id, Args: json.RawMessage(`{"city":"` + city + `"}`)}
	}
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "Weather in Paris, London and Berlin?"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			call("Paris", "call_abc123def456ghi789jkl012"),
			call("Londres", "toolu_01A09q90qw90lq917835lq9"),
			call("Berlin", "pyd_ai_0123456789abcdef0123456789abcdef"),
		}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "get_weather", ToolCallID: "call_abc123def456ghi789jkl012", Content: "Sunny"},
			ai.RetryPromptPart{ToolName: "get_weather", ToolCallID: "toolu_01A09q90qw90lq917835lq9", Content: "English"},
			ai.ToolReturnPart{ToolName: "get_weather", ToolCallID: "pyd_ai_0123456789abcdef0123456789abcdef", Content: "Cloudy"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{call("London", "aB3dE5gH9")}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "get_weather", ToolCallID: "aB3dE5gH9", Content: "Rainy"},
		}},
	}

	for range 2 {
		if _, err := agent.Run(t.Context(), "Summarize.", struct{}{}, ai.WithMessageHistory(history)); err != nil {
			t.Fatal(err)
		}
	}

	if len(bodies) != 2 || bodies[0] != bodies[1] {
		t.Fatalf("the same history sent different bytes: %q", bodies)
	}
	var payload struct {
		Messages []struct {
			Role       string `json:"role"`
			ToolCallID string `json:"tool_call_id"`
			ToolCalls  []struct {
				ID string `json:"id"`
			} `json:"tool_calls"`
		} `json:"messages"`
	}
	if err := json.Unmarshal([]byte(bodies[0]), &payload); err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, message := range payload.Messages {
		for _, call := range message.ToolCalls {
			ids = append(ids, message.Role+":"+call.ID)
		}
		if message.Role == "tool" {
			ids = append(ids, "tool:"+message.ToolCallID)
		}
	}
	want := []string{
		"assistant:2pESUE4jw", "assistant:l4LUGq9qS", "assistant:9qOPVjwIC",
		"tool:2pESUE4jw", "tool:l4LUGq9qS", "tool:9qOPVjwIC", "assistant:aB3dE5gH9", "tool:aB3dE5gH9",
	}
	if !reflect.DeepEqual(ids, want) {
		t.Fatalf("wire IDs = %q, want %q", ids, want)
	}
	if id := history[1].(ai.ModelResponse).Parts[0].(ai.ToolCallPart).ToolCallID; id != "call_abc123def456ghi789jkl012" {
		t.Fatalf("stored history changed: %q", id)
	}
}
