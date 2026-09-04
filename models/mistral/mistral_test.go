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

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/mistral"
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
