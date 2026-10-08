package openai_test

import (
	"encoding/json"
	"net/http"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestCompatibleChatCacheKeepsWideTurnPrefix(t *testing.T) {
	var body map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`))
	})
	calls := make([]ai.ResponsePart, 12)
	returns := make([]ai.RequestPart, 12)
	for index := range calls {
		calls[index] = ai.ToolCallPart{ToolName: "lookup", ToolCallID: "call", Args: []byte(`{}`)}
		returns[index] = ai.ToolReturnPart{ToolName: "lookup", ToolCallID: "call", Content: "result"}
	}
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.TextContent{Text: "old"}, ai.CachePoint{TTL: ai.CachePointTTL1Hour},
		}}}},
		ai.ModelResponse{Parts: calls}, ai.ModelRequest{Parts: returns},
	}
	ctx := openai.WithChatPromptCache(t.Context(), openai.ChatPromptCache{
		MessagesTTL: "5m", IncludeTTL: true, MaxPoints: 4,
		ExplicitMarkerStyle: openai.ChatPromptCacheMarkerControl,
	})
	if _, err := model.Request(ctx, messages, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	wire := body["messages"].([]any)
	if wire[0].(map[string]any)["content"].([]any)[0].(map[string]any)["cache_control"].(map[string]any)["ttl"] != "1h" {
		t.Fatal("library caching replaced the explicit prefix TTL")
	}
	messages = []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{
		Contents: []ai.UserContent{
			ai.TextContent{Text: "prefix"}, ai.CachePoint{TTL: ai.CachePointTTL1Hour},
			ai.TextContent{Text: "new tail"},
		},
	}}}}
	if _, err := model.Request(ctx, messages, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	parts := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if parts[1].(map[string]any)["cache_control"].(map[string]any)["ttl"] != "5m" {
		t.Fatal("new conversation tail not cached")
	}
}
