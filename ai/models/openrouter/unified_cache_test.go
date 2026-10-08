package openrouter_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/openrouter"
)

func TestUnifiedOpenRouterCaching(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body = nil
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"route","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer server.Close()
	noMessages := false
	for _, test := range []struct {
		name, model string
		cache       *ai.CacheConfig
		local       bool
		want        bool
		hour        bool
	}{
		{name: "default", model: "anthropic/claude-sonnet-4-5", cache: &ai.CacheConfig{}, want: true},
		{name: "hour", model: "anthropic/claude-sonnet-4-5", cache: &ai.CacheConfig{Retention: ai.CacheRetention1Hour}, want: true, hour: true},
		{name: "stable prefix", model: "anthropic/claude-sonnet-4-5", cache: &ai.CacheConfig{Messages: &noMessages}, want: true},
		{name: "Gemini", model: "google/gemini-2.5-pro", cache: &ai.CacheConfig{Retention: ai.CacheRetention1Hour}, want: true},
		{name: "Gemini3", model: "google/gemini-3", cache: &ai.CacheConfig{}, want: true},
		{name: "Gemini4", model: "google/gemini-4", cache: &ai.CacheConfig{}, want: true},
		{name: "old Gemini", model: "google/gemini-2.0", cache: &ai.CacheConfig{}},
		{name: "unsupported", model: "meta/llama", cache: &ai.CacheConfig{}},
		{name: "disabled", model: "anthropic/claude-sonnet-4-5", cache: &ai.CacheConfig{Retention: ai.CacheRetentionDisabled}},
		{name: "provider disabled", model: "anthropic/claude-sonnet-4-5", cache: &ai.CacheConfig{}, local: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := openrouter.NewModel(test.model, openrouter.WithBaseURL(server.URL))
			settings := ai.ModelSettings{Cache: test.cache}
			if test.local {
				settings.ExtraBody = map[string]any{"openrouter_cache_messages": false}
			}
			params := ai.ModelRequestParams{Instructions: "stable", Tools: []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}}}, Settings: settings}
			if _, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hi"}}}}, params); err != nil {
				t.Fatal(err)
			}
			encoded, _ := json.Marshal(body)
			if strings.Contains(string(encoded), "cache_control") != test.want {
				t.Fatalf("body=%s", encoded)
			}
			if strings.HasPrefix(test.model, "google/") && strings.Contains(string(encoded), `"ttl"`) {
				t.Fatalf("Gemini got TTL: %s", encoded)
			}
			wantKnown := test.want && strings.HasPrefix(test.model, "anthropic/")
			retention, known := ai.ResolveCacheRetention(model, &settings)
			want := 5 * time.Minute
			if test.hour {
				want = time.Hour
			}
			if known != wantKnown || known && retention != want {
				t.Fatalf("retention=%s %v", retention, known)
			}
			if model.CachingNotEnabled(settings) {
				t.Fatal("configured route reported missing")
			}
		})
	}
	model := openrouter.NewModel("anthropic/claude-sonnet-4-5", openrouter.WithBaseURL(server.URL))
	if !model.CachingNotEnabled(ai.ModelSettings{}) || openrouter.NewModel("google/gemini-2.5-pro").CachingNotEnabled(ai.ModelSettings{}) {
		t.Fatal("incorrect configuration diagnostic")
	}
	bad := ai.ModelSettings{Cache: &ai.CacheConfig{Retention: "week"}}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: bad}); err == nil {
		t.Fatal("invalid cache accepted")
	}
	if _, known := model.PromptCacheRetention(bad); known {
		t.Fatal("invalid retention claimed")
	}
	calls := make([]ai.ResponsePart, 12)
	returns := make([]ai.RequestPart, 12)
	for index := range calls {
		calls[index] = ai.ToolCallPart{ToolName: "lookup", ToolCallID: "call", Args: []byte(`{}`)}
		returns[index] = ai.ToolReturnPart{ToolName: "lookup", ToolCallID: "call", Content: "result"}
	}
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "old"}}}, ai.ModelResponse{Parts: calls}, ai.ModelRequest{Parts: returns}}
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{Settings: ai.ModelSettings{Cache: &ai.CacheConfig{}}}); err != nil {
		t.Fatal(err)
	}
	wire := body["messages"].([]any)
	if wire[0].(map[string]any)["content"].([]any)[0].(map[string]any)["cache_control"] == nil {
		t.Fatal("wide turn lost previous tail breakpoint")
	}
}
