package openai_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

func TestChatWebSearch(t *testing.T) {
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		if body["stream"] == true {
			response.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(response,
				"data: {\"model\":\"gpt-4o-search-preview\",\"choices\":["+
					"{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\n"+
					"data: [DONE]\n\n",
			)
			return
		}
		_, _ = io.WriteString(response, `{
			"model":"gpt-4o-search-preview",
			"choices":[{"message":{"content":"done"},"finish_reason":"stop"}]
		}`)
	}))
	defer server.Close()

	location := &ai.WebSearchUserLocation{City: "Utrecht", Country: "NL"}
	externalWebAccess := false
	model := openai.NewModel(
		"gpt-4o-search-preview", openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client()),
	)
	webSearch := ai.WebSearchTool{
		SearchContextSize: ai.WebSearchContextLow,
		UserLocation:      location,
		AllowedDomains:    []string{"example.com"},
		BlockedDomains:    []string{"blocked.example"},
		MaxUses:           2,
		ExternalWebAccess: &externalWebAccess,
	}
	agent := ai.NewAgent[struct{}, string](model, ai.WithNativeTools(webSearch))
	agent.AddTool(ai.NewSimpleTool[struct{}]("local_search", func(context.Context, struct{}) (string, error) {
		return "local", nil
	}, ai.WithNativeFallback(webSearch)))
	location.City = "changed"
	if _, err := agent.Run(t.Context(), "search", struct{}{}); err != nil {
		t.Fatal(err)
	}

	streamLocation := &ai.WebSearchUserLocation{Region: "South Holland", Timezone: "Europe/Amsterdam"}
	if _, err := collect(t, model, ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		&ai.WebSearchTool{UserLocation: streamLocation},
	}}); err != nil {
		t.Fatal(err)
	}

	first := bodies[0]["web_search_options"].(map[string]any)
	firstLocation := first["user_location"].(map[string]any)
	firstApproximate := firstLocation["approximate"].(map[string]any)
	if first["search_context_size"] != "low" || firstLocation["type"] != "approximate" ||
		firstApproximate["city"] != "Utrecht" || firstApproximate["country"] != "NL" ||
		first["allowed_domains"] != nil || first["blocked_domains"] != nil || first["max_uses"] != nil ||
		first["external_web_access"] != nil || bodies[0]["tools"] != nil {
		t.Fatalf("unexpected static web search options: %#v", bodies[0])
	}
	second := bodies[1]["web_search_options"].(map[string]any)
	secondApproximate := second["user_location"].(map[string]any)["approximate"].(map[string]any)
	if second["search_context_size"] != "medium" || secondApproximate["region"] != "South Holland" ||
		secondApproximate["timezone"] != "Europe/Amsterdam" {
		t.Fatalf("unexpected streamed web search options: %#v", bodies[1])
	}
}

func TestChatWebSearchSupportOverride(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		requests++
		_, _ = io.WriteString(response, `{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	future := openai.NewModel(
		"future-search", openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client()),
		openai.WithChatWebSearchSupport(true),
	)
	if _, err := future.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
	}); err != nil {
		t.Fatal(err)
	}
	disabled := openai.NewModel(
		"gpt-4o-search-preview", openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client()),
		openai.WithChatWebSearchSupport(false),
	)
	if _, err := disabled.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
	}); err == nil {
		t.Fatal("expected explicit web search disablement to fail")
	}
	if _, err := future.Request(t.Context(), nil, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
		Settings: ai.ModelSettings{ExtraBody: map[string]any{
			"web_search_options": map[string]any{"search_context_size": "high"},
		}},
	}); err == nil || !strings.Contains(err.Error(), `field "web_search_options" conflicts`) {
		t.Fatalf("unexpected web search options conflict: %v", err)
	}
	fallback := openai.NewModel("gpt-5", openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client()))
	if _, err := fallback.Request(t.Context(), nil, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{
			Name: "local_search", Schema: map[string]any{"type": "object"},
			NativeFallbackFor: ai.WebSearchTool{}.UniqueID(),
		}},
		NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
	}); err != nil {
		t.Fatal(err)
	}
	if requests != 2 {
		t.Fatalf("unexpected transport requests: %d", requests)
	}
}
