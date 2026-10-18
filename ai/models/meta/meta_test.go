package meta_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/meta"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestProviderConfig(t *testing.T) {
	t.Setenv("META_API_KEY", "key")
	provider := meta.NewProviderConfig()
	if provider.Name != "meta" || provider.BaseURL != "https://api.meta.ai/v1" || provider.APIKey != "key" {
		t.Fatalf("unexpected provider: %#v", provider)
	}
}

type responseServer struct {
	*httptest.Server
	mu        sync.Mutex
	bodies    []map[string]any
	headers   []http.Header
	paths     []string
	responses []string
}

func newResponseServer(t *testing.T, responses ...string) *responseServer {
	t.Helper()
	captured := &responseServer{responses: responses}
	captured.Server = httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		captured.mu.Lock()
		defer captured.mu.Unlock()
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		captured.bodies = append(captured.bodies, body)
		captured.headers = append(captured.headers, request.Header.Clone())
		captured.paths = append(captured.paths, request.URL.Path)
		index := len(captured.bodies) - 1
		if index >= len(captured.responses) {
			t.Errorf("unexpected request %d", index)
			response.WriteHeader(http.StatusInternalServerError)
			return
		}
		if strings.HasPrefix(captured.responses[index], "event:") {
			response.Header().Set("Content-Type", "text/event-stream")
		}
		_, _ = io.WriteString(response, captured.responses[index])
	}))
	t.Cleanup(captured.Close)
	return captured
}

func (server *responseServer) options() []meta.Option {
	return []meta.Option{
		meta.WithBaseURL(server.URL), meta.WithAPIKey("secret"), meta.WithHTTPClient(server.Client()),
	}
}

func textResponse(text string) string {
	return fmt.Sprintf(`{"id":"resp_2","model":"muse-spark-1.3","status":"completed","service_tier":"auto",`+
		`"output":[{"type":"message","id":"msg_2","role":"assistant","status":"completed",`+
		`"content":[{"type":"output_text","text":%q,"annotations":[]}]}],`+
		`"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}`, text)
}

const toolCallResponse = `{"id":"resp_1","model":"muse-spark-1.3","status":"completed","service_tier":"auto","output":[
	{"type":"reasoning","id":"rs_1:rs_a","encrypted_content":"opaque","summary":[],"status":"completed"},
	{"type":"message","id":"msg_1","role":"assistant","phase":"commentary","status":"completed",
	 "content":[{"type":"output_text","text":"I'll check the weather in Paris.","annotations":[]}]},
	{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Paris\"}","status":"completed"}
],"usage":{"input_tokens":544,"output_tokens":404,"output_tokens_details":{"reasoning_tokens":335},"total_tokens":948}}`

func TestMuseToolCallReplaysPhaseAndEncryptedReasoning(t *testing.T) {
	server := newResponseServer(t, toolCallResponse, textResponse("It is sunny and 21C in Paris."))
	agent := ai.NewAgent[struct{}, string](meta.NewModel("muse-spark-1.3", server.options()...))
	ai.AddSimpleTool(agent, "get_weather", func(_ context.Context, args struct {
		City string `json:"city"`
	}) (string, error) {
		return "Sunny and 21C in " + args.City + ".", nil
	})
	result, err := agent.Run(t.Context(), "What is the weather in Paris?", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "It is sunny and 21C in Paris." {
		t.Fatalf("unexpected output %q", result.Output)
	}
	if server.paths[0] != "/responses" || server.headers[0].Get("Authorization") != "Bearer secret" {
		t.Fatalf("unexpected request %s %v", server.paths[0], server.headers[0])
	}
	for _, body := range server.bodies {
		if fmt.Sprint(body["include"]) != "[reasoning.encrypted_content]" {
			t.Fatalf("encrypted reasoning was not requested: %#v", body)
		}
	}
	input := server.bodies[1]["input"].([]any)
	reasoning := input[1].(map[string]any)
	message := input[2].(map[string]any)
	if reasoning["type"] != "reasoning" || reasoning["id"] != "rs_1:rs_a" || reasoning["encrypted_content"] != "opaque" ||
		message["phase"] != "commentary" || input[3].(map[string]any)["call_id"] != "call_1" {
		t.Fatalf("reasoning and phase were not replayed: %#v", input)
	}
}

func TestMuseOutputToolFallsBackToAutoToolChoice(t *testing.T) {
	for _, newModel := range []func(string, ...meta.Option) *meta.Model{meta.NewModel, meta.NewChatModel} {
		server := newResponseServer(t, "{}")
		_, _ = newModel("muse-spark-1.3", server.options()...).Request(t.Context(), nil, ai.ModelRequestParams{
			OutputTool: &ai.ToolDefinition{Name: "final_result", Schema: map[string]any{"type": "object"}},
		})
		if server.bodies[0]["tool_choice"] != "auto" {
			t.Fatalf("output tool was forced: %#v", server.bodies[0])
		}
	}
}

func TestMuseThinkingAndSampling(t *testing.T) {
	temperature := 0.3
	tests := []struct {
		name     string
		thinking *ai.ThinkingSettings
		effort   any
	}{
		{name: "disabled keeps reasoning on", thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled}},
		{name: "unified effort", thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelLow}, effort: "low"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := newResponseServer(t, textResponse("ok"), "{}")
			params := ai.ModelRequestParams{Settings: ai.ModelSettings{Temperature: &temperature, Thinking: test.thinking}}
			if _, err := meta.NewModel("muse-spark-1.3", server.options()...).Request(t.Context(), nil, params); err != nil {
				t.Fatal(err)
			}
			_, _ = meta.NewChatModel("muse-spark-1.3", server.options()...).Request(t.Context(), nil, params)
			responses, chat := server.bodies[0], server.bodies[1]
			reasoning, _ := responses["reasoning"].(map[string]any)
			if reasoning["effort"] != test.effort || chat["reasoning_effort"] != test.effort {
				t.Fatalf("unexpected reasoning: %#v %#v", responses, chat)
			}
			if responses["temperature"] != 0.3 || chat["temperature"] != 0.3 {
				t.Fatalf("sampling was dropped: %#v %#v", responses, chat)
			}
		})
	}
}

func TestMuseNativeTools(t *testing.T) {
	server := newResponseServer(t, textResponse("ok"), textResponse("ok"))
	model := meta.NewModel("muse-spark-1.3", server.options()...)
	if !model.SupportsNativeTool(ai.WebSearchTool{}) || model.SupportsNativeTool(ai.ImageGenerationTool{}) ||
		model.SupportsNativeTool(nil) {
		t.Fatal("Muse should support web search only")
	}
	if !meta.NewChatModel("muse-spark-1.3").SupportsNativeTool(ai.WebSearchTool{}) {
		t.Fatal("Muse chat should support web search")
	}
	params := ai.ModelRequestParams{NativeTools: []ai.NativeTool{
		ai.WebSearchTool{}, ai.ImageGenerationTool{Optional: true},
	}}
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	if tools := server.bodies[0]["tools"].([]any); len(tools) != 1 || tools[0].(map[string]any)["type"] != "web_search" {
		t.Fatalf("unexpected native tools: %#v", tools)
	}
	required := ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.ImageGenerationTool{}}}
	if _, err := model.Request(t.Context(), nil, required); err == nil ||
		err.Error() != `meta: native tool "image_generation" is not supported by model "muse-spark-1.3"` {
		t.Fatalf("unexpected error: %v", err)
	}
	duplicate := ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.WebSearchTool{}, ai.WebSearchTool{}}}
	if _, err := model.Request(t.Context(), nil, duplicate); err == nil {
		t.Fatal("expected duplicate native tool error")
	}
	if _, err := model.StreamRequest(t.Context(), nil, duplicate); err == nil {
		t.Fatal("expected streamed duplicate native tool error")
	}
}

func TestMuseStream(t *testing.T) {
	server := newResponseServer(t, strings.Join([]string{
		`event: response.output_item.added`,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","content":[]}}`,
		``,
		`event: response.output_text.delta`,
		`data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"item_id":"msg_1","delta":"1, 2, 3"}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","response":` + textResponse("1, 2, 3") + `}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n"))
	thinking := ai.ModelRequestParams{Settings: ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelDisabled}}}
	events, err := meta.NewModel("muse-spark-1.3", server.options()...).StreamRequest(t.Context(), nil, thinking)
	if err != nil {
		t.Fatal(err)
	}
	text := ""
	for event, err := range events {
		if err != nil {
			t.Fatal(err)
		}
		if delta, ok := event.(ai.TextDeltaEvent); ok {
			text += delta.Delta
		}
	}
	if text != "1, 2, 3" || server.bodies[0]["stream"] != true || server.bodies[0]["reasoning"] != nil {
		t.Fatalf("unexpected stream %q %#v", text, server.bodies[0])
	}
}

func TestNonMuseModelsUseEndpointDefaults(t *testing.T) {
	server := newResponseServer(t, textResponse("ok"), "{}")
	params := ai.ModelRequestParams{Settings: ai.ModelSettings{Thinking: &ai.ThinkingSettings{Level: ai.ThinkingLevelHigh}}}
	model := meta.NewModel("muse-image-1.0", server.options()...)
	if _, err := model.Request(t.Context(), nil, params); err != nil {
		t.Fatal(err)
	}
	_, _ = meta.NewChatModel("muse-image-1.0", server.options()...).Request(t.Context(), nil, params)
	if server.bodies[0]["include"] != nil || server.bodies[0]["reasoning"] != nil || server.bodies[1]["reasoning_effort"] != nil {
		t.Fatalf("non-Muse request used Muse behavior: %#v", server.bodies)
	}
	if !model.SupportsNativeTool(ai.ImageGenerationTool{}) || meta.NewChatModel("muse-image-1.0").SupportsNativeTool(ai.WebSearchTool{}) {
		t.Fatal("non-Muse native tools should follow the endpoint")
	}
}

func TestModelProfileAndOptions(t *testing.T) {
	t.Setenv("META_API_KEY", "")
	temperature := 0.5
	server := newResponseServer(t, textResponse("ok"))
	model := meta.NewModel("muse-spark-1.3",
		meta.WithProvider(openai.ProviderConfig{Name: "gateway", BaseURL: server.URL, APIKey: "gateway-key"}),
		meta.WithHTTPClient(server.Client()),
		meta.WithDefaultSettings(ai.ModelSettings{Temperature: &temperature}),
	)
	profile := model.ModelProfile()
	if model.ProviderName() != "gateway" || profile.SupportsImageOutput || profile.DefaultOutputMode != ai.OutputModeTool ||
		meta.NewModel("muse-spark-1.3").ModelProfile().ContextWindow != 1_048_576 {
		t.Fatalf("unexpected profile %s %#v", model.ProviderName(), profile)
	}
	if _, err := ai.RequestModel(t.Context(), model, nil, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	if server.headers[0].Get("Authorization") != "Bearer gateway-key" || server.bodies[0]["temperature"] != 0.5 {
		t.Fatalf("unexpected request %v %#v", server.headers[0], server.bodies[0])
	}
}
