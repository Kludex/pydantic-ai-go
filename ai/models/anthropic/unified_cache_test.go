package anthropic_test

import (
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/anthropic"
)

func TestUnifiedAnthropicCaching(t *testing.T) {
	messagesFlag := false
	var body map[string]any
	model := newServer(t, func(w http.ResponseWriter, r *http.Request) {
		body = nil
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"model":"m","content":[{"type":"text","text":"ok"}],"usage":{}}`))
	})
	for _, test := range []struct {
		name   string
		cache  *ai.CacheConfig
		local  bool
		ttl    string
		static bool
	}{
		{name: "default", cache: &ai.CacheConfig{}, ttl: "5m"},
		{name: "snap down", cache: &ai.CacheConfig{Retention: ai.CacheRetention30Minutes}, ttl: "5m"},
		{name: "hour", cache: &ai.CacheConfig{Retention: ai.CacheRetention1Hour}, ttl: "1h"},
		{name: "stable prefix", cache: &ai.CacheConfig{Retention: ai.CacheRetention1Hour, Messages: &messagesFlag}, ttl: "1h", static: true},
		{name: "disabled", cache: &ai.CacheConfig{Retention: ai.CacheRetentionDisabled}},
		{name: "provider disabled", cache: &ai.CacheConfig{Retention: ai.CacheRetention1Hour}, local: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			settings := ai.ModelSettings{Cache: test.cache}
			if test.local {
				settings.ExtraBody = map[string]any{"anthropic_cache_instructions": false}
			}
			params := ai.ModelRequestParams{Instructions: "stable\n\ndynamic", InstructionParts: []ai.InstructionPart{{Content: "stable"}, {Content: "dynamic", Dynamic: true}}, Tools: []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}}}, Settings: settings}
			if _, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}}}}, params); err != nil {
				t.Fatal(err)
			}
			switch {
			case test.ttl == "":
				if body["cache_control"] != nil {
					t.Fatalf("unexpected automatic cache: %+v", body)
				}
			case test.static:
				if body["cache_control"] != nil || body["system"].([]any)[0].(map[string]any)["cache_control"].(map[string]any)["ttl"] != test.ttl || body["system"].([]any)[1].(map[string]any)["cache_control"] != nil || body["tools"].([]any)[0].(map[string]any)["cache_control"] == nil {
					t.Fatalf("static boundaries=%+v", body)
				}
			default:
				if body["cache_control"].(map[string]any)["ttl"] != test.ttl ||
					body["system"].([]any)[0].(map[string]any)["cache_control"].(map[string]any)["ttl"] != test.ttl ||
					body["tools"].([]any)[0].(map[string]any)["cache_control"].(map[string]any)["ttl"] != test.ttl {
					t.Fatalf("cache=%+v", body)
				}
			}
			retention, known := ai.ResolveCacheRetention(model, &settings)
			want := 5 * time.Minute
			if test.ttl == "1h" {
				want = time.Hour
			}
			if (test.ttl != "") != known || known && retention != want {
				t.Fatalf("retention=%s known=%v", retention, known)
			}
			if model.CachingNotEnabled(settings) {
				t.Fatal("configured cache reported missing")
			}
		})
	}
	if !model.CachingNotEnabled(ai.ModelSettings{}) {
		t.Fatal("unconfigured cache not reported")
	}
	bad := ai.ModelSettings{Cache: &ai.CacheConfig{Retention: "week"}}
	if _, err := model.Request(t.Context(), nil, ai.ModelRequestParams{Settings: bad}); err == nil {
		t.Fatal("invalid cache accepted")
	}
	if _, known := model.PromptCacheRetention(bad); known {
		t.Fatal("invalid retention claimed")
	}
	hour := ai.ModelSettings{Cache: &ai.CacheConfig{Retention: ai.CacheRetention1Hour}}
	if _, err := model.Request(t.Context(), []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{
		Contents: []ai.UserContent{ai.TextContent{Text: "early"}, ai.CachePoint{}, ai.TextContent{Text: "late"}},
	}}}}, ai.ModelRequestParams{Settings: hour}); err != nil {
		t.Fatal(err)
	}
	early := body["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	if early["cache_control"].(map[string]any)["ttl"] != "1h" || body["cache_control"].(map[string]any)["ttl"] != "1h" {
		t.Fatalf("earlier 5m breakpoint was not raised: %+v", body)
	}
}

func TestLegacyBedrockUnifiedCacheWideTurn(t *testing.T) {
	var body map[string]any
	client := &legacyBedrockClient{invoke: func(input *bedrockruntime.InvokeModelInput) (*bedrockruntime.InvokeModelOutput, error) {
		if err := json.Unmarshal(input.Body, &body); err != nil {
			t.Fatal(err)
		}
		return &bedrockruntime.InvokeModelOutput{Body: []byte(`{"model":"m","content":[{"type":"text","text":"ok"}],"usage":{}}`)}, nil
	}}
	for _, name := range []string{"us.anthropic.claude-sonnet-4-v1:0", "us.anthropic.claude-sonnet-4-5-v1:0"} {
		model := anthropic.NewLegacyBedrockModel(name, anthropic.LegacyBedrockConfig{Client: client})
		calls := make([]ai.ResponsePart, 12)
		returns := make([]ai.RequestPart, 12)
		for index := range calls {
			calls[index] = ai.ToolCallPart{ToolName: "lookup", ToolCallID: "call", Args: []byte(`{}`)}
			returns[index] = ai.ToolReturnPart{ToolName: "lookup", ToolCallID: "call", Content: "result"}
		}
		messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "old"}}}, ai.ModelResponse{Parts: calls}, ai.ModelRequest{Parts: returns}}
		params := ai.ModelRequestParams{Instructions: "stable", Tools: []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}}}, Settings: ai.ModelSettings{Cache: &ai.CacheConfig{Retention: ai.CacheRetention1Hour}}}
		if _, err := model.Request(t.Context(), messages, params); err != nil {
			t.Fatal(err)
		}
		wire := body["messages"].([]any)
		if body["cache_control"] != nil || wire[0].(map[string]any)["content"].([]any)[0].(map[string]any)["cache_control"] == nil || wire[2].(map[string]any)["content"].([]any)[11].(map[string]any)["cache_control"] == nil {
			t.Fatalf("previous request tail lost: %+v", body)
		}
		want := 5 * time.Minute
		if name == "us.anthropic.claude-sonnet-4-5-v1:0" {
			want = time.Hour
		}
		if got, known := ai.ResolveCacheRetention(model, &params.Settings); !known || got != want {
			t.Fatalf("retention=%s %v", got, known)
		}
		if _, err := model.Request(t.Context(), messages[:1], params); err != nil {
			t.Fatal(err)
		}
	}
}

func TestLegacyBedrockAutomaticCachingByModel(t *testing.T) {
	var body map[string]any
	client := &legacyBedrockClient{invoke: func(input *bedrockruntime.InvokeModelInput) (*bedrockruntime.InvokeModelOutput, error) {
		body = nil
		if err := json.Unmarshal(input.Body, &body); err != nil {
			t.Fatal(err)
		}
		return &bedrockruntime.InvokeModelOutput{Body: []byte(`{"model":"m","content":[{"type":"text","text":"ok"}],"usage":{}}`)}, nil
	}}
	explicit, err := (anthropic.Settings{Cache: anthropic.CacheTTL5Minutes}).Build()
	if err != nil {
		t.Fatal(err)
	}
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}}}}
	for _, test := range []struct {
		name      string
		options   []anthropic.Option
		automatic bool
	}{
		{name: "us.anthropic.claude-opus-4-8", automatic: true},
		{name: "global.anthropic.claude-sonnet-5-5-v1:0", automatic: true},
		{name: "us.anthropic.claude-sonnet-4-5-v1:0"},
		{name: "us.anthropic.claude-sonnet-4-5-v1:0", options: []anthropic.Option{anthropic.WithAutomaticCaching(true)}, automatic: true},
		{name: "us.anthropic.claude-opus-4-8", options: []anthropic.Option{anthropic.WithAutomaticCaching(false)}},
	} {
		model := anthropic.NewLegacyBedrockModel(test.name, anthropic.LegacyBedrockConfig{Client: client}, test.options...)
		if model.ModelProfile().SupportsAutoCache != test.automatic {
			t.Fatalf("%s: profile automatic caching = %v", test.name, !test.automatic)
		}
		for _, settings := range []ai.ModelSettings{explicit, {Cache: &ai.CacheConfig{}}} {
			if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{Settings: settings}); err != nil {
				t.Fatal(err)
			}
			block := body["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
			if (body["cache_control"] != nil) != test.automatic || (block["cache_control"] != nil) == test.automatic {
				t.Fatalf("%s %+v: cache_control=%v block=%v", test.name, settings, body["cache_control"], block)
			}
		}
	}
	var direct map[string]any
	model := newServerWithOptions(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&direct)
		_, _ = w.Write([]byte(`{"model":"m","content":[{"type":"text","text":"ok"}],"usage":{}}`))
	}, anthropic.WithAutomaticCaching(false))
	if _, err := model.Request(t.Context(), messages, ai.ModelRequestParams{Settings: explicit}); err != nil {
		t.Fatal(err)
	}
	if direct["cache_control"] != nil || direct["messages"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)["cache_control"] == nil {
		t.Fatalf("gateway without automatic caching: %+v", direct)
	}
}
