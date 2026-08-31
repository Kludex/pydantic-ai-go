package anthropic

import (
	"reflect"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
)

func TestCacheSettingsBuildAndExtract(t *testing.T) {
	settings, err := (Settings{
		Common: ai.ModelSettings{MaxTokens: 42, ExtraBody: map[string]any{
			"custom": map[string]any{"enabled": true},
		}},
		Cache:                CacheTTL5Minutes,
		CacheInstructions:    CacheTTL1Hour,
		CacheToolDefinitions: CacheTTL5Minutes,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	cleaned, cache, err := extractCacheSettings(settings)
	if err != nil {
		t.Fatal(err)
	}
	if cleaned.MaxTokens != 42 || !reflect.DeepEqual(cleaned.ExtraBody, map[string]any{
		"custom": map[string]any{"enabled": true},
	}) || cache.Automatic != CacheTTL5Minutes || cache.Instructions != CacheTTL1Hour ||
		cache.Messages != "" || cache.ToolDefinitions != CacheTTL5Minutes {
		t.Fatalf("unexpected cache settings: cleaned=%#v cache=%#v", cleaned, cache)
	}
	empty, err := (Settings{}).Build()
	if err != nil || empty.ExtraBody != nil {
		t.Fatalf("unexpected empty settings: %#v %v", empty, err)
	}
	if control := promptCacheControl(""); control != nil {
		t.Fatalf("empty TTL produced cache control: %#v", control)
	}
}

func TestCacheSettingsValidation(t *testing.T) {
	tests := []struct {
		name     string
		settings Settings
		want     string
	}{
		{name: "automatic messages", settings: Settings{
			Cache: CacheTTL5Minutes, CacheMessages: CacheTTL5Minutes,
		}, want: "mutually exclusive"},
		{name: "ttl", settings: Settings{CacheInstructions: "1d"}, want: "invalid cache TTL"},
		{name: "reserved", settings: Settings{Common: ai.ModelSettings{ExtraBody: map[string]any{
			cacheMessagesSetting: CacheTTL5Minutes,
		}}}, want: "is reserved"},
		{name: "wire conflict", settings: Settings{
			Common: ai.ModelSettings{ExtraBody: map[string]any{"cache_control": map[string]any{}}},
			Cache:  CacheTTL5Minutes,
		}, want: "conflicts with typed settings"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := test.settings.Build()
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected settings error: %v", err)
			}
		})
	}
	for name, value := range map[string]any{
		cacheSetting:             42,
		cacheInstructionsSetting: CacheTTL("1d"),
	} {
		t.Run("extract "+name, func(t *testing.T) {
			_, _, err := extractCacheSettings(ai.ModelSettings{ExtraBody: map[string]any{name: value}})
			if err == nil || !strings.Contains(err.Error(), "cache") {
				t.Fatalf("unexpected extraction error: %v", err)
			}
		})
	}
	_, _, err := extractCacheSettings(ai.ModelSettings{ExtraBody: map[string]any{
		cacheSetting: CacheTTL5Minutes, cacheMessagesSetting: CacheTTL5Minutes,
	}})
	if err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("unexpected extraction conflict: %v", err)
	}
	malformed := ai.ModelSettings{ExtraBody: map[string]any{cacheSetting: 42}}
	if _, err := NewModel("model").Request(t.Context(), nil, ai.ModelRequestParams{Settings: malformed}); err == nil ||
		!strings.Contains(err.Error(), "must use CacheTTL") {
		t.Fatalf("unexpected malformed request error: %v", err)
	}
	if duration, ok := NewModel("model").PromptCacheRetention(malformed); ok || duration != 0 {
		t.Fatalf("malformed settings reported retention: %s %v", duration, ok)
	}
}

func TestCachePlacementEdges(t *testing.T) {
	settings, err := (Settings{
		CacheInstructions:    CacheTTL5Minutes,
		CacheMessages:        CacheTTL5Minutes,
		CacheToolDefinitions: CacheTTL5Minutes,
	}).Build()
	if err != nil {
		t.Fatal(err)
	}
	model := NewModel("model")
	request, err := model.buildPayload(t.Context(), nil, ai.ModelRequestParams{
		Instructions: "stable", Settings: settings,
	})
	if err != nil {
		t.Fatal(err)
	}
	system := request.System.([]contentBlock)
	if system[0].CacheControl == nil || len(request.Messages) != 0 || len(request.Tools) != 0 {
		t.Fatalf("cache placement without messages or tools changed: %#v", request)
	}
	request, err = model.buildPayload(t.Context(), nil, ai.ModelRequestParams{
		Instructions:     "dynamic",
		InstructionParts: []ai.InstructionPart{{Content: "dynamic", Dynamic: true}},
		Settings:         settings,
	})
	if err != nil {
		t.Fatal(err)
	}
	if request.System.([]contentBlock)[0].CacheControl != nil {
		t.Fatalf("dynamic instruction was cached: %#v", request.System)
	}

	messages := []messageParam{{Role: "assistant", Content: []contentBlock{{Type: "thinking"}}}}
	addAnthropicMessageCachePoint(messages, CacheTTL5Minutes)
	if messages[0].Content[0].CacheControl != nil {
		t.Fatal("unsupported message block received cache control")
	}
	messages[0].Content = append(messages[0].Content, contentBlock{Type: "text", Text: "answer"})
	addAnthropicMessageCachePoint(messages, CacheTTL5Minutes)
	if messages[0].Content[1].CacheControl == nil {
		t.Fatal("cacheable message block did not receive cache control")
	}
	if err := limitAnthropicCachePoints(&messagesRequest{
		Tools: []toolParam{
			{CacheControl: promptCacheControl(CacheTTL5Minutes)},
			{CacheControl: promptCacheControl(CacheTTL5Minutes)},
			{CacheControl: promptCacheControl(CacheTTL5Minutes)},
			{CacheControl: promptCacheControl(CacheTTL5Minutes)},
		},
	}, true); err == nil || !strings.Contains(err.Error(), "exceeding the maximum") {
		t.Fatalf("unexpected reserved cache-point error: %v", err)
	}
}
