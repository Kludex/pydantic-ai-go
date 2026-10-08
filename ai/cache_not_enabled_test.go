package ai_test

import (
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

type configuredCacheModel struct{ *ai.ModelWrapper }

func (model configuredCacheModel) CachingNotEnabled(settings ai.ModelSettings) bool {
	return settings.Cache == nil
}

func TestCacheConfigurationTelemetry(t *testing.T) {
	for _, test := range []struct {
		name     string
		tokens   int
		cache    *ai.CacheConfig
		explicit bool
		want     bool
	}{
		{name: "long unconfigured", tokens: 4096, want: true},
		{name: "short", tokens: 4095},
		{name: "enabled", tokens: 4096, cache: &ai.CacheConfig{}},
		{name: "deliberately disabled", tokens: 4096, cache: &ai.CacheConfig{Retention: ai.CacheRetentionDisabled}},
		{name: "manual", tokens: 4096, explicit: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			exporter, provider := cacheExporter(t)
			responses := []*ai.ModelResponse{cacheResponse(0, 0), cacheResponse(0, 0)}
			for _, response := range responses {
				response.Usage.InputTokens = test.tokens
			}
			model := ai.WrapModel(configuredCacheModel{ai.WrapModel(cacheModel(responses...))})
			agent := ai.NewAgent[struct{}, string](model, ai.WithModelSettings(ai.ModelSettings{Cache: test.cache}),
				ai.WithCapabilities(ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(provider))))
			parts := []ai.UserContent{ai.TextContent{Text: "prefix"}}
			if test.explicit {
				parts = append(parts, ai.CachePoint{})
			}
			result, err := agent.RunParts(t.Context(), parts, struct{}{})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := agent.Run(t.Context(), "next", struct{}{}, ai.WithConversation(result.Conversation())); err != nil {
				t.Fatal(err)
			}
			spans := cacheChatSpans(exporter)
			if (cacheAttributes(spans[0])["pydantic_ai.cache.not_enabled"] == true) != test.want {
				t.Fatalf("health=%+v", cacheAttributes(spans[0]))
			}
			if test.want && (len(spans[0].Events) != 1 || spans[0].Events[0].Name != "pydantic_ai.cache.not_enabled") {
				t.Fatal("missing not-enabled event")
			}
			if len(spans[1].Events) != 0 {
				t.Fatal("repeated configuration alert")
			}
		})
	}
	model := configuredCacheModel{ai.WrapModel(cacheModel(cacheResponse(0, 0), cacheResponse(0, 5000), cacheResponse(100, 0)))}
	_, spans := runCacheTurns(t, model, 3)
	if len(spans[0].Events) != 1 || cacheAttributes(spans[2])["pydantic_ai.cache.collapsed"] != true {
		t.Fatal("unconfigured mark prevented later health monitoring")
	}
}
