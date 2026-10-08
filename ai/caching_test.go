package ai_test

import (
	"context"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestPortableCacheResolution(t *testing.T) {
	for _, test := range []struct {
		name      string
		cache     *ai.CacheConfig
		profile   ai.ModelProfile
		want      ai.CacheRetention
		nilResult bool
		invalid   bool
	}{
		{name: "unset", nilResult: true},
		{name: "unsupported", cache: &ai.CacheConfig{}, nilResult: true},
		{name: "disabled", cache: &ai.CacheConfig{Retention: ai.CacheRetentionDisabled}, nilResult: true},
		{name: "invalid", cache: &ai.CacheConfig{Retention: "1d"}, invalid: true},
		{name: "default", cache: &ai.CacheConfig{}, profile: ai.ModelProfile{SupportsCache: true}},
		{name: "no tiers", cache: &ai.CacheConfig{Retention: ai.CacheRetention1Hour}, profile: ai.ModelProfile{SupportsCache: true}},
		{name: "honored", cache: &ai.CacheConfig{Retention: ai.CacheRetention1Hour}, profile: ai.ModelProfile{SupportsCache: true, SupportedCacheRetentions: []ai.CacheRetention{ai.CacheRetention5Minutes, ai.CacheRetention1Hour}}, want: ai.CacheRetention1Hour},
		{name: "snap down", cache: &ai.CacheConfig{Retention: ai.CacheRetention30Minutes}, profile: ai.ModelProfile{SupportsCache: true, SupportedCacheRetentions: []ai.CacheRetention{ai.CacheRetention1Hour, ai.CacheRetention5Minutes}}, want: ai.CacheRetention5Minutes},
		{name: "snap up", cache: &ai.CacheConfig{Retention: ai.CacheRetention5Minutes}, profile: ai.ModelProfile{SupportsCache: true, SupportedCacheRetentions: []ai.CacheRetention{ai.CacheRetention1Hour, ai.CacheRetention30Minutes}}, want: ai.CacheRetention30Minutes},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := ai.ResolveCacheConfig(ai.ModelSettings{Cache: test.cache}, test.profile)
			if (err != nil) != test.invalid || !test.invalid && ((got == nil) != test.nilResult || got != nil && got.Retention != test.want) {
				t.Fatalf("cache=%+v err=%v", got, err)
			}
		})
	}
	messages := false
	settings := ai.ModelSettings{Cache: &ai.CacheConfig{Retention: ai.CacheRetention1Hour, Messages: &messages}}
	cloned := settings.Clone()
	resolved, err := ai.ResolveCacheConfig(settings, ai.ModelProfile{SupportsCache: true})
	if err != nil {
		t.Fatal(err)
	}
	*cloned.Cache.Messages = true
	*resolved.Messages = true
	cloned.Cache.Retention = ai.CacheRetentionDisabled
	if *settings.Cache.Messages || settings.Cache.Retention != ai.CacheRetention1Hour {
		t.Fatal("cache settings alias their source")
	}
}

func TestCachingCapabilityComposition(t *testing.T) {
	messages := false
	seen := 0
	model := fakes.NewFunctionModel(func(_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
		seen++
		want := ai.CacheRetention1Hour
		if seen == 2 {
			want = ai.CacheRetentionDisabled
		}
		if params.Settings.Cache == nil || params.Settings.Cache.Retention != want || *params.Settings.Cache.Messages {
			t.Fatalf("cache=%+v", params.Settings.Cache)
		}
		*params.Settings.Cache.Messages = true
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model,
		ai.WithCapabilities(ai.CombineCapabilities(ai.Caching{}, ai.Caching{Retention: ai.CacheRetention1Hour, Messages: &messages})))
	for _, options := range [][]ai.RunOption{nil, {ai.WithRunCapabilities(ai.Caching{Retention: ai.CacheRetentionDisabled, Messages: &messages})}} {
		if _, err := agent.Run(t.Context(), "test", struct{}{}, options...); err != nil {
			t.Fatal(err)
		}
	}
	if messages {
		t.Fatal("capability settings were mutated")
	}
	if (ai.Caching{ID: "other"}).CapabilityID() != "other" {
		t.Fatal("custom identity ignored")
	}
	if _, err := (ai.Caching{}).CombineCapabilities(nil); err == nil {
		t.Fatal("empty combination accepted")
	}
	if err := (ai.Caching{Retention: "week"}).Setup(&ai.CapabilityRegistry{}); err == nil {
		t.Fatal("invalid retention accepted")
	}
	if _, err := agent.Run(t.Context(), "test", struct{}{}, ai.WithRunModelSettings(ai.ModelSettings{Cache: &ai.CacheConfig{Retention: "week"}})); err == nil {
		t.Fatal("invalid run retention accepted")
	}
}

func TestCacheOutlookHonorsProviderTiers(t *testing.T) {
	now := time.Now()
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{ai.TextContent{Text: "prefix"}, ai.CachePoint{TTL: ai.CachePointTTL1Hour}}}}},
		ai.ModelResponse{Timestamp: now.Add(-40 * time.Minute)},
	}
	profile := ai.ModelProfile{SupportsCache: true, SupportedCacheRetentions: []ai.CacheRetention{ai.CacheRetention30Minutes}, DefaultCacheRetention: 30 * time.Minute}
	if got := ai.PromptCacheOutlook(history, &profile, nil, now); got != ai.CacheOutlookCold {
		t.Fatalf("unsupported TTL extended outlook: %s", got)
	}
	for _, tier := range []ai.CacheRetention{"", ai.CacheRetentionDisabled, "week"} {
		func() {
			defer func() {
				if recover() == nil {
					t.Error("invalid profile tier accepted")
				}
			}()
			ai.NewProfiledModel(fakes.NewTestModel(), ai.ModelProfile{DefaultOutputMode: ai.OutputModeTool, SupportedCacheRetentions: []ai.CacheRetention{tier}})
		}()
	}
	profile.DefaultOutputMode = ai.OutputModeTool
	model := ai.NewProfiledModel(fakes.NewTestModel(), profile)
	profile.SupportedCacheRetentions[0] = ai.CacheRetention1Hour
	got := model.ModelProfile()
	got.SupportedCacheRetentions[0] = ai.CacheRetention5Minutes
	if model.ModelProfile().SupportedCacheRetentions[0] != ai.CacheRetention30Minutes {
		t.Fatal("profile tiers alias caller values")
	}
}
