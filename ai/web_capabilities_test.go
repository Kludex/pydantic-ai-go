package ai_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func webCapabilityTool(t *testing.T, called *int) ai.Tool[struct{}] {
	t.Helper()
	return ai.NewSimpleTool[struct{}]("local_search", func(context.Context, struct{}) (string, error) {
		*called++
		return "local result", nil
	})
}

func TestWebSearchCapabilitySelectsNativeOrLocal(t *testing.T) {
	localCalls := 0
	local := ai.NewFunctionToolset(webCapabilityTool(t, &localCalls))
	nativeModel := &selectiveNativeModel{supported: map[string]bool{"web_search": true}}
	agent := ai.NewAgent[struct{}, string](nativeModel, ai.WithCapabilities(
		ai.NewWebSearchCapability(ai.WebSearchCapabilityConfig[struct{}]{
			Native: ai.WebSearchTool{SearchContextSize: ai.WebSearchContextHigh}, Local: local,
		}),
	))
	result, err := agent.Run(t.Context(), "search", struct{}{})
	if err != nil || result.Output != "done" || localCalls != 0 || len(nativeModel.requests) != 1 ||
		len(nativeModel.requests[0].NativeTools) != 1 || len(nativeModel.requests[0].Tools) != 0 {
		t.Fatalf("native search was not selected: result=%#v calls=%d requests=%#v err=%v",
			result, localCalls, nativeModel.requests, err)
	}

	localModel := &selectiveNativeModel{supported: map[string]bool{}, callLocal: true}
	agent = ai.NewAgent[struct{}, string](localModel, ai.WithCapabilities(
		ai.NewWebSearchCapability(ai.WebSearchCapabilityConfig[struct{}]{
			Native: ai.WebSearchTool{SearchContextSize: ai.WebSearchContextHigh}, Local: local,
		}),
	))
	result, err = agent.Run(t.Context(), "search", struct{}{})
	if err != nil || result.Output != "done" || localCalls != 1 || len(localModel.requests) != 2 ||
		len(localModel.requests[0].NativeTools) != 0 || len(localModel.requests[0].Tools) != 1 {
		t.Fatalf("local search was not selected: result=%#v calls=%d requests=%#v err=%v",
			result, localCalls, localModel.requests, err)
	}
}

func TestWebSearchCapabilityNativeConstraints(t *testing.T) {
	externalAccess := false
	for name, native := range map[string]ai.WebSearchTool{
		"blocked domains":      {BlockedDomains: []string{}},
		"allowed domains":      {AllowedDomains: []string{"example.com"}},
		"maximum uses":         {MaxUses: 2},
		"external web access":  {ExternalWebAccess: &externalAccess},
		"multiple constraints": {BlockedDomains: []string{"bad.test"}, MaxUses: 2},
	} {
		t.Run(name, func(t *testing.T) {
			localCalls := 0
			model := &selectiveNativeModel{supported: map[string]bool{}, callLocal: true}
			agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
				ai.NewWebSearchCapability(ai.WebSearchCapabilityConfig[struct{}]{
					Native: native, Local: ai.NewFunctionToolset(webCapabilityTool(t, &localCalls)),
				}),
			))
			_, err := agent.Run(t.Context(), "search", struct{}{})
			if err == nil || !strings.Contains(err.Error(), `native tool(s) "web_search"`) ||
				localCalls != 0 || len(model.requests) != 0 {
				t.Fatalf("native constraint used local fallback: calls=%d requests=%d err=%v",
					localCalls, len(model.requests), err)
			}
		})
	}
}

func TestWebSearchCapabilityAllowsPortableLocalSettings(t *testing.T) {
	localCalls := 0
	model := &selectiveNativeModel{supported: map[string]bool{}, callLocal: true}
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.NewWebSearchCapability(ai.WebSearchCapabilityConfig[struct{}]{
			Native: ai.WebSearchTool{
				SearchContextSize: ai.WebSearchContextHigh,
				UserLocation:      &ai.WebSearchUserLocation{Country: "PT"},
			},
			Local: ai.NewFunctionToolset(webCapabilityTool(t, &localCalls)),
		}),
	))
	if _, err := agent.Run(t.Context(), "search", struct{}{}); err != nil || localCalls != 1 {
		t.Fatalf("portable settings suppressed local search: calls=%d err=%v", localCalls, err)
	}
}

func TestWebFetchCapabilityClassifiesConstraints(t *testing.T) {
	localCalls := 0
	local := ai.NewFunctionToolset(webCapabilityTool(t, &localCalls))
	model := &selectiveNativeModel{supported: map[string]bool{}, callLocal: true}
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.NewWebFetchCapability(ai.WebFetchCapabilityConfig[struct{}]{
			Native: ai.WebFetchTool{
				AllowedDomains: []string{"example.com"}, BlockedDomains: []string{"bad.test"},
				EnableCitations: true, MaxContentTokens: 1000,
			},
			Local: local,
		}),
	))
	if _, err := agent.Run(t.Context(), "fetch", struct{}{}); err != nil || localCalls != 1 {
		t.Fatalf("portable fetch settings suppressed local fallback: calls=%d err=%v", localCalls, err)
	}

	localCalls = 0
	model = &selectiveNativeModel{supported: map[string]bool{}, callLocal: true}
	agent = ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.NewWebFetchCapability(ai.WebFetchCapabilityConfig[struct{}]{
			Native: ai.WebFetchTool{MaxUses: 1}, Local: local,
		}),
	))
	_, err := agent.Run(t.Context(), "fetch", struct{}{})
	if err == nil || !strings.Contains(err.Error(), `native tool(s) "web_fetch"`) || localCalls != 0 {
		t.Fatalf("native fetch constraint used local fallback: calls=%d err=%v", localCalls, err)
	}
}

func TestDynamicWebCapabilities(t *testing.T) {
	searchCalls, localCalls := 0, 0
	local := ai.NewFunctionToolset(nativeOrLocalSearchTool(t, &localCalls))
	model := &selectiveNativeModel{supported: map[string]bool{"web_search": true}}
	search := ai.NewDynamicWebSearchCapability(
		func(_ context.Context, rc *ai.RunContext[nativeOrLocalDeps]) (ai.WebSearchTool, error) {
			searchCalls++
			return ai.WebSearchTool{UserLocation: &ai.WebSearchUserLocation{City: rc.Deps.Location}}, nil
		},
		local,
	)
	agent := ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(search))
	if _, err := agent.Run(t.Context(), "search", nativeOrLocalDeps{Location: "Porto"}); err != nil ||
		searchCalls != 1 || localCalls != 0 {
		t.Fatalf("dynamic native search failed: resolves=%d local=%d err=%v", searchCalls, localCalls, err)
	}
	resolved := model.requests[0].NativeTools[0].(ai.WebSearchTool)
	if resolved.UserLocation == nil || resolved.UserLocation.City != "Porto" {
		t.Fatalf("unexpected dynamic search: %#v", resolved)
	}

	fetchCalls := 0
	model = &selectiveNativeModel{supported: map[string]bool{}, callLocal: true}
	fetch := ai.NewDynamicWebFetchCapability(
		func(context.Context, *ai.RunContext[nativeOrLocalDeps]) (ai.WebFetchTool, error) {
			fetchCalls++
			return ai.WebFetchTool{AllowedDomains: []string{"example.com"}}, nil
		},
		local,
	)
	agent = ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(fetch))
	if _, err := agent.Run(t.Context(), "fetch", nativeOrLocalDeps{}); err != nil ||
		fetchCalls != 2 || localCalls != 1 {
		t.Fatalf("dynamic local fetch failed: resolves=%d local=%d err=%v", fetchCalls, localCalls, err)
	}
}

func TestDynamicWebCapabilityErrors(t *testing.T) {
	assertNativeOrLocalPanic(t, "dynamic web-search resolver must not be nil", func() {
		ai.NewDynamicWebSearchCapability[nativeOrLocalDeps](nil, nil)
	})
	assertNativeOrLocalPanic(t, "dynamic web-fetch resolver must not be nil", func() {
		ai.NewDynamicWebFetchCapability[nativeOrLocalDeps](nil, nil)
	})

	resolveErr := errors.New("resolve failed")
	model := &selectiveNativeModel{supported: map[string]bool{"web_search": true}}
	agent := ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(
		ai.NewDynamicWebSearchCapability(
			func(context.Context, *ai.RunContext[nativeOrLocalDeps]) (ai.WebSearchTool, error) {
				return ai.WebSearchTool{}, resolveErr
			},
			ai.NewFunctionToolset(nativeOrLocalSearchTool(t, new(int))),
		),
	))
	if _, err := agent.Run(t.Context(), "search", nativeOrLocalDeps{}); !errors.Is(err, resolveErr) {
		t.Fatalf("unexpected dynamic resolver error: %v", err)
	}

	local := ai.NewFunctionToolset(nativeOrLocalSearchTool(t, new(int)))
	model = &selectiveNativeModel{supported: map[string]bool{"web_search": true}}
	agent = ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(
		ai.NewDynamicWebSearchCapability(
			func(context.Context, *ai.RunContext[nativeOrLocalDeps]) (ai.WebSearchTool, error) {
				return ai.WebSearchTool{AllowedDomains: []string{"example.com"}}, nil
			},
			local,
		),
	))
	if _, err := agent.Run(t.Context(), "search", nativeOrLocalDeps{}); err == nil ||
		!strings.Contains(err.Error(), "use WithNativeRequired") || len(model.requests) != 0 {
		t.Fatalf("dynamic search silently degraded native constraints: %v", err)
	}

	model = &selectiveNativeModel{supported: map[string]bool{"web_fetch": true}}
	agent = ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(
		ai.NewDynamicWebFetchCapability(
			func(context.Context, *ai.RunContext[nativeOrLocalDeps]) (ai.WebFetchTool, error) {
				return ai.WebFetchTool{MaxUses: 1}, nil
			},
			local,
		),
	))
	if _, err := agent.Run(t.Context(), "fetch", nativeOrLocalDeps{}); err == nil ||
		!strings.Contains(err.Error(), "use WithNativeRequired") || len(model.requests) != 0 {
		t.Fatalf("dynamic fetch silently degraded native constraints: %v", err)
	}

	model = &selectiveNativeModel{supported: map[string]bool{}}
	agent = ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(
		ai.NewDynamicWebFetchCapability(
			func(context.Context, *ai.RunContext[nativeOrLocalDeps]) (ai.WebFetchTool, error) {
				return ai.WebFetchTool{}, nil
			},
			nil,
		),
	))
	if _, err := agent.Run(t.Context(), "fetch", nativeOrLocalDeps{}); err == nil ||
		!strings.Contains(err.Error(), "have no local fallback") {
		t.Fatalf("dynamic fetch without local fallback succeeded: %v", err)
	}

	model = &selectiveNativeModel{supported: map[string]bool{}}
	agent = ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(
		ai.NewDynamicWebSearchCapability(
			func(context.Context, *ai.RunContext[nativeOrLocalDeps]) (ai.WebSearchTool, error) {
				return ai.WebSearchTool{}, nil
			},
			local,
			ai.WithNativeRequired("allowed domains"),
		),
	))
	if _, err := agent.Run(t.Context(), "search", nativeOrLocalDeps{}); err == nil ||
		!strings.Contains(err.Error(), "have no local fallback") {
		t.Fatalf("dynamic native requirement used local fallback: %v", err)
	}
}

func TestWebCapabilitiesRequireNativeWithoutLocal(t *testing.T) {
	for name, capability := range map[string]ai.Capability{
		"search": ai.NewWebSearchCapability(ai.WebSearchCapabilityConfig[struct{}]{}),
		"fetch":  ai.NewWebFetchCapability(ai.WebFetchCapabilityConfig[struct{}]{}),
	} {
		t.Run(name, func(t *testing.T) {
			model := &selectiveNativeModel{supported: map[string]bool{}}
			agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(capability))
			if _, err := agent.Run(t.Context(), "use web", struct{}{}); err == nil ||
				!strings.Contains(err.Error(), "have no local fallback") || len(model.requests) != 0 {
				t.Fatalf("missing local fallback was ignored: requests=%d err=%v", len(model.requests), err)
			}

			supported := &selectiveNativeModel{supported: map[string]bool{
				"web_search": true, "web_fetch": true,
			}}
			agent = ai.NewAgent[struct{}, string](supported, ai.WithCapabilities(capability))
			if _, err := agent.Run(t.Context(), "use web", struct{}{}); err != nil ||
				len(supported.requests) != 1 || len(supported.requests[0].NativeTools) != 1 {
				t.Fatalf("supported native-only capability failed: requests=%#v err=%v", supported.requests, err)
			}
		})
	}

	assertNativeOrLocalPanic(t, "requires native support", func() {
		ai.NewAgent[struct{}, string](
			&selectiveNativeModel{supported: map[string]bool{"web_search": true}},
			ai.WithCapabilities(ai.NewWebSearchCapability(ai.WebSearchCapabilityConfig[struct{}]{
				Native: ai.WebSearchTool{Optional: true},
			})),
		)
	})
}
