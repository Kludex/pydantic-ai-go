package ai_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

type nativeFallbackHook struct {
	uniqueID string
}

func (nativeFallbackHook) Setup(*ai.CapabilityRegistry) error { return nil }

func (hook nativeFallbackHook) BeforeModelRequest(
	_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
) (ai.ModelRequestContext, error) {
	for index := range request.Params.Tools {
		request.Params.Tools[index].NativeFallbackFor = hook.uniqueID
	}
	return request, nil
}

func TestNativeFallbackToolOption(t *testing.T) {
	web := ai.WebSearchTool{}
	fallback := ai.NewSimpleTool[struct{}]("local_search", func(context.Context, struct{}) (string, error) {
		return "local", nil
	}, ai.WithNativeFallback(web))
	companion := ai.NewSimpleTool[struct{}]("managed", func(context.Context, struct{}) (string, error) {
		return "managed", nil
	}, ai.WithNativeCompanion(web))
	if fallback.Definition().NativeFallbackFor != web.UniqueID() ||
		companion.Definition().NativeCompanionFor != web.UniqueID() {
		t.Fatalf("native preferences were not configured: %#v %#v", fallback.Definition(), companion.Definition())
	}
	encoded, err := json.Marshal([]ai.ToolDefinition{fallback.Definition(), companion.Definition()})
	if err != nil {
		t.Fatal(err)
	}
	if text := string(encoded); !strings.Contains(text, `"unless_native":"web_search"`) ||
		!strings.Contains(text, `"with_native":"web_search"`) {
		t.Fatalf("native preferences were not serialized: %s", text)
	}
	for _, input := range []string{
		`{"name":"tool","parameters_json_schema":{"type":"object"},"unless_native":"current"}`,
		`{"name":"tool","prefer_native":"legacy-native"}`,
		`{"name":"tool","prefer_builtin":"legacy-builtin"}`,
	} {
		var definition ai.ToolDefinition
		if err := json.Unmarshal([]byte(input), &definition); err != nil || definition.NativeFallbackFor == "" ||
			definition.Name != "tool" {
			t.Fatalf("native preference did not decode: %#v, %v", definition, err)
		}
	}
	var precedence ai.ToolDefinition
	if err := json.Unmarshal([]byte(
		`{"unless_native":"","prefer_native":"legacy","with_native":"manager"}`,
	), &precedence); err != nil || precedence.NativeFallbackFor != "" || precedence.NativeCompanionFor != "manager" {
		t.Fatalf("canonical native preference did not win: %#v, %v", precedence, err)
	}
	if err := json.Unmarshal([]byte(`{"name":[]}`), &precedence); err == nil {
		t.Fatal("expected malformed tool definition to fail")
	}

	var nilWebSearch *ai.WebSearchTool
	defer func() {
		if recovered := recover(); recovered == nil || !strings.Contains(recovered.(string), "native tool must not be nil") {
			t.Fatalf("unexpected invalid preference panic: %v", recovered)
		}
	}()
	_ = ai.WithNativeFallback(nilWebSearch)
}

func TestNativePreferenceToolsets(t *testing.T) {
	web := ai.WebSearchTool{}
	first := ai.NewSimpleTool[struct{}]("first", func(context.Context, struct{}) (string, error) {
		return "first", nil
	})
	second := ai.NewSimpleTool[struct{}]("second", func(context.Context, struct{}) (string, error) {
		return "second", nil
	})
	base := ai.NewFunctionToolset(first, second)
	fallbacks, err := ai.NativeFallbackToolset(base, web).Tools(t.Context(), &ai.RunContext[struct{}]{})
	if err != nil {
		t.Fatal(err)
	}
	companions, err := ai.NativeCompanionToolset(base, web).Tools(t.Context(), &ai.RunContext[struct{}]{})
	if err != nil {
		t.Fatal(err)
	}
	for index := range fallbacks {
		if fallbacks[index].Definition().NativeFallbackFor != web.UniqueID() ||
			companions[index].Definition().NativeCompanionFor != web.UniqueID() {
			t.Fatalf("toolset preference missing: %#v %#v", fallbacks[index], companions[index])
		}
	}
}

func TestAgentUsesLocalNativeFallback(t *testing.T) {
	web := ai.WebSearchTool{}
	model := &selectiveNativeModel{supported: map[string]bool{}, callLocal: true}
	called := 0
	fallback := ai.NewSimpleTool[struct{}]("local_search", func(context.Context, struct{}) (string, error) {
		called++
		return "local result", nil
	}, ai.WithNativeFallback(web))
	agent := ai.NewAgent[struct{}, string](model, ai.WithNativeTools(web))
	agent.AddTool(fallback)
	result, err := agent.Run(t.Context(), "search", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "done" || called != 1 || len(model.requests) != 2 {
		t.Fatalf("unexpected fallback run: %#v, calls=%d requests=%d", result, called, len(model.requests))
	}
	for _, params := range model.requests {
		if len(params.NativeTools) != 0 || len(params.Tools) != 1 || params.Tools[0].Name != "local_search" {
			t.Fatalf("unexpected fallback request: %#v", params)
		}
	}
}

func TestNativePreferencesResolveAfterRequestHooks(t *testing.T) {
	web := ai.WebSearchTool{}
	model := &selectiveNativeModel{supported: map[string]bool{}, callLocal: true}
	local := ai.NewSimpleTool[struct{}]("local_search", func(context.Context, struct{}) (string, error) {
		return "local", nil
	})
	agent := ai.NewAgent[struct{}, string](
		model, ai.WithNativeTools(web), ai.WithCapabilities(nativeFallbackHook{uniqueID: web.UniqueID()}),
	)
	agent.AddTool(local)
	if _, err := agent.Run(t.Context(), "search", struct{}{}); err != nil {
		t.Fatal(err)
	}
	if len(model.requests) != 2 || len(model.requests[0].NativeTools) != 0 || len(model.requests[0].Tools) != 1 {
		t.Fatalf("request hook preferences were not resolved: %#v", model.requests)
	}
}

func TestAgentRejectsUnsupportedNativeSibling(t *testing.T) {
	web := ai.WebSearchTool{}
	model := &selectiveNativeModel{supported: map[string]bool{}}
	fallback := ai.NewSimpleTool[struct{}]("local_search", func(context.Context, struct{}) (string, error) {
		return "local", nil
	}, ai.WithNativeFallback(web))
	agent := ai.NewAgent[struct{}, string](
		model, ai.WithNativeTools(web, ai.CodeExecutionTool{}),
	)
	agent.AddTool(fallback)
	if _, err := agent.Run(t.Context(), "search", struct{}{}); err == nil ||
		!strings.Contains(err.Error(), "no local fallback") {
		t.Fatalf("unexpected unsupported sibling error: %v", err)
	}
	if len(model.requests) != 0 {
		t.Fatal("unsupported sibling reached model")
	}
}

func TestAgentPrefersSupportedNativeTool(t *testing.T) {
	web := ai.WebSearchTool{}
	model := &selectiveNativeModel{supported: map[string]bool{web.UniqueID(): true}}
	fallback := ai.NewSimpleTool[struct{}]("local_search", func(context.Context, struct{}) (string, error) {
		t.Fatal("local fallback executed")
		return "", nil
	}, ai.WithNativeFallback(web))
	agent := ai.NewAgent[struct{}, string](model, ai.WithNativeTools(web))
	agent.AddTool(fallback)
	if _, err := agent.Run(t.Context(), "search", struct{}{}); err != nil {
		t.Fatal(err)
	}
	if len(model.requests) != 1 || len(model.requests[0].Tools) != 0 || len(model.requests[0].NativeTools) != 1 {
		t.Fatalf("unexpected native request: %#v", model.requests)
	}
}

func TestDirectRequestResolvesNativeFallback(t *testing.T) {
	web := ai.WebSearchTool{}
	model := &selectiveNativeModel{supported: map[string]bool{}}
	_, err := ai.RequestModel(t.Context(), model, nil, ai.ModelRequestParams{
		Tools:       []ai.ToolDefinition{{Name: "local_search", NativeFallbackFor: web.UniqueID()}},
		NativeTools: []ai.NativeTool{web},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(model.requests) != 1 || len(model.requests[0].NativeTools) != 0 || len(model.requests[0].Tools) != 1 {
		t.Fatalf("direct request did not resolve fallback: %#v", model.requests)
	}

	wrapper := ai.WrapModel(&selectiveNativeModel{supported: map[string]bool{web.UniqueID(): true}})
	resolved, err := ai.ResolveNativeToolPreferences(wrapper, ai.ModelRequestParams{
		Tools:       []ai.ToolDefinition{{Name: "local_search", NativeFallbackFor: web.UniqueID()}},
		NativeTools: []ai.NativeTool{web},
	})
	if err != nil || len(resolved.Tools) != 0 || len(resolved.NativeTools) != 1 {
		t.Fatalf("model wrapper did not delegate native support: %#v, %v", resolved, err)
	}
	unknown := ai.WrapModel(unknownNativeSupportModel{})
	resolved, err = ai.ResolveNativeToolPreferences(unknown, ai.ModelRequestParams{
		Tools:       []ai.ToolDefinition{{Name: "local_search", NativeFallbackFor: web.UniqueID()}},
		NativeTools: []ai.NativeTool{web},
	})
	if err != nil || len(resolved.Tools) != 1 || len(resolved.NativeTools) != 1 {
		t.Fatalf("model wrapper guessed unknown support: %#v, %v", resolved, err)
	}
}
