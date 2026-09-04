package ai_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

type selectiveNativeModel struct {
	supported     map[string]bool
	requests      []ai.ModelRequestParams
	callLocal     bool
	requestErr    error
	supportChecks int
}

func (*selectiveNativeModel) Name() string { return "selective" }

func (model *selectiveNativeModel) SupportsNativeTool(tool ai.NativeTool) bool {
	model.supportChecks++
	return tool != nil && model.supported[tool.UniqueID()]
}

func (model *selectiveNativeModel) Request(
	_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	model.requests = append(model.requests, ai.ModelRequestContext{Params: params}.Clone().Params)
	if model.requestErr != nil {
		return nil, model.requestErr
	}
	if model.callLocal && len(messages) == 1 {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "local_search", Args: json.RawMessage(`{}`), ToolCallID: "call-1",
		}}}, nil
	}
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
}

type nativeFallbackAPIError struct{}

func (nativeFallbackAPIError) Error() string         { return "try fallback" }
func (nativeFallbackAPIError) IsModelAPIError() bool { return true }

type unknownNativeSupportModel struct{}

func (unknownNativeSupportModel) Name() string { return "unknown" }
func (unknownNativeSupportModel) Request(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
}

func TestResolveNativeToolPreferences(t *testing.T) {
	web := ai.WebSearchTool{}
	code := ai.CodeExecutionTool{}
	params := ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{
			{Name: "local_search", Schema: map[string]any{"type": "object"}, NativeFallbackFor: web.UniqueID()},
			{Name: "local_code", NativeFallbackFor: code.UniqueID()},
			{Name: "regular"},
			{Name: "managed", NativeCompanionFor: web.UniqueID()},
			{Name: "orphan", NativeCompanionFor: "missing"},
		},
		DeferredTools: []ai.ToolDefinition{
			{Name: "hidden_search", NativeFallbackFor: web.UniqueID()},
			{Name: "hidden_code", NativeFallbackFor: code.UniqueID()},
		},
		NativeTools: []ai.NativeTool{web, code},
	}
	model := &selectiveNativeModel{supported: map[string]bool{web.UniqueID(): true}}
	resolved, err := ai.ResolveNativeToolPreferences(model, params)
	if err != nil {
		t.Fatal(err)
	}
	if len(resolved.NativeTools) != 1 || resolved.NativeTools[0].Kind() != web.Kind() {
		t.Fatalf("unexpected native tools: %#v", resolved.NativeTools)
	}
	if names := toolNames(resolved.Tools); strings.Join(names, ",") != "local_code,regular,managed,orphan" {
		t.Fatalf("unexpected tools: %v", names)
	}
	if resolved.Tools[2].NativeCompanionFor != web.UniqueID() || resolved.Tools[3].NativeCompanionFor != "" {
		t.Fatalf("unexpected companion markers: %#v", resolved.Tools)
	}
	if names := toolNames(resolved.DeferredTools); strings.Join(names, ",") != "hidden_code" {
		t.Fatalf("unexpected deferred tools: %v", names)
	}
	resolved.Tools[0].Name = "changed"
	resolved.Tools[0].Schema = map[string]any{"changed": true}
	if params.Tools[1].Name != "local_code" || params.Tools[0].Schema["changed"] != nil {
		t.Fatalf("resolver mutated input: %#v", params.Tools)
	}
}

func TestResolveNativeToolPreferencesRejectsUnsupportedSibling(t *testing.T) {
	web := ai.WebSearchTool{}
	model := &selectiveNativeModel{supported: map[string]bool{}}
	params := ai.ModelRequestParams{
		Tools:       []ai.ToolDefinition{{Name: "local_search", NativeFallbackFor: web.UniqueID()}},
		NativeTools: []ai.NativeTool{web, ai.CodeExecutionTool{}, ai.ImageGenerationTool{}},
	}
	_, err := ai.ResolveNativeToolPreferences(model, params)
	if err == nil || !strings.Contains(err.Error(), `native tool(s) "code_execution"`) ||
		!strings.Contains(err.Error(), `"image_generation"`) || !strings.Contains(err.Error(), `model "selective"`) {
		t.Fatalf("unexpected unsupported sibling error: %v", err)
	}
	if _, err := ai.RequestModel(t.Context(), model, nil, params); err == nil ||
		!strings.Contains(err.Error(), "no local fallback") {
		t.Fatalf("direct request accepted unsupported sibling: %v", err)
	}
	if _, err := ai.ResolveNativeToolPreferences(model, ai.ModelRequestParams{
		NativeTools: []ai.NativeTool{(*ai.WebSearchTool)(nil)},
	}); err == nil || !strings.Contains(err.Error(), "native tool must not be nil") {
		t.Fatalf("unexpected invalid native-tool error: %v", err)
	}

	resolved, err := ai.ResolveNativeToolPreferences(model, ai.ModelRequestParams{
		Tools:       []ai.ToolDefinition{{Name: "local_search", NativeFallbackFor: web.UniqueID()}},
		NativeTools: []ai.NativeTool{web, ai.CodeExecutionTool{Optional: true}},
	})
	if err != nil || len(resolved.NativeTools) != 0 || len(resolved.Tools) != 1 {
		t.Fatalf("optional unsupported tool was not omitted: %#v, %v", resolved, err)
	}
}

func TestResolveNativeToolPreferencesPointersAndDeferredOnly(t *testing.T) {
	web := &ai.WebSearchTool{AllowedDomains: []string{"example.com"}}
	model := &selectiveNativeModel{supported: map[string]bool{web.UniqueID(): true}}
	resolved, err := ai.ResolveNativeToolPreferences(model, ai.ModelRequestParams{
		Tools:       []ai.ToolDefinition{{Name: "fallback", NativeFallbackFor: web.UniqueID()}},
		NativeTools: []ai.NativeTool{web},
	})
	if err != nil {
		t.Fatal(err)
	}
	resolvedWeb, ok := resolved.NativeTools[0].(*ai.WebSearchTool)
	if !ok || len(resolved.Tools) != 0 {
		t.Fatalf("pointer native tool was not preserved: %#v", resolved)
	}
	resolvedWeb.AllowedDomains[0] = "changed"
	if web.AllowedDomains[0] != "example.com" {
		t.Fatal("pointer native tool was not detached")
	}
	if _, err := ai.ResolveNativeToolPreferences(model, resolved); err != nil || model.supportChecks != 1 {
		t.Fatalf("resolved preferences ran twice: checks=%d, err=%v", model.supportChecks, err)
	}

	deferred, err := ai.ResolveNativeToolPreferences(model, ai.ModelRequestParams{
		DeferredTools: []ai.ToolDefinition{{Name: "orphan", NativeCompanionFor: "missing"}},
	})
	if err != nil || deferred.DeferredTools[0].NativeCompanionFor != "" {
		t.Fatalf("deferred-only companion was not cleared: %#v, %v", deferred, err)
	}
}

func TestFallbackModelsResolveNativeToolsPerCandidate(t *testing.T) {
	web := ai.WebSearchTool{}
	primary := &selectiveNativeModel{
		supported: map[string]bool{web.UniqueID(): true}, requestErr: nativeFallbackAPIError{},
	}
	secondary := &selectiveNativeModel{supported: map[string]bool{}}
	model := ai.NewFallbackModel(primary, ai.WithFallbackModels(secondary))
	_, err := ai.RequestModel(t.Context(), model, nil, ai.ModelRequestParams{
		Tools:       []ai.ToolDefinition{{Name: "local_search", NativeFallbackFor: web.UniqueID()}},
		NativeTools: []ai.NativeTool{web},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(primary.requests) != 1 || len(primary.requests[0].Tools) != 0 ||
		len(primary.requests[0].NativeTools) != 1 {
		t.Fatalf("primary candidate did not prefer native tool: %#v", primary.requests)
	}
	if len(secondary.requests) != 1 || len(secondary.requests[0].Tools) != 1 ||
		len(secondary.requests[0].NativeTools) != 0 {
		t.Fatalf("secondary candidate did not use fallback: %#v", secondary.requests)
	}
}

func TestResolveNativeToolPreferencesUnknownSupport(t *testing.T) {
	params := ai.ModelRequestParams{
		Tools:       []ai.ToolDefinition{{Name: "fallback", NativeFallbackFor: "web_search"}},
		NativeTools: []ai.NativeTool{ai.WebSearchTool{}},
	}
	resolved, err := ai.ResolveNativeToolPreferences(unknownNativeSupportModel{}, params)
	if err != nil || len(resolved.Tools) != 1 || len(resolved.NativeTools) != 1 {
		t.Fatalf("unknown support changed request: %#v, %v", resolved, err)
	}
	if _, err := ai.ResolveNativeToolPreferences(nil, params); err != nil {
		t.Fatalf("nil model support should remain unknown: %v", err)
	}
}

func toolNames(definitions []ai.ToolDefinition) []string {
	names := make([]string, len(definitions))
	for index, definition := range definitions {
		names[index] = definition.Name
	}
	return names
}
