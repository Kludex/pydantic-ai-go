package fakes_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestFunctionModelNativeToolSupport(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{}, nil
	})
	if !model.SupportsNativeTool(ai.WebSearchTool{}) || model.SupportsNativeTool((*ai.WebSearchTool)(nil)) {
		t.Fatal("unexpected function-model native-tool support")
	}
}

func TestFunctionModelDefaults(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "hi"}}}, nil
	})
	if model.Name() != "function-model" {
		t.Fatal("unexpected name")
	}
	resp, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.Requests != 1 {
		t.Fatalf("expected default usage, got %+v", resp.Usage)
	}
}

func TestFunctionModelPreservesUsageAndErrors(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Usage: ai.Usage{Requests: 3}}, nil
	})
	resp, err := model.Request(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil || resp.Usage.Requests != 3 {
		t.Fatalf("usage overwritten: %+v %v", resp, err)
	}
	failing := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return nil, errors.New("down")
	})
	if _, err := failing.Request(t.Context(), nil, ai.ModelRequestParams{}); err == nil {
		t.Fatal("expected error")
	}
}

func TestTestModelName(t *testing.T) {
	if fakes.NewTestModel().Name() != "test-model" {
		t.Fatal("unexpected name")
	}
}

func TestTestModelCustomOutputText(t *testing.T) {
	model := fakes.NewTestModel()
	model.CustomOutputText = "custom"
	resp, err := model.Request(t.Context(), nil, ai.ModelRequestParams{AllowText: true})
	if err != nil || resp.Text() != "custom" {
		t.Fatalf("unexpected response %+v %v", resp, err)
	}
}

func TestTestModelGeneratesValuesForAllSchemaTypes(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"s":      map[string]any{"type": "string"},
			"date":   map[string]any{"type": "string", "format": "date-time"},
			"binary": map[string]any{"type": "string", "contentEncoding": "base64"},
			"i":      map[string]any{"type": "integer"},
			"n":      map[string]any{"type": "number"},
			"b":      map[string]any{"type": "boolean"},
			"a":      map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
			"o":      map[string]any{"type": "object", "properties": map[string]any{"x": map[string]any{"type": "string"}}},
			"free":   map[string]any{"type": "object"},
			"e":      map[string]any{"type": "string", "enum": []string{"one", "two"}},
			"eany":   map[string]any{"type": "string", "enum": []any{"three", "four"}},
			"maybe":  map[string]any{"anyOf": []any{map[string]any{"type": "string"}, map[string]any{"type": "null"}}},
			"weird":  map[string]any{"type": "mystery"},
			"raw":    "not a schema",
		},
	}
	model := fakes.NewTestModel()
	resp, err := model.Request(t.Context(), nil, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{Name: "t", Schema: schema}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var args map[string]any
	if err := json.Unmarshal(resp.ToolCalls()[0].Args, &args); err != nil {
		t.Fatal(err)
	}
	expected := map[string]any{
		"s": "a", "date": "2000-01-01T00:00:00Z", "binary": "YQ==", "i": 0.0, "n": 0.0, "b": false,
		"e": "one", "eany": "three", "maybe": "a", "raw": "a", "weird": nil,
	}
	for k, v := range expected {
		got, ok := args[k]
		if !ok || got != v {
			t.Fatalf("field %s: expected %v, got %v", k, v, got)
		}
	}
	if _, ok := args["a"].([]any); !ok {
		t.Fatalf("array field: %v", args["a"])
	}
	if o, ok := args["o"].(map[string]any); !ok || o["x"] != "a" {
		t.Fatalf("object field: %v", args["o"])
	}
	if free, ok := args["free"].(map[string]any); !ok || len(free) != 0 {
		t.Fatalf("free object field: %v", args["free"])
	}
}

func TestTestModelToolLoopAndOutputTool(t *testing.T) {
	toolSchema := map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}}
	params := ai.ModelRequestParams{
		Tools:      []ai.ToolDefinition{{Name: "search", Schema: toolSchema}},
		OutputTool: &ai.ToolDefinition{Name: "final_result", Schema: toolSchema},
	}
	model := fakes.NewTestModel()

	first, err := model.Request(t.Context(), nil, params)
	if err != nil {
		t.Fatal(err)
	}
	if first.ToolCalls()[0].ToolName != "search" {
		t.Fatalf("expected search call, got %+v", first.ToolCalls())
	}

	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "go"}}},
		*first,
		ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{ToolName: "search", Content: "found"}}},
	}
	second, err := model.Request(t.Context(), history, params)
	if err != nil {
		t.Fatal(err)
	}
	if second.ToolCalls()[0].ToolName != "final_result" {
		t.Fatalf("expected output tool call, got %+v", second.ToolCalls())
	}

	model.CustomOutputArgs = json.RawMessage(`{"q":"custom"}`)
	third, err := model.Request(t.Context(), history, params)
	if err != nil {
		t.Fatal(err)
	}
	if string(third.ToolCalls()[0].Args) != `{"q":"custom"}` {
		t.Fatalf("custom output args not used: %s", third.ToolCalls()[0].Args)
	}
}

func TestTestModelTextAfterToolCalls(t *testing.T) {
	model := fakes.NewTestModel()
	history := []ai.ModelMessage{
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.TextPart{Content: "thinking..."},
			ai.ToolCallPart{ToolName: "search", Args: json.RawMessage(`{}`)},
		}},
	}
	resp, err := model.Request(t.Context(), history, ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{Name: "search", Schema: map[string]any{"type": "object"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "success" {
		t.Fatalf("expected success text, got %q", resp.Text())
	}
}
