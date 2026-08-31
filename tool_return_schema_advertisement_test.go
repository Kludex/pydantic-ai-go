package ai_test

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type returnValue struct {
	Value string `json:"value"`
}

func TestIncludeToolReturnSchemasCapability(t *testing.T) {
	selectorCalls := map[string]int{}
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		definitions := append([]ai.ToolDefinition(nil), params.Tools...)
		definitions = append(definitions, params.DeferredTools...)
		for _, definition := range definitions {
			switch definition.Name {
			case "selected":
				if definition.IncludeReturnSchema == nil || !*definition.IncludeReturnSchema {
					t.Fatalf("selected schema was not included: %+v", definition)
				}
			case "omitted", "explicitly_omitted":
				if definition.IncludeReturnSchema == nil || *definition.IncludeReturnSchema {
					t.Fatalf("schema was not omitted: %+v", definition)
				}
			case "deferred":
				if definition.IncludeReturnSchema == nil || !*definition.IncludeReturnSchema {
					t.Fatalf("deferred schema was not included: %+v", definition)
				}
			}
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	capability := ai.IncludeToolReturnSchemas{Select: func(
		_ context.Context, _ *ai.RunInfo, definition ai.ToolDefinition,
	) (bool, error) {
		selectorCalls[definition.Name]++
		definition.ReturnSchema["mutated"] = true
		return definition.Name != "omitted", nil
	}}
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(capability))
	agent.AddTool(ai.NewSimpleTool[struct{}](
		"selected", func(context.Context, struct{}) (returnValue, error) { return returnValue{}, nil },
	))
	agent.AddTool(ai.NewSimpleTool[struct{}](
		"omitted", func(context.Context, struct{}) (returnValue, error) { return returnValue{}, nil },
	))
	agent.AddTool(ai.NewSimpleTool[struct{}](
		"explicitly_omitted", func(context.Context, struct{}) (returnValue, error) { return returnValue{}, nil },
		ai.WithReturnSchemaIncluded(false),
	))
	agent.AddTool(ai.NewSimpleTool[struct{}](
		"deferred", func(context.Context, struct{}) (returnValue, error) { return returnValue{}, nil },
		ai.WithDeferredLoading(),
	))
	if _, err := agent.Run(t.Context(), "hello", struct{}{}); err != nil {
		t.Fatal(err)
	}
	if selectorCalls["selected"] != 1 || selectorCalls["omitted"] != 1 || selectorCalls["deferred"] != 1 ||
		selectorCalls["explicitly_omitted"] != 0 {
		t.Fatalf("unexpected selector calls: %v", selectorCalls)
	}
}

func TestIncludeAllToolReturnSchemasAndSelectorError(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(params.Tools) != 1 || params.Tools[0].IncludeReturnSchema == nil ||
			!*params.Tools[0].IncludeReturnSchema {
			t.Fatalf("default capability did not include schema: %+v", params.Tools)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(ai.IncludeToolReturnSchemas{}))
	agent.AddTool(ai.NewSimpleTool[struct{}](
		"tool", func(context.Context, struct{}) (returnValue, error) { return returnValue{}, nil },
	))
	if _, err := agent.Run(t.Context(), "hello", struct{}{}); err != nil {
		t.Fatal(err)
	}

	selectorErr := errors.New("selector failed")
	failed := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(ai.IncludeToolReturnSchemas{
		Select: func(context.Context, *ai.RunInfo, ai.ToolDefinition) (bool, error) {
			return false, selectorErr
		},
	}))
	failed.AddTool(ai.NewSimpleTool[struct{}](
		"tool", func(context.Context, struct{}) (returnValue, error) { return returnValue{}, nil },
	))
	if _, err := failed.Run(t.Context(), "hello", struct{}{}); !errors.Is(err, selectorErr) {
		t.Fatalf("unexpected selector error: %v", err)
	}
	failedDeferred := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(ai.IncludeToolReturnSchemas{
		Select: func(context.Context, *ai.RunInfo, ai.ToolDefinition) (bool, error) {
			return false, selectorErr
		},
	}))
	failedDeferred.AddTool(ai.NewSimpleTool[struct{}](
		"deferred", func(context.Context, struct{}) (returnValue, error) { return returnValue{}, nil },
		ai.WithDeferredLoading(),
	))
	if _, err := failedDeferred.Run(t.Context(), "hello", struct{}{}); !errors.Is(err, selectorErr) {
		t.Fatalf("unexpected deferred selector error: %v", err)
	}
}

func TestPrepareToolReturnSchema(t *testing.T) {
	included := true
	definition := ai.ToolDefinition{
		Name: "lookup", Description: "Looks up a value.",
		ReturnSchema: map[string]any{"type": "object"}, IncludeReturnSchema: &included,
	}
	native, err := ai.PrepareToolReturnSchema(definition, true)
	if err != nil || native.ReturnSchema["type"] != "object" || native.Description != definition.Description {
		t.Fatalf("unexpected native definition: %+v err=%v", native, err)
	}
	native.ReturnSchema["type"] = "changed"
	if definition.ReturnSchema["type"] != "object" {
		t.Fatal("native definition was not detached")
	}
	fallback, err := ai.PrepareToolReturnSchema(definition, false)
	if err != nil || fallback.ReturnSchema != nil || !strings.Contains(fallback.Description, `"type": "object"`) ||
		!strings.HasPrefix(fallback.Description, "Looks up a value.\n\nReturn schema:\n\n") {
		t.Fatalf("unexpected fallback definition: %+v err=%v", fallback, err)
	}
	withoutDescription := definition
	withoutDescription.Description = ""
	fallback, err = ai.PrepareToolReturnSchema(withoutDescription, false)
	if err != nil || !strings.HasPrefix(fallback.Description, "Return schema:\n\n") {
		t.Fatalf("unexpected undescribed fallback: %+v err=%v", fallback, err)
	}
	for _, omitted := range []ai.ToolDefinition{
		{Name: "default", ReturnSchema: map[string]any{"type": "string"}},
		{Name: "missing", IncludeReturnSchema: &included},
		{Name: "empty", ReturnSchema: map[string]any{}, IncludeReturnSchema: &included},
	} {
		prepared, err := ai.PrepareToolReturnSchema(omitted, false)
		if err != nil || prepared.ReturnSchema != nil {
			t.Fatalf("unexpected omitted schema: %+v err=%v", prepared, err)
		}
	}
	invalid := definition
	invalid.ReturnSchema = map[string]any{"bad": make(chan struct{})}
	if _, err := ai.PrepareToolReturnSchema(invalid, false); err == nil ||
		!strings.Contains(err.Error(), `marshal return schema for tool "lookup"`) {
		t.Fatalf("unexpected marshal error: %v", err)
	}
}

type returnSchemaErrorToolset struct{ err error }

func (toolset returnSchemaErrorToolset) Tools(
	context.Context, *ai.RunContext[struct{}],
) ([]ai.Tool[struct{}], error) {
	return nil, toolset.err
}

func TestReturnSchemaToolsetError(t *testing.T) {
	toolsetErr := errors.New("toolset failed")
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	agent.AddToolset(ai.WithToolReturnSchemas[struct{}](returnSchemaErrorToolset{err: toolsetErr}))
	if _, err := agent.Run(t.Context(), "hello", struct{}{}); !errors.Is(err, toolsetErr) {
		t.Fatalf("unexpected toolset error: %v", err)
	}
}

func TestReturnSchemaOptionIsDetached(t *testing.T) {
	included := true
	tool := ai.NewSimpleTool[struct{}](
		"tool", func(context.Context, struct{}) (returnValue, error) { return returnValue{}, nil },
		ai.WithReturnSchemaIncluded(true),
	)
	definition := tool.Definition()
	if !reflect.DeepEqual(definition.IncludeReturnSchema, &included) {
		t.Fatalf("unexpected inclusion option: %+v", definition)
	}
	*definition.IncludeReturnSchema = false
	if !*tool.Definition().IncludeReturnSchema {
		t.Fatal("tool inclusion option was not detached")
	}
}
