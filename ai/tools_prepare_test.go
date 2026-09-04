package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestPrepareToolsFiltersByRunDependencies(t *testing.T) {
	var toolNames []string
	model := fakes.NewFunctionModel(func(_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
		for _, tool := range params.Tools {
			toolNames = append(toolNames, tool.Name)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[bool, string](model)
	ai.AddSimpleTool(agent, "answer", func(context.Context, struct{}) (string, error) { return "42", nil })
	agent.AddToolsPrepareFunc(func(
		_ context.Context, rc *ai.RunContext[bool], tools []ai.ToolDefinition,
	) ([]ai.ToolDefinition, error) {
		if !rc.Deps {
			return nil, nil
		}
		return tools, nil
	})
	if _, err := agent.Run(t.Context(), "hidden", false); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Run(t.Context(), "visible", true); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(toolNames, []string{"answer"}) {
		t.Fatalf("unexpected prepared tools %v", toolNames)
	}
}

func TestPrepareToolsRunsEveryStepWithFreshDefinitions(t *testing.T) {
	calls := 0
	model := fakes.NewFunctionModel(func(_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
		property := params.Tools[0].Schema["properties"].(map[string]any)["value"].(map[string]any)
		if property["description"] != "prepared" {
			t.Fatalf("tool schema was not prepared: %v", property)
		}
		calls++
		if calls == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				strategyCall("work", "call", `{"value":"x"}`),
			}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddRawTool(ai.ToolDefinition{
		Name: "work",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"value": map[string]any{"type": "string", "examples": []any{"x"}},
			},
		},
	}, func(context.Context, json.RawMessage) (any, error) { return "done", nil })
	prepareCalls := 0
	agent.AddToolsPrepareFunc(func(
		_ context.Context, rc *ai.RunContext[deps], tools []ai.ToolDefinition,
	) ([]ai.ToolDefinition, error) {
		prepareCalls++
		if len(rc.Messages()) != 2*prepareCalls-1 {
			t.Fatalf("prepare saw %d messages on step %d", len(rc.Messages()), prepareCalls)
		}
		property := tools[0].Schema["properties"].(map[string]any)["value"].(map[string]any)
		if _, exists := property["description"]; exists {
			t.Fatal("prepared schema mutation leaked into the next step")
		}
		property["description"] = "prepared"
		return tools, nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if prepareCalls != 2 {
		t.Fatalf("expected preparation on both steps, got %d", prepareCalls)
	}
}

func TestPrepareToolsHooksComposeInRegistrationOrder(t *testing.T) {
	var exposed int
	model := fakes.NewFunctionModel(func(_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
		exposed = len(params.Tools)
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) { return "", nil })
	agent.AddToolsPrepareFunc(func(
		_ context.Context, _ *ai.RunContext[deps], tools []ai.ToolDefinition,
	) ([]ai.ToolDefinition, error) {
		tools[0].Description = "first"
		return tools, nil
	})
	agent.AddToolsPrepareFunc(func(
		_ context.Context, _ *ai.RunContext[deps], tools []ai.ToolDefinition,
	) ([]ai.ToolDefinition, error) {
		if tools[0].Description != "first" {
			t.Fatalf("second hook did not see first hook result: %+v", tools[0])
		}
		return []ai.ToolDefinition{}, nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if exposed != 0 {
		t.Fatalf("expected no tools, got %d", exposed)
	}
}

func TestPrepareToolsReceivesCurrentOutputRetry(t *testing.T) {
	var retries []int
	calls := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		calls++
		if calls == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{strategyCall("retry", "call", `{}`)}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "retry", func(context.Context, struct{}) (string, error) {
		return "", ai.Retryf("again")
	})
	agent.AddToolsPrepareFunc(func(
		_ context.Context, rc *ai.RunContext[deps], tools []ai.ToolDefinition,
	) ([]ai.ToolDefinition, error) {
		retries = append(retries, rc.Retry)
		return tools, nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(retries, []int{0, 0}) {
		t.Fatalf("function retry leaked into output retry context: %v", retries)
	}
}

func TestPrepareToolsPreservesNilRawSchema(t *testing.T) {
	model := fakes.NewFunctionModel(func(_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
		if params.Tools[0].Schema != nil {
			t.Fatalf("expected nil schema, got %v", params.Tools[0].Schema)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddRawTool(ai.ToolDefinition{Name: "raw"}, func(context.Context, json.RawMessage) (any, error) {
		return nil, nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
}

func TestPrepareToolsFailuresStopBeforeModelRequest(t *testing.T) {
	boom := errors.New("boom")
	tests := []struct {
		name    string
		prepare ai.ToolsPrepareFunc[deps]
		want    error
	}{
		{
			name: "callback error",
			prepare: func(context.Context, *ai.RunContext[deps], []ai.ToolDefinition) ([]ai.ToolDefinition, error) {
				return nil, boom
			},
			want: boom,
		},
		{
			name: "unknown tool",
			prepare: func(context.Context, *ai.RunContext[deps], []ai.ToolDefinition) ([]ai.ToolDefinition, error) {
				return []ai.ToolDefinition{{Name: "unknown"}}, nil
			},
		},
		{
			name: "duplicate tool",
			prepare: func(_ context.Context, _ *ai.RunContext[deps], tools []ai.ToolDefinition) ([]ai.ToolDefinition, error) {
				return append(tools, tools[0]), nil
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			requested := false
			model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
				requested = true
				return nil, fmt.Errorf("model should not be called")
			})
			agent := ai.NewAgent[deps, string](model)
			ai.AddSimpleTool(agent, "known", func(context.Context, struct{}) (string, error) { return "", nil })
			agent.AddToolsPrepareFunc(test.prepare)
			_, err := agent.Run(t.Context(), "go", deps{})
			if test.want != nil {
				if !errors.Is(err, test.want) {
					t.Fatalf("expected %v, got %v", test.want, err)
				}
			} else if err == nil {
				t.Fatal("expected preparation error")
			}
			if requested {
				t.Fatal("model was called after preparation failed")
			}
		})
	}
}
