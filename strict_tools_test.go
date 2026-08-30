package ai_test

import (
	"context"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestStrictToolOptions(t *testing.T) {
	var definitions []ai.ToolDefinition
	model := fakes.NewFunctionModel(func(_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
		definitions = params.Tools
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "strict", func(context.Context, struct{}) (string, error) { return "", nil }, ai.WithStrict())
	ai.AddSimpleTool(agent, "loose", func(context.Context, struct{}) (string, error) { return "", nil }, ai.WithoutStrict())
	ai.AddSimpleTool(agent, "default", func(context.Context, struct{}) (string, error) { return "", nil })
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if definitions[0].Strict == nil || !*definitions[0].Strict {
		t.Fatalf("strict option not applied: %+v", definitions[0])
	}
	if definitions[1].Strict == nil || *definitions[1].Strict {
		t.Fatalf("without-strict option not applied: %+v", definitions[1])
	}
	if definitions[2].Strict != nil {
		t.Fatalf("default strict value must stay nil: %+v", definitions[2])
	}
}
