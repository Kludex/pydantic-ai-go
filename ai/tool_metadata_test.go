package ai_test

import (
	"context"
	"encoding/json"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestToolMetadataIsLocalAndPreparedFromFreshCopies(t *testing.T) {
	metadata := map[string]any{
		"owner":  "agent",
		"nested": map[string]any{"stable": true},
	}
	requests := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if len(params.Tools) != 1 {
			t.Fatalf("unexpected tools: %+v", params.Tools)
		}
		got := params.Tools[0].Metadata
		if got["owner"] != "prepared" || got["step"] != requests ||
			got["global"] != true || got["outside"] != nil ||
			got["nested"].(map[string]any)["stable"] != true {
			t.Fatalf("unexpected metadata on request %d: %+v", requests, got)
		}
		if requests == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "call", Args: []byte(`{}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddPreparedTool(
		agent,
		"work",
		func(context.Context, *ai.RunContext[deps], struct{}) (string, error) { return "done", nil },
		func(_ context.Context, rc *ai.RunContext[deps], def ai.ToolDefinition) (*ai.ToolDefinition, error) {
			if def.Metadata["owner"] != "agent" || def.Metadata["step"] != nil {
				t.Fatalf("metadata leaked between steps: %+v", def.Metadata)
			}
			def.Metadata["owner"] = "prepared"
			def.Metadata["step"] = rc.Usage().Requests + 1
			return &def, nil
		},
		ai.WithToolMetadata(metadata),
	)
	metadata["outside"] = true
	metadata["nested"].(map[string]any)["stable"] = false
	agent.AddToolsPrepareFunc(func(
		_ context.Context, _ *ai.RunContext[deps], tools []ai.ToolDefinition,
	) ([]ai.ToolDefinition, error) {
		tools[0].Metadata["global"] = true
		return tools, nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if requests != 2 {
		t.Fatalf("expected two requests, got %d", requests)
	}
}

func TestRawToolDefinitionIsClonedAtRegistration(t *testing.T) {
	definition := ai.ToolDefinition{
		Name: "raw",
		Schema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
		Metadata: map[string]any{"owner": "original"},
	}
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if params.Tools[0].Metadata["owner"] != "original" || params.Tools[0].Schema["type"] != "object" {
			t.Fatalf("registered definition was mutated externally: %+v", params.Tools[0])
		}
		wire, err := json.Marshal(params.Tools[0])
		if err != nil {
			t.Fatal(err)
		}
		if string(wire) == "" || json.Valid(wire) == false {
			t.Fatalf("invalid tool definition JSON: %s", wire)
		}
		var decoded map[string]any
		if err := json.Unmarshal(wire, &decoded); err != nil {
			t.Fatal(err)
		}
		if _, exists := decoded["metadata"]; exists {
			t.Fatalf("metadata leaked into provider JSON: %s", wire)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddRawTool(definition, func(context.Context, json.RawMessage) (any, error) { return nil, nil })
	definition.Metadata["owner"] = "outside"
	definition.Schema["type"] = "string"
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
}
