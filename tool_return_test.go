package ai_test

import (
	"context"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type richToolArgs struct {
	Name string `json:"name"`
}

type richToolResult struct {
	Value string `json:"value"`
}

func TestReflectedToolReturnSchemas(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		definitions := make(map[string]ai.ToolDefinition, len(params.Tools))
		for _, definition := range params.Tools {
			definitions[definition.Name] = definition
		}
		structured := definitions["structured"].ReturnSchema
		if structured["type"] != "object" ||
			structured["properties"].(map[string]any)["value"].(map[string]any)["type"] != "string" {
			t.Fatalf("unexpected structured return schema: %+v", structured)
		}
		if definitions["scalar"].ReturnSchema["type"] != "string" ||
			definitions["rich"].ReturnSchema != nil || definitions["unsupported"].ReturnSchema != nil ||
			definitions["rich_custom"].ReturnSchema["type"] != "object" {
			t.Fatalf("unexpected reflected return schemas: %+v", definitions)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	structured := ai.NewSimpleTool[deps](
		"structured", func(context.Context, struct{}) (richToolResult, error) {
			return richToolResult{Value: "ok"}, nil
		},
	)
	definition := structured.Definition()
	definition.ReturnSchema["type"] = "changed"
	if structured.Definition().ReturnSchema["type"] != "object" {
		t.Fatal("Tool.Definition shared its return schema")
	}
	agent.AddTool(structured)
	ai.AddSimpleTool(agent, "scalar", func(context.Context, struct{}) (string, error) {
		return "ok", nil
	})
	ai.AddSimpleTool(agent, "rich", func(context.Context, struct{}) (*ai.ToolReturn, error) {
		return &ai.ToolReturn{ReturnValue: "ok"}, nil
	})
	customSchema := map[string]any{"type": "object"}
	ai.AddSimpleTool(agent, "rich_custom", func(context.Context, struct{}) (ai.ToolReturn, error) {
		return ai.ToolReturn{ReturnValue: richToolResult{Value: "ok"}}, nil
	}, ai.WithReturnSchema(customSchema))
	customSchema["type"] = "changed"
	ai.AddSimpleTool(agent, "unsupported", func(context.Context, struct{}) (chan int, error) {
		return nil, nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
}

func TestNestedRichToolReturnsAreRejected(t *testing.T) {
	for name, value := range map[string]any{
		"value":   []any{ai.ToolReturn{ReturnValue: "nested"}},
		"pointer": []*ai.ToolReturn{nil},
	} {
		t.Run(name, func(t *testing.T) {
			model := fakes.NewFunctionModel(func(
				context.Context, []ai.ModelMessage, ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
					ToolName: "nested", ToolCallID: "nested", Args: []byte(`{}`),
				}}}, nil
			})
			agent := ai.NewAgent[deps, string](model)
			ai.AddSimpleTool(agent, "nested", func(context.Context, struct{}) (any, error) {
				return value, nil
			})
			_, err := agent.Run(t.Context(), "go", deps{})
			if err == nil || !strings.Contains(err.Error(), "return value contains nested ToolReturn") {
				t.Fatalf("unexpected nested return error: %v", err)
			}
		})
	}
}

func TestRichToolReturnsPreserveValueContentAndMetadata(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(messages) == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{
					ToolName: "lookup", ToolCallID: "first", ToolKind: ai.ToolPartKindToolSearch,
					Args: []byte(`{"name":"one"}`),
				},
				ai.ToolCallPart{
					ToolName: "lookup", ToolCallID: "second", ToolKind: ai.ToolPartKindCapabilityLoad,
					Args: []byte(`{"name":"two"}`),
				},
			}}, nil
		}

		parts := messages[len(messages)-1].(ai.ModelRequest).Parts
		if len(parts) != 4 {
			t.Fatalf("unexpected rich tool request parts: %+v", parts)
		}
		first := parts[0].(ai.ToolReturnPart)
		second := parts[1].(ai.ToolReturnPart)
		if first.Content != "value-one" || first.Metadata["name"] != "one" ||
			first.ToolKind != ai.ToolPartKindToolSearch {
			t.Fatalf("unexpected first tool return: %+v", first)
		}
		if second.Content != "value-two" || second.Metadata["name"] != "two" ||
			second.ToolKind != ai.ToolPartKindCapabilityLoad {
			t.Fatalf("unexpected second tool return: %+v", second)
		}
		firstContent := parts[2].(ai.UserPromptPart).Contents
		secondContent := parts[3].(ai.UserPromptPart).Contents
		if firstContent[0].(ai.TextContent).Text != "details-one" ||
			secondContent[0].(ai.TextContent).Text != "details-two" {
			t.Fatalf("unexpected trailing content: %+v %+v", firstContent, secondContent)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "lookup", func(_ context.Context, args richToolArgs) (ai.ToolReturn, error) {
		return ai.ToolReturn{
			ReturnValue: "value-" + args.Name,
			Content:     []ai.UserContent{ai.TextContent{Text: "details-" + args.Name}},
			Metadata:    map[string]any{"name": args.Name},
		}, nil
	})

	result, err := agent.Run(t.Context(), "look up both", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "done" {
		t.Fatalf("unexpected output %q", result.Output)
	}
}

func TestRichToolReturnPointerAndNilPointer(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		switch request {
		case 1:
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "rich", ToolCallID: "rich", Args: []byte(`{}`)},
			}}, nil
		case 2:
			part := messages[len(messages)-1].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart)
			if part.Content != "value" {
				t.Fatalf("unexpected pointer return: %+v", part)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "nil", ToolCallID: "nil", Args: []byte(`{}`)},
			}}, nil
		default:
			part := messages[len(messages)-1].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart)
			if part.Content != nil {
				t.Fatalf("nil rich return was not normalized: %+v", part)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		}
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "rich", func(context.Context, struct{}) (*ai.ToolReturn, error) {
		return &ai.ToolReturn{ReturnValue: "value"}, nil
	})
	ai.AddSimpleTool(agent, "nil", func(context.Context, struct{}) (*ai.ToolReturn, error) {
		return nil, nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
}

func TestToolReturnKindAndDeniedOutcomeRoundTrip(t *testing.T) {
	original := ai.ToolReturnPart{
		ToolName: "lookup", ToolCallID: "call", ToolKind: ai.ToolPartKindToolSearch,
		Content: "denied", Outcome: ai.ToolReturnOutcomeDenied,
	}
	data, err := ai.MarshalMessages([]ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{original}}})
	if err != nil {
		t.Fatal(err)
	}
	messages, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	part := messages[0].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart)
	if part.ToolKind != ai.ToolPartKindToolSearch || part.Outcome != ai.ToolReturnOutcomeDenied {
		t.Fatalf("tool return fields did not round trip: %+v", part)
	}
}
