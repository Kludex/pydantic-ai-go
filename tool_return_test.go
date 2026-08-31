package ai_test

import (
	"context"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type richToolArgs struct {
	Name string `json:"name"`
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
