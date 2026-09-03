package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type delegationArgs struct {
	Prompt string `json:"prompt"`
}

func TestAgentDelegationPropagatesUsage(t *testing.T) {
	for name, requestLimit := range map[string]int{"complete": 0, "parent limit": 3} {
		t.Run(name, func(t *testing.T) {
			childRequest := 0
			childModel := fakes.NewFunctionModel(func(
				_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				childRequest++
				if childRequest == 1 {
					return &ai.ModelResponse{
						Parts: []ai.ResponsePart{ai.ToolCallPart{
							ToolName: "lookup", ToolCallID: "lookup-1", Args: json.RawMessage(`{}`),
						}},
						Usage: ai.Usage{Requests: 1, InputTokens: 2, Details: map[string]int{"child": 1}},
					}, nil
				}
				return &ai.ModelResponse{
					Parts: []ai.ResponsePart{ai.TextPart{Content: "child result"}},
					Usage: ai.Usage{Requests: 1, OutputTokens: 3, Details: map[string]int{"child": 1}},
				}, nil
			})
			child := ai.NewAgent[deps, string](childModel)
			ai.AddSimpleTool(child, "lookup", func(context.Context, struct{}) (string, error) {
				return "found", nil
			})

			parentRequest := 0
			parentModel := fakes.NewFunctionModel(func(
				_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				parentRequest++
				if parentRequest == 1 {
					return &ai.ModelResponse{
						Parts: []ai.ResponsePart{ai.ToolCallPart{
							ToolName: "delegate", ToolCallID: "delegate-1",
							Args: json.RawMessage(`{"prompt":"research"}`),
						}},
						Usage: ai.Usage{Requests: 1, InputTokens: 5, Details: map[string]int{"parent": 1}},
					}, nil
				}
				part := messages[len(messages)-1].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart)
				if part.Content != "child result" {
					t.Fatalf("unexpected delegated result: %#v", part.Content)
				}
				return &ai.ModelResponse{
					Parts: []ai.ResponsePart{ai.TextPart{Content: "parent result"}},
					Usage: ai.Usage{Requests: 1, OutputTokens: 7, Details: map[string]int{"parent": 1}},
				}, nil
			})
			options := []ai.Option(nil)
			if requestLimit != 0 {
				options = append(options, ai.WithUsageLimits(ai.UsageLimits{RequestLimit: requestLimit}))
			}
			parent := ai.NewAgent[deps, string](parentModel, options...)
			ai.AddTool(parent, "delegate", func(
				ctx context.Context, rc *ai.RunContext[deps], args delegationArgs,
			) (ai.ToolReturn, error) {
				result, err := child.Run(ctx, args.Prompt, rc.Deps)
				if err != nil {
					return ai.ToolReturn{}, err
				}
				return ai.ToolReturn{ReturnValue: result.Output, Usage: result.Usage()}, nil
			})

			result, err := parent.Run(t.Context(), "delegate research", deps{})
			if requestLimit != 0 {
				if !errors.Is(err, ai.ErrUsageLimitExceeded) || result != nil || parentRequest != 1 {
					t.Fatalf("unexpected limited result=%+v requests=%d err=%v", result, parentRequest, err)
				}
				return
			}
			if err != nil || result.Output != "parent result" {
				t.Fatalf("unexpected result=%+v err=%v", result, err)
			}
			usage := result.Usage()
			if usage.Requests != 4 || usage.ToolCalls != 2 || usage.InputTokens != 7 || usage.OutputTokens != 10 ||
				usage.Details["child"] != 2 || usage.Details["parent"] != 2 {
				t.Fatalf("unexpected delegated usage: %+v", usage)
			}
		})
	}
}

func TestConcurrentDelegationAccumulatesUsage(t *testing.T) {
	childModel := fakes.NewTestModel()
	childModel.CustomOutputText = "child result"
	child := ai.NewAgent[deps, string](childModel)
	parentRequest := 0
	parentModel := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		parentRequest++
		if parentRequest == 1 {
			return &ai.ModelResponse{Usage: ai.Usage{Requests: 1}, Parts: []ai.ResponsePart{
				ai.ToolCallPart{
					ToolName: "delegate", ToolCallID: "delegate-1", Args: json.RawMessage(`{"prompt":"one"}`),
				},
				ai.ToolCallPart{
					ToolName: "delegate", ToolCallID: "delegate-2", Args: json.RawMessage(`{"prompt":"two"}`),
				},
			}}, nil
		}
		request := messages[len(messages)-1].(ai.ModelRequest)
		if len(request.Parts) != 2 {
			t.Fatalf("expected two delegated results: %#v", request.Parts)
		}
		return &ai.ModelResponse{
			Usage: ai.Usage{Requests: 1}, Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}},
		}, nil
	})
	parent := ai.NewAgent[deps, string](parentModel)
	ai.AddTool(parent, "delegate", func(
		ctx context.Context, rc *ai.RunContext[deps], args delegationArgs,
	) (ai.ToolReturn, error) {
		result, err := child.Run(ctx, args.Prompt, rc.Deps)
		if err != nil {
			return ai.ToolReturn{}, err
		}
		return ai.ToolReturn{ReturnValue: result.Output, Usage: result.Usage()}, nil
	})

	result, err := parent.Run(t.Context(), "delegate twice", deps{})
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected result=%+v err=%v", result, err)
	}
	if usage := result.Usage(); usage.Requests != 4 || usage.ToolCalls != 2 {
		t.Fatalf("unexpected concurrent delegated usage: %+v", usage)
	}
}
