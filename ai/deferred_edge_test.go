package ai_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestDeferredResumeRejectsMalformedPendingHistory(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "unused"}}}, nil
	})
	for name, test := range map[string]struct {
		history []ai.ModelMessage
		results ai.DeferredToolResults
		want    string
	}{
		"empty ID": {
			history: []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "external", Args: json.RawMessage(`{}`)}}}},
			results: ai.DeferredToolResults{Calls: map[string]any{"": "result"}}, want: "empty tool call ID",
		},
		"duplicate ID": {
			history: []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "external", ToolCallID: "duplicate", Args: json.RawMessage(`{}`)},
				ai.ToolCallPart{ToolName: "external", ToolCallID: "duplicate", Args: json.RawMessage(`{}`)},
			}}},
			results: ai.DeferredToolResults{Calls: map[string]any{"duplicate": "result"}}, want: "duplicate tool call ID",
		},
		"ordinary call": {
			history: []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "ordinary", ToolCallID: "ordinary", Args: json.RawMessage(`{}`),
			}}}},
			results: ai.DeferredToolResults{Calls: map[string]any{"ordinary": "result"}}, want: "is not deferred",
		},
		"legacy approval receives call result": {
			history: []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "approval", ToolCallID: "approval", Args: json.RawMessage(`{}`),
			}}}},
			results: ai.DeferredToolResults{Calls: map[string]any{"approval": "result"}}, want: "approval tool call",
		},
		"legacy external receives approval": {
			history: []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "external", ToolCallID: "external", Args: json.RawMessage(`{}`),
			}}}},
			results: ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{
				"external": ai.ApproveTool(),
			}}, want: "external tool call",
		},
		"unknown tool": {
			history: []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "missing", ToolCallID: "missing", Args: json.RawMessage(`{}`),
			}}}},
			results: ai.DeferredToolResults{Calls: map[string]any{"missing": "result"}}, want: "is not available",
		},
	} {
		t.Run(name, func(t *testing.T) {
			agent := ai.NewAgent[deps, string](model)
			ai.AddExternalTool[deps, string, struct{}, string](agent, "external")
			ai.AddSimpleTool(agent, "approval", func(context.Context, struct{}) (string, error) {
				return "approved", nil
			}, ai.WithApprovalRequired())
			ai.AddSimpleTool(agent, "ordinary", func(context.Context, struct{}) (string, error) { return "ok", nil })
			_, err := agent.Run(
				t.Context(), "continue", deps{}, ai.WithMessageHistory(test.history),
				ai.WithDeferredToolResults(test.results),
			)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("expected %q, got %v", test.want, err)
			}
		})
	}
}

func TestDeferredKindLookupSkipsLaterResponses(t *testing.T) {
	history := []ai.ModelMessage{
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "approval", ToolCallID: "approval", Args: json.RawMessage(`{}`),
		}}, Metadata: map[string]any{ai.DeferredToolKindsMetadataKey: map[string]any{"approval": "approval"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"other"}}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "later response"}}},
	}
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "approval", func(context.Context, struct{}) (string, error) {
		return "unreachable", nil
	}, ai.WithApprovalRequired())
	result, err := agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(history),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{
			"approval": ai.DenyTool("denied"),
		}}),
	)
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected interior deferred result=%+v err=%v", result, err)
	}
}

func TestDeferredResumeRejectsPreparedOutTool(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "unused"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddTool(ai.NewPreparedTool(
		"external",
		func(context.Context, *ai.RunContext[deps], struct{}) (string, error) { return "unreachable", nil },
		func(_ context.Context, rc *ai.RunContext[deps], definition ai.ToolDefinition) (*ai.ToolDefinition, error) {
			if rc.Deps.Location == "hide" {
				return nil, nil
			}
			return &definition, nil
		},
		ai.WithExternalExecution(),
	))
	history := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
		ToolName: "external", ToolCallID: "external", Args: json.RawMessage(`{}`),
	}}}}
	_, err := agent.Run(
		t.Context(), "continue", deps{Location: "hide"}, ai.WithMessageHistory(history),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Calls: map[string]any{"external": "result"}}),
	)
	if err == nil || !strings.Contains(err.Error(), "is not available") {
		t.Fatalf("expected unavailable error, got %v", err)
	}
}

func TestDeferredToolCallLimitsCountOnlyApprovedLocalExecution(t *testing.T) {
	zero := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "approval", ToolCallID: "approval", Args: json.RawMessage(`{}`)},
			ai.ToolCallPart{ToolName: "external", ToolCallID: "external", Args: json.RawMessage(`{}`)},
		}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithUsageLimits(ai.UsageLimits{ToolCallLimit: &zero}))
	ai.AddSimpleTool(agent, "approval", func(context.Context, struct{}) (string, error) { return "approved", nil }, ai.WithApprovalRequired())
	ai.AddExternalTool[deps, string, struct{}, string](agent, "external")
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || paused.Deferred() == nil {
		t.Fatalf("deferred calls should not consume the local execution limit: result=%+v err=%v", paused, err)
	}
	_, err = agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{
			Approvals: map[string]ai.ToolApproval{"approval": ai.ApproveTool()},
			Calls:     map[string]any{"external": "done"},
		}),
	)
	if err == nil || !strings.Contains(err.Error(), "tool call count 1 exceeds limit 0") {
		t.Fatalf("expected approved execution limit error, got %v", err)
	}
}

func TestDeniedApprovalDoesNotConsumeToolCallLimit(t *testing.T) {
	zero := 0
	request := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "approval", ToolCallID: "approval", Args: json.RawMessage(`{}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithUsageLimits(ai.UsageLimits{ToolCallLimit: &zero}))
	ai.AddSimpleTool(agent, "approval", func(context.Context, struct{}) (string, error) { return "unreachable", nil }, ai.WithApprovalRequired())
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	result, err := agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{
			"approval": ai.DenyTool("denied"),
		}}),
	)
	if err != nil || result.Output != "done" {
		t.Fatalf("denied call should not consume limit: result=%+v err=%v", result, err)
	}
}

func TestDeferredExternalRetryBudgetAndNestedReturn(t *testing.T) {
	for name, test := range map[string]struct {
		value any
		want  string
	}{
		"retry":      {value: ai.Retryf("again"), want: "max retries"},
		"retry part": {value: ai.RetryPromptPart{Content: "again"}, want: "max retries"},
		"nested":     {value: []any{ai.ToolReturn{ReturnValue: "nested"}}, want: "nested"},
	} {
		t.Run(name, func(t *testing.T) {
			model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
					ToolName: "external", ToolCallID: "external", Args: json.RawMessage(`{}`),
				}}}, nil
			})
			agent := ai.NewAgent[deps, string](model, ai.WithMaxRetries(0))
			ai.AddExternalTool[deps, string, struct{}, string](agent, "external")
			paused, err := agent.Run(t.Context(), "go", deps{})
			if err != nil {
				t.Fatal(err)
			}
			_, err = agent.Run(
				t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
				ai.WithDeferredToolResults(ai.DeferredToolResults{Calls: map[string]any{"external": test.value}}),
			)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("expected %s error, got %v", test.want, err)
			}
		})
	}
}

func TestDeferredHistoryIgnoresDuplicateCompletedResults(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		request++
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "ordinary", func(context.Context, struct{}) (string, error) { return "ok", nil })
	ai.AddExternalTool[deps, string, struct{}, string](agent, "external")
	history := []ai.ModelMessage{
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "ordinary", ToolCallID: "ordinary", Args: json.RawMessage(`{}`)}}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "ordinary", ToolCallID: "ordinary", Content: "one"},
			ai.ToolReturnPart{ToolName: "ordinary", ToolCallID: "ordinary", Content: "two"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "external", ToolCallID: "external", Args: json.RawMessage(`{}`)}}},
	}
	result, err := agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(history),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Calls: map[string]any{"external": "result"}}),
	)
	if err != nil || result.Output != "done" || request != 1 {
		t.Fatalf("unexpected duplicate-result resume: result=%+v err=%v", result, err)
	}
}
