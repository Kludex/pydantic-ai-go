package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type approvalErrorToolset struct{}

func (approvalErrorToolset) Tools(context.Context, *ai.RunContext[deps]) ([]ai.Tool[deps], error) {
	return nil, errors.New("approval tools unavailable")
}

func TestRequireApprovalToolsetSelectsOriginalNames(t *testing.T) {
	request := 0
	var calls []string
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			if params.Tools[0].RequiresApproval || !params.Tools[1].RequiresApproval {
				t.Fatalf("unexpected approval definitions: %+v", params.Tools)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "safe", ToolCallID: "safe", Args: json.RawMessage(`{}`)},
				ai.ToolCallPart{ToolName: "danger", ToolCallID: "danger", Args: json.RawMessage(`{}`)},
			}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	safe := ai.NewSimpleTool[deps]("safe", func(context.Context, struct{}) (string, error) {
		calls = append(calls, "safe")
		return "safe", nil
	})
	danger := ai.NewSimpleTool[deps]("danger", func(context.Context, struct{}) (string, error) {
		calls = append(calls, "danger")
		return "danger", nil
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddToolset(ai.RequireApprovalToolset(ai.NewFunctionToolset(safe, danger), "danger"))
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(calls, []string{"safe"}) || paused.Deferred() == nil ||
		paused.Deferred().Approvals[0].ToolName != "danger" {
		t.Fatalf("unexpected selective approval: calls=%v pending=%+v", calls, paused.Deferred())
	}
	result, err := agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{
			"danger": ai.ApproveTool(),
		}}),
	)
	if err != nil || result.Output != "done" || !slices.Equal(calls, []string{"safe", "danger"}) {
		t.Fatalf("unexpected approval resume: result=%+v calls=%v err=%v", result, calls, err)
	}
}

func TestRequireApprovalToolsetPropagatesToolErrors(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	agent.AddToolset(ai.RequireApprovalToolset[deps](approvalErrorToolset{}))
	if _, err := agent.Run(t.Context(), "go", deps{}); err == nil || !strings.Contains(err.Error(), "unavailable") {
		t.Fatalf("expected wrapped toolset error, got %v", err)
	}
}

func TestRequireApprovalToolsetForwardsLifecycle(t *testing.T) {
	request := 0
	var log []string
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if !strings.Contains(params.Instructions, "remote step 1") || !params.Tools[0].RequiresApproval {
			t.Fatalf("wrapper lost lifecycle behavior: instructions=%q tools=%+v", params.Instructions, params.Tools)
		}
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "work", Args: json.RawMessage(`{}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddToolset(ai.RequireApprovalToolset[deps](&lifecycleToolset{log: &log}))
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains(log, "call:work:1") || !slices.Contains(log, "close") {
		t.Fatalf("toolset executed before approval or did not close: %v", log)
	}
	result, err := agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{
			"work": ai.ApproveTool(),
		}}),
	)
	if err != nil || result.Output != "done" || !slices.Contains(log, "call:work:1") ||
		request != 2 {
		t.Fatalf("unexpected lifecycle approval: result=%+v log=%v err=%v", result, log, err)
	}
}
