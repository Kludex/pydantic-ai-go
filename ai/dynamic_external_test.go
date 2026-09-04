package ai_test

import (
	"context"
	"encoding/json"
	"strings"
	"sync/atomic"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestDynamicExternalExecutionPausesSelectedCalls(t *testing.T) {
	request := 0
	var invoked atomic.Int64
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "work", ToolCallID: "local", Args: json.RawMessage(`{"value":"local"}`)},
				ai.ToolCallPart{ToolName: "work", ToolCallID: "remote", Args: json.RawMessage(`{"value":"remote"}`)},
			}}, nil
		}
		last := messages[len(messages)-1].(ai.ModelRequest)
		if len(last.Parts) != 2 || last.Parts[0].(ai.ToolReturnPart).Content != "external result" {
			t.Fatalf("unexpected dynamic external resume: %+v", last.Parts)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddTool(agent, "work", func(
		_ context.Context, _ *ai.RunContext[deps], args deferredArgs,
	) (any, error) {
		invoked.Add(1)
		if args.Value == "remote" {
			return ai.RequestExternalToolExecution(map[string]any{"queue": "slow"}), nil
		}
		return "local result", nil
	}, ai.WithDynamicExternalExecution())
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	pending := paused.Deferred()
	if pending == nil || len(pending.Calls) != 1 || pending.Calls[0].ToolCallID != "remote" ||
		pending.Metadata["remote"]["queue"] != "slow" || invoked.Load() != 2 || paused.Usage().ToolCalls != 1 {
		t.Fatalf("unexpected dynamic external request: pending=%+v invoked=%d usage=%+v", pending, invoked.Load(), paused.Usage())
	}
	result, err := agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Calls: map[string]any{"remote": "external result"}}),
	)
	if err != nil || result.Output != "done" || invoked.Load() != 2 || result.Usage().ToolCalls != 0 {
		t.Fatalf("unexpected dynamic external result=%+v invoked=%d err=%v", result, invoked.Load(), err)
	}
}

func TestApprovedCallCanRedeferForExternalExecution(t *testing.T) {
	requests := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "work", Args: json.RawMessage(`{}`),
			}}}, nil
		}
		latest := messages[len(messages)-1].(ai.ModelRequest)
		if latest.Parts[0].(ai.ToolReturnPart).Content != "remote result" {
			t.Fatalf("unexpected external result: %+v", latest.Parts)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	invocations := 0
	ai.AddTool(agent, "work", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (any, error) {
		invocations++
		if !rc.ToolCallApproved {
			return ai.RequestToolApproval(map[string]any{"stage": "approval"}), nil
		}
		return ai.RequestExternalToolExecution(map[string]any{"stage": "external"}), nil
	}, ai.WithDynamicApproval(), ai.WithDynamicExternalExecution())
	approval, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || approval.Deferred() == nil || len(approval.Deferred().Approvals) != 1 {
		t.Fatalf("unexpected approval pause: result=%+v err=%v", approval, err)
	}
	encoded, err := ai.MarshalMessages(approval.Messages())
	if err != nil {
		t.Fatal(err)
	}
	approvalHistory, err := ai.UnmarshalMessages(encoded)
	if err != nil {
		t.Fatal(err)
	}
	_, err = agent.Run(
		t.Context(), "bypass approval", deps{}, ai.WithMessageHistory(approvalHistory),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Calls: map[string]any{"work": "bypass"}}),
	)
	if err == nil || !strings.Contains(err.Error(), "approval tool call") {
		t.Fatalf("serialized approval kind allowed external result: %v", err)
	}
	external, err := agent.Run(
		t.Context(), "approve", deps{}, ai.WithMessageHistory(approvalHistory),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{
			"work": ai.ApproveTool(),
		}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	pending := external.Deferred()
	if pending == nil || len(pending.Calls) != 1 || len(pending.Approvals) != 0 ||
		pending.Metadata["work"]["stage"] != "external" || requests != 1 || invocations != 2 {
		t.Fatalf("unexpected external re-deferral: pending=%+v requests=%d invocations=%d", pending, requests, invocations)
	}
	encoded, err = ai.MarshalMessages(external.Messages())
	if err != nil {
		t.Fatal(err)
	}
	externalHistory, err := ai.UnmarshalMessages(encoded)
	if err != nil {
		t.Fatal(err)
	}
	_, err = agent.Run(
		t.Context(), "bypass external", deps{}, ai.WithMessageHistory(externalHistory),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{
			"work": ai.ApproveTool(),
		}}),
	)
	if err == nil || !strings.Contains(err.Error(), "external tool call") {
		t.Fatalf("serialized external kind allowed approval result: %v", err)
	}
	result, err := agent.Run(
		t.Context(), "finish", deps{}, ai.WithMessageHistory(externalHistory),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Calls: map[string]any{"work": "remote result"}}),
	)
	if err != nil || result.Output != "done" || requests != 2 || invocations != 2 {
		t.Fatalf("unexpected external continuation: result=%+v requests=%d invocations=%d err=%v", result, requests, invocations, err)
	}
}

func TestDynamicExternalExecutionRequiresDeclaration(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "work", ToolCallID: "work", Args: json.RawMessage(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (ai.ExternalToolRequest, error) {
		return ai.RequestExternalToolExecution(nil), nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err == nil || !strings.Contains(err.Error(), "without WithDynamicExternalExecution") {
		t.Fatalf("expected undeclared external request error, got %v", err)
	}
}

func TestNilDynamicExternalPointerIsOrdinaryResult(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "maybe", ToolCallID: "maybe", Args: json.RawMessage(`{}`),
			}}}, nil
		}
		if last := messages[len(messages)-1].(ai.ModelRequest); last.Parts[0].(ai.ToolReturnPart).Content != (*ai.ExternalToolRequest)(nil) {
			t.Fatalf("unexpected nil external pointer result: %+v", last.Parts)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "maybe", func(context.Context, struct{}) (*ai.ExternalToolRequest, error) {
		return nil, nil
	}, ai.WithDynamicExternalExecution())
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "done" || result.Deferred() != nil {
		t.Fatalf("unexpected nil dynamic external result=%+v err=%v", result, err)
	}
}

func TestDynamicExternalConfigurationConflict(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	ai.AddSimpleTool(agent, "invalid", func(context.Context, struct{}) (string, error) { return "", nil },
		ai.WithExternalExecution(), ai.WithDynamicExternalExecution())
	if _, err := agent.Run(t.Context(), "go", deps{}); err == nil || !strings.Contains(err.Error(), "static and dynamic external") {
		t.Fatalf("expected dynamic external conflict, got %v", err)
	}
}
