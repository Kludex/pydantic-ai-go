package ai_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func deferredHookModel(t *testing.T, inspect func(ai.ToolReturnPart)) ai.Model {
	t.Helper()
	request := 0
	return fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "call", Args: json.RawMessage(`{"value":1}`),
			}}}, nil
		}
		if inspect != nil {
			inspect(messages[len(messages)-1].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart))
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
}

func TestAfterToolValidationHookCanRequestApproval(t *testing.T) {
	approved := false
	hook := ai.AfterToolValidationFunc(func(
		_ context.Context, _ *ai.RunInfo, hook ai.ToolHookContext, args any,
	) (any, error) {
		if hook.Approved {
			approved = true
			if hook.CallMetadata["reviewer"] != "human" {
				t.Fatalf("approval metadata missing from hook context: %+v", hook)
			}
			return args, nil
		}
		request := ai.RequestToolApproval(map[string]any{"reason": "sensitive"})
		return &request, nil
	})
	executed := 0
	agent := ai.NewAgent[deps, string](deferredHookModel(t, nil), ai.WithCapabilities(hook))
	ai.AddTool(agent, "work", func(
		_ context.Context, _ *ai.RunContext[deps], args hookedArgs,
	) (int, error) {
		executed = args.Value
		return args.Value, nil
	})
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || paused.Deferred() == nil || executed != 0 ||
		paused.Deferred().Metadata["call"]["reason"] != "sensitive" {
		t.Fatalf("unexpected hook approval pause=%+v executed=%d err=%v", paused, executed, err)
	}
	result, err := agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{
			Approvals: map[string]ai.ToolApproval{"call": ai.ApproveTool()},
			Metadata:  map[string]map[string]any{"call": {"reviewer": "human"}},
		}),
	)
	if err != nil || result.Output != "done" || !approved || executed != 1 {
		t.Fatalf("unexpected hook approval resume=%+v approved=%v executed=%d err=%v", result, approved, executed, err)
	}
}

func TestAfterToolValidationHookCanRequestExternalExecution(t *testing.T) {
	hook := ai.AfterToolValidationFunc(func(
		context.Context, *ai.RunInfo, ai.ToolHookContext, any,
	) (any, error) {
		return ai.RequestExternalToolExecution(map[string]any{"queue": "validation"}), nil
	})
	agent := ai.NewAgent[deps, string](deferredHookModel(t, nil), ai.WithCapabilities(hook))
	ai.AddTool(agent, "work", func(
		context.Context, *ai.RunContext[deps], hookedArgs,
	) (string, error) {
		t.Fatal("tool executed after validation requested external execution")
		return "", nil
	})
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || paused.Deferred() == nil || len(paused.Deferred().Calls) != 1 ||
		paused.Deferred().Metadata["call"]["queue"] != "validation" {
		t.Fatalf("unexpected validation external pause=%+v err=%v", paused, err)
	}
}

func TestBeforeToolExecutionHookCanRequestExternalExecution(t *testing.T) {
	hook := ai.BeforeToolExecutionFunc(func(
		_ context.Context, _ *ai.RunInfo, hook ai.ToolHookContext, args any,
	) (any, error) {
		if hook.Approved {
			return args, nil
		}
		request := ai.RequestExternalToolExecution(map[string]any{"queue": "remote"})
		return &request, nil
	})
	executed := false
	agent := ai.NewAgent[deps, string](deferredHookModel(t, func(part ai.ToolReturnPart) {
		if part.Content != "remote result" {
			t.Fatalf("unexpected external result: %+v", part)
		}
	}), ai.WithCapabilities(hook))
	ai.AddTool(agent, "work", func(
		context.Context, *ai.RunContext[deps], hookedArgs,
	) (string, error) {
		executed = true
		return "local", nil
	})
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || paused.Deferred() == nil || len(paused.Deferred().Calls) != 1 ||
		paused.Deferred().Metadata["call"]["queue"] != "remote" || executed {
		t.Fatalf("unexpected external pause=%+v executed=%v err=%v", paused, executed, err)
	}
	result, err := agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Calls: map[string]any{"call": "remote result"}}),
	)
	if err != nil || result.Output != "done" || executed {
		t.Fatalf("unexpected external resume=%+v executed=%v err=%v", result, executed, err)
	}
}

func TestAfterToolExecutionHookCanDeferAfterSideEffect(t *testing.T) {
	executions := 0
	hook := ai.AfterToolExecutionFunc(func(
		_ context.Context, _ *ai.RunInfo, hook ai.ToolHookContext, _ any, result any,
	) (any, error) {
		if !hook.Approved {
			return ai.RequestToolApproval(map[string]any{"late": true}), nil
		}
		return result, nil
	})
	agent := ai.NewAgent[deps, string](deferredHookModel(t, nil), ai.WithCapabilities(hook))
	ai.AddTool(agent, "work", func(
		context.Context, *ai.RunContext[deps], hookedArgs,
	) (int, error) {
		executions++
		return executions, nil
	})
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || paused.Deferred() == nil || executions != 1 {
		t.Fatalf("unexpected late approval pause=%+v executions=%d err=%v", paused, executions, err)
	}
	result, err := agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{
			Approvals: map[string]ai.ToolApproval{"call": ai.ApproveTool()},
		}),
	)
	if err != nil || result.Output != "done" || executions != 2 {
		t.Fatalf("unexpected late approval resume=%+v executions=%d err=%v", result, executions, err)
	}
}

func TestToolHookDeferralRejectsUnmarshalableArguments(t *testing.T) {
	invalidValidation := ai.AfterToolValidationFunc(func(
		context.Context, *ai.RunInfo, ai.ToolHookContext, any,
	) (any, error) {
		return func() {}, nil
	})
	deferValidation := ai.AfterToolValidationFunc(func(
		context.Context, *ai.RunInfo, ai.ToolHookContext, any,
	) (any, error) {
		return ai.RequestToolApproval(nil), nil
	})
	invalidExecution := ai.BeforeToolExecutionFunc(func(
		context.Context, *ai.RunInfo, ai.ToolHookContext, any,
	) (any, error) {
		return func() {}, nil
	})
	deferExecution := ai.BeforeToolExecutionFunc(func(
		context.Context, *ai.RunInfo, ai.ToolHookContext, any,
	) (any, error) {
		return ai.RequestToolApproval(nil), nil
	})
	for name, capabilities := range map[string][]ai.Capability{
		"validation": {deferValidation, invalidValidation},
		"execution":  {invalidExecution, deferExecution},
	} {
		t.Run(name, func(t *testing.T) {
			agent := ai.NewAgent[deps, string](
				deferredHookModel(t, nil), ai.WithCapabilities(capabilities...),
			)
			ai.AddTool(agent, "work", func(
				context.Context, *ai.RunContext[deps], hookedArgs,
			) (int, error) {
				return 1, nil
			})
			_, err := agent.Run(t.Context(), "go", deps{})
			if err == nil || !strings.Contains(err.Error(), "marshal") {
				t.Fatalf("unexpected deferred argument error: %v", err)
			}
		})
	}
}

func TestToolValidationErrorHookCannotDefer(t *testing.T) {
	hook := ai.ToolValidationErrorFunc(func(
		context.Context, *ai.RunInfo, ai.ToolHookContext, json.RawMessage, error,
	) (any, error) {
		return ai.RequestExternalToolExecution(nil), nil
	})
	agent := ai.NewAgent[deps, string](
		alwaysToolModel(json.RawMessage(`{"value":"bad"}`)), ai.WithCapabilities(hook),
	)
	ai.AddTool(agent, "work", func(
		context.Context, *ai.RunContext[deps], hookedArgs,
	) (int, error) {
		return 0, nil
	})
	_, err := agent.Run(t.Context(), "go", deps{})
	if err == nil || !strings.Contains(err.Error(), "validation error hooks cannot defer") {
		t.Fatalf("unexpected validation error deferral result: %v", err)
	}
}
