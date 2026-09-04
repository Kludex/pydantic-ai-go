package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestDynamicApprovalValuePausesAndResumes(t *testing.T) {
	request := 0
	var executed atomic.Int64
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "update", ToolCallID: "safe", Args: json.RawMessage(`{"value":"safe"}`)},
				ai.ToolCallPart{ToolName: "update", ToolCallID: "protected", Args: json.RawMessage(`{"value":"protected"}`)},
			}}, nil
		}
		last := messages[len(messages)-1].(ai.ModelRequest)
		if len(last.Parts) != 2 || last.Parts[0].(ai.ToolReturnPart).Content != "updated protected" {
			t.Fatalf("unexpected approved dynamic result: %+v", last.Parts)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddTool(agent, "update", func(
		_ context.Context, rc *ai.RunContext[deps], args deferredArgs,
	) (any, error) {
		if args.Value == "protected" && !rc.ToolCallApproved {
			return ai.RequestToolApproval(map[string]any{"reason": "protected"}), nil
		}
		executed.Add(1)
		return "updated " + args.Value, nil
	}, ai.WithDynamicApproval())
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	pending := paused.Deferred()
	if pending == nil || len(pending.Approvals) != 1 || pending.Approvals[0].ToolCallID != "protected" ||
		pending.Metadata["protected"]["reason"] != "protected" || executed.Load() != 1 {
		t.Fatalf("unexpected dynamic approval: pending=%+v executed=%d", pending, executed.Load())
	}
	result, err := agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{
			"protected": ai.ApproveTool(),
		}}),
	)
	if err != nil || result.Output != "done" || executed.Load() != 2 {
		t.Fatalf("unexpected dynamic resume: result=%+v executed=%d err=%v", result, executed.Load(), err)
	}
}

func TestDynamicApprovalRequiresDeclaration(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "update", ToolCallID: "update", Args: json.RawMessage(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "update", func(context.Context, struct{}) (ai.ToolApprovalRequest, error) {
		return ai.RequestToolApproval(nil), nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err == nil || !strings.Contains(err.Error(), "without WithDynamicApproval") {
		t.Fatalf("expected undeclared dynamic approval error, got %v", err)
	}
}

func TestApprovedDynamicToolCanRequestApprovalAgain(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "update", ToolCallID: "update", Args: json.RawMessage(`{}`)},
				ai.ToolCallPart{ToolName: "external", ToolCallID: "external", Args: json.RawMessage(`{}`)},
			}}, nil
		}
		latest := messages[len(messages)-1].(ai.ModelRequest)
		if len(latest.Parts) != 2 || latest.Parts[0].(ai.ToolReturnPart).Content != "updated" ||
			latest.Parts[1].(ai.UserPromptPart).Content != "continue twice" {
			t.Fatalf("unexpected final resumed request: %+v", latest.Parts)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	invocations := 0
	ai.AddTool(agent, "update", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (any, error) {
		invocations++
		if invocations <= 2 {
			return ai.RequestToolApproval(map[string]any{"stage": invocations}), nil
		}
		if !rc.ToolCallApproved {
			t.Fatal("final invocation was not approved")
		}
		return "updated", nil
	}, ai.WithDynamicApproval())
	ai.AddExternalTool[deps, string, struct{}, string](agent, "external")
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	pausedAgain, err := agent.Run(
		t.Context(), "continue once", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{
			Approvals: map[string]ai.ToolApproval{"update": ai.ApproveTool()},
			Calls:     map[string]any{"external": "external result"},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	pending := pausedAgain.Deferred()
	if pending == nil || len(pending.Approvals) != 1 || pending.Metadata["update"]["stage"] != 2 ||
		request != 1 || len(pausedAgain.Messages()) != len(paused.Messages())+1 {
		t.Fatalf("unexpected repeated approval pause: pending=%+v requests=%d messages=%+v", pending, request, pausedAgain.Messages())
	}
	latest := pausedAgain.Messages()[len(pausedAgain.Messages())-1].(ai.ModelRequest)
	if len(latest.Parts) != 1 || latest.Parts[0].(ai.ToolReturnPart).Content != "external result" {
		t.Fatalf("re-deferred history did not retain only completed sibling: %+v", latest.Parts)
	}
	result, err := agent.Run(
		t.Context(), "continue twice", deps{}, ai.WithMessageHistory(pausedAgain.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{
			"update": ai.ApproveTool(),
		}}),
	)
	if err != nil || result.Output != "done" || request != 2 || invocations != 3 {
		t.Fatalf("unexpected repeated approval result=%+v requests=%d invocations=%d err=%v", result, request, invocations, err)
	}
}

func TestRedeferredRequestCanStopStream(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "work", ToolCallID: "work", Args: json.RawMessage(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddTool(agent, "work", func(context.Context, *ai.RunContext[deps], struct{}) (any, error) {
		return ai.RequestToolApproval(nil), nil
	}, ai.WithDynamicApproval())
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	stream := agent.RunStream(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{
			"work": ai.ApproveTool(),
		}}),
	)
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(ai.DeferredToolRequestsEvent); ok {
			break
		}
	}
	if stream.Result() != nil {
		t.Fatalf("stopped re-deferred stream should have no result: %+v", stream.Result())
	}
}

func TestRedeferredHandlerError(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "work", ToolCallID: "work", Args: json.RawMessage(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddTool(agent, "work", func(context.Context, *ai.RunContext[deps], struct{}) (any, error) {
		return ai.RequestToolApproval(nil), nil
	}, ai.WithDynamicApproval())
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	handler := ai.DeferredToolHandlerFunc(func(
		context.Context, *ai.RunInfo, ai.DeferredToolRequests,
	) (*ai.DeferredToolResults, error) {
		return nil, errors.New("approval service offline")
	})
	_, err = agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()), ai.WithRunCapabilities(handler),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{
			"work": ai.ApproveTool(),
		}}),
	)
	if err == nil || !strings.Contains(err.Error(), "approval service offline") {
		t.Fatalf("unexpected re-deferred handler error: %v", err)
	}
}

func TestNilDynamicApprovalPointerIsOrdinaryResult(t *testing.T) {
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
		last := messages[len(messages)-1].(ai.ModelRequest)
		if last.Parts[0].(ai.ToolReturnPart).Content != (*ai.ToolApprovalRequest)(nil) {
			t.Fatalf("unexpected nil pointer result: %+v", last.Parts[0])
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "maybe", func(context.Context, struct{}) (*ai.ToolApprovalRequest, error) {
		return nil, nil
	}, ai.WithDynamicApproval())
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "done" || result.Deferred() != nil {
		t.Fatalf("unexpected nil dynamic approval: result=%+v err=%v", result, err)
	}
}

func TestDynamicApprovalConfigurationConflicts(t *testing.T) {
	for name, opts := range map[string][]ai.ToolOption{
		"static":   {ai.WithApprovalRequired(), ai.WithDynamicApproval()},
		"external": {ai.WithExternalExecution(), ai.WithDynamicApproval()},
	} {
		t.Run(name, func(t *testing.T) {
			agent := ai.NewAgent[deps, string](fakes.NewTestModel())
			ai.AddSimpleTool(agent, "invalid", func(context.Context, struct{}) (string, error) { return "", nil }, opts...)
			if _, err := agent.Run(t.Context(), "go", deps{}); err == nil || !strings.Contains(err.Error(), "cannot") {
				t.Fatalf("expected dynamic approval conflict, got %v", err)
			}
		})
	}
}

func TestDynamicDeferredResumeRequiresResultKind(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "work", ToolCallID: "work", Args: json.RawMessage(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddTool(agent, "work", func(context.Context, *ai.RunContext[deps], struct{}) (any, error) {
		return ai.RequestToolApproval(nil), nil
	}, ai.WithDynamicApproval(), ai.WithDynamicExternalExecution())
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{}),
	)
	if err == nil || !strings.Contains(err.Error(), "missing deferred result") {
		t.Fatalf("unexpected missing dynamic result error: %v", err)
	}
}

func TestDynamicApprovalToolset(t *testing.T) {
	request := 0
	checks := 0
	executed := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "work", ToolCallID: "safe", Args: json.RawMessage(`{"value":"safe"}`)},
				ai.ToolCallPart{ToolName: "work", ToolCallID: "review", Args: json.RawMessage(`{"value":"review"}`)},
			}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	tool := ai.NewSimpleTool[deps]("work", func(context.Context, deferredArgs) (string, error) {
		executed++
		return "worked", nil
	}, ai.WithDescription("Work on a value"), ai.WithSequential())
	toolset := ai.RequireApprovalToolsetWhen(ai.NewFunctionToolset(tool), func(
		_ context.Context,
		rc *ai.RunContext[deps],
		definition ai.ToolDefinition,
		raw json.RawMessage,
	) (*ai.ToolApprovalRequest, error) {
		checks++
		if rc.ToolCallApproved || definition.Description != "Work on a value" {
			t.Fatalf("unexpected approval check context: rc=%+v definition=%+v", rc, definition)
		}
		var args deferredArgs
		if err := json.Unmarshal(raw, &args); err != nil {
			return nil, err
		}
		if args.Value == "safe" {
			raw[0] = 'X'
		}
		if args.Value == "review" {
			request := ai.RequestToolApproval(map[string]any{"policy": "review"})
			return &request, nil
		}
		return nil, nil
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddToolset(toolset)
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || paused.Deferred() == nil || checks != 2 || executed != 1 ||
		paused.Deferred().Metadata["review"]["policy"] != "review" {
		t.Fatalf("unexpected toolset approval: result=%+v checks=%d executed=%d err=%v", paused, checks, executed, err)
	}
	result, err := agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{
			"review": ai.ApproveTool(),
		}}),
	)
	if err != nil || result.Output != "done" || checks != 2 || executed != 2 {
		t.Fatalf("unexpected toolset resume: result=%+v checks=%d executed=%d err=%v", result, checks, executed, err)
	}
}

func TestDynamicApprovalToolsetErrors(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected nil approval function panic")
		}
	}()
	_ = ai.RequireApprovalToolsetWhen[deps](ai.NewFunctionToolset[deps](), nil)
}

func TestDynamicApprovalToolsetCheckError(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "work", ToolCallID: "work", Args: json.RawMessage(`{}`),
		}}}, nil
	})
	tool := ai.NewSimpleTool[deps]("work", func(context.Context, struct{}) (string, error) { return "", nil })
	agent := ai.NewAgent[deps, string](model)
	agent.AddToolset(ai.RequireApprovalToolsetWhen(ai.NewFunctionToolset(tool), func(
		context.Context, *ai.RunContext[deps], ai.ToolDefinition, json.RawMessage,
	) (*ai.ToolApprovalRequest, error) {
		return nil, errors.New("policy offline")
	}))
	if _, err := agent.Run(t.Context(), "go", deps{}); err == nil || !strings.Contains(err.Error(), "policy offline") {
		t.Fatalf("expected policy error, got %v", err)
	}
}
