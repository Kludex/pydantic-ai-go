package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestDeferredHandlerResolvesCallsInline(t *testing.T) {
	request := 0
	executed := false
	handlerCalls := 0
	var capturedEvents []ai.StreamEvent
	handler := ai.DeferredToolHandlerFunc(func(
		_ context.Context, info *ai.RunInfo, requests ai.DeferredToolRequests,
	) (*ai.DeferredToolResults, error) {
		handlerCalls++
		if info.RunID == "" || len(requests.Approvals) != 1 || len(requests.Calls) != 1 {
			t.Fatalf("unexpected handler input: info=%+v requests=%+v", info, requests)
		}
		requests.Approvals[0].Args[0] = 'X'
		return &ai.DeferredToolResults{
			Approvals: map[string]ai.ToolApproval{"approval": ai.ApproveTool()},
			Calls:     map[string]any{"external": "remote"},
			Metadata:  map[string]map[string]any{"approval": {"source": "handler"}},
		}, nil
	})
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "approval", ToolCallID: "approval", Args: json.RawMessage(`{}`)},
				ai.ToolCallPart{ToolName: "external", ToolCallID: "external", Args: json.RawMessage(`{}`)},
			}}, nil
		}
		last := messages[len(messages)-1].(ai.ModelRequest)
		if len(last.Parts) != 2 || last.Parts[0].(ai.ToolReturnPart).Content != "approved" ||
			last.Parts[1].(ai.ToolReturnPart).Content != "remote" {
			t.Fatalf("unexpected inline results: %+v", last.Parts)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(handler))
	ai.AddTool(agent, "approval", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (string, error) {
		executed = true
		if !rc.ToolCallApproved || rc.ToolCallMetadata["source"] != "handler" {
			t.Fatalf("missing inline approval context: %+v", rc)
		}
		return "approved", nil
	}, ai.WithApprovalRequired())
	ai.AddExternalTool[deps, string, struct{}, string](agent, "external")
	stream := agent.RunStream(t.Context(), "go", deps{})
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		capturedEvents = append(capturedEvents, event)
	}
	if stream.Result() == nil || stream.Result().Output != "done" || stream.Result().Deferred() != nil ||
		!executed || handlerCalls != 1 || request != 2 {
		t.Fatalf("unexpected inline result=%+v executed=%v handlers=%d requests=%d", stream.Result(), executed, handlerCalls, request)
	}
	var functionCalls, functionResults, requestEvents, resultEvents int
	for _, event := range capturedEvents {
		switch event.(type) {
		case ai.FunctionToolCallEvent:
			functionCalls++
		case ai.FunctionToolResultEvent:
			functionResults++
		case ai.DeferredToolRequestsEvent:
			requestEvents++
		case ai.DeferredToolResultsEvent:
			resultEvents++
		}
	}
	if functionCalls != 2 || functionResults != 2 || requestEvents != 1 || resultEvents != 1 {
		t.Fatalf("unexpected inline events: calls=%d results=%d requests=%d handled=%d events=%+v", functionCalls, functionResults, requestEvents, resultEvents, capturedEvents)
	}
}

func TestDeferredHandlersResolveRedeferredCallInline(t *testing.T) {
	approvalHandler := ai.DeferredToolHandlerFunc(func(
		_ context.Context, _ *ai.RunInfo, requests ai.DeferredToolRequests,
	) (*ai.DeferredToolResults, error) {
		if len(requests.Approvals) != 1 || len(requests.Calls) != 0 {
			t.Fatalf("approval handler received unexpected requests: %+v", requests)
		}
		return &ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{"work": ai.ApproveTool()}}, nil
	})
	externalHandler := ai.DeferredToolHandlerFunc(func(
		_ context.Context, _ *ai.RunInfo, requests ai.DeferredToolRequests,
	) (*ai.DeferredToolResults, error) {
		if len(requests.Approvals) != 0 || len(requests.Calls) != 1 ||
			requests.Metadata["work"]["queue"] != "remote" {
			t.Fatalf("external handler did not receive re-deferred call: %+v", requests)
		}
		return &ai.DeferredToolResults{Calls: map[string]any{"work": "remote result"}}, nil
	})
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "work", Args: json.RawMessage(`{}`),
			}}}, nil
		}
		latest := messages[len(messages)-1].(ai.ModelRequest)
		if latest.Parts[0].(ai.ToolReturnPart).Content != "remote result" {
			t.Fatalf("unexpected re-deferred result: %+v", latest.Parts)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(approvalHandler, externalHandler))
	ai.AddTool(agent, "work", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (any, error) {
		if !rc.ToolCallApproved {
			t.Fatal("approval was not propagated")
		}
		return ai.RequestExternalToolExecution(map[string]any{"queue": "remote"}), nil
	}, ai.WithApprovalRequired(), ai.WithDynamicExternalExecution())
	stream := agent.RunStream(t.Context(), "go", deps{})
	var requestEvents, resultEvents int
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		switch event.(type) {
		case ai.DeferredToolRequestsEvent:
			requestEvents++
		case ai.DeferredToolResultsEvent:
			resultEvents++
		}
	}
	if stream.Result() == nil || stream.Result().Output != "done" || requestEvents != 1 || resultEvents != 2 {
		t.Fatalf("unexpected re-deferred inline result=%+v request events=%d result events=%d", stream.Result(), requestEvents, resultEvents)
	}
}

func TestDeferredHandlersComposeAndBubbleRemainingCalls(t *testing.T) {
	var seen []string
	empty := ai.DeferredToolHandlerFunc(func(
		_ context.Context, _ *ai.RunInfo, requests ai.DeferredToolRequests,
	) (*ai.DeferredToolResults, error) {
		seen = append(seen, "empty")
		return &ai.DeferredToolResults{}, nil
	})
	approval := ai.DeferredToolHandlerFunc(func(
		_ context.Context, _ *ai.RunInfo, requests ai.DeferredToolRequests,
	) (*ai.DeferredToolResults, error) {
		seen = append(seen, "approval")
		if len(requests.Approvals) != 1 || len(requests.Calls) != 1 {
			t.Fatalf("first resolving handler did not receive full remainder: %+v", requests)
		}
		return &ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{
			"approval": ai.DenyTool("not allowed"),
		}}, nil
	})
	decline := ai.DeferredToolHandlerFunc(func(
		_ context.Context, _ *ai.RunInfo, requests ai.DeferredToolRequests,
	) (*ai.DeferredToolResults, error) {
		seen = append(seen, "decline")
		if len(requests.Approvals) != 0 || len(requests.Calls) != 1 {
			t.Fatalf("handler received resolved calls: %+v", requests)
		}
		return nil, nil
	})
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "approval", ToolCallID: "approval", Args: json.RawMessage(`{}`)},
			ai.ToolCallPart{ToolName: "external", ToolCallID: "external", Args: json.RawMessage(`{}`)},
		}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(empty, approval, decline))
	ai.AddSimpleTool(agent, "approval", func(context.Context, struct{}) (string, error) {
		return "unreachable", nil
	}, ai.WithApprovalRequired())
	ai.AddExternalTool[deps, string, struct{}, string](agent, "external")
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	pending := result.Deferred()
	if pending == nil || len(pending.Calls) != 1 || len(pending.Approvals) != 0 ||
		!slices.Equal(seen, []string{"empty", "approval", "decline"}) {
		t.Fatalf("unexpected partial handling: pending=%+v seen=%v", pending, seen)
	}
	last := result.Messages()[len(result.Messages())-1].(ai.ModelRequest)
	if len(last.Parts) != 1 || last.Parts[0].(ai.ToolReturnPart).Outcome != ai.ToolReturnOutcomeDenied {
		t.Fatalf("resolved denial missing from history: %+v", last)
	}
}

func TestDeferredHandlerCanResolveExternalAndLeaveApproval(t *testing.T) {
	handler := ai.DeferredToolHandlerFunc(func(
		context.Context, *ai.RunInfo, ai.DeferredToolRequests,
	) (*ai.DeferredToolResults, error) {
		return &ai.DeferredToolResults{Calls: map[string]any{"external": "done"}}, nil
	})
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "approval", ToolCallID: "approval", Args: json.RawMessage(`{}`)},
			ai.ToolCallPart{ToolName: "external", ToolCallID: "external", Args: json.RawMessage(`{}`)},
		}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(handler))
	ai.AddSimpleTool(
		agent, "approval", func(context.Context, struct{}) (string, error) { return "approved", nil },
		ai.WithApprovalRequired(), ai.WithApprovalMetadata(map[string]any{"reason": "sensitive"}),
	)
	ai.AddExternalTool[deps, string, struct{}, string](agent, "external")
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Deferred() == nil || len(result.Deferred().Approvals) != 1 || len(result.Deferred().Calls) != 0 ||
		result.Deferred().Metadata["approval"]["reason"] != "sensitive" {
		t.Fatalf("unexpected external-only handling: result=%+v err=%v", result, err)
	}
}

func TestDeferredHandlerErrors(t *testing.T) {
	for name, handler := range map[string]ai.DeferredToolHandlerFunc{
		"handler": func(context.Context, *ai.RunInfo, ai.DeferredToolRequests) (*ai.DeferredToolResults, error) {
			return nil, errors.New("resolver offline")
		},
		"result": func(context.Context, *ai.RunInfo, ai.DeferredToolRequests) (*ai.DeferredToolResults, error) {
			return &ai.DeferredToolResults{Calls: map[string]any{"unknown": "bad"}}, nil
		},
	} {
		t.Run(name, func(t *testing.T) {
			model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
					ToolName: "external", ToolCallID: "external", Args: json.RawMessage(`{}`),
				}}}, nil
			})
			agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(handler))
			ai.AddExternalTool[deps, string, struct{}, string](agent, "external")
			_, err := agent.Run(t.Context(), "go", deps{})
			if err == nil || name == "handler" && !strings.Contains(err.Error(), "resolver offline") ||
				name == "result" && !strings.Contains(err.Error(), "does not match") {
				t.Fatalf("unexpected handler error: %v", err)
			}
		})
	}
}

func TestDeferredResultsEventCanStopStream(t *testing.T) {
	handler := ai.DeferredToolHandlerFunc(func(
		context.Context, *ai.RunInfo, ai.DeferredToolRequests,
	) (*ai.DeferredToolResults, error) {
		return &ai.DeferredToolResults{Calls: map[string]any{"external": "done"}}, nil
	})
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "external", ToolCallID: "external", Args: json.RawMessage(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(handler))
	ai.AddExternalTool[deps, string, struct{}, string](agent, "external")
	stream := agent.RunStream(t.Context(), "go", deps{})
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(ai.DeferredToolResultsEvent); ok {
			break
		}
	}
	if stream.Result() != nil {
		t.Fatalf("stopped inline stream should have no result: %+v", stream.Result())
	}
}
