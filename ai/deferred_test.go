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

type deferredArgs struct {
	Value string `json:"value"`
}

type deferredReply struct {
	Value string `json:"value"`
}

type unsupportedApproval struct{}

func (unsupportedApproval) ToolApprovalKind() string { return "unsupported" }

func TestDeferredApprovalAndExternalCallResume(t *testing.T) {
	request := 0
	ordinaryCalls := 0
	approvedCalls := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if len(params.Tools) != 3 {
			t.Fatalf("deferred tools should use ordinary provider definitions: %+v", params.Tools)
		}
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "approve", ToolCallID: "approval-1", Args: json.RawMessage(`{"value":"original"}`)},
				ai.ToolCallPart{ToolName: "ordinary", ToolCallID: "ordinary-1", Args: json.RawMessage(`{"value":"now"}`)},
				ai.ToolCallPart{ToolName: "external", ToolCallID: "external-1", Args: json.RawMessage(`{"value":"later"}`)},
			}}, nil
		}
		last := messages[len(messages)-1].(ai.ModelRequest)
		if len(last.Parts) != 4 {
			t.Fatalf("expected two results, rich content, and prompt: %+v", last.Parts)
		}
		approved := last.Parts[0].(ai.ToolReturnPart)
		external := last.Parts[1].(ai.ToolReturnPart)
		content := last.Parts[2].(ai.UserPromptPart)
		prompt := last.Parts[3].(ai.UserPromptPart)
		if approved.Content != "approved: replacement" || approved.Outcome != ai.ToolReturnOutcomeSuccess ||
			external.Content != "remote result" || external.Metadata["source"] != "worker" ||
			content.Contents[0].(ai.TextContent).Text != "remote details" || prompt.Content != "continue" {
			t.Fatalf("unexpected resumed request: %+v", last.Parts)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddTool(agent, "approve", func(
		_ context.Context, rc *ai.RunContext[deps], args deferredArgs,
	) (string, error) {
		approvedCalls++
		if !rc.ToolCallApproved || rc.ToolCallMetadata["reviewer"] != "Ada" {
			t.Fatalf("approval context missing: %+v", rc)
		}
		return "approved: " + args.Value, nil
	}, ai.WithApprovalRequired())
	ai.AddTool(agent, "ordinary", func(
		_ context.Context, _ *ai.RunContext[deps], args deferredArgs,
	) (string, error) {
		ordinaryCalls++
		return "ordinary: " + args.Value, nil
	})
	ai.AddExternalTool[deps, string, deferredArgs, deferredReply](agent, "external")

	paused, err := agent.Run(t.Context(), "start", deps{})
	if err != nil {
		t.Fatal(err)
	}
	pending := paused.Deferred()
	if pending == nil || len(pending.Approvals) != 1 || len(pending.Calls) != 1 ||
		pending.Approvals[0].ToolCallID != "approval-1" || pending.Calls[0].ToolCallID != "external-1" ||
		ordinaryCalls != 1 || approvedCalls != 0 || paused.Usage().ToolCalls != 1 {
		t.Fatalf("unexpected deferred result: pending=%+v ordinary=%d approved=%d usage=%+v", pending, ordinaryCalls, approvedCalls, paused.Usage())
	}
	pending.Approvals[0].Args[0] = 'X'
	if paused.Deferred().Approvals[0].Args[0] == 'X' {
		t.Fatal("Deferred returned shared arguments")
	}
	override, err := ai.ApproveToolWithArgs(deferredArgs{Value: "replacement"})
	if err != nil {
		t.Fatal(err)
	}
	results := ai.DeferredToolResults{
		Approvals: map[string]ai.ToolApproval{"approval-1": &override},
		Calls: map[string]any{"external-1": ai.ToolReturn{
			ReturnValue: "remote result",
			Content:     []ai.UserContent{ai.TextContent{Text: "remote details"}},
			Metadata:    map[string]any{"source": "worker"},
		}},
		Metadata: map[string]map[string]any{"approval-1": {"reviewer": "Ada"}},
	}
	resumed, err := agent.Run(
		t.Context(), "continue", deps{},
		ai.WithMessageHistory(paused.Messages()), ai.WithDeferredToolResults(results),
	)
	if err != nil {
		t.Fatal(err)
	}
	results.Metadata["approval-1"]["reviewer"] = "changed"
	if resumed.Output != "done" || resumed.Deferred() != nil || approvedCalls != 1 ||
		resumed.Usage().ToolCalls != 1 || request != 2 {
		t.Fatalf("unexpected resumed result: %+v calls=%d requests=%d", resumed, approvedCalls, request)
	}
}

func TestDeferredApprovalDenialDoesNotExecute(t *testing.T) {
	request := 0
	executed := false
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "danger", ToolCallID: "danger-1", Args: json.RawMessage(`{}`),
			}}}, nil
		}
		last := messages[len(messages)-1].(ai.ModelRequest)
		part := last.Parts[0].(ai.ToolReturnPart)
		if part.Content != "The tool call was denied." || part.Outcome != ai.ToolReturnOutcomeDenied {
			t.Fatalf("unexpected denial: %+v", part)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "safe"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "danger", func(context.Context, struct{}) (string, error) {
		executed = true
		return "bad", nil
	}, ai.WithApprovalRequired())
	paused, err := agent.Run(t.Context(), "start", deps{})
	if err != nil {
		t.Fatal(err)
	}
	denied := ai.DenyTool("")
	resumed, err := agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{
			"danger-1": &denied,
		}}),
	)
	if err != nil || resumed.Output != "safe" || executed || resumed.Usage().ToolCalls != 0 {
		t.Fatalf("unexpected denied resume: result=%+v executed=%v err=%v", resumed, executed, err)
	}
}

func TestExternalResultKinds(t *testing.T) {
	for name, supplied := range map[string]struct {
		result      any
		wantPart    any
		wantContent string
	}{
		"failed":      {result: ai.ToolFailedf("worker failed"), wantPart: ai.ToolReturnPart{}, wantContent: "worker failed"},
		"retry error": {result: ai.Retryf("try another worker"), wantPart: ai.RetryPromptPart{}, wantContent: "try another worker"},
		"retry part":  {result: ai.RetryPromptPart{Content: "retry directly"}, wantPart: ai.RetryPromptPart{}, wantContent: "retry directly"},
		"return part": {result: ai.ToolReturnPart{Content: "direct"}, wantPart: ai.ToolReturnPart{}, wantContent: "direct"},
	} {
		t.Run(name, func(t *testing.T) {
			request := 0
			model := fakes.NewFunctionModel(func(
				_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				request++
				if request == 1 {
					return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
						ToolName: "job", ToolCallID: "job-1", Args: json.RawMessage(`{}`),
					}}}, nil
				}
				last := messages[len(messages)-1].(ai.ModelRequest)
				switch part := last.Parts[0].(type) {
				case ai.ToolReturnPart:
					if _, ok := supplied.wantPart.(ai.ToolReturnPart); !ok || part.Content != supplied.wantContent ||
						part.ToolName != "job" || part.ToolCallID != "job-1" {
						t.Fatalf("unexpected return: %+v", part)
					}
				case ai.RetryPromptPart:
					if _, ok := supplied.wantPart.(ai.RetryPromptPart); !ok || part.Content != supplied.wantContent ||
						part.ToolName != "job" || part.ToolCallID != "job-1" {
						t.Fatalf("unexpected retry: %+v", part)
					}
				default:
					t.Fatalf("unexpected part %T", part)
				}
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
			})
			agent := ai.NewAgent[deps, string](model)
			agent.AddTool(ai.NewRawExternalTool[deps](ai.ToolDefinition{
				Name: "job", Schema: map[string]any{"type": "object"},
			}, ai.WithDescription("External job")))
			paused, err := agent.Run(t.Context(), "start", deps{})
			if err != nil {
				t.Fatal(err)
			}
			result, err := agent.Run(
				t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
				ai.WithDeferredToolResults(ai.DeferredToolResults{Calls: map[string]any{"job-1": supplied.result}}),
			)
			if err != nil || result.Output != "done" {
				t.Fatalf("unexpected result=%+v err=%v", result, err)
			}
		})
	}
}

func TestDeferredOutputPreemptsPendingCalls(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "external", ToolCallID: "external-1", Args: json.RawMessage(`{}`)},
			ai.ToolCallPart{ToolName: "final_result", ToolCallID: "final-1", Args: json.RawMessage(`{"value":"done"}`)},
		}}, nil
	})
	agent := ai.NewAgent[deps, deferredReply](model, ai.WithEndStrategy(ai.EndStrategyGraceful))
	ai.AddExternalTool[deps, deferredReply, struct{}, string](agent, "external")
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output.Value != "done" || result.Deferred() != nil {
		t.Fatalf("unexpected output-preemption result=%+v err=%v", result, err)
	}
	last := result.Messages()[len(result.Messages())-1].(ai.ModelRequest)
	if part := last.Parts[0].(ai.ToolReturnPart); part.Content != "Tool not executed - a final result was already processed." {
		t.Fatalf("unexpected deferred stub: %+v", part)
	}
}

func TestDeferredStreamEvent(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "external", ToolCallID: "external-1", Args: json.RawMessage(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddExternalTool[deps, string, struct{}, string](agent, "external")
	stream := agent.RunStream(t.Context(), "go", deps{})
	var events []ai.StreamEvent
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, event)
	}
	if stream.Result() == nil || stream.Result().Deferred() == nil || len(events) < 2 {
		t.Fatalf("missing deferred stream result: events=%+v result=%+v", events, stream.Result())
	}
	deferred, ok := events[len(events)-1].(ai.DeferredToolRequestsEvent)
	if !ok || deferred.Requests.Calls[0].ToolCallID != "external-1" {
		t.Fatalf("unexpected final event: %+v", events[len(events)-1])
	}
}

func TestDeferredStreamCanStopAtPendingEvent(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "external", ToolCallID: "external-1", Args: json.RawMessage(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddExternalTool[deps, string, struct{}, string](agent, "external")
	stream := agent.RunStream(t.Context(), "go", deps{})
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(ai.DeferredToolRequestsEvent); ok {
			break
		}
	}
	if stream.Result() != nil {
		t.Fatalf("stopped stream should not expose a result: %+v", stream.Result())
	}
}

func TestDeferredResultValidation(t *testing.T) {
	call := ai.ToolCallPart{ToolName: "approval", ToolCallID: "call-1", Args: json.RawMessage(`{}`)}
	newAgent := func(parts []ai.ResponsePart, opts ...ai.ToolOption) (*ai.Agent[deps, string], *ai.RunResult[string]) {
		t.Helper()
		model := fakes.NewFunctionModel(func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{Parts: parts}, nil
		})
		agent := ai.NewAgent[deps, string](model)
		ai.AddSimpleTool(agent, "approval", func(context.Context, struct{}) (string, error) { return "ok", nil }, opts...)
		paused, err := agent.Run(t.Context(), "start", deps{})
		if err != nil {
			t.Fatal(err)
		}
		return agent, paused
	}
	cases := map[string]struct {
		parts   []ai.ResponsePart
		opts    []ai.ToolOption
		results ai.DeferredToolResults
		want    string
	}{
		"missing":             {[]ai.ResponsePart{call}, []ai.ToolOption{ai.WithApprovalRequired()}, ai.DeferredToolResults{}, "missing approval"},
		"unknown":             {[]ai.ResponsePart{call}, []ai.ToolOption{ai.WithApprovalRequired()}, ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{"other": ai.ApproveTool()}}, "does not match"},
		"unknown call result": {[]ai.ResponsePart{call}, []ai.ToolOption{ai.WithApprovalRequired()}, ai.DeferredToolResults{Calls: map[string]any{"other": "result"}}, "does not match"},
		"overlap":             {[]ai.ResponsePart{call}, []ai.ToolOption{ai.WithApprovalRequired()}, ai.DeferredToolResults{Calls: map[string]any{"call-1": "x"}, Approvals: map[string]ai.ToolApproval{"call-1": ai.ApproveTool()}}, "appears in calls and approvals"},
		"wrong kind":          {[]ai.ResponsePart{call}, []ai.ToolOption{ai.WithApprovalRequired()}, ai.DeferredToolResults{Calls: map[string]any{"call-1": "x"}}, "received an external result"},
		"unsupported":         {[]ai.ResponsePart{call}, []ai.ToolOption{ai.WithApprovalRequired()}, ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{"call-1": unsupportedApproval{}}}, "unsupported approval"},
		"external missing":    {[]ai.ResponsePart{call}, []ai.ToolOption{ai.WithExternalExecution()}, ai.DeferredToolResults{}, "missing result"},
		"external approval":   {[]ai.ResponsePart{call}, []ai.ToolOption{ai.WithExternalExecution()}, ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{"call-1": ai.ApproveTool()}}, "received an approval"},
		"metadata":            {[]ai.ResponsePart{call}, []ai.ToolOption{ai.WithApprovalRequired()}, ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{"call-1": ai.ApproveTool()}, Metadata: map[string]map[string]any{"other": {}}}, "metadata"},
		"nil approved":        {[]ai.ResponsePart{call}, []ai.ToolOption{ai.WithApprovalRequired()}, ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{"call-1": (*ai.ToolApproved)(nil)}}, "must not be nil"},
		"nil denied":          {[]ai.ResponsePart{call}, []ai.ToolOption{ai.WithApprovalRequired()}, ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{"call-1": (*ai.ToolDenied)(nil)}}, "must not be nil"},
		"nil interface":       {[]ai.ResponsePart{call}, []ai.ToolOption{ai.WithApprovalRequired()}, ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{"call-1": nil}}, "must not be nil"},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			agent, paused := newAgent(test.parts, test.opts...)
			_, err := agent.Run(
				t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
				ai.WithDeferredToolResults(test.results),
			)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("expected %q, got %v", test.want, err)
			}
		})
	}
}

func TestDeferredCallIdentityAndConfigurationErrors(t *testing.T) {
	for name, calls := range map[string][]ai.ResponsePart{
		"empty": {ai.ToolCallPart{ToolName: "external", Args: json.RawMessage(`{}`)}},
		"duplicate": {
			ai.ToolCallPart{ToolName: "external", ToolCallID: "same", Args: json.RawMessage(`{}`)},
			ai.ToolCallPart{ToolName: "external", ToolCallID: "same", Args: json.RawMessage(`{}`)},
		},
	} {
		t.Run(name, func(t *testing.T) {
			model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
				return &ai.ModelResponse{Parts: calls}, nil
			})
			agent := ai.NewAgent[deps, string](model)
			ai.AddExternalTool[deps, string, struct{}, string](agent, "external")
			if _, err := agent.Run(t.Context(), "go", deps{}); err == nil || !strings.Contains(err.Error(), name) {
				t.Fatalf("expected %s error, got %v", name, err)
			}
		})
	}

	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "unused"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "invalid", func(context.Context, struct{}) (string, error) { return "", nil },
		ai.WithApprovalRequired(), ai.WithExternalExecution())
	if _, err := agent.Run(t.Context(), "go", deps{}); err == nil || !strings.Contains(err.Error(), "cannot require approval") {
		t.Fatalf("expected incompatible deferral error, got %v", err)
	}
}

func TestDeferredApprovalOverrideValidationRetries(t *testing.T) {
	request := 0
	executed := false
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "approval", ToolCallID: "approval-1", Args: json.RawMessage(`{"value":"valid"}`),
			}}}, nil
		}
		last := messages[len(messages)-1].(ai.ModelRequest)
		if retry, ok := last.Parts[0].(ai.RetryPromptPart); !ok || len(retry.Errors) == 0 {
			t.Fatalf("expected override validation retry: %+v", last.Parts)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "corrected"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "approval", func(context.Context, deferredArgs) (string, error) {
		executed = true
		return "unexpected", nil
	}, ai.WithApprovalRequired())
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	result, err := agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{
			"approval-1": ai.ToolApproved{OverrideArgs: json.RawMessage(`{"value":1}`)},
		}}),
	)
	if err != nil || result.Output != "corrected" || executed {
		t.Fatalf("unexpected override result=%+v executed=%v err=%v", result, executed, err)
	}
}

func TestApproveToolWithArgsError(t *testing.T) {
	if ai.ApproveTool().ToolApprovalKind() != "approved" || ai.DenyTool("no").ToolApprovalKind() != "denied" {
		t.Fatal("unexpected approval kinds")
	}
	if _, err := ai.ApproveToolWithArgs(make(chan int)); err == nil || !strings.Contains(err.Error(), "marshal") {
		t.Fatalf("expected marshal error, got %v", err)
	}
}

func TestDeferredResultOptionsDetachEverySupportedValue(t *testing.T) {
	approved := &ai.ToolApproved{OverrideArgs: json.RawMessage(`{"value":"approved"}`)}
	denied := &ai.ToolDenied{Message: "denied"}
	var nilApproved *ai.ToolApproved
	var nilDenied *ai.ToolDenied
	var nilReturn *ai.ToolReturn
	values := ai.DeferredToolResults{
		Approvals: map[string]ai.ToolApproval{
			"approved": approved, "denied": denied, "nil-approved": nilApproved, "nil-denied": nilDenied,
			"nil": nil,
		},
		Calls: map[string]any{
			"return-pointer": &ai.ToolReturn{ReturnValue: map[string]any{"value": "x"}},
			"nil-return":     nilReturn,
			"retry": ai.RetryPromptPart{Errors: []ai.ValidationError{{
				Location: []any{"value"}, Input: map[string]any{"bad": true}, Context: map[string]any{"rule": "x"},
			}}},
			"part":  ai.ToolReturnPart{Metadata: map[string]any{"source": "worker"}},
			"raw":   json.RawMessage(`{"value":"raw"}`),
			"plain": map[string]any{"value": []any{"plain"}},
		},
	}
	_ = ai.WithDeferredToolResults(values)
	approved.OverrideArgs[0] = 'X'
	denied.Message = "changed"
}

func TestDeferredResultsWithoutPendingCallsAreAllowed(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	result, err := agent.Run(t.Context(), "go", deps{}, ai.WithDeferredToolResults(ai.DeferredToolResults{}))
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected empty deferred results: result=%+v err=%v", result, err)
	}
}

func TestDeferredHistoryKeepsCompletedSibling(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "ordinary", ToolCallID: "ordinary", Args: json.RawMessage(`{}`)},
			ai.ToolCallPart{ToolName: "external", ToolCallID: "external", Args: json.RawMessage(`{}`)},
		}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "ordinary", func(context.Context, struct{}) (string, error) { return "complete", nil })
	ai.AddExternalTool[deps, string, struct{}, string](agent, "external")
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if got := result.NewMessages(); len(got) != 3 {
		t.Fatalf("expected prompt, response, and completed sibling return: %+v", got)
	}
	last := result.NewMessages()[2].(ai.ModelRequest)
	if len(last.Parts) != 1 || last.Parts[0].(ai.ToolReturnPart).ToolCallID != "ordinary" {
		t.Fatalf("unexpected completed sibling history: %+v", last)
	}
	if !slices.Equal([]string{result.Deferred().Calls[0].ToolCallID}, []string{"external"}) {
		t.Fatalf("unexpected pending call: %+v", result.Deferred())
	}
}

func TestDeferredPlainErrorIsReturned(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		request++
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "external", ToolCallID: "external", Args: json.RawMessage(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddExternalTool[deps, string, struct{}, string](agent, "external")
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Calls: map[string]any{"external": errors.New("worker offline")}}),
	)
	if err == nil || !strings.Contains(err.Error(), "worker offline") || request != 1 {
		t.Fatalf("expected worker error before another model request, got %v", err)
	}
}
