package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type hookedArgs struct {
	Value int `json:"value"`
}

type fullToolCapability struct {
	name string
	log  *[]string
}

func (*fullToolCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (capability *fullToolCapability) BeforeToolValidation(
	_ context.Context, _ *ai.RunInfo, hook ai.ToolHookContext, raw json.RawMessage,
) (json.RawMessage, error) {
	*capability.log = append(*capability.log, capability.name+":before-validation")
	if capability.name == "outer" {
		cloned := hook.Clone()
		cloned.Call.Args[0] = '['
		cloned.Call.ProviderDetails["source"] = "changed"
		cloned.Definition.Metadata["owner"] = "changed"
		if string(hook.Call.Args) != `{"value":1}` || hook.Call.ProviderDetails["source"] != "model" ||
			hook.Definition.Metadata["owner"] != "agent" {
			return nil, errors.New("tool hook context clone mutated its source")
		}
	}
	var args hookedArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, err
	}
	args.Value++
	return json.Marshal(args)
}

func (capability *fullToolCapability) WrapToolValidation(
	ctx context.Context,
	_ *ai.RunInfo,
	_ ai.ToolHookContext,
	raw json.RawMessage,
	next ai.ToolValidationFunc,
) (any, error) {
	*capability.log = append(*capability.log, capability.name+":validation-wrapper-before")
	var args hookedArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, err
	}
	args.Value++
	raw, err := json.Marshal(args)
	if err != nil {
		return nil, err
	}
	validated, err := next(ctx, raw)
	*capability.log = append(*capability.log, capability.name+":validation-wrapper-after")
	return validated, err
}

func (capability *fullToolCapability) AfterToolValidation(
	_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, validated any,
) (any, error) {
	*capability.log = append(*capability.log, capability.name+":after-validation")
	args := validated.(hookedArgs)
	args.Value++
	return args, nil
}

func (capability *fullToolCapability) BeforeToolExecution(
	_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, validated any,
) (any, error) {
	*capability.log = append(*capability.log, capability.name+":before-execution")
	args := validated.(hookedArgs)
	args.Value++
	return args, nil
}

func (capability *fullToolCapability) WrapToolExecution(
	ctx context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, validated any, next ai.ToolExecutionFunc,
) (any, error) {
	*capability.log = append(*capability.log, capability.name+":execution-wrapper-before")
	args := validated.(hookedArgs)
	args.Value++
	result, err := next(ctx, args)
	*capability.log = append(*capability.log, capability.name+":execution-wrapper-after")
	return result, err
}

func (capability *fullToolCapability) AfterToolExecution(
	_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, _ any, result any,
) (any, error) {
	*capability.log = append(*capability.log, capability.name+":after-execution")
	return result.(int) + 1, nil
}

func TestToolValidationAndExecutionHooksUseMiddlewareOrder(t *testing.T) {
	var log []string
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "call", Args: json.RawMessage(`{"value":1}`),
				ProviderDetails: map[string]any{"source": "model"},
			}}}, nil
		}
		part := messages[len(messages)-1].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart)
		if part.Content != 13 {
			t.Fatalf("unexpected transformed tool return: %+v", part)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	outer := &fullToolCapability{name: "outer", log: &log}
	inner := &fullToolCapability{name: "inner", log: &log}
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(outer, inner))
	ai.AddToolWithArgsValidator(
		agent,
		"work",
		func(_ context.Context, _ *ai.RunContext[deps], args hookedArgs) (int, error) {
			log = append(log, "tool")
			return args.Value, nil
		},
		func(_ context.Context, _ *ai.RunContext[deps], args hookedArgs) error {
			log = append(log, fmt.Sprintf("validator:%d", args.Value))
			return nil
		},
		ai.WithToolMetadata(map[string]any{"owner": "agent"}),
	)
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected hook result=%+v err=%v", result, err)
	}
	want := []string{
		"outer:before-validation", "inner:before-validation",
		"outer:validation-wrapper-before", "inner:validation-wrapper-before", "validator:5",
		"inner:validation-wrapper-after", "outer:validation-wrapper-after",
		"inner:after-validation", "outer:after-validation",
		"outer:before-execution", "inner:before-execution",
		"outer:execution-wrapper-before", "inner:execution-wrapper-before", "tool",
		"inner:execution-wrapper-after", "outer:execution-wrapper-after",
		"inner:after-execution", "outer:after-execution",
	}
	if !slices.Equal(log, want) {
		t.Fatalf("unexpected hook order:\n got %v\nwant %v", log, want)
	}
}

func TestToolValidationErrorHooksRecoverInsideOut(t *testing.T) {
	var log []string
	inner := ai.ToolValidationErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, _ json.RawMessage, err error,
	) (any, error) {
		log = append(log, "inner: "+err.Error())
		return nil, fmt.Errorf("inner: %w", err)
	})
	outer := ai.ToolValidationErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, raw json.RawMessage, err error,
	) (any, error) {
		log = append(log, "outer: "+err.Error())
		if string(raw) != `{"value":"bad"}` {
			return nil, fmt.Errorf("unexpected raw arguments %s", raw)
		}
		return hookedArgs{Value: 7}, nil
	})
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(messages) == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "call", Args: json.RawMessage(`{"value":"bad"}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(outer, inner))
	ai.AddTool(agent, "work", func(
		_ context.Context, _ *ai.RunContext[deps], args hookedArgs,
	) (int, error) {
		return args.Value, nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "done" || result.Usage().ToolCalls != 1 ||
		len(log) != 2 || !strings.HasPrefix(log[0], "inner:") || !strings.HasPrefix(log[1], "outer: inner:") {
		t.Fatalf("unexpected validation recovery result=%+v log=%v err=%v", result, log, err)
	}
}

func TestApprovalDefersOnlyAfterTypedValidation(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		switch request {
		case 1:
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "first", Args: json.RawMessage(`{"value":1}`),
			}}}, nil
		case 2:
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "second", Args: json.RawMessage(`{"value":2}`),
			}}}, nil
		default:
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		}
	})
	validationCalls := 0
	executedWith := 0
	agent := ai.NewAgent[deps, string](model)
	ai.AddToolWithArgsValidator(
		agent,
		"work",
		func(_ context.Context, _ *ai.RunContext[deps], args hookedArgs) (int, error) {
			executedWith = args.Value
			return args.Value, nil
		},
		func(_ context.Context, _ *ai.RunContext[deps], args hookedArgs) error {
			validationCalls++
			if args.Value == 1 {
				return ai.Retryf("value must be greater than one")
			}
			return nil
		},
		ai.WithApprovalRequired(),
	)
	paused, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || paused.Deferred() == nil || validationCalls != 2 || executedWith != 0 ||
		string(paused.Deferred().Approvals[0].Args) != `{"value":2}` {
		t.Fatalf(
			"tool deferred before validation: result=%+v validations=%d executed=%d err=%v",
			paused, validationCalls, executedWith, err,
		)
	}
	result, err := agent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(paused.Messages()),
		ai.WithDeferredToolResults(ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{
			"second": ai.ApproveTool(),
		}}),
	)
	if err != nil || result.Output != "done" || validationCalls != 3 || executedWith != 2 {
		t.Fatalf(
			"validated approval did not resume: result=%+v validations=%d executed=%d err=%v",
			result, validationCalls, executedWith, err,
		)
	}
}

func TestToolExecutionErrorHooksRecoverAndPostProcess(t *testing.T) {
	var log []string
	inner := ai.ToolExecutionErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, _ any, err error,
	) (any, error) {
		log = append(log, "inner: "+err.Error())
		return nil, fmt.Errorf("inner: %w", err)
	})
	outer := ai.ToolExecutionErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, args any, err error,
	) (any, error) {
		log = append(log, "outer: "+err.Error())
		return args.(hookedArgs).Value, nil
	})
	after := ai.AfterToolExecutionFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, _ any, result any,
	) (any, error) {
		log = append(log, "after")
		return result.(int) + 1, nil
	})
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(messages) == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "call", Args: json.RawMessage(`{"value":7}`),
			}}}, nil
		}
		part := messages[len(messages)-1].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart)
		if part.Content != 8 {
			t.Fatalf("unexpected recovered return: %+v", part)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(outer, after, inner))
	ai.AddTool(agent, "work", func(
		context.Context, *ai.RunContext[deps], hookedArgs,
	) (int, error) {
		return 0, errors.New("execution failed")
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	want := []string{"inner: execution failed", "outer: inner: execution failed", "after"}
	if err != nil || result.Output != "done" || !slices.Equal(log, want) {
		t.Fatalf("unexpected execution recovery result=%+v log=%v err=%v", result, log, err)
	}
}
