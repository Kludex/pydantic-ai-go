package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type toolSetupCapability struct {
	setup func(*ai.CapabilityRegistry) error
}

func (capability *toolSetupCapability) Setup(registry *ai.CapabilityRegistry) error {
	return capability.setup(registry)
}

type invalidToolCallArgsWrapper struct{}

func (invalidToolCallArgsWrapper) Setup(*ai.CapabilityRegistry) error { return nil }

func (invalidToolCallArgsWrapper) WrapToolCall(
	ctx context.Context, _ *ai.RunInfo, call ai.ToolCallPart, next ai.ToolCallFunc,
) (any, error) {
	call.Args = json.RawMessage(`{"value":"bad"}`)
	return next(ctx, call)
}

type toolCallArgsWrapper struct{}

func (toolCallArgsWrapper) Setup(*ai.CapabilityRegistry) error { return nil }

func (toolCallArgsWrapper) WrapToolCall(
	ctx context.Context, _ *ai.RunInfo, call ai.ToolCallPart, next ai.ToolCallFunc,
) (any, error) {
	call.Args = json.RawMessage(`{"value":9}`)
	return next(ctx, call)
}

func toolThenTextModel(t *testing.T, raw json.RawMessage, inspect func(ai.ToolReturnPart)) ai.Model {
	t.Helper()
	request := 0
	return fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "call", Args: raw,
			}}}, nil
		}
		if inspect != nil {
			inspect(messages[len(messages)-1].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart))
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
}

func alwaysToolModel(raw json.RawMessage) ai.Model {
	return fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "work", ToolCallID: "call", Args: raw,
		}}}, nil
	})
}

func TestToolCallWrapperRevalidationPropagatesHookErrors(t *testing.T) {
	before := ai.BeforeToolValidationFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, raw json.RawMessage,
	) (json.RawMessage, error) {
		if string(raw) == `{"value":9}` {
			return raw, errors.New("wrapper arguments rejected")
		}
		return raw, nil
	})
	agent := ai.NewAgent[deps, string](
		toolThenTextModel(t, json.RawMessage(`{"value":1}`), nil),
		ai.WithCapabilities(before, toolCallArgsWrapper{}),
	)
	ai.AddTool(agent, "work", func(
		context.Context, *ai.RunContext[deps], hookedArgs,
	) (int, error) {
		return 0, nil
	})
	_, err := agent.Run(t.Context(), "go", deps{})
	if err == nil || !strings.Contains(err.Error(), "wrapper arguments rejected") {
		t.Fatalf("unexpected wrapper hook error: %v", err)
	}
}

func TestToolCallWrapperInvalidArgumentsAreRevalidated(t *testing.T) {
	agent := ai.NewAgent[deps, string](
		alwaysToolModel(json.RawMessage(`{"value":1}`)), ai.WithCapabilities(invalidToolCallArgsWrapper{}),
	)
	ai.AddTool(agent, "work", func(
		context.Context, *ai.RunContext[deps], hookedArgs,
	) (int, error) {
		t.Fatal("tool executed with invalid wrapper arguments")
		return 0, nil
	})
	_, err := agent.Run(t.Context(), "go", deps{})
	if !errors.Is(err, ai.ErrMaxRetriesExceeded) {
		t.Fatalf("unexpected wrapper validation error: %v", err)
	}
}

func TestToolHookFunctionAdaptersAndToolCallRevalidation(t *testing.T) {
	var calls []string
	beforeValidation := ai.BeforeToolValidationFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, raw json.RawMessage,
	) (json.RawMessage, error) {
		calls = append(calls, "before-validation")
		return raw, nil
	})
	afterValidation := ai.AfterToolValidationFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, args any,
	) (any, error) {
		calls = append(calls, "after-validation")
		return args, nil
	})
	beforeExecution := ai.BeforeToolExecutionFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, args any,
	) (any, error) {
		calls = append(calls, "before-execution")
		return args, nil
	})
	model := toolThenTextModel(t, json.RawMessage(`{"value":1}`), func(part ai.ToolReturnPart) {
		if part.Content != 9 {
			t.Fatalf("tool-call wrapper arguments were not revalidated: %+v", part)
		}
	})
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(
		beforeValidation, afterValidation, beforeExecution, toolCallArgsWrapper{},
	))
	ai.AddTool(agent, "work", func(
		_ context.Context, _ *ai.RunContext[deps], args hookedArgs,
	) (int, error) {
		return args.Value, nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "done" || len(calls) != 5 {
		t.Fatalf("unexpected adapter result=%+v calls=%v err=%v", result, calls, err)
	}
}

func TestToolValidationHookRetryPaths(t *testing.T) {
	for name, capability := range map[string]ai.Capability{
		"before": ai.BeforeToolValidationFunc(func(
			_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, raw json.RawMessage,
		) (json.RawMessage, error) {
			return raw, ai.Retryf("before rejected")
		}),
		"after": ai.AfterToolValidationFunc(func(
			_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, args any,
		) (any, error) {
			return args, ai.Retryf("after rejected")
		}),
	} {
		t.Run(name, func(t *testing.T) {
			agent := ai.NewAgent[deps, string](
				alwaysToolModel(json.RawMessage(`{"value":1}`)), ai.WithCapabilities(capability),
			)
			ai.AddTool(agent, "work", func(
				context.Context, *ai.RunContext[deps], hookedArgs,
			) (int, error) {
				t.Fatal("tool executed after validation hook retry")
				return 0, nil
			})
			_, err := agent.Run(t.Context(), "go", deps{})
			if !errors.Is(err, ai.ErrMaxRetriesExceeded) {
				t.Fatalf("unexpected validation retry error: %v", err)
			}
		})
	}
}

func TestToolExecutionHookRetryAndFailureBypassErrorHooks(t *testing.T) {
	for name, capability := range map[string]ai.Capability{
		"before retry": ai.BeforeToolExecutionFunc(func(
			_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, args any,
		) (any, error) {
			return args, ai.Retryf("not now")
		}),
		"after retry": ai.AfterToolExecutionFunc(func(
			_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, _ any, result any,
		) (any, error) {
			return result, ai.Retryf("reject result")
		}),
	} {
		t.Run(name, func(t *testing.T) {
			errorHookCalls := 0
			onError := ai.ToolExecutionErrorFunc(func(
				_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, _ any, err error,
			) (any, error) {
				errorHookCalls++
				return nil, err
			})
			agent := ai.NewAgent[deps, string](
				alwaysToolModel(json.RawMessage(`{"value":1}`)), ai.WithCapabilities(capability, onError),
			)
			ai.AddTool(agent, "work", func(
				context.Context, *ai.RunContext[deps], hookedArgs,
			) (int, error) {
				return 1, nil
			})
			_, err := agent.Run(t.Context(), "go", deps{})
			if !errors.Is(err, ai.ErrMaxRetriesExceeded) || errorHookCalls != 0 {
				t.Fatalf("unexpected execution retry error=%v error hooks=%d", err, errorHookCalls)
			}
		})
	}

	errorHookCalls := 0
	onError := ai.ToolExecutionErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, _ any, err error,
	) (any, error) {
		errorHookCalls++
		return nil, err
	})
	agent := ai.NewAgent[deps, string](
		toolThenTextModel(t, json.RawMessage(`{"value":1}`), nil), ai.WithCapabilities(onError),
	)
	ai.AddTool(agent, "work", func(
		context.Context, *ai.RunContext[deps], hookedArgs,
	) (int, error) {
		return 0, ai.ToolFailedf("denied")
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "done" || errorHookCalls != 0 {
		t.Fatalf("tool failure reached execution error hook result=%+v calls=%d err=%v", result, errorHookCalls, err)
	}
}

func TestToolHooksSurfaceInvalidTransformedTypesAndValues(t *testing.T) {
	for name, capability := range map[string]ai.Capability{
		"typed arguments": ai.AfterToolValidationFunc(func(
			context.Context, *ai.RunInfo, ai.ToolHookContext, any,
		) (any, error) {
			return "wrong", nil
		}),
		"unmarshalable arguments": ai.AfterToolValidationFunc(func(
			context.Context, *ai.RunInfo, ai.ToolHookContext, any,
		) (any, error) {
			return func() {}, nil
		}),
		"after execution error": ai.AfterToolExecutionFunc(func(
			context.Context, *ai.RunInfo, ai.ToolHookContext, any, any,
		) (any, error) {
			return nil, errors.New("post-process failed")
		}),
	} {
		t.Run(name, func(t *testing.T) {
			agent := ai.NewAgent[deps, string](
				toolThenTextModel(t, json.RawMessage(`{"value":1}`), nil), ai.WithCapabilities(capability),
			)
			ai.AddTool(agent, "work", func(
				context.Context, *ai.RunContext[deps], hookedArgs,
			) (int, error) {
				return 1, nil
			})
			_, err := agent.Run(t.Context(), "go", deps{})
			if err == nil {
				t.Fatal("expected transformed value error")
			}
		})
	}
}

func TestRawAndCapabilityToolHooksRejectWrongValidatedTypes(t *testing.T) {
	wrongType := ai.AfterToolValidationFunc(func(
		context.Context, *ai.RunInfo, ai.ToolHookContext, any,
	) (any, error) {
		return 42, nil
	})

	t.Run("raw", func(t *testing.T) {
		agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(wrongType))
		agent.AddRawTool(
			ai.ToolDefinition{Name: "raw", Schema: map[string]any{"type": "object"}},
			func(context.Context, json.RawMessage) (any, error) { return "unused", nil },
		)
		_, err := agent.Run(t.Context(), "go", deps{})
		if err == nil || !strings.Contains(err.Error(), "expected json.RawMessage") {
			t.Fatalf("unexpected raw type error: %v", err)
		}
	})

	t.Run("capability", func(t *testing.T) {
		capability := &toolSetupCapability{setup: func(reg *ai.CapabilityRegistry) error {
			reg.AddTool(
				ai.ToolDefinition{Name: "capability", Schema: map[string]any{"type": "object"}},
				func(context.Context, json.RawMessage) (any, error) { return "unused", nil },
			)
			return nil
		}}
		agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(capability, wrongType))
		_, err := agent.Run(t.Context(), "go", deps{})
		if err == nil || !strings.Contains(err.Error(), "expected json.RawMessage") {
			t.Fatalf("unexpected capability type error: %v", err)
		}
	})

	t.Run("run capability", func(t *testing.T) {
		capability := &toolSetupCapability{setup: func(reg *ai.CapabilityRegistry) error {
			reg.AddTool(
				ai.ToolDefinition{Name: "run_capability", Schema: map[string]any{"type": "object"}},
				func(context.Context, json.RawMessage) (any, error) { return "unused", nil },
			)
			return nil
		}}
		agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(wrongType))
		_, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunCapabilities(capability))
		if err == nil || !strings.Contains(err.Error(), "expected json.RawMessage") {
			t.Fatalf("unexpected run capability type error: %v", err)
		}
	})
}

func TestToolSearchRejectsInvalidValidatedHookType(t *testing.T) {
	hidden := ai.NewSimpleTool[deps]("hidden", func(context.Context, struct{}) (string, error) {
		return "hidden", nil
	}, ai.WithDeferredLoading())
	wrongType := ai.AfterToolValidationFunc(func(
		context.Context, *ai.RunInfo, ai.ToolHookContext, any,
	) (any, error) {
		return 42, nil
	})
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: ai.ToolSearchName, ToolCallID: "search", Args: json.RawMessage(`{"queries":["hidden"]}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(wrongType))
	agent.AddToolset(ai.WithToolSearch(ai.NewFunctionToolset(hidden), ai.ToolSearchConfig[deps]{}))
	_, err := agent.Run(t.Context(), "go", deps{})
	if err == nil || !strings.Contains(err.Error(), "validated tool search arguments") {
		t.Fatalf("unexpected search hook type error: %v", err)
	}
}

func TestApprovalToolsetRejectsUnmarshalableExecutionHookArgs(t *testing.T) {
	invalid := ai.BeforeToolExecutionFunc(func(
		context.Context, *ai.RunInfo, ai.ToolHookContext, any,
	) (any, error) {
		return func() {}, nil
	})
	tool := ai.NewSimpleTool[deps]("work", func(context.Context, hookedArgs) (string, error) {
		return "done", nil
	})
	toolset := ai.RequireApprovalToolsetWhen(ai.NewFunctionToolset(tool), func(
		context.Context, *ai.RunContext[deps], ai.ToolDefinition, json.RawMessage,
	) (*ai.ToolApprovalRequest, error) {
		return nil, nil
	})
	agent := ai.NewAgent[deps, string](
		toolThenTextModel(t, json.RawMessage(`{"value":1}`), nil), ai.WithCapabilities(invalid),
	)
	agent.AddToolset(toolset)
	_, err := agent.Run(t.Context(), "go", deps{})
	if err == nil || !strings.Contains(err.Error(), "marshal validated arguments") {
		t.Fatalf("unexpected approval hook argument error: %v", err)
	}
}

func TestValidationErrorHookMayReturnAnInvalidType(t *testing.T) {
	recoverWrong := ai.ToolValidationErrorFunc(func(
		context.Context, *ai.RunInfo, ai.ToolHookContext, json.RawMessage, error,
	) (any, error) {
		return "wrong", nil
	})
	executionErrors := ai.ToolExecutionErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, _ any, err error,
	) (any, error) {
		return nil, fmt.Errorf("observed: %w", err)
	})
	agent := ai.NewAgent[deps, string](
		toolThenTextModel(t, json.RawMessage(`{"value":"bad"}`), nil),
		ai.WithCapabilities(recoverWrong, executionErrors),
	)
	ai.AddTool(agent, "work", func(
		context.Context, *ai.RunContext[deps], hookedArgs,
	) (int, error) {
		return 1, nil
	})
	_, err := agent.Run(t.Context(), "go", deps{})
	if err == nil || !strings.Contains(err.Error(), "observed:") || !strings.Contains(err.Error(), "expected") {
		t.Fatalf("unexpected recovered type error: %v", err)
	}
}
