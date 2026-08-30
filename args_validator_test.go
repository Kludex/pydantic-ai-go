package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type validatedArgs struct {
	Value int `json:"value"`
}

func TestToolArgsValidatorRetriesBeforeExecution(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request <= 2 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "validated", ToolCallID: "call", Args: json.RawMessage(`{"value":42}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[int, string](model)
	var retries []int
	executions := 0
	ai.AddToolWithArgsValidator(
		agent,
		"validated",
		func(_ context.Context, runContext *ai.RunContext[int], args validatedArgs) (int, error) {
			executions++
			if runContext.Retry != 1 || args.Value != 42 {
				t.Fatalf("tool received unexpected context or args: rc=%+v args=%+v", runContext, args)
			}
			return args.Value, nil
		},
		func(_ context.Context, runContext *ai.RunContext[int], args validatedArgs) error {
			retries = append(retries, runContext.Retry)
			if runContext.Deps != 7 || runContext.ToolCallID != "call" || runContext.MaxRetries != 2 || args.Value != 42 {
				t.Fatalf("validator received unexpected context or args: rc=%+v args=%+v", runContext, args)
			}
			if runContext.Retry == 0 {
				return ai.Retryf("value needs confirmation")
			}
			return nil
		},
		ai.WithToolMaxRetries(2),
	)
	result, err := agent.Run(t.Context(), "go", 7)
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "done" || executions != 1 || !slices.Equal(retries, []int{0, 1}) {
		t.Fatalf("unexpected validator lifecycle: output=%q executions=%d retries=%v", result.Output, executions, retries)
	}
	firstReturn := result.Messages()[2].(ai.ModelRequest).Parts[0]
	if retry, ok := firstReturn.(ai.RetryPromptPart); !ok || retry.Content != "value needs confirmation" {
		t.Fatalf("validator retry was not returned to the model: %+v", firstReturn)
	}
}

func TestToolArgsValidatorRejectsDecodedAndTerminalFailures(t *testing.T) {
	t.Run("invalid JSON skips validator", func(t *testing.T) {
		request := 0
		model := fakes.NewFunctionModel(func(
			_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			request++
			if request == 1 {
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
					ToolName: "validated", ToolCallID: "call", Args: json.RawMessage(`{"value":`),
				}}}, nil
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		})
		agent := ai.NewAgent[deps, string](model)
		validatorCalled := false
		ai.AddToolWithArgsValidator(
			agent, "validated",
			func(context.Context, *ai.RunContext[deps], validatedArgs) (int, error) { return 0, nil },
			func(context.Context, *ai.RunContext[deps], validatedArgs) error {
				validatorCalled = true
				return nil
			},
		)
		if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
			t.Fatal(err)
		}
		if validatorCalled {
			t.Fatal("validator ran before JSON decoding succeeded")
		}
	})

	t.Run("terminal failure", func(t *testing.T) {
		request := 0
		model := fakes.NewFunctionModel(func(
			_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			request++
			if request == 1 {
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
					ToolName: "validated", ToolCallID: "call", Args: json.RawMessage(`{"value":1}`),
				}}}, nil
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		})
		agent := ai.NewAgent[deps, string](model)
		executed := false
		ai.AddToolWithArgsValidator(
			agent, "validated",
			func(context.Context, *ai.RunContext[deps], validatedArgs) (int, error) {
				executed = true
				return 0, nil
			},
			func(context.Context, *ai.RunContext[deps], validatedArgs) error {
				return ai.ToolFailedf("policy denied")
			},
		)
		result, err := agent.Run(t.Context(), "go", deps{})
		if err != nil {
			t.Fatal(err)
		}
		part := result.Messages()[2].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart)
		if executed || part.Outcome != ai.ToolReturnOutcomeFailed || part.Content != "policy denied" ||
			result.Usage().ToolCalls != 0 {
			t.Fatalf("unexpected failed validation result: executed=%v part=%+v usage=%+v", executed, part, result.Usage())
		}
	})

	t.Run("ordinary error", func(t *testing.T) {
		model := fakes.NewFunctionModel(func(
			_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "validated", ToolCallID: "call", Args: json.RawMessage(`{"value":1}`),
			}}}, nil
		})
		agent := ai.NewAgent[deps, string](model)
		ai.AddToolWithArgsValidator(
			agent, "validated",
			func(context.Context, *ai.RunContext[deps], validatedArgs) (int, error) { return 0, nil },
			func(context.Context, *ai.RunContext[deps], validatedArgs) error { return errors.New("policy broke") },
		)
		if _, err := agent.Run(t.Context(), "go", deps{}); err == nil ||
			err.Error() != `ai: tool "validated": policy broke` {
			t.Fatalf("unexpected validator error: %v", err)
		}
	})
}

func TestRawToolArgsValidatorError(t *testing.T) {
	model := fakes.NewTestModel()
	agent := ai.NewAgent[deps, string](model)
	agent.AddRawToolWithArgsValidator(
		ai.ToolDefinition{Name: "raw", Schema: map[string]any{"type": "object"}},
		func(context.Context, json.RawMessage) (any, error) {
			t.Fatal("raw tool ran after validation failure")
			return nil, nil
		},
		func(context.Context, *ai.RunContext[deps], json.RawMessage) error {
			return errors.New("raw policy broke")
		},
	)
	if _, err := agent.Run(t.Context(), "go", deps{}); err == nil ||
		err.Error() != `ai: tool "raw": raw policy broke` {
		t.Fatalf("unexpected raw validator error: %v", err)
	}
}

func TestPreparedSimpleAndRawToolArgsValidators(t *testing.T) {
	model := fakes.NewTestModel()
	agent := ai.NewAgent[deps, string](model)
	var calls []string
	ai.AddPreparedToolWithArgsValidator(
		agent, "prepared",
		func(context.Context, *ai.RunContext[deps], validatedArgs) (string, error) {
			calls = append(calls, "prepared tool")
			return "ok", nil
		},
		func(context.Context, *ai.RunContext[deps], validatedArgs) error {
			calls = append(calls, "prepared validator")
			return nil
		},
		func(_ context.Context, _ *ai.RunContext[deps], tool ai.ToolDefinition) (*ai.ToolDefinition, error) {
			calls = append(calls, "prepare")
			return &tool, nil
		},
	)
	ai.AddSimpleToolWithArgsValidator(
		agent, "simple",
		func(context.Context, validatedArgs) (string, error) {
			calls = append(calls, "simple tool")
			return "ok", nil
		},
		func(context.Context, *ai.RunContext[deps], validatedArgs) error {
			calls = append(calls, "simple validator")
			return nil
		},
	)
	agent.AddRawToolWithArgsValidator(
		ai.ToolDefinition{Name: "raw", Schema: map[string]any{"type": "object"}},
		func(_ context.Context, raw json.RawMessage) (any, error) {
			calls = append(calls, "raw tool "+string(raw))
			return "ok", nil
		},
		func(_ context.Context, _ *ai.RunContext[deps], raw json.RawMessage) error {
			calls = append(calls, "raw validator "+string(raw))
			return nil
		},
	)
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	want := []string{
		"prepare", "prepared validator", "prepared tool",
		"prepare", "simple validator", "simple tool",
		"prepare", "raw validator {}", "raw tool {}", "prepare",
	}
	if !slices.Equal(calls, want) {
		t.Fatalf("unexpected prepare/validation/execution order: %v", calls)
	}
}
