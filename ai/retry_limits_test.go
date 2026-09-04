package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestFunctionRetriesAreTrackedPerTool(t *testing.T) {
	step := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		calls := []string{"first", "second", "first", "second"}
		if step < len(calls) {
			name := calls[step]
			step++
			return &ai.ModelResponse{Parts: []ai.ResponsePart{strategyCall(name, name, `{}`)}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	var firstRetries, secondRetries []int
	ai.AddTool(agent, "first", func(_ context.Context, rc *ai.RunContext[deps], _ struct{}) (string, error) {
		firstRetries = append(firstRetries, rc.Retry)
		if rc.Retry == 0 {
			return "", ai.Retryf("again")
		}
		return "first", nil
	})
	ai.AddTool(agent, "second", func(_ context.Context, rc *ai.RunContext[deps], _ struct{}) (string, error) {
		secondRetries = append(secondRetries, rc.Retry)
		if rc.Retry == 0 {
			return "", ai.Retryf("again")
		}
		return "second", nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(firstRetries, []int{0, 1}) || !slices.Equal(secondRetries, []int{0, 1}) {
		t.Fatalf("retry counters were shared: first=%v second=%v", firstRetries, secondRetries)
	}
}

func TestFunctionAndOutputRetriesHaveSeparateBudgets(t *testing.T) {
	step := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		step++
		switch step {
		case 1:
			return &ai.ModelResponse{Parts: []ai.ResponsePart{strategyCall("work", "tool", `{}`)}}, nil
		case 2:
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{bad`}}}, nil
		default:
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"value":2}`}}}, nil
		}
	})
	agent := ai.NewAgent[deps, strategyOutput](model, ai.WithOutputMode(ai.OutputModeNative))
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) {
		return "", ai.Retryf("again")
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.Value != 2 {
		t.Fatalf("unexpected output %+v", result.Output)
	}
}

func TestRetryLimitPrecedence(t *testing.T) {
	t.Run("run overrides agent", func(t *testing.T) {
		calls := 0
		model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
			calls++
			if calls == 1 {
				return &ai.ModelResponse{Parts: []ai.ResponsePart{strategyCall("work", "tool", `{}`)}}, nil
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		})
		agent := ai.NewAgent[deps, string](model, ai.WithRetryLimits(ai.RetryLimits{}))
		ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) {
			return "", ai.Retryf("again")
		})
		_, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunRetryLimits(ai.RetryLimits{Tools: 1, Output: 1}))
		if err != nil {
			t.Fatal(err)
		}
	})

	t.Run("tool overrides run", func(t *testing.T) {
		model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{strategyCall("work", "tool", `{}`)}}, nil
		})
		agent := ai.NewAgent[deps, string](model)
		ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) {
			return "", ai.Retryf("again")
		}, ai.WithToolMaxRetries(0))
		_, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunRetryLimits(ai.RetryLimits{Tools: 3, Output: 3}))
		if !errors.Is(err, ai.ErrMaxRetriesExceeded) {
			t.Fatalf("expected per-tool limit to win, got %v", err)
		}
	})
}

func TestRawToolAcceptsRetryOption(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{strategyCall("raw", "tool", `{}`)}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddRawTool(
		ai.ToolDefinition{Name: "raw", Schema: map[string]any{"type": "object"}},
		func(context.Context, json.RawMessage) (any, error) { return nil, ai.Retryf("again") },
		ai.WithToolMaxRetries(0),
	)
	if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, ai.ErrMaxRetriesExceeded) {
		t.Fatalf("expected raw tool retry limit, got %v", err)
	}
}

func TestToolRunContextReportsRetryLimit(t *testing.T) {
	calls := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		calls++
		if calls <= 3 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{strategyCall("work", "tool", `{}`)}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	var retries, maximums []int
	ai.AddTool(agent, "work", func(_ context.Context, rc *ai.RunContext[deps], _ struct{}) (string, error) {
		retries = append(retries, rc.Retry)
		maximums = append(maximums, rc.MaxRetries)
		if rc.Retry < 2 {
			return "", ai.Retryf("again")
		}
		return "done", nil
	}, ai.WithToolMaxRetries(2))
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(retries, []int{0, 1, 2}) || !slices.Equal(maximums, []int{2, 2, 2}) {
		t.Fatalf("unexpected contexts: retries=%v maximums=%v", retries, maximums)
	}
}

func TestOutputRunContextReportsRetryLimit(t *testing.T) {
	value := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		value++
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			strategyCall("final_result", "output", fmt.Sprintf(`{"value":%d}`, value)),
		}}, nil
	})
	agent := ai.NewAgent[deps, strategyOutput](model, ai.WithRetryLimits(ai.RetryLimits{Tools: 0, Output: 2}))
	var retries, maximums []int
	agent.AddOutputValidator(func(_ context.Context, rc *ai.RunContext[deps], output strategyOutput) error {
		retries = append(retries, rc.Retry)
		maximums = append(maximums, rc.MaxRetries)
		if output.Value < 3 {
			return ai.Retryf("too small")
		}
		return nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.Value != 3 || !slices.Equal(retries, []int{0, 1, 2}) || !slices.Equal(maximums, []int{2, 2, 2}) {
		t.Fatalf("unexpected output contexts: result=%+v retries=%v maximums=%v", result.Output, retries, maximums)
	}
}

func TestZeroOutputRetryLimit(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{bad`}}}, nil
	})
	agent := ai.NewAgent[deps, strategyOutput](model, ai.WithMaxRetries(0), ai.WithOutputMode(ai.OutputModeNative))
	if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, ai.ErrMaxRetriesExceeded) {
		t.Fatalf("expected zero output budget to fail, got %v", err)
	}
}

func TestNegativeRetryLimitsPanic(t *testing.T) {
	tests := []struct {
		name string
		run  func()
	}{
		{
			name: "agent",
			run: func() {
				ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithRetryLimits(ai.RetryLimits{Tools: -1}))
			},
		},
		{
			name: "run",
			run: func() {
				agent := ai.NewAgent[deps, string](fakes.NewTestModel())
				_, _ = agent.Run(context.Background(), "go", deps{}, ai.WithRunRetryLimits(ai.RetryLimits{Output: -1}))
			},
		},
		{
			name: "tool",
			run: func() {
				ai.WithToolMaxRetries(-1)
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			test.run()
		})
	}
}
