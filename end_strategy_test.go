package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type strategyOutput struct {
	Value int `json:"value"`
}

func strategyCall(name, id, args string) ai.ResponsePart {
	return ai.ToolCallPart{ToolName: name, ToolCallID: id, Args: json.RawMessage(args)}
}

func trailingRequest(t *testing.T, result *ai.RunResult[strategyOutput], offset int) ai.ModelRequest {
	t.Helper()
	messages := result.Messages()
	return messages[len(messages)-1-offset].(ai.ModelRequest)
}

func TestGracefulEndStrategyIsDefault(t *testing.T) {
	model := fakes.NewFunctionModel(func(_ context.Context, msgs []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
		if len(msgs) != 1 {
			t.Fatalf("graceful strategy should finish after one response, got %d messages", len(msgs))
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			strategyCall("before", "c1", `{}`),
			strategyCall("final_result", "out1", `{"value":1}`),
			strategyCall("after", "c2", `{}`),
			strategyCall("final_result", "out2", `{"value":2}`),
		}}, nil
	})
	agent := ai.NewAgent[deps, strategyOutput](model)
	var mu sync.Mutex
	var events []string
	record := func(event string) {
		mu.Lock()
		defer mu.Unlock()
		events = append(events, event)
	}
	ai.AddSimpleTool(agent, "before", func(context.Context, struct{}) (string, error) {
		record("before")
		return "before", nil
	})
	ai.AddSimpleTool(agent, "after", func(context.Context, struct{}) (string, error) {
		record("after")
		return "after", nil
	})
	agent.AddOutputValidator(func(_ context.Context, rc *ai.RunContext[deps], _ strategyOutput) error {
		record(rc.ToolCallID)
		return nil
	})

	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.Value != 1 {
		t.Fatalf("expected first output, got %+v", result.Output)
	}
	if !slices.Equal(events, []string{"before", "out1", "after"}) {
		t.Fatalf("unexpected execution order %v", events)
	}
	parts := trailingRequest(t, result, 0).Parts
	if len(parts) != 4 {
		t.Fatalf("expected a result for every call, got %+v", parts)
	}
	if got := parts[1].(ai.ToolReturnPart).Content; got != "Final result processed." {
		t.Fatalf("unexpected winning output status %q", got)
	}
	if got := parts[3].(ai.ToolReturnPart).Content; got != "Output tool not used - a final result was already processed." {
		t.Fatalf("unexpected skipped output status %q", got)
	}
}

func TestEarlyEndStrategySkipsFunctionTools(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			strategyCall("before", "c1", `{}`),
			strategyCall("final_result", "out", `{"value":1}`),
			strategyCall("after", "c2", `{}`),
			strategyCall("final_result", "later", `{"value":2}`),
		}}, nil
	})
	agent := ai.NewAgent[deps, strategyOutput](model, ai.WithEndStrategy(ai.EndStrategyEarly))
	ran := false
	tool := func(context.Context, struct{}) (string, error) {
		ran = true
		return "", nil
	}
	ai.AddSimpleTool(agent, "before", tool)
	ai.AddSimpleTool(agent, "after", tool)

	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if ran {
		t.Fatal("early strategy executed a function tool")
	}
	parts := trailingRequest(t, result, 0).Parts
	for _, index := range []int{0, 2} {
		if got := parts[index].(ai.ToolReturnPart).Content; got != "Tool not executed - a final result was already processed." {
			t.Fatalf("unexpected skipped tool status %q", got)
		}
	}
	if got := parts[3].(ai.ToolReturnPart).Content; got != "Output tool not used - a final result was already processed." {
		t.Fatalf("unexpected skipped output status %q", got)
	}
}

func TestEarlyEndStrategyRunsFunctionsWhenEveryOutputFails(t *testing.T) {
	calls := 0
	model := fakes.NewFunctionModel(func(_ context.Context, msgs []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
		calls++
		if calls == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				strategyCall("work", "c1", `{}`),
				strategyCall("final_result", "bad", `{"value":"wrong"}`),
			}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			strategyCall("final_result", "good", `{"value":2}`),
		}}, nil
	})
	agent := ai.NewAgent[deps, strategyOutput](model, ai.WithEndStrategy(ai.EndStrategyEarly))
	workRan := false
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) {
		workRan = true
		return "done", nil
	})

	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if !workRan || result.Output.Value != 2 {
		t.Fatalf("unexpected result %+v, work ran: %v", result.Output, workRan)
	}
	firstResults := trailingRequest(t, result, 2).Parts
	if _, ok := firstResults[0].(ai.ToolReturnPart); !ok {
		t.Fatalf("function result lost emission order: %+v", firstResults)
	}
	if _, ok := firstResults[1].(ai.RetryPromptPart); !ok {
		t.Fatalf("output retry lost emission order: %+v", firstResults)
	}
}

func TestExhaustiveEndStrategyRunsEveryOutputConcurrently(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			strategyCall("final_result", "first", `{"value":1}`),
			strategyCall("final_result", "second", `{"value":2}`),
		}}, nil
	})
	agent := ai.NewAgent[deps, strategyOutput](model, ai.WithEndStrategy(ai.EndStrategyExhaustive))
	var started atomic.Int32
	bothStarted := make(chan struct{})
	agent.AddOutputValidator(func(ctx context.Context, _ *ai.RunContext[deps], _ strategyOutput) error {
		if started.Add(1) == 2 {
			close(bothStarted)
		}
		select {
		case <-bothStarted:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	})

	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	result, err := agent.Run(ctx, "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.Value != 1 {
		t.Fatalf("expected first output by emission order, got %+v", result.Output)
	}
	parts := trailingRequest(t, result, 0).Parts
	if got := parts[1].(ai.ToolReturnPart).Content; got != "Output tool processed, but its value will not be the final result of the agent run." {
		t.Fatalf("unexpected non-winning output status %q", got)
	}
}

func TestFunctionRetrySuppressesGracefulAndExhaustiveOutput(t *testing.T) {
	for _, strategy := range []ai.EndStrategy{ai.EndStrategyGraceful, ai.EndStrategyExhaustive} {
		t.Run(string(strategy), func(t *testing.T) {
			calls := 0
			model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
				calls++
				if calls == 1 {
					return &ai.ModelResponse{Parts: []ai.ResponsePart{
						strategyCall("final_result", "first", `{"value":1}`),
						strategyCall("retry", "tool", `{}`),
					}}, nil
				}
				return &ai.ModelResponse{Parts: []ai.ResponsePart{
					strategyCall("final_result", "second", `{"value":2}`),
				}}, nil
			})
			agent := ai.NewAgent[deps, strategyOutput](model, ai.WithEndStrategy(strategy))
			ai.AddSimpleTool(agent, "retry", func(context.Context, struct{}) (string, error) {
				return "", ai.Retryf("try again")
			})

			result, err := agent.Run(t.Context(), "go", deps{})
			if err != nil {
				t.Fatal(err)
			}
			if result.Output.Value != 2 || calls != 2 {
				t.Fatalf("retry did not suppress first output: %+v after %d calls", result.Output, calls)
			}
			firstResults := trailingRequest(t, result, 2).Parts
			if got := firstResults[0].(ai.ToolReturnPart).Content; got != "Output not used as the final result - addressing tool retries from this round first." {
				t.Fatalf("unexpected retry-wins status %q", got)
			}
		})
	}
}

func TestOutputRetryDoesNotSuppressExhaustiveWinner(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			strategyCall("final_result", "bad", `{"value":"wrong"}`),
			strategyCall("final_result", "good", `{"value":2}`),
		}}, nil
	})
	agent := ai.NewAgent[deps, strategyOutput](model, ai.WithEndStrategy(ai.EndStrategyExhaustive))
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.Value != 2 {
		t.Fatalf("output retry suppressed valid output: %+v", result.Output)
	}
}

func TestEarlyNativeOutputSkipsFunctionTools(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.TextPart{Content: `{"value":1}`},
			strategyCall("work", "tool", `{}`),
		}}, nil
	})
	agent := ai.NewAgent[deps, strategyOutput](
		model,
		ai.WithOutputMode(ai.OutputModeNative),
		ai.WithEndStrategy(ai.EndStrategyEarly),
	)
	workRan := false
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) {
		workRan = true
		return "", nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if workRan || result.Output.Value != 1 {
		t.Fatalf("unexpected result %+v, work ran: %v", result.Output, workRan)
	}
	if got := trailingRequest(t, result, 0).Parts[0].(ai.ToolReturnPart).Content; got != "Tool not executed - a final result was already processed." {
		t.Fatalf("unexpected skipped tool status %q", got)
	}
}

func TestGracefulAndExhaustiveIgnoreNativeOutputAlongsideTools(t *testing.T) {
	for _, strategy := range []ai.EndStrategy{ai.EndStrategyGraceful, ai.EndStrategyExhaustive} {
		t.Run(string(strategy), func(t *testing.T) {
			calls := 0
			model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
				calls++
				if calls == 1 {
					return &ai.ModelResponse{Parts: []ai.ResponsePart{
						ai.TextPart{Content: `{"value":1}`},
						strategyCall("work", "tool", `{}`),
					}}, nil
				}
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"value":2}`}}}, nil
			})
			agent := ai.NewAgent[deps, strategyOutput](
				model,
				ai.WithOutputMode(ai.OutputModeNative),
				ai.WithEndStrategy(strategy),
			)
			workRan := false
			ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) {
				workRan = true
				return "done", nil
			})
			result, err := agent.Run(t.Context(), "go", deps{})
			if err != nil {
				t.Fatal(err)
			}
			if !workRan || result.Output.Value != 2 {
				t.Fatalf("unexpected result %+v, work ran: %v", result.Output, workRan)
			}
		})
	}
}

func TestInvalidEarlyNativeOutputFallsThroughToTools(t *testing.T) {
	calls := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		calls++
		if calls == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.TextPart{Content: `{bad`},
				strategyCall("work", "tool", `{}`),
			}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"value":2}`}}}, nil
	})
	agent := ai.NewAgent[deps, strategyOutput](
		model,
		ai.WithOutputMode(ai.OutputModeNative),
		ai.WithEndStrategy(ai.EndStrategyEarly),
	)
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) { return "done", nil })
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.Value != 2 {
		t.Fatalf("unexpected output %+v", result.Output)
	}
	if _, ok := trailingRequest(t, result, 1).Parts[0].(ai.ToolReturnPart); !ok {
		t.Fatalf("invalid native output surfaced a retry: %+v", result.Messages())
	}
}

func TestEarlyNativeValidatorRetryFallsThroughToTools(t *testing.T) {
	calls := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		calls++
		if calls == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.TextPart{Content: `{"value":1}`},
				strategyCall("work", "tool", `{}`),
			}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"value":2}`}}}, nil
	})
	agent := ai.NewAgent[deps, strategyOutput](
		model,
		ai.WithOutputMode(ai.OutputModeNative),
		ai.WithEndStrategy(ai.EndStrategyEarly),
	)
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) { return "done", nil })
	agent.AddOutputValidator(func(_ context.Context, _ *ai.RunContext[deps], output strategyOutput) error {
		if output.Value == 1 {
			return ai.Retryf("not one")
		}
		return nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.Value != 2 {
		t.Fatalf("unexpected output %+v", result.Output)
	}
	if _, ok := trailingRequest(t, result, 1).Parts[0].(ai.ToolReturnPart); !ok {
		t.Fatalf("candidate validation surfaced a retry: %+v", result.Messages())
	}
}

func TestEarlyPlainTextDoesNotPreemptTools(t *testing.T) {
	calls := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		calls++
		if calls == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.TextPart{Content: "working"},
				strategyCall("work", "tool", `{}`),
			}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithEndStrategy(ai.EndStrategyEarly))
	workRan := false
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) {
		workRan = true
		return "done", nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if !workRan || result.Output != "done" {
		t.Fatalf("unexpected output %q, work ran: %v", result.Output, workRan)
	}
}

func TestEarlyNativeOutputValidatorErrorStopsRun(t *testing.T) {
	boom := errors.New("boom")
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.TextPart{Content: `{"value":1}`},
			strategyCall("work", "tool", `{}`),
		}}, nil
	})
	agent := ai.NewAgent[deps, strategyOutput](
		model,
		ai.WithOutputMode(ai.OutputModeNative),
		ai.WithEndStrategy(ai.EndStrategyEarly),
	)
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) { return "done", nil })
	agent.AddOutputValidator(func(context.Context, *ai.RunContext[deps], strategyOutput) error { return boom })
	if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, boom) {
		t.Fatalf("expected validator error, got %v", err)
	}
}

func TestEndStrategyToolErrors(t *testing.T) {
	boom := errors.New("boom")
	tests := []struct {
		name     string
		strategy ai.EndStrategy
		parts    []ai.ResponsePart
		setup    func(*ai.Agent[deps, strategyOutput])
	}{
		{
			name:     "early output",
			strategy: ai.EndStrategyEarly,
			parts:    []ai.ResponsePart{strategyCall("final_result", "out", `{"value":1}`)},
			setup: func(agent *ai.Agent[deps, strategyOutput]) {
				agent.AddOutputValidator(func(context.Context, *ai.RunContext[deps], strategyOutput) error {
					return boom
				})
			},
		},
		{
			name:     "early function after failed output",
			strategy: ai.EndStrategyEarly,
			parts: []ai.ResponsePart{
				strategyCall("final_result", "out", `{"value":"wrong"}`),
				strategyCall("fails", "tool", `{}`),
			},
			setup: func(agent *ai.Agent[deps, strategyOutput]) {
				ai.AddSimpleTool(agent, "fails", func(context.Context, struct{}) (string, error) { return "", boom })
			},
		},
		{
			name:     "exhaustive batch before barrier",
			strategy: ai.EndStrategyExhaustive,
			parts: []ai.ResponsePart{
				strategyCall("fails", "tool", `{}`),
				strategyCall("barrier", "barrier", `{}`),
			},
			setup: func(agent *ai.Agent[deps, strategyOutput]) {
				ai.AddSimpleTool(agent, "fails", func(context.Context, struct{}) (string, error) { return "", boom })
				ai.AddSimpleTool(agent, "barrier", func(context.Context, struct{}) (string, error) { return "", nil }, ai.WithSequential())
			},
		},
		{
			name:     "exhaustive barrier",
			strategy: ai.EndStrategyExhaustive,
			parts:    []ai.ResponsePart{strategyCall("fails", "tool", `{}`)},
			setup: func(agent *ai.Agent[deps, strategyOutput]) {
				ai.AddSimpleTool(agent, "fails", func(context.Context, struct{}) (string, error) { return "", boom }, ai.WithSequential())
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
				return &ai.ModelResponse{Parts: test.parts}, nil
			})
			agent := ai.NewAgent[deps, strategyOutput](model, ai.WithEndStrategy(test.strategy))
			test.setup(agent)
			if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, boom) {
				t.Fatalf("expected tool error, got %v", err)
			}
		})
	}
}

func TestExhaustiveSequentialToolSucceeds(t *testing.T) {
	calls := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		calls++
		if calls == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{strategyCall("work", "tool", `{}`)}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			strategyCall("final_result", "out", `{"value":1}`),
		}}, nil
	})
	agent := ai.NewAgent[deps, strategyOutput](model, ai.WithEndStrategy(ai.EndStrategyExhaustive))
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) { return "done", nil }, ai.WithSequential())
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
}

func TestInvalidEndStrategyPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected invalid strategy to panic")
		}
	}()
	ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithEndStrategy("later"))
}
