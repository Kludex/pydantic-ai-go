package ai_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
)

func TestRunStreamCommitsTextBeforeToolProcessing(t *testing.T) {
	tests := []struct {
		strategy ai.EndStrategy
		calls    int32
	}{
		{strategy: ai.EndStrategyEarly, calls: 0},
		{strategy: ai.EndStrategyGraceful, calls: 2},
		{strategy: ai.EndStrategyExhaustive, calls: 2},
	}
	for _, test := range tests {
		t.Run(string(test.strategy), func(t *testing.T) {
			model := newStreamingModel(func([]ai.ModelMessage) []ai.StreamEvent {
				return []ai.StreamEvent{
					ai.ToolCallStartEvent{ToolName: "before", ToolCallID: "a"},
					ai.ToolCallDeltaEvent{ArgsDelta: `{}`},
					ai.TextDeltaEvent{Delta: "committed"},
					ai.ToolCallStartEvent{ToolName: "after", ToolCallID: "b"},
					ai.ToolCallDeltaEvent{ArgsDelta: `{}`},
					ai.FinishEvent{},
				}
			})
			agent := ai.NewAgent[deps, string](model, ai.WithEndStrategy(test.strategy))
			var calls atomic.Int32
			ai.AddSimpleTool(agent, "before", func(context.Context, struct{}) (string, error) {
				calls.Add(1)
				return "before", nil
			})
			ai.AddSimpleTool(agent, "after", func(context.Context, struct{}) (string, error) {
				calls.Add(1)
				return "after", nil
			})
			result, err := consumeRunStream(t, agent)
			if err != nil {
				t.Fatal(err)
			}
			if result.Output != "committed" || calls.Load() != test.calls {
				t.Fatalf("unexpected output %q or calls %d", result.Output, calls.Load())
			}
			if len(result.Messages()) != 3 {
				t.Fatalf("tool calls were not settled in history: %v", result.Messages())
			}
		})
	}
}

func TestRunStreamCommitsFirstOutputTool(t *testing.T) {
	tests := []struct {
		strategy       ai.EndStrategy
		functionCalls  int32
		validatorCalls int
		secondStatus   string
	}{
		{strategy: ai.EndStrategyEarly, validatorCalls: 1, secondStatus: "Output tool not used - a final result was already processed."},
		{
			strategy: ai.EndStrategyGraceful, functionCalls: 2, validatorCalls: 1,
			secondStatus: "Output tool not used - a final result was already processed.",
		},
		{
			strategy: ai.EndStrategyExhaustive, functionCalls: 2, validatorCalls: 2,
			secondStatus: "Output tool processed, but its value will not be the final result of the agent run.",
		},
	}
	for _, test := range tests {
		t.Run(string(test.strategy), func(t *testing.T) {
			model := newStreamingModel(func([]ai.ModelMessage) []ai.StreamEvent {
				return []ai.StreamEvent{
					ai.ToolCallStartEvent{ToolName: "before", ToolCallID: "a"},
					ai.ToolCallDeltaEvent{ArgsDelta: `{}`},
					ai.ToolCallStartEvent{ToolName: "final_result", ToolCallID: "first"},
					ai.ToolCallDeltaEvent{ArgsDelta: `{"city":"first","temp_c":1}`},
					ai.ToolCallStartEvent{ToolName: "final_result", ToolCallID: "second"},
					ai.ToolCallDeltaEvent{ArgsDelta: `{"city":"second","temp_c":2}`},
					ai.ToolCallStartEvent{ToolName: "after", ToolCallID: "b"},
					ai.ToolCallDeltaEvent{ArgsDelta: `{}`},
					ai.FinishEvent{},
				}
			})
			agent := ai.NewAgent[deps, weather](model, ai.WithEndStrategy(test.strategy))
			var functionCalls atomic.Int32
			ai.AddSimpleTool(agent, "before", func(context.Context, struct{}) (string, error) {
				functionCalls.Add(1)
				return "before", nil
			})
			ai.AddSimpleTool(agent, "after", func(context.Context, struct{}) (string, error) {
				functionCalls.Add(1)
				return "after", nil
			}, ai.WithSequential())
			var mutex sync.Mutex
			var validated []string
			agent.AddOutputValidator(func(_ context.Context, _ *ai.RunContext[deps], output weather) error {
				mutex.Lock()
				defer mutex.Unlock()
				validated = append(validated, output.City)
				return nil
			})
			result, err := consumeRunStream(t, agent)
			if err != nil {
				t.Fatal(err)
			}
			if result.Output.City != "first" || functionCalls.Load() != test.functionCalls {
				t.Fatalf("unexpected result %+v or calls %d", result.Output, functionCalls.Load())
			}
			mutex.Lock()
			validatorCalls := len(validated)
			mutex.Unlock()
			if validatorCalls != test.validatorCalls {
				t.Fatalf("expected %d validator calls, got %v", test.validatorCalls, validated)
			}
			request := result.Messages()[2].(ai.ModelRequest)
			winner := request.Parts[1].(ai.ToolReturnPart)
			second := request.Parts[2].(ai.ToolReturnPart)
			if winner.Content != "Final result processed." || second.Content != test.secondStatus {
				t.Fatalf("unexpected output statuses winner=%v second=%v", winner.Content, second.Content)
			}
		})
	}
}

func TestRunStreamCommittedOutputCannotBeRevokedByRetries(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.StreamEvent {
		return []ai.StreamEvent{
			ai.TextDeltaEvent{Delta: "done"},
			ai.ToolCallStartEvent{ToolName: "retry", ToolCallID: "r"},
			ai.ToolCallDeltaEvent{ArgsDelta: `{}`},
			ai.FinishEvent{},
		}
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "retry", func(context.Context, struct{}) (string, error) {
		return "", ai.Retryf("again")
	})
	result, err := consumeRunStream(t, agent)
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "done" {
		t.Fatalf("retry revoked committed output: %+v", result)
	}
	part := result.Messages()[2].(ai.ModelRequest).Parts[0]
	if _, ok := part.(ai.RetryPromptPart); !ok {
		t.Fatalf("function retry was not preserved: %T", part)
	}
}

func TestRunStreamCommittedExhaustiveOutputFailures(t *testing.T) {
	tests := []struct {
		name      string
		args      string
		validator func(weather) error
		limit     int
		wantRetry bool
		status    string
	}{
		{name: "invalid JSON exceeds retries", args: `{`, limit: 0, status: "Output tool not used - output failed validation."},
		{name: "validator error", args: `{"city":"bad","temp_c":2}`, validator: func(weather) error {
			return errors.New("bad output")
		}, limit: 1, status: "Output tool not used - output failed validation."},
		{name: "validator retry", args: `{"city":"retry","temp_c":2}`, validator: func(weather) error {
			return ai.Retryf("again")
		}, limit: 1, wantRetry: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := newStreamingModel(func([]ai.ModelMessage) []ai.StreamEvent {
				return []ai.StreamEvent{
					ai.ToolCallStartEvent{ToolName: "final_result", ToolCallID: "first"},
					ai.ToolCallDeltaEvent{ArgsDelta: `{"city":"first","temp_c":1}`},
					ai.ToolCallStartEvent{ToolName: "final_result", ToolCallID: "second"},
					ai.ToolCallDeltaEvent{ArgsDelta: test.args},
					ai.FinishEvent{},
				}
			})
			agent := ai.NewAgent[deps, weather](
				model, ai.WithEndStrategy(ai.EndStrategyExhaustive),
				ai.WithRetryLimits(ai.RetryLimits{Tools: 1, Output: test.limit}),
			)
			if test.validator != nil {
				agent.AddOutputValidator(func(_ context.Context, _ *ai.RunContext[deps], output weather) error {
					if output.City == "first" {
						return nil
					}
					return test.validator(output)
				})
			}
			result, err := consumeRunStream(t, agent)
			if err != nil {
				t.Fatal(err)
			}
			part := result.Messages()[2].(ai.ModelRequest).Parts[1]
			if test.wantRetry {
				if _, ok := part.(ai.RetryPromptPart); !ok {
					t.Fatalf("expected retry part, got %T", part)
				}
			} else if part.(ai.ToolReturnPart).Content != test.status {
				t.Fatalf("unexpected failure status: %v", part)
			}
		})
	}
}

func TestRunStreamCommittedOutputCancellation(t *testing.T) {
	for name, cancellation := range map[string]error{
		"canceled": context.Canceled, "deadline": context.DeadlineExceeded,
	} {
		t.Run(name, func(t *testing.T) {
			model := newStreamingModel(func([]ai.ModelMessage) []ai.StreamEvent {
				return []ai.StreamEvent{
					ai.ToolCallStartEvent{ToolName: "final_result", ToolCallID: "first"},
					ai.ToolCallDeltaEvent{ArgsDelta: `{"city":"first","temp_c":1}`},
					ai.ToolCallStartEvent{ToolName: "final_result", ToolCallID: "second"},
					ai.ToolCallDeltaEvent{ArgsDelta: `{"city":"second","temp_c":2}`}, ai.FinishEvent{},
				}
			})
			agent := ai.NewAgent[deps, weather](model, ai.WithEndStrategy(ai.EndStrategyExhaustive))
			agent.AddOutputValidator(func(_ context.Context, _ *ai.RunContext[deps], output weather) error {
				if output.City == "second" {
					return cancellation
				}
				return nil
			})
			if _, err := consumeRunStream(t, agent); !errors.Is(err, cancellation) {
				t.Fatalf("expected %v, got %v", cancellation, err)
			}
		})
	}
}

func TestRunStreamValidationFailuresDoNotRetry(t *testing.T) {
	tests := []struct {
		name   string
		agent  func(*streamingModel) *ai.Agent[deps, weather]
		events []ai.StreamEvent
	}{
		{
			name: "native JSON",
			agent: func(model *streamingModel) *ai.Agent[deps, weather] {
				return ai.NewAgent[deps, weather](model, ai.WithOutputMode(ai.OutputModeNative))
			},
			events: []ai.StreamEvent{ai.TextDeltaEvent{Delta: `{`}, ai.FinishEvent{}},
		},
		{
			name: "output tool JSON",
			agent: func(model *streamingModel) *ai.Agent[deps, weather] {
				return ai.NewAgent[deps, weather](model)
			},
			events: []ai.StreamEvent{
				ai.TextDeltaEvent{Delta: "not an output"},
				ai.ToolCallStartEvent{ToolName: "final_result", ToolCallID: "bad"},
				ai.ToolCallDeltaEvent{ArgsDelta: `{`}, ai.FinishEvent{},
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			requests := 0
			model := newStreamingModel(func([]ai.ModelMessage) []ai.StreamEvent {
				requests++
				return test.events
			})
			_, err := consumeRunStream(t, test.agent(model))
			var unexpected *ai.UnexpectedModelBehaviorError
			if !errors.As(err, &unexpected) || !strings.Contains(err.Error(), "retries are not supported") {
				t.Fatalf("unexpected validation error %v", err)
			}
			if requests != 1 {
				t.Fatalf("streaming validation retried %d requests", requests)
			}
		})
	}
}

func TestRunStreamValidatorFailures(t *testing.T) {
	tests := []struct {
		name       string
		validator  func(weather) error
		unexpected bool
	}{
		{name: "retry", validator: func(weather) error { return ai.Retryf("again") }, unexpected: true},
		{name: "error", validator: func(weather) error { return errors.New("broken") }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := newStreamingModel(func([]ai.ModelMessage) []ai.StreamEvent {
				return []ai.StreamEvent{ai.TextDeltaEvent{Delta: `{"city":"ok","temp_c":1}`}, ai.FinishEvent{}}
			})
			agent := ai.NewAgent[deps, weather](model, ai.WithOutputMode(ai.OutputModeNative))
			agent.AddOutputValidator(func(_ context.Context, _ *ai.RunContext[deps], output weather) error {
				return test.validator(output)
			})
			_, err := consumeRunStream(t, agent)
			var unexpected *ai.UnexpectedModelBehaviorError
			if errors.As(err, &unexpected) != test.unexpected {
				t.Fatalf("unexpected error type: %v", err)
			}
		})
	}
}

func TestRunStreamOutputToolValidatorFailures(t *testing.T) {
	for name, validator := range map[string]func(weather) error{
		"retry": func(weather) error { return ai.Retryf("again") },
		"error": func(weather) error { return errors.New("broken") },
	} {
		t.Run(name, func(t *testing.T) {
			model := newStreamingModel(func([]ai.ModelMessage) []ai.StreamEvent {
				return []ai.StreamEvent{
					ai.ToolCallStartEvent{ToolName: "final_result", ToolCallID: "result"},
					ai.ToolCallDeltaEvent{ArgsDelta: `{"city":"ok","temp_c":1}`}, ai.FinishEvent{},
				}
			})
			agent := ai.NewAgent[deps, weather](model)
			agent.AddOutputValidator(func(_ context.Context, _ *ai.RunContext[deps], output weather) error {
				return validator(output)
			})
			if _, err := consumeRunStream(t, agent); err == nil {
				t.Fatal("expected validator error")
			}
		})
	}
}

func TestRunStreamCommittedFlushErrors(t *testing.T) {
	t.Run("graceful before output", func(t *testing.T) {
		model := newStreamingModel(func([]ai.ModelMessage) []ai.StreamEvent {
			return []ai.StreamEvent{
				ai.ToolCallStartEvent{ToolName: "bad", ToolCallID: "bad"}, ai.ToolCallDeltaEvent{ArgsDelta: `{}`},
				ai.ToolCallStartEvent{ToolName: "final_result", ToolCallID: "result"},
				ai.ToolCallDeltaEvent{ArgsDelta: `{"city":"ok","temp_c":1}`}, ai.FinishEvent{},
			}
		})
		agent := ai.NewAgent[deps, weather](model)
		ai.AddSimpleTool(agent, "bad", func(context.Context, struct{}) (string, error) {
			return "", errors.New("failed")
		})
		if _, err := consumeRunStream(t, agent); err == nil {
			t.Fatal("expected pending batch error")
		}
	})

	t.Run("exhaustive before barrier", func(t *testing.T) {
		model := newStreamingModel(func([]ai.ModelMessage) []ai.StreamEvent {
			return []ai.StreamEvent{
				ai.TextDeltaEvent{Delta: "done"},
				ai.ToolCallStartEvent{ToolName: "bad", ToolCallID: "bad"}, ai.ToolCallDeltaEvent{ArgsDelta: `{}`},
				ai.ToolCallStartEvent{ToolName: "gate", ToolCallID: "gate"}, ai.ToolCallDeltaEvent{ArgsDelta: `{}`},
				ai.FinishEvent{},
			}
		})
		agent := ai.NewAgent[deps, string](model, ai.WithEndStrategy(ai.EndStrategyExhaustive))
		ai.AddSimpleTool(agent, "bad", func(context.Context, struct{}) (string, error) {
			return "", errors.New("failed")
		})
		ai.AddSimpleTool(agent, "gate", func(context.Context, struct{}) (string, error) { return "ok", nil }, ai.WithSequential())
		if _, err := consumeRunStream(t, agent); err == nil {
			t.Fatal("expected pre-barrier batch error")
		}
	})
}

func TestRunStreamCommittedFunctionErrors(t *testing.T) {
	tests := []struct {
		name     string
		strategy ai.EndStrategy
		second   bool
		barrier  bool
	}{
		{name: "graceful final batch", strategy: ai.EndStrategyGraceful},
		{name: "graceful barrier", strategy: ai.EndStrategyGraceful, barrier: true},
		{name: "exhaustive concurrent", strategy: ai.EndStrategyExhaustive, second: true},
		{name: "exhaustive barrier", strategy: ai.EndStrategyExhaustive, barrier: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := newStreamingModel(func([]ai.ModelMessage) []ai.StreamEvent {
				events := []ai.StreamEvent{
					ai.TextDeltaEvent{Delta: "done"},
					ai.ToolCallStartEvent{ToolName: "bad", ToolCallID: "bad"}, ai.ToolCallDeltaEvent{ArgsDelta: `{}`},
				}
				if test.second {
					events = append(events,
						ai.ToolCallStartEvent{ToolName: "good", ToolCallID: "good"}, ai.ToolCallDeltaEvent{ArgsDelta: `{}`},
					)
				}
				return append(events, ai.FinishEvent{})
			})
			agent := ai.NewAgent[deps, string](model, ai.WithEndStrategy(test.strategy))
			opts := []ai.ToolOption{}
			if test.barrier {
				opts = append(opts, ai.WithSequential())
			}
			ai.AddSimpleTool(agent, "bad", func(context.Context, struct{}) (string, error) {
				return "", errors.New("failed")
			}, opts...)
			if test.second {
				ai.AddSimpleTool(agent, "good", func(context.Context, struct{}) (string, error) { return "ok", nil })
			}
			if _, err := consumeRunStream(t, agent); err == nil || !strings.Contains(err.Error(), "failed") {
				t.Fatalf("expected tool error, got %v", err)
			}
		})
	}
}

func consumeRunStream[Output any](
	t *testing.T, agent *ai.Agent[deps, Output],
) (*ai.RunResult[Output], error) {
	t.Helper()
	stream := agent.RunStream(t.Context(), "go", deps{})
	var streamErr error
	for _, err := range stream.Events() {
		if err != nil {
			streamErr = err
		}
	}
	return stream.Result(), streamErr
}
