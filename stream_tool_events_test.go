package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
)

func TestRunStreamFunctionToolEventsUseCallAndCompletionOrder(t *testing.T) {
	model := newStreamingModel(func(messages []ai.ModelMessage) []ai.ModelStreamEvent {
		if len(messages) == 1 {
			return []ai.ModelStreamEvent{
				ai.ToolCallStartEvent{PartID: "slow", ToolName: "slow", ToolCallID: "slow"},
				ai.ToolCallDeltaEvent{PartID: "slow", ArgsDelta: `{}`},
				ai.ToolCallStartEvent{PartID: "fast", ToolName: "fast", ToolCallID: "fast"},
				ai.ToolCallDeltaEvent{PartID: "fast", ArgsDelta: `{}`},
				ai.FinishEvent{},
			}
		}
		return []ai.ModelStreamEvent{ai.TextDeltaEvent{PartID: "text", Delta: "done"}, ai.FinishEvent{}}
	})
	agent := ai.NewAgent[deps, string](model)
	releaseSlow := make(chan struct{})
	ai.AddSimpleTool(agent, "slow", func(context.Context, struct{}) (string, error) {
		<-releaseSlow
		return "slow", nil
	})
	ai.AddSimpleTool(agent, "fast", func(context.Context, struct{}) (string, error) {
		return "fast", nil
	})

	stream := agent.RunStream(t.Context(), "go", deps{})
	var calls, results []string
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		switch event := event.(type) {
		case ai.FunctionToolCallEvent:
			if event.ArgsValid != nil {
				t.Fatalf("argument validity should be unknown without argument validators: %+v", event)
			}
			calls = append(calls, event.Part.ToolName)
		case ai.FunctionToolResultEvent:
			part := event.Part.(ai.ToolReturnPart)
			results = append(results, part.ToolName)
			if part.ToolName == "fast" {
				close(releaseSlow)
			}
		}
	}
	if !slices.Equal(calls, []string{"slow", "fast"}) {
		t.Fatalf("call events lost model order: %v", calls)
	}
	if !slices.Equal(results, []string{"fast", "slow"}) {
		t.Fatalf("result events lost completion order: %v", results)
	}
	if stream.Result().Output != "done" {
		t.Fatalf("unexpected result: %+v", stream.Result())
	}
}

type collectingStreamCapability struct {
	events []ai.StreamEvent
}

func (*collectingStreamCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (c *collectingStreamCapability) ProcessStreamEvent(
	_ context.Context, _ *ai.RunInfo, event ai.StreamEvent,
) (ai.StreamEvent, error) {
	c.events = append(c.events, event)
	return event, nil
}

func TestRunEventProcessorObservesOutputToolRetryWins(t *testing.T) {
	model := newStreamingModel(func(messages []ai.ModelMessage) []ai.ModelStreamEvent {
		city := "first"
		if len(messages) > 1 {
			city = "second"
		}
		events := []ai.ModelStreamEvent{
			ai.ToolCallStartEvent{PartID: "output", ToolName: "final_result", ToolCallID: city},
			ai.ToolCallDeltaEvent{
				PartID: "output", ArgsDelta: `{"city":"` + city + `","temp_c":1}`,
			},
		}
		if city == "first" {
			events = append(events,
				ai.ToolCallStartEvent{PartID: "retry", ToolName: "retry", ToolCallID: "retry"},
				ai.ToolCallDeltaEvent{PartID: "retry", ArgsDelta: `{}`},
			)
		}
		return append(events, ai.FinishEvent{})
	})
	collector := &collectingStreamCapability{}
	agent := ai.NewAgent[deps, weather](model, ai.WithCapabilities(collector))
	ai.AddSimpleTool(agent, "retry", func(context.Context, struct{}) (string, error) {
		return "", ai.Retryf("again")
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	var outputCalls, outputResults int
	var retryWon bool
	for _, rawEvent := range collector.events {
		switch event := rawEvent.(type) {
		case ai.OutputToolCallEvent:
			outputCalls++
			if event.Part.ToolName != "final_result" {
				t.Fatalf("unexpected output call: %+v", event)
			}
		case ai.OutputToolResultEvent:
			outputResults++
			if part, ok := event.Part.(ai.ToolReturnPart); ok &&
				part.Content == "Output not used as the final result - addressing tool retries from this round first." {
				retryWon = true
			}
		case ai.FunctionToolResultEvent:
			if _, ok := event.Part.(ai.RetryPromptPart); !ok {
				t.Fatalf("expected function retry result, got %T", event.Part)
			}
		}
	}
	if outputCalls != 2 || outputResults != 2 || !retryWon {
		t.Fatalf("unexpected output events: calls=%d results=%d retryWon=%v", outputCalls, outputResults, retryWon)
	}
	if result.Output.City != "second" {
		t.Fatalf("unexpected final output: %+v", result.Output)
	}
}

type rejectToolEventCapability struct {
	target string
}

func (*rejectToolEventCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (c *rejectToolEventCapability) ProcessStreamEvent(
	_ context.Context, _ *ai.RunInfo, event ai.StreamEvent,
) (ai.StreamEvent, error) {
	kind := ""
	switch event.(type) {
	case ai.FunctionToolCallEvent:
		kind = "function-call"
	case ai.FunctionToolResultEvent:
		kind = "function-result"
	case ai.OutputToolCallEvent:
		kind = "output-call"
	case ai.OutputToolResultEvent:
		kind = "output-result"
	}
	if kind == c.target {
		return nil, errors.New("rejected " + kind)
	}
	return event, nil
}

func TestRunStreamConsumerBreaksOnToolEvents(t *testing.T) {
	for _, target := range []string{"function-call", "function-result"} {
		t.Run(target, func(t *testing.T) {
			model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
				return []ai.ModelStreamEvent{
					ai.ToolCallStartEvent{PartID: "work", ToolName: "work", ToolCallID: "work"},
					ai.ToolCallDeltaEvent{PartID: "work", ArgsDelta: `{}`}, ai.FinishEvent{},
				}
			})
			agent := ai.NewAgent[deps, string](model)
			called := false
			ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) {
				called = true
				return "done", nil
			})
			stream := agent.RunStream(t.Context(), "go", deps{})
			for event := range stream.Events() {
				kind := ""
				switch event.(type) {
				case ai.FunctionToolCallEvent:
					kind = "function-call"
				case ai.FunctionToolResultEvent:
					kind = "function-result"
				}
				if kind == target {
					break
				}
			}
			if stream.Result() != nil || called != (target == "function-result") {
				t.Fatalf("unexpected break behavior: result=%+v called=%v", stream.Result(), called)
			}
		})
	}
}

func TestToolEventProcessorErrorsStopOutputPaths(t *testing.T) {
	t.Run("committed output call", func(t *testing.T) {
		capability := &rejectToolEventCapability{target: "output-call"}
		agent := ai.NewAgent[deps, weather](newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
			return []ai.ModelStreamEvent{
				ai.ToolCallStartEvent{PartID: "output", ToolName: "final_result", ToolCallID: "output"},
				ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `{"city":"x","temp_c":1}`}, ai.FinishEvent{},
			}
		}), ai.WithCapabilities(capability))
		stream := agent.RunStream(t.Context(), "go", deps{})
		var got error
		for _, err := range stream.Events() {
			if err != nil {
				got = err
			}
		}
		if got == nil || !strings.Contains(got.Error(), "output-call") {
			t.Fatalf("expected output-call processor error, got %v", got)
		}
	})

	t.Run("regular run output result", func(t *testing.T) {
		capability := &rejectToolEventCapability{target: "output-result"}
		agent := ai.NewAgent[deps, weather](newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
			return []ai.ModelStreamEvent{
				ai.ToolCallStartEvent{PartID: "output", ToolName: "final_result", ToolCallID: "output"},
				ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `{"city":"x","temp_c":1}`}, ai.FinishEvent{},
			}
		}), ai.WithCapabilities(capability))
		if _, err := agent.Run(t.Context(), "go", deps{}); err == nil || !strings.Contains(err.Error(), "output-result") {
			t.Fatalf("expected output-result processor error, got %v", err)
		}
	})

	for _, strategy := range []ai.EndStrategy{
		ai.EndStrategyEarly, ai.EndStrategyGraceful, ai.EndStrategyExhaustive,
	} {
		t.Run("committed "+string(strategy), func(t *testing.T) {
			capability := &rejectToolEventCapability{target: "output-result"}
			agent := ai.NewAgent[deps, weather](newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
				return []ai.ModelStreamEvent{
					ai.ToolCallStartEvent{PartID: "output", ToolName: "final_result", ToolCallID: "output"},
					ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `{"city":"x","temp_c":1}`}, ai.FinishEvent{},
				}
			}), ai.WithCapabilities(capability), ai.WithEndStrategy(strategy))
			stream := agent.RunStream(t.Context(), "go", deps{})
			var got error
			for _, err := range stream.Events() {
				if err != nil {
					got = err
				}
			}
			if got == nil || !strings.Contains(got.Error(), "output-result") {
				t.Fatalf("expected committed output-result error, got %v", got)
			}
		})
	}
}

func TestEarlyNativeToolEventProcessorErrors(t *testing.T) {
	for _, target := range []string{"function-call", "function-result"} {
		t.Run(target, func(t *testing.T) {
			capability := &rejectToolEventCapability{target: target}
			agent := ai.NewAgent[deps, weather](newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
				return []ai.ModelStreamEvent{
					ai.TextDeltaEvent{PartID: "output", Delta: `{"city":"x","temp_c":1}`},
					ai.ToolCallStartEvent{PartID: "work", ToolName: "work", ToolCallID: "work"},
					ai.ToolCallDeltaEvent{PartID: "work", ArgsDelta: `{}`}, ai.FinishEvent{},
				}
			}), ai.WithCapabilities(capability), ai.WithOutputMode(ai.OutputModeNative),
				ai.WithEndStrategy(ai.EndStrategyEarly))
			ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) {
				t.Fatal("early native output executed skipped tool")
				return "", nil
			})
			if _, err := agent.Run(t.Context(), "go", deps{}); err == nil || !strings.Contains(err.Error(), target) {
				t.Fatalf("expected %s processor error, got %v", target, err)
			}
		})
	}
}

func TestRunStreamEarlyCommittedOutputEmitsSkippedToolResult(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.TextDeltaEvent{PartID: "text", Delta: "done"},
			ai.ToolCallStartEvent{PartID: "work", ToolName: "work", ToolCallID: "work"},
			ai.ToolCallDeltaEvent{PartID: "work", ArgsDelta: string(json.RawMessage(`{}`))},
			ai.FinishEvent{},
		}
	})
	agent := ai.NewAgent[deps, string](model, ai.WithEndStrategy(ai.EndStrategyEarly))
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) {
		t.Fatal("skipped tool executed")
		return "", nil
	})
	stream := agent.RunStream(t.Context(), "go", deps{})
	var result ai.FunctionToolResultEvent
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(ai.FunctionToolResultEvent); ok {
			result = event
		}
	}
	part := result.Part.(ai.ToolReturnPart)
	if part.Content != "Tool not executed - a final result was already processed." {
		t.Fatalf("unexpected skipped result event: %+v", part)
	}
}
