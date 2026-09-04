package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type rejectEnqueuedCapability struct{}

func (rejectEnqueuedCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (rejectEnqueuedCapability) ProcessStreamEvent(
	_ context.Context, _ *ai.RunInfo, event ai.StreamEvent,
) (ai.StreamEvent, error) {
	if _, ok := event.(ai.EnqueuedMessagesEvent); ok {
		return nil, errors.New("rejected enqueued messages")
	}
	return event, nil
}

func TestStreamedOutputRedirectsForEnqueuedMessage(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: []string{"first", "second"}[request-1]}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	validated := 0
	agent.AddOutputValidator(func(_ context.Context, rc *ai.RunContext[deps], _ string) error {
		validated++
		if validated == 1 {
			_, err := rc.Enqueue(ai.TextContent{Text: "steer stream"})
			return err
		}
		return nil
	})
	stream := agent.RunStream(t.Context(), "go", deps{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	if stream.Result() == nil || stream.Result().Output != "second" || request != 2 {
		t.Fatalf("unexpected streamed enqueue redirect result=%+v requests=%d", stream.Result(), request)
	}
}

func TestStreamedOutputRedirectCanStopAtEnqueuedMessage(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "first"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddOutputValidator(func(_ context.Context, rc *ai.RunContext[deps], _ string) error {
		_, err := rc.Enqueue(ai.TextContent{Text: "steer stream"})
		return err
	})
	stream := agent.RunStream(t.Context(), "go", deps{})
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(ai.EnqueuedMessagesEvent); ok {
			break
		}
	}
	if stream.Result() != nil {
		t.Fatalf("stopped redirect stream should have no result: %+v", stream.Result())
	}
}

func TestTextOutputRedirectPropagatesEnqueuedEventError(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "first"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(rejectEnqueuedCapability{}))
	agent.AddOutputValidator(func(_ context.Context, rc *ai.RunContext[deps], _ string) error {
		_, err := rc.Enqueue(ai.TextContent{Text: "steer text"})
		return err
	})
	_, err := agent.Run(t.Context(), "go", deps{})
	if err == nil || !strings.Contains(err.Error(), "rejected enqueued messages") {
		t.Fatalf("unexpected text redirect event error: %v", err)
	}
}

func TestEarlyNativeOutputRedirectsForEnqueuedMessage(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.TextPart{Content: `{"value":1}`},
				ai.ToolCallPart{ToolName: "work", ToolCallID: "work", Args: json.RawMessage(`{}`)},
			}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"value":2}`}}}, nil
	})
	agent := ai.NewAgent[deps, strategyOutput](model, ai.WithOutputMode(ai.OutputModeNative), ai.WithEndStrategy(ai.EndStrategyEarly))
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) {
		return "unreachable", nil
	})
	validated := 0
	agent.AddOutputValidator(func(_ context.Context, rc *ai.RunContext[deps], _ strategyOutput) error {
		validated++
		if validated == 1 {
			_, err := rc.Enqueue(ai.TextContent{Text: "steer native"})
			return err
		}
		return nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output.Value != 2 || request != 2 {
		t.Fatalf("unexpected native enqueue redirect result=%+v requests=%d err=%v", result, request, err)
	}
}

func TestEarlyNativeRedirectPropagatesEnqueuedEventError(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.TextPart{Content: `{"value":1}`},
			ai.ToolCallPart{ToolName: "work", ToolCallID: "work", Args: json.RawMessage(`{}`)},
		}}, nil
	})
	agent := ai.NewAgent[deps, strategyOutput](
		model, ai.WithCapabilities(rejectEnqueuedCapability{}),
		ai.WithOutputMode(ai.OutputModeNative), ai.WithEndStrategy(ai.EndStrategyEarly),
	)
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) { return "unreachable", nil })
	agent.AddOutputValidator(func(_ context.Context, rc *ai.RunContext[deps], _ strategyOutput) error {
		_, err := rc.Enqueue(ai.TextContent{Text: "steer native"})
		return err
	})
	_, err := agent.Run(t.Context(), "go", deps{})
	if err == nil || !strings.Contains(err.Error(), "rejected enqueued messages") {
		t.Fatalf("unexpected native redirect event error: %v", err)
	}
}

func TestOutputToolRedirectsForEnqueuedMessage(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "final_result", ToolCallID: "output", Args: json.RawMessage([]string{`{"value":1}`, `{"value":2}`}[request-1]),
		}}}, nil
	})
	agent := ai.NewAgent[deps, strategyOutput](model)
	validated := 0
	agent.AddOutputValidator(func(_ context.Context, rc *ai.RunContext[deps], _ strategyOutput) error {
		validated++
		if validated == 1 {
			_, err := rc.Enqueue(ai.TextContent{Text: "steer output"})
			return err
		}
		return nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output.Value != 2 || request != 2 {
		t.Fatalf("unexpected output-tool enqueue redirect result=%+v requests=%d err=%v", result, request, err)
	}
}

func TestOutputToolRedirectPropagatesEnqueuedEventError(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "final_result", ToolCallID: "output", Args: json.RawMessage(`{"value":1}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, strategyOutput](model, ai.WithCapabilities(rejectEnqueuedCapability{}))
	agent.AddOutputValidator(func(_ context.Context, rc *ai.RunContext[deps], _ strategyOutput) error {
		_, err := rc.Enqueue(ai.TextContent{Text: "steer output"})
		return err
	})
	_, err := agent.Run(t.Context(), "go", deps{})
	if err == nil || !strings.Contains(err.Error(), "rejected enqueued messages") {
		t.Fatalf("unexpected output redirect event error: %v", err)
	}
}
