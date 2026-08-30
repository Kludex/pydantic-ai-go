package ai_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestOutputToolConfigurationAndPreparation(t *testing.T) {
	strict := true
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if params.OutputTool == nil || params.OutputTool.Name != "result_step" ||
			params.OutputTool.Description != "Return the weather." || !params.OutputTool.Sequential ||
			params.OutputTool.Strict == nil || !*params.OutputTool.Strict ||
			params.OutputTool.Metadata["step"] != request {
			t.Fatalf("unexpected prepared output tool on request %d: %+v", request, params.OutputTool)
		}
		args := []byte(`{"city":`)
		if request == 2 {
			args = []byte(`{"city":"Oslo","temp_c":3}`)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: params.OutputTool.Name, ToolCallID: "result", Args: args,
		}}}, nil
	})
	agent := ai.NewAgent[deps, weather](model, ai.WithOutputTool(ai.OutputToolConfig{
		Name: "weather_result", Description: "Return the weather.", Sequential: true, Strict: &strict,
	}))
	strict = false
	agent.AddOutputToolPrepareFunc(func(
		_ context.Context, rc *ai.RunContext[deps], tool ai.ToolDefinition,
	) (*ai.ToolDefinition, error) {
		if tool.Name != "weather_result" || tool.Metadata != nil {
			t.Fatalf("output definition leaked between requests: %+v", tool)
		}
		tool.Name = "result_step"
		tool.Metadata = map[string]any{"step": rc.RunStep}
		return &tool, nil
	})
	result, err := agent.Run(t.Context(), "weather", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != (weather{City: "Oslo", TempC: 3}) || request != 2 {
		t.Fatalf("unexpected output result=%+v requests=%d", result.Output, request)
	}
}

func TestConfiguredOutputToolStreams(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.ToolCallStartEvent{PartID: "output", ToolName: "weather_result", ToolCallID: "result"},
			ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `{"city":"Oslo","temp_c":3}`},
			ai.FinishEvent{},
		}
	})
	agent := ai.NewAgent[deps, weather](model, ai.WithOutputTool(ai.OutputToolConfig{Name: "weather_result"}))
	stream := agent.RunStream(t.Context(), "weather", deps{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	if stream.Result() == nil || stream.Result().Output.TempC != 3 {
		t.Fatalf("configured output tool did not stream: %+v", stream.Result())
	}
}

func TestOutputToolPreparationMayOmitOneStep(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			if params.OutputTool != nil {
				t.Fatalf("output tool was not omitted: %+v", params.OutputTool)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "not structured"}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: params.OutputTool.Name, ToolCallID: "result", Args: []byte(`{"city":"Oslo","temp_c":3}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, weather](model)
	agent.AddOutputToolPrepareFunc(func(
		_ context.Context, rc *ai.RunContext[deps], tool ai.ToolDefinition,
	) (*ai.ToolDefinition, error) {
		if rc.RunStep == 1 {
			return nil, nil
		}
		return &tool, nil
	})
	result, err := agent.Run(t.Context(), "weather", deps{})
	if err != nil || result.Output.TempC != 3 || request != 2 {
		t.Fatalf("output omission did not recover: result=%+v err=%v requests=%d", result, err, request)
	}
}

func TestOutputToolPreparationErrors(t *testing.T) {
	tests := map[string]struct {
		prepare ai.OutputToolPrepareFunc[deps]
		want    string
	}{
		"error": {
			prepare: func(context.Context, *ai.RunContext[deps], ai.ToolDefinition) (*ai.ToolDefinition, error) {
				return nil, errors.New("prepare failed")
			},
			want: "ai: prepare output tool: prepare failed",
		},
		"empty name": {
			prepare: func(
				_ context.Context, _ *ai.RunContext[deps], tool ai.ToolDefinition,
			) (*ai.ToolDefinition, error) {
				tool.Name = ""
				return &tool, nil
			},
			want: "ai: prepared output tool name must not be empty",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			agent := ai.NewAgent[deps, weather](fakes.NewTestModel())
			agent.AddOutputToolPrepareFunc(test.prepare)
			if _, err := agent.Run(t.Context(), "go", deps{}); err == nil || err.Error() != test.want {
				t.Fatalf("got error %v, want %q", err, test.want)
			}
		})
	}
}

func TestSequentialOutputToolIsExecutionBarrier(t *testing.T) {
	validatorStarted := make(chan struct{})
	releaseValidator := make(chan struct{})
	toolStarted := make(chan struct{})
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{
				ToolName: params.OutputTool.Name, ToolCallID: "result",
				Args: []byte(`{"city":"Oslo","temp_c":3}`),
			},
			ai.ToolCallPart{ToolName: "work", ToolCallID: "work", Args: []byte(`{}`)},
		}}, nil
	})
	agent := ai.NewAgent[deps, weather](
		model,
		ai.WithEndStrategy(ai.EndStrategyExhaustive),
		ai.WithOutputTool(ai.OutputToolConfig{Sequential: true}),
	)
	agent.AddOutputValidator(func(context.Context, *ai.RunContext[deps], weather) error {
		close(validatorStarted)
		<-releaseValidator
		return nil
	})
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) {
		close(toolStarted)
		return "done", nil
	})
	go func() {
		<-validatorStarted
		select {
		case <-toolStarted:
			t.Error("function tool overlapped sequential output validation")
		case <-time.After(20 * time.Millisecond):
		}
		close(releaseValidator)
	}()
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output.TempC != 3 {
		t.Fatalf("unexpected sequential output result=%+v err=%v", result, err)
	}
	select {
	case <-toolStarted:
	default:
		t.Fatal("function tool did not run after output validation")
	}
}
