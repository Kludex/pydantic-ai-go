package ai_test

import (
	"context"
	"errors"
	"reflect"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func textModel(name, output string, inspect func(ai.ModelRequestParams)) ai.Model {
	return namedFunctionModel{
		name: name,
		request: func(_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
			if inspect != nil {
				inspect(params)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: output}}}, nil
		},
	}
}

type namedFunctionModel struct {
	name    string
	request func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error)
}

func (m namedFunctionModel) Name() string { return m.name }

func (m namedFunctionModel) Request(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return m.request(ctx, messages, params)
}

func TestRunOverridesModelWithoutMutatingAgent(t *testing.T) {
	agent := ai.NewAgent[deps, string](textModel("default", "default", nil))
	override := textModel("override", "override", nil)

	result, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunModel(override))
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "override" || result.Messages()[1].(ai.ModelResponse).ModelName != "override" {
		t.Fatalf("run did not use override model: %+v", result)
	}

	result, err = agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "default" || result.Messages()[1].(ai.ModelResponse).ModelName != "default" {
		t.Fatalf("override mutated agent model: %+v", result)
	}
}

func TestRunStreamOverridesModel(t *testing.T) {
	defaultModel := textModel("default", "default", func(ai.ModelRequestParams) {
		t.Fatal("default model was called")
	})
	agent := ai.NewAgent[deps, string](defaultModel)
	stream := agent.RunStream(t.Context(), "go", deps{}, ai.WithRunModel(textModel("stream", "streamed", nil)))
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	if stream.Result() == nil || stream.Result().Output != "streamed" {
		t.Fatalf("stream did not use override model: %+v", stream.Result())
	}
}

func TestRunMergesSettingsAndAppendsInstructions(t *testing.T) {
	baseTemperature := 0.2
	runTemperature := 0.8
	runTopP := 0.7
	runSeed := 42
	runParallel := false
	var got ai.ModelRequestParams
	model := textModel("settings", "ok", func(params ai.ModelRequestParams) { got = params })
	agent := ai.NewAgent[deps, string](
		model,
		ai.WithInstructions("Base instructions."),
		ai.WithModelSettings(ai.ModelSettings{
			MaxTokens: 100, Temperature: &baseTemperature, StopSequences: []string{"base"},
		}),
	)
	agent.AddInstructionsFunc(func(_ context.Context, runContext *ai.RunContext[deps]) (string, error) {
		if runContext.Model.Name() != "settings" || runContext.ModelSettings.MaxTokens != 200 ||
			runContext.UsageLimits.RequestLimit != 3 {
			t.Fatalf("dynamic instructions saw stale run configuration: %+v", runContext)
		}
		return "Dynamic instructions.", nil
	})
	result, err := agent.Run(
		t.Context(), "go", deps{},
		ai.WithRunInstructions("Run instructions."),
		ai.WithRunUsageLimits(ai.UsageLimits{RequestLimit: 3}),
		ai.WithRunModelSettings(ai.ModelSettings{
			MaxTokens: 200, Temperature: &runTemperature, TopP: &runTopP, Seed: &runSeed,
			StopSequences: []string{}, ParallelToolCalls: &runParallel,
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "ok" ||
		got.Instructions != "Base instructions.\n\nRun instructions.\n\nDynamic instructions." {
		t.Fatalf("unexpected run parameters: output=%q instructions=%q", result.Output, got.Instructions)
	}
	wantInstructions := []ai.InstructionPart{
		{Content: "Base instructions.", ID: ai.AgentInstructionID()},
		{Content: "Run instructions."},
		{Content: "Dynamic instructions.", Dynamic: true},
	}
	if !reflect.DeepEqual(got.InstructionParts, wantInstructions) {
		t.Fatalf("instruction boundaries were lost: got=%+v want=%+v", got.InstructionParts, wantInstructions)
	}
	want := ai.ModelSettings{
		MaxTokens: 200, Temperature: &runTemperature, TopP: &runTopP, Seed: &runSeed,
		StopSequences: []string{}, ParallelToolCalls: &runParallel,
	}
	if !reflect.DeepEqual(got.Settings, want) {
		t.Fatalf("settings were not merged: got=%+v want=%+v", got.Settings, want)
	}
}

func TestRunSettingsInheritUnspecifiedAgentDefaults(t *testing.T) {
	temperature := 0.2
	topP := 0.7
	var got ai.ModelSettings
	agent := ai.NewAgent[deps, string](
		textModel("settings", "ok", func(params ai.ModelRequestParams) { got = params.Settings }),
		ai.WithModelSettings(ai.ModelSettings{MaxTokens: 100, Temperature: &temperature, StopSequences: []string{"base"}}),
	)
	if _, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunModelSettings(ai.ModelSettings{TopP: &topP})); err != nil {
		t.Fatal(err)
	}
	if got.MaxTokens != 100 || got.Temperature == nil || *got.Temperature != temperature ||
		got.TopP == nil || *got.TopP != topP || !reflect.DeepEqual(got.StopSequences, []string{"base"}) {
		t.Fatalf("unspecified settings did not inherit defaults: %+v", got)
	}
}

func TestRunOverridesAndDisablesAgentUsageLimits(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "ok"}}, Usage: ai.Usage{Requests: 1, InputTokens: 5},
		}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithUsageLimits(ai.UsageLimits{InputTokenLimit: 1}))

	if _, err := agent.Run(
		t.Context(), "go", deps{}, ai.WithRunUsageLimits(ai.UsageLimits{InputTokenLimit: 5}),
	); err != nil {
		t.Fatalf("run limit did not replace agent limit: %v", err)
	}
	if _, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunUsageLimits(ai.UsageLimits{})); err != nil {
		t.Fatalf("zero run limit did not disable agent limit: %v", err)
	}
	if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, ai.ErrUsageLimitExceeded) {
		t.Fatalf("agent limit was mutated: %v", err)
	}
}

func TestRunOverridesStructuredOutputMode(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if params.OutputSchema != nil && params.OutputTool == nil {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"city":"Oslo","temp_c":3}`}}}, nil
		}
		if params.OutputTool == nil || params.OutputSchema != nil {
			t.Fatalf("unexpected output mode: tool=%+v schema=%+v", params.OutputTool, params.OutputSchema)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: params.OutputTool.Name, ToolCallID: "result", Args: []byte(`{"city":"Oslo","temp_c":3}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, weather](model)

	result, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunOutputMode(ai.OutputModeNative))
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != (weather{City: "Oslo", TempC: 3}) {
		t.Fatalf("native override returned %+v", result.Output)
	}
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatalf("output override mutated agent mode: %v", err)
	}
}

func TestInvalidOutputModesPanic(t *testing.T) {
	for name, build := range map[string]func(){
		"agent": func() {
			ai.NewAgent[deps, weather](fakes.NewTestModel(), ai.WithOutputMode(ai.OutputMode(99)))
		},
		"run": func() {
			agent := ai.NewAgent[deps, weather](fakes.NewTestModel())
			if _, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunOutputMode(ai.OutputMode(99))); err != nil {
				t.Fatal(err)
			}
		},
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected invalid output mode panic")
				}
			}()
			build()
		})
	}
}
