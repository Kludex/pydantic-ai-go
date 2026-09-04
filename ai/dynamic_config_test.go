package ai_test

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type dynamicConfigCapability struct{}

func (dynamicConfigCapability) Setup(reg *ai.CapabilityRegistry) error {
	topP := 0.4
	reg.AddInstructions("Capability static.")
	reg.AddModelSettings(ai.ModelSettings{TopP: &topP})
	return nil
}

func (dynamicConfigCapability) ModelSettings(
	_ context.Context, ri *ai.RunInfo, current ai.ModelSettings,
) (ai.ModelSettings, error) {
	seed := current.MaxTokens
	if current.TopP == nil || *current.TopP != 0.4 {
		return ai.ModelSettings{}, fmt.Errorf("static capability settings missing: %+v", current)
	}
	return ai.ModelSettings{Seed: &seed, StopSequences: []string{fmt.Sprintf("cap-%d", ri.Usage().Requests)}}, nil
}

func (dynamicConfigCapability) Instructions(_ context.Context, ri *ai.RunInfo) (string, error) {
	return fmt.Sprintf("Capability dynamic %d.", ri.Usage().Requests), nil
}

func TestDynamicSettingsAndInstructionsResolveEveryStep(t *testing.T) {
	baseTemperature := 0.1
	runTemperature := 0.9
	var got []ai.ModelRequestParams
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		got = append(got, params)
		if len(got) == 1 {
			return &ai.ModelResponse{
				Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "work", ToolCallID: "call", Args: []byte(`{}`)}},
				Usage: ai.Usage{Requests: 1},
			}, nil
		}
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}, Usage: ai.Usage{Requests: 1},
		}, nil
	})
	agent := ai.NewAgent[deps, string](
		model,
		ai.WithInstructions("Agent static."),
		ai.WithModelSettings(ai.ModelSettings{MaxTokens: 10, Temperature: &baseTemperature}),
		ai.WithCapabilities(dynamicConfigCapability{}),
	)
	agent.AddModelSettingsFunc(func(_ context.Context, rc *ai.RunContext[deps]) (ai.ModelSettings, error) {
		if rc.ModelSettings.MaxTokens != 10 || rc.Deps.Location != "deps" {
			t.Fatalf("agent settings callback saw wrong layer: %+v", rc)
		}
		return ai.ModelSettings{MaxTokens: 100 + rc.Usage().Requests}, nil
	})
	agent.AddInstructionsFunc(func(_ context.Context, rc *ai.RunContext[deps]) (string, error) {
		return fmt.Sprintf("Agent dynamic %d/%d.", rc.Usage().Requests, rc.ModelSettings.MaxTokens), nil
	})
	var preparedSettings []ai.ModelSettings
	agent.AddToolsPrepareFunc(func(
		_ context.Context, rc *ai.RunContext[deps], tools []ai.ToolDefinition,
	) ([]ai.ToolDefinition, error) {
		preparedSettings = append(preparedSettings, rc.ModelSettings)
		return tools, nil
	})
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) { return "done", nil })

	result, err := agent.Run(
		t.Context(), "go", deps{Location: "deps"},
		ai.WithRunInstructions("Run static."),
		ai.WithRunModelSettings(ai.ModelSettings{Temperature: &runTemperature}),
		ai.WithRunModelSettingsFunc(func(_ context.Context, rc *ai.RunContext[deps]) (ai.ModelSettings, error) {
			if rc.ModelSettings.Seed == nil || *rc.ModelSettings.Seed != 100+rc.Usage().Requests ||
				rc.ModelSettings.Temperature == nil || *rc.ModelSettings.Temperature != runTemperature {
				t.Fatalf("run settings callback saw wrong layers: %+v", rc.ModelSettings)
			}
			return ai.ModelSettings{MaxTokens: 200 + rc.Usage().Requests}, nil
		}),
		ai.WithRunInstructionsFunc(func(_ context.Context, rc *ai.RunContext[deps]) (string, error) {
			return fmt.Sprintf("Run dynamic %d/%d.", rc.Usage().Requests, rc.ModelSettings.MaxTokens), nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "done" || len(got) != 2 || len(preparedSettings) != 2 {
		t.Fatalf("unexpected dynamic run: result=%+v requests=%d prepares=%d", result, len(got), len(preparedSettings))
	}
	for index, params := range got {
		if params.Settings.MaxTokens != 200+index || params.Settings.Seed == nil ||
			*params.Settings.Seed != 100+index || params.Settings.Temperature == nil ||
			*params.Settings.Temperature != runTemperature ||
			!reflect.DeepEqual(params.Settings.StopSequences, []string{fmt.Sprintf("cap-%d", index)}) {
			t.Fatalf("unexpected settings for step %d: %+v", index, params.Settings)
		}
		wantInstructions := fmt.Sprintf(
			"Agent static.\n\nCapability static.\n\nRun static.\n\nAgent dynamic %d/%d.\n\nCapability dynamic %d.\n\nRun dynamic %d/%d.",
			index, 200+index, index, index, 200+index,
		)
		if params.Instructions != wantInstructions {
			t.Fatalf("unexpected instructions for step %d: %q", index, params.Instructions)
		}
		if preparedSettings[index].MaxTokens != 200+index {
			t.Fatalf("tool preparation saw stale settings for step %d: %+v", index, preparedSettings[index])
		}
	}
}

type failingSettingsCapability struct{}

func (failingSettingsCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (failingSettingsCapability) ModelSettings(
	context.Context, *ai.RunInfo, ai.ModelSettings,
) (ai.ModelSettings, error) {
	return ai.ModelSettings{}, errors.New("capability settings failed")
}

func TestDynamicModelSettingsErrors(t *testing.T) {
	tests := map[string]struct {
		agent   *ai.Agent[deps, string]
		options []ai.RunOption
		want    string
	}{
		"agent": {
			agent: func() *ai.Agent[deps, string] {
				agent := ai.NewAgent[deps, string](fakes.NewTestModel())
				agent.AddModelSettingsFunc(func(context.Context, *ai.RunContext[deps]) (ai.ModelSettings, error) {
					return ai.ModelSettings{}, errors.New("agent settings failed")
				})
				return agent
			}(),
			want: "ai: model settings: agent settings failed",
		},
		"capability": {
			agent: ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(failingSettingsCapability{})),
			want:  "ai: model settings: capability settings failed",
		},
		"run": {
			agent: ai.NewAgent[deps, string](fakes.NewTestModel()),
			options: []ai.RunOption{ai.WithRunModelSettingsFunc(
				func(context.Context, *ai.RunContext[deps]) (ai.ModelSettings, error) {
					return ai.ModelSettings{}, errors.New("run settings failed")
				},
			)},
			want: "ai: model settings: run settings failed",
		},
		"dependency mismatch": {
			agent: ai.NewAgent[deps, string](fakes.NewTestModel()),
			options: []ai.RunOption{ai.WithRunModelSettingsFunc(
				func(context.Context, *ai.RunContext[string]) (ai.ModelSettings, error) {
					return ai.ModelSettings{}, nil
				},
			)},
			want: "ai: model settings: ai: run model settings dependencies do not match agent",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if _, err := test.agent.Run(t.Context(), "go", deps{}, test.options...); err == nil || err.Error() != test.want {
				t.Fatalf("got error %v, want %q", err, test.want)
			}
		})
	}
}

func TestRunInstructionsFuncErrors(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	_, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunInstructionsFunc(
		func(context.Context, *ai.RunContext[deps]) (string, error) {
			return "", errors.New("run instructions failed")
		},
	))
	if err == nil || err.Error() != "ai: instructions: run instructions failed" {
		t.Fatalf("unexpected run instruction error: %v", err)
	}

	_, err = agent.Run(t.Context(), "go", deps{}, ai.WithRunInstructionsFunc(
		func(context.Context, *ai.RunContext[string]) (string, error) { return "", nil },
	))
	if err == nil || err.Error() != "ai: instructions: ai: run instructions dependencies do not match agent" {
		t.Fatalf("unexpected dependency error: %v", err)
	}
}
