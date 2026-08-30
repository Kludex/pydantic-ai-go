package ai_test

import (
	"context"
	"slices"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type modelWithDefaults struct {
	ai.Model
	defaults ai.ModelSettings
}

func (m modelWithDefaults) DefaultModelSettings() ai.ModelSettings { return m.defaults }

func TestSelectedModelDefaultsResolveBeforeAgentAndRunSettings(t *testing.T) {
	temperature := 0.2
	topP := 0.7
	var requested []ai.ModelSettings
	modelA := modelWithDefaults{Model: fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requested = append(requested, params.Settings.Clone())
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "work", ToolCallID: "call", Args: []byte(`{}`),
		}}}, nil
	}), defaults: ai.ModelSettings{MaxTokens: 100, Temperature: &temperature, StopSequences: []string{"a"}}}
	modelB := modelWithDefaults{Model: fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requested = append(requested, params.Settings.Clone())
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	}), defaults: ai.ModelSettings{MaxTokens: 200, Temperature: &temperature, StopSequences: []string{"b"}}}

	agentTemperature := 0.4
	agent := ai.NewAgent[deps, string](modelA, ai.WithModelSettings(ai.ModelSettings{Temperature: &agentTemperature}))
	agent.AddModelSelector(func(
		_ context.Context, selection ai.ModelSelectionContext[deps],
	) (ai.ModelSelection, error) {
		if selection.Step == 2 {
			return ai.ModelSelection{Model: modelB}, nil
		}
		return ai.ModelSelection{}, nil
	})
	var seen []ai.ModelSettings
	agent.AddModelSettingsFunc(func(_ context.Context, rc *ai.RunContext[deps]) (ai.ModelSettings, error) {
		seen = append(seen, rc.ModelSettings.Clone())
		return ai.ModelSettings{TopP: &topP}, nil
	})
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) { return "ok", nil })
	runMaxTokens := 300
	result, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunModelSettings(ai.ModelSettings{MaxTokens: runMaxTokens}))
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected adaptive-default result=%+v err=%v", result, err)
	}
	if len(seen) != 2 || seen[0].MaxTokens != 100 || seen[1].MaxTokens != 200 ||
		*seen[0].Temperature != agentTemperature || *seen[1].Temperature != agentTemperature ||
		!slices.Equal(seen[0].StopSequences, []string{"a"}) ||
		!slices.Equal(seen[1].StopSequences, []string{"b"}) {
		t.Fatalf("dynamic settings did not see selected model defaults: %+v", seen)
	}
	for _, settings := range seen {
		if settings.TopP != nil {
			t.Fatalf("dynamic callback saw its own later setting: %+v", settings)
		}
	}
	if len(requested) != 2 || requested[0].MaxTokens != runMaxTokens || requested[1].MaxTokens != runMaxTokens ||
		*requested[0].TopP != topP || *requested[1].TopP != topP {
		t.Fatalf("agent and run settings did not override model defaults: %+v", requested)
	}
}

func TestModelSettingsCloneIsDetached(t *testing.T) {
	temperature := 0.1
	topP := 0.2
	seed := 3
	parallel := true
	original := ai.ModelSettings{
		Temperature: &temperature, TopP: &topP, Seed: &seed,
		StopSequences: []string{"stop"}, ParallelToolCalls: &parallel,
	}
	cloned := original.Clone()
	*cloned.Temperature = 1
	*cloned.TopP = 1
	*cloned.Seed = 10
	cloned.StopSequences[0] = "changed"
	*cloned.ParallelToolCalls = false
	if *original.Temperature != 0.1 || *original.TopP != 0.2 || *original.Seed != 3 ||
		original.StopSequences[0] != "stop" || !*original.ParallelToolCalls {
		t.Fatalf("clone mutated original settings: %+v", original)
	}
}
