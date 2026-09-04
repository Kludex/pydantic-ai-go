package ai_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestRunMetadataMergesRecomputesAndIsDetached(t *testing.T) {
	requests := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "inspect", ToolCallID: "call", Args: []byte(`{}`),
			}}}, nil
		}
		for _, part := range messages[len(messages)-1].(ai.ModelRequest).Parts {
			if _, ok := part.(ai.ToolReturnPart); ok {
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
			}
		}
		return nil, errors.New("tool result missing")
	})
	base := map[string]any{
		"environment": "test", "shared": "agent",
		"nested": map[string]any{"stable": true},
	}
	runValues := map[string]any{"shared": "run", "run": true}
	agent := ai.NewAgent[struct{}, string](model, ai.WithMetadata(base))
	base["environment"] = "mutated"
	runValues["run"] = false
	agentCalls := 0
	agent.AddMetadataFunc(func(_ context.Context, rc *ai.RunContext[struct{}]) (map[string]any, error) {
		agentCalls++
		if rc.Prompt.Content != "metadata prompt" || rc.Metadata["environment"] != "test" {
			t.Fatalf("unexpected agent metadata context: %+v", rc)
		}
		return map[string]any{"requests": rc.Usage().Requests, "agent_dynamic": true}, nil
	})
	runCalls := 0
	runMetadata := ai.WithRunMetadataFunc(func(
		_ context.Context, rc *ai.RunContext[struct{}],
	) (map[string]any, error) {
		runCalls++
		if rc.Metadata["shared"] != "run" || rc.Metadata["agent_dynamic"] != true {
			t.Fatalf("run metadata did not see prior layers: %+v", rc.Metadata)
		}
		return map[string]any{"run_dynamic": rc.Usage().Requests}, nil
	})
	before := ai.BeforeRunFunc(func(_ context.Context, info *ai.RunInfo) error {
		metadata := info.Metadata()
		metadata["environment"] = "capability mutation"
		prompt := info.Prompt()
		prompt.Content = "capability mutation"
		return nil
	})
	ai.AddTool(agent, "inspect", func(_ context.Context, rc *ai.RunContext[struct{}], _ struct{}) (string, error) {
		if rc.Metadata["shared"] != "run" || rc.Prompt.Content != "metadata prompt" {
			t.Fatalf("tool did not receive run metadata and prompt: %+v", rc)
		}
		rc.Metadata["shared"] = "tool mutation"
		rc.Metadata["nested"].(map[string]any)["stable"] = false
		return "ok", nil
	})
	runStatic := ai.WithRunMetadata(runValues)
	runValues["run"] = "mutated after option"
	result, err := agent.Run(
		t.Context(), "metadata prompt", struct{}{}, runStatic, runMetadata, ai.WithRunCapabilities(before),
	)
	if err != nil {
		t.Fatal(err)
	}
	metadata := result.Metadata()
	if metadata["environment"] != "test" || metadata["shared"] != "run" || metadata["run"] != false ||
		metadata["requests"] != 2 || metadata["run_dynamic"] != 2 || metadata["agent_dynamic"] != true ||
		metadata["nested"].(map[string]any)["stable"] != true {
		t.Fatalf("unexpected final metadata: %+v", metadata)
	}
	metadata["shared"] = "result mutation"
	if result.Metadata()["shared"] != "run" {
		t.Fatal("result metadata aliases caller mutation")
	}
	if agentCalls != 2 || runCalls != 2 {
		t.Fatalf("metadata functions did not run twice: agent=%d run=%d", agentCalls, runCalls)
	}

}

func TestRunMetadataAvailableOnManualRun(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](
		fakes.NewTestModel(), ai.WithMetadata(map[string]any{"static": true}),
	)
	agent.AddMetadataFunc(func(_ context.Context, rc *ai.RunContext[struct{}]) (map[string]any, error) {
		return map[string]any{"requests": rc.Usage().Requests}, nil
	})
	run, err := agent.StartRun(t.Context(), "prompt", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if run.Metadata()["requests"] != 0 {
		t.Fatalf("unexpected initial metadata: %+v", run.Metadata())
	}
	if _, err := drainAgentRun(run); err != nil {
		t.Fatal(err)
	}
	if run.Metadata()["requests"] != 1 || run.Result().Metadata()["static"] != true {
		t.Fatalf("unexpected completed metadata: run=%+v result=%+v", run.Metadata(), run.Result().Metadata())
	}
}

func TestRunMetadataErrorsAndDependencyValidation(t *testing.T) {
	boom := errors.New("metadata failed")
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	agent.AddMetadataFunc(func(context.Context, *ai.RunContext[struct{}]) (map[string]any, error) {
		return nil, boom
	})
	if _, err := agent.Run(t.Context(), "prompt", struct{}{}); !errors.Is(err, boom) ||
		!strings.Contains(err.Error(), "ai: metadata") {
		t.Fatalf("unexpected agent metadata error: %v", err)
	}

	completionAgent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	completionAgent.AddMetadataFunc(func(_ context.Context, rc *ai.RunContext[struct{}]) (map[string]any, error) {
		if rc.Usage().Requests > 0 {
			return nil, boom
		}
		return map[string]any{"initial": true}, nil
	})
	if _, err := completionAgent.Run(t.Context(), "prompt", struct{}{}); !errors.Is(err, boom) {
		t.Fatalf("unexpected completion metadata error: %v", err)
	}

	mismatch := ai.WithRunMetadataFunc(func(context.Context, *ai.RunContext[string]) (map[string]any, error) {
		return nil, nil
	})
	if _, err := ai.NewAgent[struct{}, string](fakes.NewTestModel()).Run(
		t.Context(), "prompt", struct{}{}, mismatch,
	); err == nil || !strings.Contains(err.Error(), "dependencies do not match") {
		t.Fatalf("unexpected metadata dependency error: %v", err)
	}

	assertOutputFunctionPanics(t, func() {
		ai.NewAgent[struct{}, string](fakes.NewTestModel()).AddMetadataFunc(nil)
	})
	assertOutputFunctionPanics(t, func() {
		ai.WithRunMetadataFunc[struct{}](nil)
	})
}
