package ai_test

import (
	"context"
	"errors"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestUnknownToolWithNoAvailableTools(t *testing.T) {
	var retry ai.RetryPromptPart
	model := fakes.NewFunctionModel(func(_ context.Context, msgs []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
		if len(msgs) == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{strategyCall("missing", "bad", `{}`)}}, nil
		}
		retry = msgs[len(msgs)-1].(ai.ModelRequest).Parts[0].(ai.RetryPromptPart)
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if retry.Content != "Unknown tool name: 'missing'. No tools available." {
		t.Fatalf("unexpected retry %q", retry.Content)
	}
}

func TestPreparedOutToolCannotBeExecuted(t *testing.T) {
	var retry ai.RetryPromptPart
	model := fakes.NewFunctionModel(func(_ context.Context, msgs []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
		if len(msgs) == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{strategyCall("hidden", "bad", `{}`)}}, nil
		}
		retry = msgs[len(msgs)-1].(ai.ModelRequest).Parts[0].(ai.RetryPromptPart)
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ran := false
	ai.AddSimpleTool(agent, "hidden", func(context.Context, struct{}) (string, error) {
		ran = true
		return "", nil
	})
	agent.AddToolsPrepareFunc(func(
		context.Context, *ai.RunContext[deps], []ai.ToolDefinition,
	) ([]ai.ToolDefinition, error) {
		return nil, nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if ran {
		t.Fatal("tool omitted from the current step was executed")
	}
	if retry.Content != "Unknown tool name: 'hidden'. No tools available." {
		t.Fatalf("unexpected retry %q", retry.Content)
	}
}

func TestUnknownToolDoesNotSuppressOutput(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			strategyCall("final_result", "output", `{"value":1}`),
			strategyCall("missing", "bad", `{}`),
		}}, nil
	})
	agent := ai.NewAgent[deps, strategyOutput](model)
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.Value != 1 {
		t.Fatalf("unknown tool suppressed output %+v", result.Output)
	}
	parts := trailingRequest(t, result, 0).Parts
	retry := parts[1].(ai.RetryPromptPart)
	if retry.Content != "Unknown tool name: 'missing'. Available tools: 'final_result'" {
		t.Fatalf("unexpected retry %q", retry.Content)
	}
}

func TestUnknownToolUsesPerNameRetryBudget(t *testing.T) {
	step := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		step++
		name := "first"
		if step == 2 {
			name = "second"
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{strategyCall(name, name, `{}`)}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	_, err := agent.Run(t.Context(), "go", deps{})
	if !errors.Is(err, ai.ErrMaxRetriesExceeded) {
		t.Fatalf("expected third first-name call to exhaust its budget, got %v", err)
	}
	if step != 3 {
		t.Fatalf("unknown names shared a retry budget after %d requests", step)
	}
}

func TestToolFailedReturnsTerminalResultWithoutRetry(t *testing.T) {
	var returned ai.ToolReturnPart
	model := fakes.NewFunctionModel(func(_ context.Context, msgs []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
		if len(msgs) == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{strategyCall("work", "call", `{}`)}}, nil
		}
		returned = msgs[len(msgs)-1].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart)
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "recovered"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithRetryLimits(ai.RetryLimits{}))
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) {
		return "", ai.ToolFailedf("resource %d is gone", 42)
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "recovered" || returned.Content != "resource 42 is gone" || returned.Outcome != ai.ToolReturnOutcomeFailed {
		t.Fatalf("unexpected result %q and tool return %+v", result.Output, returned)
	}
	data, err := ai.MarshalMessages([]ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{returned}}})
	if err != nil {
		t.Fatal(err)
	}
	messages, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	roundTrip := messages[0].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart)
	if roundTrip.Outcome != ai.ToolReturnOutcomeFailed {
		t.Fatalf("failed outcome did not round trip: %s", data)
	}
}

func TestToolFailedError(t *testing.T) {
	err := ai.ToolFailedf("failed %s", "now")
	var failed *ai.ToolFailedError
	if !errors.As(err, &failed) || err.Error() != "ai: tool failed: failed now" {
		t.Fatalf("unexpected error %v", err)
	}
}
