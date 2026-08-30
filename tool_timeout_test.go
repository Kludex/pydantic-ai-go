package ai_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestToolTimeoutBecomesRetryPrompt(t *testing.T) {
	var retry ai.RetryPromptPart
	model := fakes.NewFunctionModel(func(_ context.Context, msgs []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
		if len(msgs) == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{strategyCall("slow", "call", `{}`)}}, nil
		}
		retry = msgs[len(msgs)-1].(ai.ModelRequest).Parts[0].(ai.RetryPromptPart)
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "recovered"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "slow", func(ctx context.Context, _ struct{}) (string, error) {
		<-ctx.Done()
		return "", ctx.Err()
	}, ai.WithToolTimeout(time.Millisecond))
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "recovered" || retry.Content != "Timed out after 1ms." {
		t.Fatalf("unexpected output %q and retry %+v", result.Output, retry)
	}
}

func TestToolTimeoutUsesToolRetryBudget(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{strategyCall("slow", "call", `{}`)}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "slow", func(ctx context.Context, _ struct{}) (string, error) {
		<-ctx.Done()
		return "", nil
	}, ai.WithToolTimeout(time.Millisecond), ai.WithToolMaxRetries(0))
	if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, ai.ErrMaxRetriesExceeded) {
		t.Fatalf("expected timeout to consume tool budget, got %v", err)
	}
}

func TestParentCancellationIsNotToolTimeout(t *testing.T) {
	started := make(chan struct{})
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{strategyCall("slow", "call", `{}`)}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "slow", func(ctx context.Context, _ struct{}) (string, error) {
		close(started)
		<-ctx.Done()
		return "", ctx.Err()
	}, ai.WithToolTimeout(time.Hour))
	ctx, cancel := context.WithCancel(t.Context())
	go func() {
		<-started
		cancel()
	}()
	if _, err := agent.Run(ctx, "go", deps{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected parent cancellation, got %v", err)
	}
}

func TestToolTimeoutMustBePositive(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	ai.WithToolTimeout(0)
}
