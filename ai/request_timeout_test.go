package ai_test

import (
	"context"
	"errors"
	"iter"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestModelRequestTimeout(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		ctx context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	})
	agent := ai.NewAgent[deps, string](model, ai.WithModelSettings(ai.ModelSettings{
		RequestTimeout: 10 * time.Millisecond,
	}))
	started := time.Now()
	if _, err := agent.Run(t.Context(), "wait", deps{}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected request deadline, got %v", err)
	}
	if time.Since(started) > time.Second {
		t.Fatal("request timeout was not applied promptly")
	}
}

type waitingStreamingModel struct{}

func (waitingStreamingModel) Name() string { return "waiting-stream" }

func (waitingStreamingModel) Request(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return nil, errors.New("non-streaming request used")
}

func (waitingStreamingModel) StreamRequest(
	ctx context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		<-ctx.Done()
		yield(nil, ctx.Err())
	}, nil
}

func TestStreamingModelRequestTimeout(t *testing.T) {
	agent := ai.NewAgent[deps, string](waitingStreamingModel{}, ai.WithModelSettings(ai.ModelSettings{
		RequestTimeout: 10 * time.Millisecond,
	}))
	stream := agent.RunStream(t.Context(), "wait", deps{})
	var got error
	for _, err := range stream.Events() {
		if err != nil {
			got = err
		}
	}
	if !errors.Is(got, context.DeadlineExceeded) || stream.Result() != nil {
		t.Fatalf("expected streaming request deadline, got result=%+v err=%v", stream.Result(), got)
	}
}

func TestRunRequestTimeoutOverridesAgentDefault(t *testing.T) {
	var got time.Duration
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		got = params.Settings.RequestTimeout
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithModelSettings(ai.ModelSettings{RequestTimeout: time.Second}))
	if _, err := agent.Run(
		t.Context(), "go", deps{}, ai.WithRunModelSettings(ai.ModelSettings{RequestTimeout: 2 * time.Second}),
	); err != nil {
		t.Fatal(err)
	}
	if got != 2*time.Second {
		t.Fatalf("request timeout was not merged: %s", got)
	}
}

func TestNegativeModelRequestTimeoutFailsBeforeRequest(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		t.Fatal("model was called")
		return nil, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithModelSettings(ai.ModelSettings{RequestTimeout: -time.Second}))
	if _, err := agent.Run(t.Context(), "go", deps{}); err == nil ||
		err.Error() != "ai: request timeout must be non-negative, got -1s" {
		t.Fatalf("unexpected timeout validation error: %v", err)
	}
}

func TestParentCancellationPrecedesModelRequestTimeout(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		ctx context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	})
	agent := ai.NewAgent[deps, string](model, ai.WithModelSettings(ai.ModelSettings{RequestTimeout: time.Hour}))
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := agent.Run(ctx, "wait", deps{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("parent cancellation was replaced by request timeout: %v", err)
	}
}
