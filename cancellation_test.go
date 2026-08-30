package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestRunContextCancelDrainsConcurrentTools(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "cancel", Args: json.RawMessage(`{}`), ToolCallID: "cancel"},
				ai.ToolCallPart{ToolName: "sibling", Args: json.RawMessage(`{}`), ToolCallID: "sibling"},
			},
			Usage: ai.Usage{Requests: 1, InputTokens: 2},
		}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	started := make(chan struct{})
	drained := make(chan struct{})
	ai.AddTool(agent, "cancel", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (string, error) {
		<-started
		rc.Cancel()
		rc.Cancel()
		return "discarded", nil
	})
	ai.AddSimpleTool(agent, "sibling", func(ctx context.Context, _ struct{}) (string, error) {
		close(started)
		defer close(drained)
		<-ctx.Done()
		return "", ctx.Err()
	})

	result, err := agent.Run(t.Context(), "go", deps{})
	if result != nil || !errors.Is(err, ai.ErrRunCancelled) {
		t.Fatalf("expected run cancellation, got result=%+v err=%v", result, err)
	}
	select {
	case <-drained:
	default:
		t.Fatal("Run returned before the sibling tool was drained")
	}
	var cancelled *ai.RunCancelledError
	if !errors.As(err, &cancelled) {
		t.Fatalf("expected RunCancelledError, got %T", err)
	}
	if len(cancelled.Messages()) != 2 || cancelled.Usage().InputTokens != 2 {
		t.Fatalf("cancellation snapshot lost state: messages=%v usage=%+v", cancelled.Messages(), cancelled.Usage())
	}
	messages := cancelled.Messages()
	messages[0] = nil
	if cancelled.Messages()[0] == nil {
		t.Fatal("Messages did not return a copy")
	}
}

func TestRunContextCancelFromOutputValidator(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "discarded"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddOutputValidator(func(_ context.Context, rc *ai.RunContext[deps], _ string) error {
		rc.Cancel()
		return nil
	})
	if result, err := agent.Run(t.Context(), "go", deps{}); result != nil || !errors.Is(err, ai.ErrRunCancelled) {
		t.Fatalf("validator cancellation was recoverable: result=%+v err=%v", result, err)
	}
}

func TestRunContextCancelFromDynamicInstructions(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "discarded"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddInstructionsFunc(func(_ context.Context, rc *ai.RunContext[deps]) (string, error) {
		rc.Cancel()
		return "cancelled", nil
	})
	if result, err := agent.Run(t.Context(), "go", deps{}); result != nil || !errors.Is(err, ai.ErrRunCancelled) {
		t.Fatalf("instruction cancellation was recoverable: result=%+v err=%v", result, err)
	}
}

func TestRunContextCancelAfterRunIsNoOp(t *testing.T) {
	var runContext *ai.RunContext[deps]
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	agent.AddInstructionsFunc(func(_ context.Context, rc *ai.RunContext[deps]) (string, error) {
		runContext = rc
		return "", nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output == "" {
		t.Fatalf("unexpected run result=%+v err=%v", result, err)
	}
	runContext.Cancel()
	runContext.Cancel()
	var detached ai.RunContext[deps]
	detached.Cancel()
}

func TestRunStreamContextCancellation(t *testing.T) {
	model := &cancelStreamingModel{Model: fakes.NewTestModel()}
	agent := ai.NewAgent[deps, string](model)
	agent.AddOutputValidator(func(_ context.Context, rc *ai.RunContext[deps], _ string) error {
		rc.Cancel()
		return nil
	})
	stream := agent.RunStream(t.Context(), "go", deps{})
	var got error
	for _, err := range stream.Events() {
		if err != nil {
			got = err
		}
	}
	if stream.Result() != nil || !errors.Is(got, ai.ErrRunCancelled) {
		t.Fatalf("unexpected streamed cancellation result=%+v err=%v", stream.Result(), got)
	}
}

func TestExternalContextCancellationRemainsContextError(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		ctx context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, ctx.Err()
	})
	agent := ai.NewAgent[deps, string](model)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err := agent.Run(ctx, "go", deps{})
	if !errors.Is(err, context.Canceled) || errors.Is(err, ai.ErrRunCancelled) {
		t.Fatalf("external cancellation was misclassified: %v", err)
	}
}

type cancelStreamingModel struct{ ai.Model }

func (m *cancelStreamingModel) StreamRequest(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (iter.Seq2[ai.StreamEvent, error], error) {
	return func(yield func(ai.StreamEvent, error) bool) {
		if yield(ai.TextDeltaEvent{PartID: "text", Delta: "discarded"}, nil) {
			yield(ai.FinishEvent{}, nil)
		}
	}, nil
}
