package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestRunContextCancelDrainsConcurrentTools(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "fast", Args: json.RawMessage(`{}`), ToolCallID: "fast"},
				ai.ToolCallPart{ToolName: "cancel", Args: json.RawMessage(`{}`), ToolCallID: "cancel"},
				ai.ToolCallPart{ToolName: "sibling", Args: json.RawMessage(`{}`), ToolCallID: "sibling"},
			},
			Usage: ai.Usage{Requests: 1, InputTokens: 2},
		}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithMetadata(map[string]any{"run": "cancelled"}))
	started := make(chan struct{})
	drained := make(chan struct{})
	ai.AddSimpleTool(agent, "fast", func(context.Context, struct{}) (string, error) {
		return "completed", nil
	}, ai.WithSequential())
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
	if len(cancelled.Messages()) != 3 || cancelled.Usage().InputTokens != 2 ||
		cancelled.Metadata()["run"] != "cancelled" {
		t.Fatalf(
			"cancellation snapshot lost state: messages=%v usage=%+v metadata=%+v",
			cancelled.Messages(), cancelled.Usage(), cancelled.Metadata(),
		)
	}
	cancelled.Metadata()["run"] = "mutated"
	if cancelled.Metadata()["run"] != "cancelled" {
		t.Fatal("cancellation metadata aliases caller mutation")
	}
	interrupted := cancelled.Messages()[2].(ai.ModelRequest)
	if interrupted.State != ai.RequestStateInterrupted || len(interrupted.Parts) != 1 {
		t.Fatalf("completed sibling result was not retained: %+v", interrupted)
	}
	part := interrupted.Parts[0].(ai.ToolReturnPart)
	if part.ToolName != "fast" || part.Content != "completed" {
		t.Fatalf("wrong completed result retained: %+v", part)
	}
	messages := cancelled.Messages()
	messages[0] = nil
	if cancelled.Messages()[0] == nil {
		t.Fatal("Messages did not return a copy")
	}

	var resumed []ai.ModelMessage
	resumeModel := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		resumed = messages
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "resumed"}}}, nil
	})
	resumeAgent := ai.NewAgent[deps, string](resumeModel)
	resumeResult, err := resumeAgent.Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(cancelled.Messages()),
	)
	if err != nil || resumeResult.Output != "resumed" {
		t.Fatalf("cancellation history did not resume: result=%+v err=%v", resumeResult, err)
	}
	request := resumed[len(resumed)-2].(ai.ModelRequest)
	if len(request.Parts) != 3 {
		t.Fatalf("dangling tool calls were not repaired beside completed results: %+v", request)
	}
	for index, name := range []string{"cancel", "sibling"} {
		interrupted := request.Parts[index+1].(ai.ToolReturnPart)
		if interrupted.ToolName != name || interrupted.Outcome != ai.ToolReturnOutcomeInterrupted ||
			interrupted.Metadata[ai.SynthesizedToolReturnMetadataKey] != true {
			t.Fatalf("unexpected synthesized return: %+v", interrupted)
		}
	}
	prompt := resumed[len(resumed)-1].(ai.ModelRequest)
	if len(prompt.Parts) != 1 || prompt.Parts[0].(ai.UserPromptPart).Content != "continue" {
		t.Fatalf("resume prompt was not preserved after repairs: %+v", prompt)
	}
}

func TestMessageHistoryRepairRecognizesExistingResults(t *testing.T) {
	history := []ai.ModelMessage{
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "with_id", ToolCallID: "call", Args: json.RawMessage(`{}`)},
			ai.ToolCallPart{ToolName: "without_id", Args: json.RawMessage(`{}`)},
		}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.RetryPromptPart{ToolName: "with_id", ToolCallID: "call", Content: "again"},
			ai.ToolReturnPart{ToolName: "without_id", Content: "done"},
		}},
	}
	var seen []ai.ModelMessage
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		seen = messages
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	if _, err := agent.Run(t.Context(), "continue", deps{}, ai.WithMessageHistory(history)); err != nil {
		t.Fatal(err)
	}
	request := seen[len(seen)-1].(ai.ModelRequest)
	if len(request.Parts) != 1 {
		t.Fatalf("already-settled calls were synthesized again: %+v", request)
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

func TestRunStreamCommittedCancellationRetainsCompletedTools(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.TextDeltaEvent{PartID: "text", Delta: "discarded"},
			ai.ToolCallStartEvent{PartID: "fast", ToolName: "fast", ToolCallID: "fast"},
			ai.ToolCallDeltaEvent{PartID: "fast", ArgsDelta: `{}`},
			ai.ToolCallStartEvent{PartID: "cancel", ToolName: "cancel", ToolCallID: "cancel"},
			ai.ToolCallDeltaEvent{PartID: "cancel", ArgsDelta: `{}`},
			ai.FinishEvent{},
		}
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "fast", func(context.Context, struct{}) (string, error) {
		return "completed", nil
	}, ai.WithSequential())
	ai.AddTool(agent, "cancel", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (string, error) {
		rc.Cancel()
		return "discarded", nil
	})
	stream := agent.RunStream(t.Context(), "go", deps{})
	var got error
	for _, err := range stream.Events() {
		if err != nil {
			got = err
		}
	}
	var cancelled *ai.RunCancelledError
	if !errors.As(got, &cancelled) {
		t.Fatalf("expected streamed RunCancelledError, got %v", got)
	}
	request := cancelled.Messages()[2].(ai.ModelRequest)
	if request.State != ai.RequestStateInterrupted || request.Parts[0].(ai.ToolReturnPart).ToolName != "fast" {
		t.Fatalf("completed streamed tool was not retained: %+v", request)
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
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		if yield(ai.TextDeltaEvent{PartID: "text", Delta: "discarded"}, nil) {
			yield(ai.FinishEvent{}, nil)
		}
	}, nil
}
