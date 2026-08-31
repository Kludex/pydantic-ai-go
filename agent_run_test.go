package ai_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func drainAgentRun[Deps, Output any](run *ai.AgentRun[Deps, Output]) ([]ai.StreamEvent, error) {
	var events []ai.StreamEvent
	for {
		event, ok, err := run.Next()
		if err != nil || !ok {
			return events, err
		}
		events = append(events, event)
	}
}

func TestAgentRunManualProgressionAndExternalEnqueue(t *testing.T) {
	requests := 0
	var seen []ai.ModelMessage
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		seen = messages
		return &ai.ModelResponse{
			ModelName: "gpt-5", ProviderName: "openai", Usage: ai.Usage{Requests: 1, InputTokens: requests},
			Parts: []ai.ResponsePart{ai.TextPart{Content: fmt.Sprintf("answer %d", requests)}},
		}, nil
	})
	run, err := ai.NewAgent[deps, string](model).StartRun(t.Context(), "first", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if run.RunID() == "" || run.ConversationID() == "" || run.Result() != nil || run.Err() != nil || !run.Usage().IsZero() {
		t.Fatalf("unexpected initial run state: result=%+v err=%v usage=%+v", run.Result(), run.Err(), run.Usage())
	}
	enqueueID, err := run.Enqueue(ai.UserPromptPart{Content: "second"})
	if err != nil || enqueueID == "" {
		t.Fatalf("enqueue failed: id=%q err=%v", enqueueID, err)
	}
	if emptyID, err := run.Enqueue(); err != nil || emptyID != "" {
		t.Fatalf("empty enqueue was not a no-op: id=%q err=%v", emptyID, err)
	}

	event, ok, err := run.Next()
	if err != nil || !ok {
		t.Fatalf("first step failed: event=%T ok=%t err=%v", event, ok, err)
	}
	if _, ok := event.(ai.EnqueuedMessagesEvent); !ok {
		t.Fatalf("expected enqueue boundary first, got %T", event)
	}
	if run.Result() != nil || run.Usage().InputTokens != 0 {
		t.Fatalf("run advanced beyond the yielded boundary: result=%+v usage=%+v", run.Result(), run.Usage())
	}
	events, err := drainAgentRun(run)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) == 0 || run.Result() == nil || run.Result().Output != "answer 1" || run.Err() != nil {
		t.Fatalf("unexpected completed run: events=%d result=%+v err=%v", len(events), run.Result(), run.Err())
	}
	if run.Usage().Requests != 1 || run.Usage().CostUSD == nil {
		t.Fatalf("manual run usage was not live: %+v", run.Usage())
	}
	if requests != 1 || len(seen) != 2 || seen[1].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content != "second" {
		t.Fatalf("external enqueue was not delivered: requests=%d messages=%+v", requests, seen)
	}
	if _, err := run.Enqueue(ai.UserPromptPart{Content: "late"}); err == nil {
		t.Fatal("expected enqueue after completion to fail")
	}
	if err := run.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestAgentRunCanEnqueueWhenPausedAtStreamEvent(t *testing.T) {
	requests := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 2 {
			last := messages[len(messages)-1].(ai.ModelRequest)
			if last.Parts[0].(ai.UserPromptPart).Content != "idle" {
				t.Fatalf("unexpected idle message: %+v", last)
			}
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: fmt.Sprintf("answer %d", requests)}}}, nil
	})
	run, err := ai.NewAgent[deps, string](model).StartRunParts(
		t.Context(), []ai.UserContent{ai.TextContent{Text: "first"}}, deps{},
	)
	if err != nil {
		t.Fatal(err)
	}
	event, ok, err := run.Next()
	if err != nil || !ok {
		t.Fatalf("first event failed: %T %t %v", event, ok, err)
	}
	if _, ok := event.(ai.PartStartEvent); !ok {
		t.Fatalf("unexpected first event %T", event)
	}
	id, err := run.EnqueueWhenIdle(ai.UserPromptPart{Content: "idle"})
	if err != nil || id == "" {
		t.Fatalf("idle enqueue failed: %q %v", id, err)
	}
	events, err := drainAgentRun(run)
	if err != nil {
		t.Fatal(err)
	}
	if requests != 2 || run.Result() == nil || run.Result().Output != "answer 2" {
		t.Fatalf("idle enqueue did not redirect: requests=%d events=%d result=%+v", requests, len(events), run.Result())
	}
}

func TestAgentRunEventsAndErrors(t *testing.T) {
	want := errors.New("model failed")
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, want
	})
	run, err := ai.NewAgent[deps, string](model).StartRun(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	var got error
	for _, err := range run.Events() {
		got = err
	}
	if !errors.Is(got, want) || !errors.Is(run.Err(), want) || run.Result() != nil {
		t.Fatalf("unexpected terminal error: events=%v run=%v result=%+v", got, run.Err(), run.Result())
	}
	if _, ok, err := run.Next(); ok || err != nil {
		t.Fatalf("completed progression did not stay closed: ok=%t err=%v", ok, err)
	}
}

func TestAgentRunCancelAndClose(t *testing.T) {
	started := make(chan struct{})
	model := fakes.NewFunctionModel(func(
		ctx context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		close(started)
		<-ctx.Done()
		return nil, ctx.Err()
	})
	run, err := ai.NewAgent[deps, string](model).StartRun(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		_, _, err := run.Next()
		done <- err
	}()
	<-started
	run.Cancel()
	if err := <-done; !errors.Is(err, ai.ErrRunCancelled) {
		t.Fatalf("unexpected cancellation error: %v", err)
	}
	if err := run.Close(); err != nil {
		t.Fatal(err)
	}

	abandoned, err := ai.NewAgent[deps, string](fakes.NewTestModel()).StartRun(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if err := abandoned.Close(); err != nil {
		t.Fatal(err)
	}
	if !errors.Is(abandoned.Err(), ai.ErrRunCancelled) {
		t.Fatalf("abandoned run did not cancel: %v", abandoned.Err())
	}
}

func TestResumeRunAndStartRunSetupError(t *testing.T) {
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "go"}}},
		ai.ModelResponse{State: ai.ModelResponseStateSuspended, ProviderResponseID: "job"},
	}
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(messages) != 2 || messages[1].(ai.ModelResponse).State != ai.ModelResponseStateSuspended {
			t.Fatalf("suspended history not replayed: %+v", messages)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	run, err := ai.NewAgent[deps, string](model).ResumeRun(t.Context(), history, deps{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := drainAgentRun(run); err != nil || run.Result() == nil || run.Result().Output != "done" {
		t.Fatalf("resume iteration failed: result=%+v err=%v", run.Result(), err)
	}

	_, err = ai.NewAgent[deps, string](model).StartRun(
		t.Context(), "go", deps{}, ai.WithRunModel(model), ai.WithRunModelID("other"),
	)
	if err == nil {
		t.Fatal("expected setup error")
	}
}

func TestAgentRunEnqueueValidation(t *testing.T) {
	run, err := ai.NewAgent[deps, string](fakes.NewTestModel()).StartRun(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = run.Close() }()
	if _, err := run.EnqueueWithPriority(ai.PendingMessagePriority("bad"), ai.UserPromptPart{Content: "x"}); err == nil {
		t.Fatal("expected invalid priority error")
	}
	if _, err := run.Enqueue(ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "bad"}}}); err == nil {
		t.Fatal("expected response-ending enqueue error")
	}
}

func TestAgentRunEventsCompletionAndEarlyStop(t *testing.T) {
	completed, err := ai.NewAgent[deps, string](fakes.NewTestModel()).StartRun(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, err := range completed.Events() {
		if err != nil {
			t.Fatal(err)
		}
		count++
	}
	if count == 0 || completed.Result() == nil {
		t.Fatalf("events did not complete the run: count=%d result=%+v", count, completed.Result())
	}

	stopped, err := ai.NewAgent[deps, string](fakes.NewTestModel()).StartRun(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	for range stopped.Events() {
		break
	}
	if !errors.Is(stopped.Err(), ai.ErrRunCancelled) {
		t.Fatalf("early event stop did not close the run: %v", stopped.Err())
	}
}

func TestAgentRunReportsCleanupErrors(t *testing.T) {
	closeErr := errors.New("close failed")
	log := []string{}
	model := &lifecycleModel{
		name: "closing", log: &log, closeErr: closeErr,
		request: func([]ai.ModelMessage) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		},
	}
	run, err := ai.NewAgent[deps, string](model).StartRun(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := drainAgentRun(run); !errors.Is(err, closeErr) {
		t.Fatalf("cleanup error was not yielded: %v", err)
	}
	if !errors.Is(run.Err(), closeErr) || run.Result() != nil || !errors.Is(run.Close(), closeErr) {
		t.Fatalf("cleanup state was not retained: err=%v close=%v result=%+v", run.Err(), run.Close(), run.Result())
	}

	requestErr := errors.New("request failed")
	model = &lifecycleModel{
		name: "both", log: &log, closeErr: closeErr,
		request: func([]ai.ModelMessage) (*ai.ModelResponse, error) { return nil, requestErr },
	}
	run, err = ai.NewAgent[deps, string](model).StartRun(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := drainAgentRun(run); !errors.Is(err, requestErr) {
		t.Fatalf("request error was not yielded: %v", err)
	}
	if !errors.Is(run.Err(), requestErr) || !errors.Is(run.Err(), closeErr) {
		t.Fatalf("request and cleanup errors were not joined: %v", run.Err())
	}
}

func TestAgentRunCloseWhilePaused(t *testing.T) {
	run, err := ai.NewAgent[deps, string](fakes.NewTestModel()).StartRun(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok, err := run.Next(); err != nil || !ok {
		t.Fatalf("failed to reach pause: ok=%t err=%v", ok, err)
	}
	if err := run.Close(); err != nil {
		t.Fatal(err)
	}
	if !errors.Is(run.Err(), ai.ErrRunCancelled) {
		t.Fatalf("paused run was not cancelled: %v", run.Err())
	}
}
