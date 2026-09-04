package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func twoToolModel(t *testing.T, names ...string) *fakes.FunctionModel {
	t.Helper()
	return fakes.NewFunctionModel(func(_ context.Context, msgs []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
		if len(msgs) == 1 {
			parts := make([]ai.ResponsePart, 0, len(names))
			for _, name := range names {
				parts = append(parts, ai.ToolCallPart{ToolName: name, Args: json.RawMessage(`{}`), ToolCallID: "call_" + name})
			}
			return &ai.ModelResponse{Parts: parts}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
}

func TestIndependentToolCallsRunConcurrently(t *testing.T) {
	model := twoToolModel(t, "first", "second")
	agent := ai.NewAgent[deps, string](model)
	var started atomic.Int32
	bothStarted := make(chan struct{})
	tool := func(ctx context.Context, _ struct{}) (string, error) {
		if started.Add(1) == 2 {
			close(bothStarted)
		}
		select {
		case <-bothStarted:
			return "ok", nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}
	ai.AddSimpleTool(agent, "first", tool)
	ai.AddSimpleTool(agent, "second", tool)

	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	if _, err := agent.Run(ctx, "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if started.Load() != 2 {
		t.Fatalf("expected both tools to start, got %d", started.Load())
	}
}

func TestConcurrentToolResultsPreserveModelOrder(t *testing.T) {
	var returned []string
	model := fakes.NewFunctionModel(func(_ context.Context, msgs []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
		if len(msgs) == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "first", Args: json.RawMessage(`{}`), ToolCallID: "c1"},
				ai.ToolCallPart{ToolName: "second", Args: json.RawMessage(`{}`), ToolCallID: "c2"},
			}}, nil
		}
		request := msgs[len(msgs)-1].(ai.ModelRequest)
		for _, part := range request.Parts {
			returned = append(returned, part.(ai.ToolReturnPart).ToolName)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	secondDone := make(chan struct{})
	ai.AddSimpleTool(agent, "first", func(ctx context.Context, _ struct{}) (string, error) {
		select {
		case <-secondDone:
			return "first", nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	})
	ai.AddSimpleTool(agent, "second", func(context.Context, struct{}) (string, error) {
		close(secondDone)
		return "second", nil
	})

	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(returned, []string{"first", "second"}) {
		t.Fatalf("unexpected result order %v", returned)
	}
}

func TestSequentialToolCreatesBarrier(t *testing.T) {
	model := twoToolModel(t, "before", "barrier", "after")
	agent := ai.NewAgent[deps, string](model)
	var mu sync.Mutex
	var events []string
	record := func(event string) {
		mu.Lock()
		defer mu.Unlock()
		events = append(events, event)
	}
	ai.AddSimpleTool(agent, "before", func(context.Context, struct{}) (string, error) {
		record("before")
		return "", nil
	})
	ai.AddSimpleTool(agent, "barrier", func(context.Context, struct{}) (string, error) {
		record("barrier")
		return "", nil
	}, ai.WithSequential())
	ai.AddSimpleTool(agent, "after", func(context.Context, struct{}) (string, error) {
		record("after")
		return "", nil
	})

	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(events, []string{"before", "barrier", "after"}) {
		t.Fatalf("unexpected execution order %v", events)
	}
}

func TestAgentWideSequentialToolExecution(t *testing.T) {
	model := twoToolModel(t, "first", "second")
	agent := ai.NewAgent[deps, string](model, ai.WithSequentialToolExecution())
	var mu sync.Mutex
	var events []string
	tool := func(name string) func(context.Context, struct{}) (string, error) {
		return func(context.Context, struct{}) (string, error) {
			mu.Lock()
			defer mu.Unlock()
			events = append(events, name)
			return "", nil
		}
	}
	ai.AddSimpleTool(agent, "first", tool("first"))
	ai.AddSimpleTool(agent, "second", tool("second"))
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(events, []string{"first", "second"}) {
		t.Fatalf("unexpected execution order %v", events)
	}
}

func TestToolCallContextIsIsolated(t *testing.T) {
	model := twoToolModel(t, "first", "second")
	agent := ai.NewAgent[deps, string](model)
	ids := make(chan string, 2)
	tool := func(_ context.Context, rc *ai.RunContext[deps], _ struct{}) (string, error) {
		ids <- rc.ToolCallID
		return "", nil
	}
	ai.AddTool(agent, "first", tool)
	ai.AddTool(agent, "second", tool)
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	close(ids)
	got := make([]string, 0, 2)
	for id := range ids {
		got = append(got, id)
	}
	slices.Sort(got)
	if !slices.Equal(got, []string{"call_first", "call_second"}) {
		t.Fatalf("tool call IDs leaked between contexts: %v", got)
	}
}

func TestErrorBeforeSequentialBarrierStopsExecution(t *testing.T) {
	model := twoToolModel(t, "fails", "peer", "barrier")
	agent := ai.NewAgent[deps, string](model)
	boom := errors.New("boom")
	ai.AddSimpleTool(agent, "fails", func(context.Context, struct{}) (string, error) {
		return "", boom
	})
	ai.AddSimpleTool(agent, "peer", func(context.Context, struct{}) (string, error) {
		return "", nil
	})
	barrierRan := false
	ai.AddSimpleTool(agent, "barrier", func(context.Context, struct{}) (string, error) {
		barrierRan = true
		return "", nil
	}, ai.WithSequential())

	_, err := agent.Run(t.Context(), "go", deps{})
	if !errors.Is(err, boom) {
		t.Fatalf("expected tool error, got %v", err)
	}
	if barrierRan {
		t.Fatal("sequential barrier ran after an earlier call failed")
	}
}

func TestToolErrorCancelsConcurrentCalls(t *testing.T) {
	model := twoToolModel(t, "fails", "waits")
	agent := ai.NewAgent[deps, string](model)
	boom := errors.New("boom")
	ai.AddSimpleTool(agent, "fails", func(context.Context, struct{}) (string, error) {
		return "", boom
	})
	cancelled := make(chan struct{})
	ai.AddSimpleTool(agent, "waits", func(ctx context.Context, _ struct{}) (string, error) {
		<-ctx.Done()
		close(cancelled)
		return "", ctx.Err()
	})

	_, err := agent.Run(t.Context(), "go", deps{})
	if !errors.Is(err, boom) {
		t.Fatalf("expected tool error, got %v", err)
	}
	select {
	case <-cancelled:
	default:
		t.Fatal("concurrent tool did not observe cancellation")
	}
}
