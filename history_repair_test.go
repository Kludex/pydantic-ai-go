package ai_test

import (
	"context"
	"reflect"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestRunRepairsInteriorAndTrailingDanglingToolCalls(t *testing.T) {
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "future", ToolCallID: "future-1", Content: "out of place"},
			ai.UserPromptPart{Content: "start"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "a", ToolCallID: "a-1", Args: []byte(`{}`)},
			ai.ToolCallPart{ToolName: "b", ToolCallID: "b-1", Args: []byte(`{}`)},
			ai.ToolCallPart{ToolName: "retry", ToolCallID: "retry-1", Args: []byte(`{}`)},
			ai.ToolCallPart{ToolName: "future", ToolCallID: "future-1", Args: []byte(`{}`)},
		}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "b", ToolCallID: "b-1", Content: "done"},
			ai.RetryPromptPart{ToolName: "retry", ToolCallID: "retry-1", Content: "again"},
			ai.RetryPromptPart{ToolCallID: "a-1", Content: "plain validation feedback"},
			ai.UserPromptPart{Content: "middle"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "old", ToolCallID: "reused", Args: []byte(`{}`)},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "new", ToolCallID: "reused", Args: []byte(`{}`)},
		}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "new", ToolCallID: "reused", Content: "done"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "tail", ToolCallID: "tail-1", Args: []byte(`{}`)},
			ai.ToolCallPart{ToolName: "empty-id", Args: []byte(`{}`)},
		}},
	}
	var captured []ai.ModelMessage
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		captured = append([]ai.ModelMessage(nil), messages...)
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	result, err := agent.Run(t.Context(), "continue", deps{}, ai.WithMessageHistory(history))
	if err != nil {
		t.Fatal(err)
	}
	if len(captured) != 9 {
		t.Fatalf("unexpected repaired history length: %d\n%+v", len(captured), captured)
	}
	initial := captured[0].(ai.ModelRequest).Parts
	if len(initial) != 1 || initial[0].(ai.UserPromptPart).Content != "start" {
		t.Fatalf("out-of-place tool result was not dropped: %+v", initial)
	}

	interior := captured[2].(ai.ModelRequest).Parts
	if len(interior) != 6 {
		t.Fatalf("interior request was not repaired: %+v", interior)
	}
	assertSynthesizedReturn(t, interior[2], "a", "a-1")
	assertSynthesizedReturn(t, interior[3], "future", "future-1")
	if _, ok := interior[4].(ai.RetryPromptPart); !ok {
		t.Fatalf("synthesized result was not inserted before user-facing feedback: %+v", interior)
	}

	shadowedRequest := captured[4].(ai.ModelRequest)
	if len(shadowedRequest.Parts) != 1 {
		t.Fatalf("shadowed call repair was not isolated between responses: %+v", shadowedRequest)
	}
	assertSynthesizedReturn(t, shadowedRequest.Parts[0], "old", "reused")

	promptRequest := captured[8].(ai.ModelRequest)
	if len(promptRequest.Parts) != 3 {
		t.Fatalf("trailing repairs were not attached before the prompt: %+v", promptRequest)
	}
	assertSynthesizedReturn(t, promptRequest.Parts[0], "tail", "tail-1")
	assertSynthesizedReturn(t, promptRequest.Parts[1], "empty-id", "")
	if promptRequest.Parts[2].(ai.UserPromptPart).Content != "continue" {
		t.Fatalf("prompt moved ahead of synthesized results: %+v", promptRequest)
	}

	firstMessages := result.Messages()
	result, err = agent.Run(t.Context(), "continue again", deps{}, ai.WithMessageHistory(firstMessages))
	if err != nil {
		t.Fatal(err)
	}
	if countSynthesizedReturns(result.Messages()) != 5 {
		t.Fatalf("history repair was not idempotent: %+v", result.Messages())
	}
}

func TestRunDropsOrphanedToolResults(t *testing.T) {
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "orphan", ToolCallID: "one", Content: "bad"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "old"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "orphan", ToolCallID: "two", Content: "bad"},
			ai.RetryPromptPart{ToolName: "orphan", ToolCallID: "three", Content: "bad"},
			ai.RetryPromptPart{Content: "plain feedback"},
		}},
	}
	var captured []ai.ModelMessage
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		captured = append([]ai.ModelMessage(nil), messages...)
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	if _, err := ai.NewAgent[deps, string](model).Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(history),
	); err != nil {
		t.Fatal(err)
	}
	if len(captured) != 3 {
		t.Fatalf("unexpected normalized history: %+v", captured)
	}
	trailing := captured[1].(ai.ModelRequest).Parts
	if len(trailing) != 1 || trailing[0].(ai.RetryPromptPart).Content != "plain feedback" {
		t.Fatalf("orphaned results were not removed selectively: %+v", trailing)
	}
}

func TestRunKeepsEmptyTrailingRequestAfterDroppingOrphan(t *testing.T) {
	history := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.ToolReturnPart{ToolName: "orphan", ToolCallID: "one", Content: "bad"},
	}}}
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(messages) != 2 || len(messages[0].(ai.ModelRequest).Parts) != 0 {
			t.Fatalf("empty trailing request was not retained: %+v", messages)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	if _, err := ai.NewAgent[deps, string](model).Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(history),
	); err != nil {
		t.Fatal(err)
	}
}

func TestRunDoesNotRepairFullyMatchedHistory(t *testing.T) {
	history := []ai.ModelMessage{
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "work", ToolCallID: "call", Args: []byte(`{}`)},
		}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "work", ToolCallID: "call", Content: "done"},
		}},
	}
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if !reflect.DeepEqual(messages[:len(history)], history) {
			t.Fatalf("matched history was changed: got=%+v want=%+v", messages, history)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	if _, err := ai.NewAgent[deps, string](model).Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(history),
	); err != nil {
		t.Fatal(err)
	}
}

func assertSynthesizedReturn(t *testing.T, part ai.RequestPart, toolName, toolCallID string) {
	t.Helper()
	result, ok := part.(ai.ToolReturnPart)
	if !ok || result.ToolName != toolName || result.ToolCallID != toolCallID ||
		result.Outcome != ai.ToolReturnOutcomeInterrupted ||
		result.Metadata[ai.SynthesizedToolReturnMetadataKey] != true {
		t.Fatalf("unexpected synthesized result: %+v", part)
	}
}

func countSynthesizedReturns(messages []ai.ModelMessage) int {
	count := 0
	for _, message := range messages {
		request, ok := message.(ai.ModelRequest)
		if !ok {
			continue
		}
		for _, part := range request.Parts {
			result, ok := part.(ai.ToolReturnPart)
			if ok && result.Metadata[ai.SynthesizedToolReturnMetadataKey] == true {
				count++
			}
		}
	}
	return count
}
