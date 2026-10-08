package ai_test

import (
	"context"
	"reflect"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestRepairMessagesPublicAPI(t *testing.T) {
	timestamp := time.Date(2026, 10, 8, 12, 0, 0, 0, time.UTC)
	history := []ai.ModelMessage{
		ai.ModelResponse{Timestamp: timestamp, RunID: "run", ConversationID: "conversation", Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "first", ToolCallID: "first", Args: []byte(`{}`)},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "last", ToolCallID: "last", Args: []byte(`{}`)},
		}},
	}
	preserved := ai.RepairMessages(history, ai.MessageRepairOptions{PreserveLastResponse: true})
	if len(preserved) != 3 {
		t.Fatalf("unexpected preserved history: %+v", preserved)
	}
	interrupted := preserved[1].(ai.ModelRequest)
	assertSynthesizedReturn(t, interrupted.Parts[0], "first", "first")
	if interrupted.Timestamp != timestamp || interrupted.RunID != "run" || interrupted.ConversationID != "conversation" {
		t.Fatalf("synthesized request identity changed: %+v", interrupted)
	}
	if len(preserved[2].(ai.ModelResponse).ToolCalls()) != 1 {
		t.Fatalf("last response was repaired: %+v", preserved)
	}
	closed := ai.RepairMessages(history, ai.MessageRepairOptions{})
	if countSynthesizedReturns(closed) != 2 || countSynthesizedReturns(ai.RepairMessages(closed, ai.MessageRepairOptions{})) != 2 {
		t.Fatalf("repair was not complete and idempotent: %+v", closed)
	}
	closed[0].(ai.ModelResponse).Parts[0] = ai.TextPart{Content: "changed"}
	if _, ok := history[0].(ai.ModelResponse).Parts[0].(ai.ToolCallPart); !ok {
		t.Fatal("repair shared response parts with its input")
	}

	merged := ai.RepairMessages([]ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "one"}}, Metadata: map[string]any{"one": 1}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "two"}}, Metadata: map[string]any{"two": 2}},
	}, ai.MessageRepairOptions{})
	request := merged[0].(ai.ModelRequest)
	if request.Metadata["one"] != 1 || request.Metadata["two"] != 2 {
		t.Fatalf("request metadata was not merged: %+v", request.Metadata)
	}
}

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

func TestRunMergesConsecutiveSameRoleHistory(t *testing.T) {
	history := []ai.ModelMessage{
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "work", ToolCallID: "call", Args: []byte(`{}`)},
		}},
		ai.ModelRequest{State: ai.RequestStateInterrupted, Parts: []ai.RequestPart{
			ai.UserPromptPart{Content: "before result"},
			ai.ToolReturnPart{ToolName: "work", ToolCallID: "call", Content: "done"},
		}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.RetryPromptPart{Content: "plain feedback"},
			ai.UserPromptPart{Content: "after result"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "one"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "two"}}},
		ai.ModelResponse{ModelName: "provider", Parts: []ai.ResponsePart{ai.TextPart{Content: "three"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "four"}}},
	}
	var captured []ai.ModelMessage
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		captured = append([]ai.ModelMessage(nil), messages...)
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	result, err := ai.NewAgent[deps, string](model).Run(
		t.Context(), "continue", deps{}, ai.WithMessageHistory(history),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(captured) != 6 {
		t.Fatalf("unexpected merged history: %+v", captured)
	}
	request := captured[1].(ai.ModelRequest)
	if request.State == ai.RequestStateInterrupted || len(request.Parts) != 4 {
		t.Fatalf("consecutive requests were not merged: %+v", request)
	}
	if resultPart, ok := request.Parts[0].(ai.ToolReturnPart); !ok || resultPart.ToolCallID != "call" {
		t.Fatalf("tool results were not hoisted: %+v", request.Parts)
	}
	response := captured[2].(ai.ModelResponse)
	if len(response.Parts) != 2 || response.Parts[0].(ai.TextPart).Content != "one" ||
		response.Parts[1].(ai.TextPart).Content != "two" {
		t.Fatalf("synthetic responses were not merged: %+v", response)
	}
	if len(result.NewMessages()) != 2 ||
		result.NewMessages()[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content != "continue" {
		t.Fatalf("history merging changed the new-message boundary: %+v", result.NewMessages())
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
