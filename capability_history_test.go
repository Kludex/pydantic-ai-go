package ai_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestHistoryProcessorsComposeWithoutReplacingDurableHistory(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(messages) != 2 {
			t.Fatalf("model received %d processed messages: %#v", len(messages), messages)
		}
		current := messages[1].(ai.ModelRequest).Parts[0].(ai.UserPromptPart)
		if current.Content != "processed" {
			t.Fatalf("processors ran out of order: %#v", messages)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	first := ai.HistoryProcessor(func(
		_ context.Context, runInfo *ai.RunInfo, messages []ai.ModelMessage,
	) ([]ai.ModelMessage, error) {
		runMessages := runInfo.Messages()
		if runInfo.RunID == "" || len(runMessages) != 3 {
			t.Fatalf("processor received incomplete run info: %+v", runInfo)
		}
		runMessages[0].(ai.ModelRequest).Parts[0] = ai.UserPromptPart{Content: "also detached"}
		messages[0].(ai.ModelRequest).Parts[0] = ai.UserPromptPart{Content: "processed"}
		return messages[1:], nil
	})
	second := ai.HistoryProcessor(func(
		_ context.Context, _ *ai.RunInfo, messages []ai.ModelMessage,
	) ([]ai.ModelMessage, error) {
		messages[1].(ai.ModelRequest).Parts[0] = ai.UserPromptPart{Content: "processed"}
		return messages, nil
	})
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "original"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "old answer"}}},
	}
	result, err := ai.NewAgent[deps, string](model, ai.WithCapabilities(first, second)).Run(
		t.Context(), "current", deps{}, ai.WithMessageHistory(history),
	)
	if err != nil {
		t.Fatal(err)
	}
	messages := result.Messages()
	if len(messages) != 4 || messages[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content != "original" ||
		messages[1].(ai.ModelResponse).Parts[0].(ai.TextPart).Content != "old answer" {
		t.Fatalf("processor replaced or mutated durable history: %#v", messages)
	}
}

func TestHistoryProcessorErrors(t *testing.T) {
	if err := (ai.HistoryProcessor)(nil).Setup(&ai.CapabilityRegistry{}); err == nil ||
		err.Error() != "ai: history processor must not be nil" {
		t.Fatalf("unexpected nil processor error: %v", err)
	}
	processorErr := errors.New("processing failed")
	processor := ai.HistoryProcessor(func(
		context.Context, *ai.RunInfo, []ai.ModelMessage,
	) ([]ai.ModelMessage, error) {
		return nil, processorErr
	})
	_, err := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(processor)).Run(
		t.Context(), "go", deps{},
	)
	if !errors.Is(err, processorErr) || !strings.Contains(err.Error(), "processing failed") {
		t.Fatalf("unexpected processor error: %v", err)
	}
}
