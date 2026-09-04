package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"slices"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type explicitCompactionModel struct {
	compact func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error)
}

func (*explicitCompactionModel) Name() string { return "compact-model" }

func (*explicitCompactionModel) Request(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
}

func (model *explicitCompactionModel) CompactMessages(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return model.compact(ctx, messages, params)
}

func TestCompactModelMessagesUsesDetachedSnapshots(t *testing.T) {
	providerDetails := map[string]any{"nested": map[string]any{"value": "original"}}
	model := &explicitCompactionModel{compact: func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		messages[0].(ai.ModelRequest).Parts[0] = ai.UserPromptPart{Content: "changed"}
		params.Tools[0].Schema["changed"] = true
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.CompactionPart{Content: "Summary.", ProviderDetails: providerDetails}},
			Usage: ai.Usage{InputTokens: 2, OutputTokens: 1},
		}, nil
	}}
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "original"}}}}
	params := ai.ModelRequestParams{Tools: []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}}}}
	response, err := ai.CompactModelMessages(t.Context(), ai.WrapModel(model), messages, params)
	if err != nil || response.Parts[0].(ai.CompactionPart).Content != "Summary." {
		t.Fatalf("unexpected compaction response=%+v err=%v", response, err)
	}
	if messages[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content != "original" ||
		params.Tools[0].Schema["changed"] != nil {
		t.Fatalf("compaction mutated caller input: messages=%+v params=%+v", messages, params)
	}
	response.Parts[0].(ai.CompactionPart).ProviderDetails["nested"].(map[string]any)["value"] = "changed"
	if providerDetails["nested"].(map[string]any)["value"] != "original" {
		t.Fatal("compaction response retained provider-owned state")
	}
}

func TestCompactModelMessagesValidationAndCancellation(t *testing.T) {
	if _, err := ai.CompactModelMessages(t.Context(), nil, nil, ai.ModelRequestParams{}); !errors.Is(err, ai.ErrNoModel) {
		t.Fatalf("unexpected nil-model error: %v", err)
	}
	unsupported := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, nil
	})
	if _, err := ai.CompactModelMessages(t.Context(), unsupported, nil, ai.ModelRequestParams{}); !errors.Is(
		err, ai.ErrCompactionUnsupported,
	) {
		t.Fatalf("unexpected unsupported-compaction error: %v", err)
	}
	model := &explicitCompactionModel{compact: func(
		ctx context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	}}
	if _, err := ai.CompactModelMessages(t.Context(), model, nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		RequestTimeout: -time.Second,
	}}); err == nil || err.Error() != "ai: request timeout must be non-negative, got -1s" {
		t.Fatalf("unexpected settings error: %v", err)
	}
	if _, err := ai.CompactModelMessages(t.Context(), model, nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		RequestTimeout: time.Millisecond,
	}}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("unexpected compaction timeout: %v", err)
	}
	model.compact = func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, nil
	}
	if response, err := ai.CompactModelMessages(t.Context(), model, nil, ai.ModelRequestParams{}); response != nil || err != nil {
		t.Fatalf("unexpected nil compaction response=%+v err=%v", response, err)
	}
}

func TestUpstreamCompactionFixtureRoundTrips(t *testing.T) {
	data, err := os.ReadFile("testdata/messages/upstream_compaction.json")
	if err != nil {
		t.Fatal(err)
	}
	messages, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	response := messages[0].(ai.ModelResponse)
	part := response.Parts[0].(ai.CompactionPart)
	if part.Content != "Summary of prior work." || !part.HasContent() || part.ID != "cmp_1" ||
		part.ProviderName != "anthropic" || part.ProviderDetails["encrypted_content"] != "opaque" {
		t.Fatalf("unexpected compaction fixture: %+v", part)
	}
	encoded, err := ai.MarshalMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	roundTripped, err := ai.UnmarshalMessages(encoded)
	if err != nil {
		t.Fatal(err)
	}
	cloned := roundTripped[0].(ai.ModelResponse).Parts[0].(ai.CompactionPart)
	cloned.ProviderDetails["nested"].(map[string]any)["value"] = 2
	if part.ProviderDetails["nested"].(map[string]any)["value"] != float64(1) {
		t.Fatal("compaction provider details were not detached")
	}

	withoutContent := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.CompactionPart{
		ID: "opaque", ProviderName: "openai", ProviderDetails: map[string]any{"encrypted_content": "data"},
	}}}}
	encoded, err = ai.MarshalMessages(withoutContent)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := ai.UnmarshalMessages(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if decodedPart := decoded[0].(ai.ModelResponse).Parts[0].(ai.CompactionPart); decodedPart.HasContent() {
		t.Fatalf("contentless compaction gained content: %+v", decodedPart)
	}
}

func TestCompactionStreamingLifecycle(t *testing.T) {
	details := map[string]any{"encrypted_content": "opaque"}
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.CompactionEvent{
				PartID: "compact", Content: "Summary.", ID: "cmp_1",
				ProviderName: "openai", ProviderDetails: details,
			},
			ai.TextDeltaEvent{PartID: "text", Delta: "done"},
			ai.FinishEvent{},
		}
	})
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	var starts []ai.ResponsePart
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(ai.PartStartEvent); ok {
			starts = append(starts, event.Part)
		}
	}
	if stream.Result() == nil || stream.Result().Output != "done" || len(starts) != 2 {
		t.Fatalf("unexpected compaction stream result=%+v starts=%+v", stream.Result(), starts)
	}
	compaction := starts[0].(ai.CompactionPart)
	if compaction.Content != "Summary." || compaction.ID != "cmp_1" ||
		compaction.ProviderDetails["encrypted_content"] != "opaque" {
		t.Fatalf("unexpected streamed compaction: %+v", compaction)
	}
	details["encrypted_content"] = "changed"
	if compaction.ProviderDetails["encrypted_content"] != "opaque" {
		t.Fatal("streamed compaction retained provider map")
	}
}

func TestFallbackStreamingReplaysCompaction(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.CompactionPart{Content: "Summary.", ProviderName: "anthropic"},
			ai.TextPart{Content: "done"},
		}}, nil
	})
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	seen := false
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(ai.PartStartEvent); ok {
			if _, compacted := event.Part.(ai.CompactionPart); compacted {
				seen = true
			}
		}
	}
	if !seen || stream.Result() == nil || stream.Result().Output != "done" {
		t.Fatalf("fallback stream omitted compaction or result: seen=%v result=%+v", seen, stream.Result())
	}
}

func TestStoppingFallbackCompactionUnwindsStream(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.CompactionPart{Content: "Summary."}, ai.TextPart{Content: "done"},
		}}, nil
	})
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(ai.PartStartEvent); ok {
			if _, compacted := event.Part.(ai.CompactionPart); compacted {
				break
			}
		}
	}
	if stream.Result() != nil {
		t.Fatalf("stopped compaction stream completed: %+v", stream.Result())
	}
}

func TestDuplicateCompactionStreamIDIsRejected(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.CompactionEvent{PartID: "compact", Content: "first"},
			ai.CompactionEvent{PartID: "compact", Content: "second"},
		}
	})
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	var got error
	for _, err := range stream.Events() {
		if err != nil {
			got = err
		}
	}
	if got == nil || got.Error() != `ai: unexpected model behavior: duplicate compaction stream part "compact"` {
		t.Fatalf("unexpected duplicate compaction error: %v", got)
	}
}

func TestCompactionResetsDeferredToolVisibilityOnNextRequest(t *testing.T) {
	requests := 0
	toolCalls := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		visible := make([]string, len(params.Tools))
		for index, tool := range params.Tools {
			visible[index] = tool.Name
		}
		slices.Sort(visible)
		switch requests {
		case 1:
			if !slices.Contains(visible, "hidden") {
				t.Fatalf("pre-compaction reveal was not visible: %v", visible)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.CompactionPart{Content: "Summary.", ProviderName: "function"},
				ai.ToolCallPart{ToolName: "hidden", ToolCallID: "hidden", Args: json.RawMessage(`{}`)},
			}}, nil
		case 2:
			if slices.Contains(visible, "hidden") {
				t.Fatalf("compaction did not reset deferred visibility: %v", visible)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		default:
			t.Fatalf("unexpected request %d", requests)
			return nil, nil
		}
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "hidden", func(context.Context, struct{}) (string, error) {
		toolCalls++
		return "ran", nil
	}, ai.WithDeferredLoading())
	history := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolAvailabilityDeltaPart{
		ToolsAdded: []string{"hidden"}, ToolCallID: "reveal",
	}}}}
	result, err := agent.Run(t.Context(), "go", deps{}, ai.WithMessageHistory(history))
	if err != nil || result.Output != "done" || toolCalls != 1 || requests != 2 {
		t.Fatalf("unexpected compacted run result=%+v tools=%d requests=%d err=%v", result, toolCalls, requests, err)
	}
}

func TestPostCompactionRevealRestoresDeferredVisibility(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(params.Tools) != 1 || params.Tools[0].Name != "hidden" {
			t.Fatalf("post-compaction reveal was not visible: %+v", params.Tools)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "hidden", func(context.Context, struct{}) (string, error) {
		return "ran", nil
	}, ai.WithDeferredLoading())
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"old"}}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.CompactionPart{Content: "Summary."}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"hidden"}}}},
	}
	result, err := agent.Run(t.Context(), "go", deps{}, ai.WithMessageHistory(history))
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected post-compaction result=%+v err=%v", result, err)
	}
}
