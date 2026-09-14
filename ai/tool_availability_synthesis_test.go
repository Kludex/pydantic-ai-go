package ai_test

import (
	"context"
	"slices"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

type availabilityCaptureModel struct {
	messages []ai.ModelMessage
}

func (*availabilityCaptureModel) Name() string { return "capture" }

func (model *availabilityCaptureModel) Request(
	_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	model.messages = messages
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
}

func TestToolAvailabilitySynthesisHandlesStandaloneAndEmptyDeltas(t *testing.T) {
	prepared, err := ai.PrepareModelMessages(&availabilityCaptureModel{}, []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolAvailabilityDeltaPart{ToolCallID: "unique", ToolsAdded: []string{"hidden"}},
			ai.RetryPromptPart{Content: "plain validation"},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(prepared) != 2 || prepared[0].(ai.ModelResponse).Parts[0].(ai.ToolCallPart).ToolCallID != "unique" {
		t.Fatalf("standalone delta was not synthesized: %#v", prepared)
	}
	parts := prepared[1].(ai.ModelRequest).Parts
	if len(parts) != 2 || parts[1].(ai.RetryPromptPart).Content != "plain validation" {
		t.Fatalf("validation feedback moved: %#v", parts)
	}

	model := &availabilityCaptureModel{}
	_, err = ai.RequestModel(t.Context(), model, []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"forged"}}}},
	}, ai.ModelRequestParams{DeferredTools: []ai.ToolDefinition{{Name: "hidden", DeferLoading: true}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(model.messages) != 0 {
		t.Fatalf("empty filtered delta left an invalid request: %#v", model.messages)
	}
}

func TestToolAvailabilitySynthesisKeepsParallelResultsTogether(t *testing.T) {
	model := &availabilityCaptureModel{}
	messages := []ai.ModelMessage{
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "reveal", ToolCallID: "reveal"},
			ai.ToolCallPart{ToolName: "sibling", ToolCallID: "sibling"},
		}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "reveal", ToolCallID: "reveal", Content: "loaded"},
			ai.ToolAvailabilityDeltaPart{ToolCallID: "reveal", ToolsAdded: []string{"hidden", "forged"}},
			ai.ToolReturnPart{ToolName: "sibling", ToolCallID: "sibling", Content: "done"},
			ai.UserPromptPart{Content: "continue"},
		}},
	}
	_, err := ai.RequestModel(t.Context(), model, messages, ai.ModelRequestParams{
		DeferredTools: []ai.ToolDefinition{{Name: "hidden", DeferLoading: true}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(model.messages) != 4 {
		t.Fatalf("unexpected synthesized history: %#v", model.messages)
	}
	results := model.messages[1].(ai.ModelRequest).Parts
	if len(results) != 2 || results[0].(ai.ToolReturnPart).ToolCallID != "reveal" ||
		results[1].(ai.ToolReturnPart).ToolCallID != "sibling" {
		t.Fatalf("parallel results were split or reordered: %#v", results)
	}
	call := model.messages[2].(ai.ModelResponse).Parts[0].(ai.ToolCallPart)
	if call.ToolKind != ai.ToolPartKindToolSearch || call.ToolCallID == "reveal" {
		t.Fatalf("synthetic call reused an existing identity: %#v", call)
	}
	trailing := model.messages[3].(ai.ModelRequest).Parts
	returned := trailing[0].(ai.ToolReturnPart)
	result := returned.Content.(ai.ToolSearchResult)
	if returned.ToolCallID != call.ToolCallID || len(result.DiscoveredTools) != 1 ||
		result.DiscoveredTools[0].Name != "hidden" {
		t.Fatalf("unexpected synthetic result: %#v", returned)
	}
	if trailing[1].(ai.UserPromptPart).Content != "continue" {
		t.Fatalf("unrelated request content moved: %#v", trailing)
	}
	if slices.Contains(result.DiscoveredTools, ai.ToolSearchMatch{Name: "forged"}) {
		t.Fatal("unconfigured deferred tool was synthesized")
	}
}
