package ai_test

import (
	"context"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type nativeHistoryModel struct {
	ai.Model
	provider string
}

func (m nativeHistoryModel) NativeToolSearchProvider() string { return m.provider }

func TestForeignNativeToolSearchHistoryBecomesLocal(t *testing.T) {
	timestamp := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	original := []ai.ModelMessage{
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.NativeToolCallPart{ToolName: "web_search", ToolKind: ai.ToolPartKindCapabilityLoad},
			ai.NativeToolCallPart{
				ToolName: ai.ToolSearchName, Args: []byte(`{"queries":["first"]}`), ToolCallID: "call-1",
				ToolKind: ai.ToolPartKindToolSearch, ProviderName: "openai",
				ProviderDetails: map[string]any{"execution": "server"},
			},
			ai.NativeToolReturnPart{
				ToolName: ai.ToolSearchName, ToolCallID: "call-1", ToolKind: ai.ToolPartKindToolSearch,
				Content:  ai.ToolSearchResult{DiscoveredTools: []ai.ToolSearchMatch{{Name: "first"}}},
				Metadata: map[string]any{"source": "server"}, Timestamp: timestamp,
				Outcome: ai.ToolReturnOutcomeSuccess, ProviderName: "openai",
			},
			ai.NativeToolReturnPart{
				ToolName: ai.ToolSearchName, ToolCallID: "call-2", ToolKind: ai.ToolPartKindToolSearch,
				Content: ai.ToolSearchResult{},
			},
			ai.TextPart{Content: "after search"},
		}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "next"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.NativeToolReturnPart{
			ToolName: ai.ToolSearchName, ToolCallID: "call-3", ToolKind: ai.ToolPartKindToolSearch,
			Content: ai.ToolSearchResult{}, ProviderName: "anthropic",
		}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "merged"}}},
		ai.ModelResponse{},
	}
	var seen []ai.ModelMessage
	base := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		seen = messages
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](nativeHistoryModel{Model: base, provider: "google"})
	if _, err := agent.Run(t.Context(), "current", deps{}, ai.WithMessageHistory(original)); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 7 {
		t.Fatalf("unexpected adapted history length: %+v", seen)
	}
	first := seen[0].(ai.ModelResponse)
	if len(first.Parts) != 2 {
		t.Fatalf("unexpected first adapted response: %+v", first)
	}
	if _, ok := first.Parts[0].(ai.NativeToolCallPart); !ok {
		t.Fatalf("unrelated native part changed type: %T", first.Parts[0])
	}
	call := first.Parts[1].(ai.ToolCallPart)
	if call.ToolCallID != "call-1" || call.ProviderDetails["execution"] != "server" {
		t.Fatalf("unexpected localized call: %+v", call)
	}
	returns := seen[1].(ai.ModelRequest).Parts
	if len(returns) != 2 {
		t.Fatalf("consecutive native returns did not share a request: %+v", returns)
	}
	returned := returns[0].(ai.ToolReturnPart)
	if returned.ToolCallID != "call-1" || returned.Timestamp != timestamp ||
		returned.Outcome != ai.ToolReturnOutcomeSuccess || returned.Metadata["source"] != "server" {
		t.Fatalf("unexpected localized return: %+v", returned)
	}
	if seen[2].(ai.ModelResponse).Text() != "after search" ||
		seen[3].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content != "next" {
		t.Fatalf("response boundaries changed: %+v", seen)
	}
	merged := seen[4].(ai.ModelRequest).Parts
	if len(merged) != 2 || merged[0].(ai.ToolReturnPart).ToolCallID != "call-3" ||
		merged[1].(ai.UserPromptPart).Content != "merged" {
		t.Fatalf("generated return request did not merge with the next request: %+v", merged)
	}
	if len(seen[5].(ai.ModelResponse).Parts) != 0 ||
		seen[6].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content != "current" {
		t.Fatalf("empty response or current prompt was lost: %+v", seen)
	}

	call.Args[0] = '['
	returned.Content.(ai.ToolSearchResult).DiscoveredTools[0].Name = "changed"
	returned.Metadata["source"] = "changed"
	originalCall := original[0].(ai.ModelResponse).Parts[1].(ai.NativeToolCallPart)
	originalReturn := original[0].(ai.ModelResponse).Parts[2].(ai.NativeToolReturnPart)
	if string(originalCall.Args) != `{"queries":["first"]}` ||
		originalReturn.Content.(ai.ToolSearchResult).DiscoveredTools[0].Name != "first" ||
		originalReturn.Metadata["source"] != "server" {
		t.Fatalf("adapted history aliases durable history: %+v", original)
	}
}

func TestSameProviderNativeToolSearchHistoryIsPreserved(t *testing.T) {
	history := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.NativeToolCallPart{
			ToolName: ai.ToolSearchName, ToolCallID: "call", ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "openai",
		},
		ai.NativeToolReturnPart{
			ToolName: ai.ToolSearchName, ToolCallID: "call", ToolKind: ai.ToolPartKindToolSearch,
			ProviderName: "openai", Content: ai.ToolSearchResult{},
		},
	}}}
	base := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		parts := messages[0].(ai.ModelResponse).Parts
		if len(parts) != 2 {
			t.Fatalf("same-provider history was split: %+v", messages)
		}
		if _, ok := parts[0].(ai.NativeToolCallPart); !ok {
			t.Fatalf("same-provider call changed type: %T", parts[0])
		}
		if _, ok := parts[1].(ai.NativeToolReturnPart); !ok {
			t.Fatalf("same-provider return changed type: %T", parts[1])
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](nativeHistoryModel{Model: base, provider: "openai"})
	if _, err := agent.Run(t.Context(), "current", deps{}, ai.WithMessageHistory(history)); err != nil {
		t.Fatal(err)
	}
}

func TestModelWithoutNativeSearchReplayReceivesLocalHistory(t *testing.T) {
	history := []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{ai.NativeToolCallPart{
		ToolName: ai.ToolSearchName, ToolCallID: "call", ToolKind: ai.ToolPartKindToolSearch,
		ProviderName: "openai", Args: []byte(`{}`),
	}}}}
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if _, ok := messages[0].(ai.ModelResponse).Parts[0].(ai.ToolCallPart); !ok {
			t.Fatalf("native history was not localized for an unsupported model: %+v", messages)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	if _, err := ai.NewAgent[deps, string](model).Run(
		t.Context(), "current", deps{}, ai.WithMessageHistory(history),
	); err != nil {
		t.Fatal(err)
	}
}
