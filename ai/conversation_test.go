package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestConversationContinuation(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request := messages[len(messages)-1].(ai.ModelRequest)
		prompt := request.Parts[len(request.Parts)-1].(ai.UserPromptPart)
		text := prompt.Content
		if len(prompt.Contents) > 0 {
			text = prompt.Contents[0].(ai.TextContent).Text
		}
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.TextPart{Content: text}},
			Usage: ai.Usage{Requests: 1, InputTokens: 3, OutputTokens: 2, Details: map[string]int{"counter": 1}},
		}, nil
	}))
	root, err := agent.Run(t.Context(), "root", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	conversation := root.Conversation()
	if !reflect.DeepEqual(conversation.Messages, root.Messages()) ||
		!reflect.DeepEqual(conversation.Usage, root.Usage()) || conversation.ConversationID != root.ConversationID() ||
		conversation.DeferredToolRequests != nil {
		t.Fatalf("incomplete bundle: %+v", conversation)
	}
	conversation.Usage.ToolCalls = 3
	conversation.Usage.Details["branch"] = 1
	if root.Usage().ToolCalls != 0 || root.Usage().Details["branch"] != 0 {
		t.Fatal("conversation usage aliases the result")
	}

	parts := []ai.UserContent{ai.TextContent{Text: "branch"}}
	tests := []struct {
		name string
		run  func(...ai.RunOption) (*ai.RunResult[string], error)
	}{
		{"Run", func(opts ...ai.RunOption) (*ai.RunResult[string], error) {
			return agent.Run(t.Context(), "branch", struct{}{}, opts...)
		}},
		{"RunParts", func(opts ...ai.RunOption) (*ai.RunResult[string], error) {
			return agent.RunParts(t.Context(), parts, struct{}{}, opts...)
		}},
		{"RunStream", func(opts ...ai.RunOption) (*ai.RunResult[string], error) {
			stream := agent.RunStream(t.Context(), "branch", struct{}{}, opts...)
			for _, err := range stream.Events() {
				if err != nil {
					return nil, err
				}
			}
			if stream.Usage().Requests != 2 {
				t.Fatalf("stream lost carried usage: %+v", stream.Usage())
			}
			return stream.Result(), nil
		}},
		{"RunStreamParts", func(opts ...ai.RunOption) (*ai.RunResult[string], error) {
			stream := agent.RunStreamParts(t.Context(), parts, struct{}{}, opts...)
			for _, err := range stream.Events() {
				if err != nil {
					return nil, err
				}
			}
			return stream.Result(), nil
		}},
		{"StartRun", func(opts ...ai.RunOption) (*ai.RunResult[string], error) {
			run, err := agent.StartRun(t.Context(), "branch", struct{}{}, opts...)
			if err != nil {
				return nil, err
			}
			if run.Usage().Requests != 1 || run.Usage().ToolCalls != 3 || run.ConversationID() != root.ConversationID() {
				t.Fatalf("manual run lost carried state: %+v", run.Usage())
			}
			_, err = drainAgentRun(run)
			return run.Result(), err
		}},
		{"StartRunParts", func(opts ...ai.RunOption) (*ai.RunResult[string], error) {
			run, err := agent.StartRunParts(t.Context(), parts, struct{}{}, opts...)
			if err != nil {
				return nil, err
			}
			_, err = drainAgentRun(run)
			return run.Result(), err
		}},
		{"RunAs", func(opts ...ai.RunOption) (*ai.RunResult[string], error) {
			return ai.RunAs[string](t.Context(), agent, "branch", struct{}{}, opts...)
		}},
		{"RunStreamAs", func(opts ...ai.RunOption) (*ai.RunResult[string], error) {
			stream := ai.RunStreamAs[string](t.Context(), agent, "branch", struct{}{}, opts...)
			for _, err := range stream.Events() {
				if err != nil {
					return nil, err
				}
			}
			return stream.Result(), nil
		}},
	}
	option := ai.WithConversation(conversation)
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result, err := test.run(option)
			if err != nil {
				t.Fatal(err)
			}
			usage := result.Usage()
			if result.Output != "branch" || result.ConversationID() != conversation.ConversationID ||
				result.RunID() == root.RunID() || usage.Requests != 2 || usage.InputTokens != 6 ||
				usage.OutputTokens != 4 || usage.ToolCalls != 3 || usage.Details["counter"] != 2 ||
				!reflect.DeepEqual(result.Messages()[:len(conversation.Messages)], conversation.Messages) ||
				len(result.NewMessages()) != 2 {
				t.Fatalf("unexpected continuation: %+v", result.Conversation())
			}
			result.Messages()[0].(ai.ModelRequest).Parts[0] = ai.UserPromptPart{Content: "changed"}
		})
	}
	if conversation.Usage.Requests != 1 || conversation.Usage.Details["counter"] != 1 ||
		conversation.Messages[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content != "root" {
		t.Fatal("continuation mutated its branch point")
	}
	historyOnly, err := agent.Run(t.Context(), "history only", struct{}{}, ai.WithMessageHistory(conversation.Messages))
	if err != nil || historyOnly.Usage().Requests != 1 {
		t.Fatalf("message history must not carry usage: %+v, %v", historyOnly, err)
	}
}

func TestConversationZeroValueAndID(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	first, err := agent.Run(t.Context(), "first", struct{}{}, ai.WithConversation(ai.Conversation{}))
	if err != nil {
		t.Fatal(err)
	}
	second, err := agent.Run(t.Context(), "second", struct{}{}, ai.WithConversation(ai.Conversation{}))
	if err != nil || first.ConversationID() == "" || first.ConversationID() == second.ConversationID() {
		t.Fatalf("zero conversation did not generate a fresh ID: %+v, %v", second, err)
	}
	for _, id := range []string{"", "explicit"} {
		result, err := agent.Run(t.Context(), "continue", struct{}{}, ai.WithConversation(ai.Conversation{
			Messages: first.Messages(), ConversationID: id,
		}))
		wantID := id
		if wantID == "" {
			wantID = first.ConversationID()
		}
		if err != nil || result.ConversationID() != wantID {
			t.Fatalf("ID %q: result=%+v err=%v", id, result, err)
		}
	}
}

func TestConversationConflicts(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	for name, conflict := range map[string]ai.RunOption{
		"nil history": ai.WithMessageHistory(nil), "empty history": ai.WithMessageHistory([]ai.ModelMessage{}),
		"ID": ai.WithConversationID("id"), "new ID": ai.WithConversationID("new"),
	} {
		for _, reversed := range []bool{false, true} {
			t.Run(name, func(t *testing.T) {
				opts := []ai.RunOption{ai.WithConversation(ai.Conversation{}), conflict}
				if reversed {
					slices.Reverse(opts)
				}
				_, err := agent.Run(t.Context(), "run", struct{}{}, opts...)
				if err == nil || !strings.Contains(err.Error(), "WithConversation") {
					t.Fatalf("expected conflict, got %v", err)
				}
				run, err := agent.StartRun(t.Context(), "manual", struct{}{}, opts...)
				if err == nil || run != nil {
					t.Fatalf("expected manual conflict, got %v", err)
				}
				stream := agent.RunStream(t.Context(), "stream", struct{}{}, opts...)
				var streamErr error
				for _, err := range stream.Events() {
					streamErr = err
				}
				if streamErr == nil || stream.Result() != nil {
					t.Fatal("expected streaming conflict")
				}
			})
		}
	}
	if _, err := agent.Resume(t.Context(), nil, struct{}{}, ai.WithConversation(ai.Conversation{})); err == nil {
		t.Fatal("explicit suspended history must also conflict")
	}
}

func TestConversationUsageLimits(t *testing.T) {
	cost := 1.0
	toolLimit := 2
	for _, test := range []struct {
		name   string
		limits ai.UsageLimits
		calls  int
	}{
		{"requests", ai.UsageLimits{RequestLimit: 2}, 0},
		{"input", ai.UsageLimits{InputTokenLimit: 10}, 1},
		{"output", ai.UsageLimits{OutputTokenLimit: 10}, 1},
		{"total", ai.UsageLimits{TotalTokenLimit: 20}, 1},
		{"cost", ai.UsageLimits{CostLimitUSD: &cost}, 1},
		{"tools", ai.UsageLimits{ToolCallLimit: &toolLimit}, 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			calls, executions := 0, 0
			model := fakes.NewFunctionModel(func(
				context.Context, []ai.ModelMessage, ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				calls++
				return &ai.ModelResponse{
					Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "work", ToolCallID: "work", Args: []byte(`{}`)}},
					Usage: ai.Usage{Requests: 1, InputTokens: 1, OutputTokens: 1, CostUSD: &cost},
				}, nil
			})
			agent := ai.NewAgent[struct{}, string](model)
			ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) {
				executions++
				return "done", nil
			})
			conversation := ai.Conversation{Usage: ai.Usage{
				Requests: 2, ToolCalls: 2, InputTokens: 10, OutputTokens: 10, CostUSD: &cost,
			}}
			_, err := agent.Run(t.Context(), "continue", struct{}{}, ai.WithConversation(conversation),
				ai.WithRunUsageLimits(test.limits))
			if !errors.Is(err, ai.ErrUsageLimitExceeded) || calls != test.calls || executions != 0 ||
				conversation.Usage.Requests != 2 || *conversation.Usage.CostUSD != 1 {
				t.Fatalf("budget reset: calls=%d executions=%d err=%v", calls, executions, err)
			}
		})
	}
}

func TestConversationConcurrentBranchesAndOptionSnapshot(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	cost := 0.25
	conversation := ai.Conversation{
		Messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "root"}},
			Metadata: map[string]any{"nested": map[string]any{"value": "root"}}}},
		Usage: ai.Usage{Requests: 4, Details: map[string]int{"root": 1}, CostUSD: &cost}, ConversationID: "branches",
	}
	option := ai.WithConversation(conversation)
	conversation.Messages[0].(ai.ModelRequest).Parts[0] = ai.UserPromptPart{Content: "changed"}
	conversation.Messages[0].(ai.ModelRequest).Metadata["nested"].(map[string]any)["value"] = "changed"
	conversation.Usage.Details["root"] = 2
	cost = 0.5
	var workers sync.WaitGroup
	for range 2 {
		workers.Go(func() {
			result, err := agent.Run(t.Context(), "branch", struct{}{}, option)
			if err != nil {
				t.Error(err)
				return
			}
			bundle := result.Conversation()
			request := bundle.Messages[0].(ai.ModelRequest)
			if request.Parts[0].(ai.UserPromptPart).Content != "root" ||
				request.Metadata["nested"].(map[string]any)["value"] != "root" ||
				bundle.Usage.Requests != 5 || bundle.Usage.Details["root"] != 1 || *bundle.Usage.CostUSD != 0.25 {
				t.Errorf("shared branch state: %+v", bundle)
			}
			request.Parts[0] = ai.UserPromptPart{Content: "mutated bundle"}
			request.Metadata["nested"].(map[string]any)["value"] = "mutated bundle"
			bundle.Messages = nil
			bundle.Usage.Details["root"] = 3
			*bundle.Usage.CostUSD = 1
			original := result.Messages()[0].(ai.ModelRequest)
			if original.Parts[0].(ai.UserPromptPart).Content != "root" ||
				original.Metadata["nested"].(map[string]any)["value"] != "root" || result.Usage().Details["root"] != 1 ||
				*result.Usage().CostUSD != 0.25 {
				t.Error("extraction mutated its result")
			}
		})
	}
	workers.Wait()
}

func TestConversationProviderStream(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.ResponseMetadataEvent{Usage: ai.Usage{InputTokens: 2}},
			ai.TextDeltaEvent{Delta: "streamed"},
			ai.FinishEvent{Usage: ai.Usage{Requests: 1, InputTokens: 2, OutputTokens: 3}},
		}
	})
	agent := ai.NewAgent[struct{}, string](model)
	conversation := ai.Conversation{
		Usage: ai.Usage{Requests: 4, InputTokens: 10, OutputTokens: 5}, ConversationID: "stream",
	}
	stream := agent.RunStream(t.Context(), "continue", struct{}{}, ai.WithConversation(conversation))
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if stream.Usage().Requests < 4 || stream.Usage().InputTokens < 10 {
			t.Fatalf("in-flight usage dropped prior spend: %+v", stream.Usage())
		}
	}
	result := stream.Result().Conversation()
	if stream.Result().Output != "streamed" || result.ConversationID != "stream" ||
		result.Usage.Requests != 5 || result.Usage.InputTokens != 12 || result.Usage.OutputTokens != 8 ||
		!reflect.DeepEqual(result.Usage, stream.Usage()) || conversation.Usage.Requests != 4 {
		t.Fatalf("unexpected streamed bundle: %+v", result)
	}
}

func TestConversationJSON(t *testing.T) {
	conversation := ai.Conversation{
		Messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "read", ToolCallID: "read-1", Content: []byte{0xff, 0xfe}},
			ai.UserPromptPart{Contents: []ai.UserContent{ai.BinaryContent{Data: []byte{0x89, 0xff}, MediaType: "image/png"}}},
		}, Metadata: map[string]any{"tag": "value"}}},
		Usage: ai.Usage{Requests: 4, ToolCalls: 3, Details: map[string]int{"counter": 2}}, ConversationID: "stored",
	}
	messages, err := ai.MarshalMessages(conversation.Messages)
	if err != nil {
		t.Fatal(err)
	}
	expected, err := ai.UnmarshalMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	type thread struct {
		Owner        string          `json:"owner"`
		Conversation ai.Conversation `json:"conversation"`
	}
	data, err := json.Marshal(thread{Owner: "acme", Conversation: conversation})
	if err != nil {
		t.Fatal(err)
	}
	var restored thread
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatal(err)
	}
	if restored.Owner != "acme" || !reflect.DeepEqual(restored.Conversation.Messages, expected) ||
		!reflect.DeepEqual(restored.Conversation.Usage, conversation.Usage) || restored.Conversation.ConversationID != "stored" {
		t.Fatalf("unexpected round trip: %+v", restored)
	}
	var wire struct {
		Conversation struct {
			Messages json.RawMessage `json:"messages"`
		} `json:"conversation"`
	}
	if err := json.Unmarshal(data, &wire); err != nil || string(wire.Conversation.Messages) != string(messages) {
		t.Fatalf("message persistence boundary changed: %s, %v", data, err)
	}
	before := restored.Conversation
	for _, invalid := range []string{`{`, `[]`, `{"messages":{}}`, `{"messages":[{"kind":"unknown"}]}`} {
		if err := restored.Conversation.UnmarshalJSON([]byte(invalid)); err == nil ||
			!reflect.DeepEqual(restored.Conversation, before) {
			t.Fatalf("invalid JSON changed the receiver: %s, %v", invalid, err)
		}
	}
	for _, empty := range []string{`{}`, `{"messages":null}`, `{"messages":[]}`} {
		var conversation ai.Conversation
		if err := json.Unmarshal([]byte(empty), &conversation); err != nil || len(conversation.Messages) != 0 ||
			!conversation.Usage.IsZero() || conversation.ConversationID != "" || conversation.DeferredToolRequests != nil {
			t.Fatalf("invalid empty conversation: %+v, %v", conversation, err)
		}
	}
	conversation.Messages = []ai.ModelMessage{nil}
	if _, err := json.Marshal(conversation); err == nil {
		t.Fatal("invalid messages must fail serialization")
	}
	conversation.Messages = nil
	conversation.DeferredToolRequests = &ai.DeferredToolRequests{Metadata: map[string]map[string]any{"bad": {"value": make(chan int)}}}
	if _, err := json.Marshal(conversation); err == nil {
		t.Fatal("invalid deferred metadata must fail serialization")
	}
}
