package ai_test

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestConversationDeferredRoundTripAndResume(t *testing.T) {
	for _, mode := range []string{"ordinary", "streamed", "manual"} {
		t.Run(mode, func(t *testing.T) {
			requests, executions := 0, 0
			model := fakes.NewFunctionModel(func(
				_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				requests++
				if requests == 1 {
					return &ai.ModelResponse{Parts: []ai.ResponsePart{
						ai.ToolCallPart{ToolName: "approve", ToolCallID: "approval", Args: []byte(`{}`),
							ProviderDetails: map[string]any{"marker": []string{"original"}}},
						ai.ToolCallPart{ToolName: "external", ToolCallID: "external", Args: []byte(`{}`)},
						ai.ToolCallPart{ToolName: "local", ToolCallID: "local", Args: []byte(`{}`)},
					}}, nil
				}
				parts := messages[len(messages)-1].(ai.ModelRequest).Parts
				if parts[0].(ai.ToolReturnPart).Content != "approved" || parts[1].(ai.ToolReturnPart).Content != "remote" {
					t.Fatalf("lost deferred results: %+v", parts)
				}
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
			})
			agent := ai.NewAgent[struct{}, string](model, ai.WithSequentialToolExecution())
			ai.AddTool(agent, "approve", func(
				_ context.Context, rc *ai.RunContext[struct{}], _ struct{},
			) (string, error) {
				executions++
				if !rc.ToolCallApproved || rc.Usage().ToolCalls != 1 {
					t.Fatalf("lost carried tool accounting: %+v", rc.Usage())
				}
				return "approved", nil
			}, ai.WithApprovalRequired())
			ai.AddSimpleTool(agent, "external", func(context.Context, struct{}) (any, error) {
				return ai.RequestExternalToolExecution(map[string]any{"queue": "orders"}), nil
			}, ai.WithDynamicExternalExecution())
			ai.AddSimpleTool(agent, "local", func(context.Context, struct{}) (string, error) {
				return "local", nil
			})
			run := func(prompt string, options ...ai.RunOption) (*ai.RunResult[string], error) {
				switch mode {
				case "ordinary":
					return agent.Run(t.Context(), prompt, struct{}{}, options...)
				case "streamed":
					stream := agent.RunStream(t.Context(), prompt, struct{}{}, options...)
					for _, err := range stream.Events() {
						if err != nil {
							return nil, err
						}
					}
					return stream.Result(), nil
				default:
					driver, err := agent.StartRun(t.Context(), prompt, struct{}{}, options...)
					if err != nil {
						return nil, err
					}
					_, err = drainAgentRun(driver)
					return driver.Result(), err
				}
			}
			paused, err := run("pause")
			if err != nil {
				t.Fatal(err)
			}
			bundle := paused.Conversation()
			pending := bundle.DeferredToolRequests
			if pending == nil || len(pending.Calls) != 1 || len(pending.Approvals) != 1 ||
				pending.Metadata["external"]["queue"] != "orders" || bundle.Usage.ToolCalls != 1 || executions != 0 ||
				!reflect.DeepEqual(pending, paused.Deferred()) {
				t.Fatalf("incomplete paused bundle: %+v", bundle)
			}
			pending.Approvals[0].Args[0] = 'X'
			pending.Approvals[0].ProviderDetails["marker"].([]string)[0] = "changed"
			pending.Metadata["external"]["queue"] = "changed"
			original := paused.Deferred()
			if original.Approvals[0].Args[0] != '{' || original.Metadata["external"]["queue"] != "orders" ||
				original.Approvals[0].ProviderDetails["marker"].([]string)[0] != "original" {
				t.Fatal("pending requests alias the result")
			}
			stored, err := json.Marshal(paused.Conversation())
			if err != nil {
				t.Fatal(err)
			}
			var conversation ai.Conversation
			if err := json.Unmarshal(stored, &conversation); err != nil {
				t.Fatal(err)
			}
			pending = conversation.DeferredToolRequests
			if pending.Calls[0].ToolCallID != "external" || pending.Approvals[0].ToolCallID != "approval" ||
				pending.Metadata["external"]["queue"] != "orders" {
				t.Fatalf("lost stored requests: %+v", pending)
			}
			options := []ai.RunOption{ai.WithConversation(conversation), ai.WithDeferredToolResults(ai.DeferredToolResults{
				Calls:     map[string]any{pending.Calls[0].ToolCallID: "remote"},
				Approvals: map[string]ai.ToolApproval{pending.Approvals[0].ToolCallID: ai.ApproveTool()},
			})}
			resumed, err := run("", options...)
			if err != nil || resumed.Output != "done" || resumed.Conversation().DeferredToolRequests != nil ||
				resumed.Usage().Requests != 2 || resumed.Usage().ToolCalls != 2 || executions != 1 ||
				resumed.ConversationID() != paused.ConversationID() || conversation.Usage.ToolCalls != 1 ||
				conversation.DeferredToolRequests == nil || len(conversation.DeferredToolRequests.Approvals) != 1 {
				t.Fatalf("unexpected resumed conversation: result=%+v executions=%d err=%v", resumed, executions, err)
			}
		})
	}
}
