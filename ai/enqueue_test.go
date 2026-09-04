package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"sync"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestToolEnqueueDeliversInterleavedMessages(t *testing.T) {
	request := 0
	var enqueueID string
	explicitTimestamp := time.Date(2025, 1, 2, 3, 4, 5, 0, time.UTC)
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "notify", ToolCallID: "notify", Args: json.RawMessage(`{}`),
			}}}, nil
		}
		if len(messages) < 4 {
			t.Fatalf("enqueued exchange missing: %+v", messages)
		}
		delivered := messages[len(messages)-4:]
		first := delivered[0].(ai.ModelRequest)
		if first.RunID == "" || first.ConversationID == "" || first.Timestamp.IsZero() ||
			len(first.Parts) != 2 {
			t.Fatalf("enqueued request was not stamped: %+v", first)
		}
		prompt := first.Parts[0].(ai.UserPromptPart)
		if len(prompt.Contents) != 2 || prompt.Contents[0].(ai.TextContent).Text != "caption" ||
			prompt.Contents[1].(ai.ImageURL).URL != "https://example.com/image.png" {
			t.Fatalf("adjacent user content was not grouped: %+v", prompt)
		}
		if first.Parts[1].(ai.SystemPromptPart).Content != "incident mode" {
			t.Fatalf("request part ordering changed: %+v", first.Parts)
		}
		synthetic := delivered[1].(ai.ModelResponse)
		if synthetic.Timestamp != explicitTimestamp || synthetic.RunID == "" || synthetic.Text() != "synthetic" {
			t.Fatalf("synthetic response metadata changed: %+v", synthetic)
		}
		stampedResponse := delivered[2].(ai.ModelResponse)
		if stampedResponse.Timestamp.IsZero() || stampedResponse.Text() != "stamped synthetic" {
			t.Fatalf("synthetic response was not stamped: %+v", stampedResponse)
		}
		last := delivered[3].(ai.ModelRequest)
		if last.Parts[0].(ai.UserPromptPart).Content != "follow up" {
			t.Fatalf("enqueued sequence did not end with request: %+v", last)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddTool(agent, "notify", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (string, error) {
		var err error
		enqueueID, err = rc.Enqueue(
			ai.TextContent{Text: "caption"},
			ai.ImageURL{URL: "https://example.com/image.png"},
			ai.SystemPromptPart{Content: "incident mode"},
			ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "synthetic"}}, Timestamp: explicitTimestamp},
			ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ThinkingPart{Content: "thinking", ProviderDetails: map[string]any{"signature": "local"}},
				ai.TextPart{Content: "stamped synthetic"},
			}},
			ai.UserPromptPart{Content: "follow up"},
		)
		return "notified", err
	})
	stream := agent.RunStream(t.Context(), "go", deps{})
	var event ai.EnqueuedMessagesEvent
	for item, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if delivered, ok := item.(ai.EnqueuedMessagesEvent); ok {
			event = delivered
		}
	}
	if stream.Result() == nil || stream.Result().Output != "done" || enqueueID == "" ||
		event.EnqueueID != enqueueID || len(event.Messages) != 4 {
		t.Fatalf("unexpected enqueue result=%+v id=%q event=%+v", stream.Result(), enqueueID, event)
	}
}

func TestASAPEnqueueRedirectsFinalOutput(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 2 {
			latest := messages[len(messages)-1].(ai.ModelRequest)
			if latest.Parts[0].(ai.UserPromptPart).Contents[0].(ai.TextContent).Text != "steer" {
				t.Fatalf("ASAP redirect missing: %+v", latest)
			}
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: []string{"first", "second"}[request-1]}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	validated := 0
	agent.AddOutputValidator(func(_ context.Context, rc *ai.RunContext[deps], _ string) error {
		validated++
		if validated == 1 {
			_, err := rc.Enqueue(ai.TextContent{Text: "steer"})
			return err
		}
		return nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "second" || request != 2 || validated != 2 {
		t.Fatalf("unexpected ASAP redirect result=%+v requests=%d validations=%d err=%v", result, request, validated, err)
	}
}

func TestWhenIdleEnqueueWaitsForFinalCandidate(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		switch request {
		case 1:
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "schedule", ToolCallID: "schedule", Args: json.RawMessage(`{}`),
			}}}, nil
		case 2:
			latest := messages[len(messages)-1].(ai.ModelRequest)
			if _, isToolReturn := latest.Parts[0].(ai.ToolReturnPart); !isToolReturn {
				t.Fatalf("idle message arrived before candidate output: %+v", latest)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "candidate"}}}, nil
		default:
			latest := messages[len(messages)-1].(ai.ModelRequest)
			if latest.Parts[0].(ai.UserPromptPart).Contents[0].(ai.TextContent).Text != "follow up" {
				t.Fatalf("idle follow-up missing: %+v", latest)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "final"}}}, nil
		}
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddTool(agent, "schedule", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (string, error) {
		_, err := rc.EnqueueWhenIdle(ai.TextContent{Text: "follow up"})
		return "scheduled", err
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "final" || request != 3 {
		t.Fatalf("unexpected idle redirect result=%+v requests=%d err=%v", result, request, err)
	}
}

func TestConcurrentToolEnqueueIsSafe(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "work", ToolCallID: "one", Args: json.RawMessage(`{}`)},
				ai.ToolCallPart{ToolName: "work", ToolCallID: "two", Args: json.RawMessage(`{}`)},
			}}, nil
		}
		var values []string
		for _, message := range messages {
			request, ok := message.(ai.ModelRequest)
			if !ok || len(request.Parts) != 1 {
				continue
			}
			prompt, ok := request.Parts[0].(ai.UserPromptPart)
			if ok && len(prompt.Contents) == 1 {
				values = append(values, prompt.Contents[0].(ai.TextContent).Text)
			}
		}
		slices.Sort(values)
		if !slices.Equal(values, []string{"one", "two"}) {
			t.Fatalf("concurrent enqueue lost messages: %v", values)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	var idsMu sync.Mutex
	var ids []string
	ai.AddTool(agent, "work", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (string, error) {
		id, err := rc.Enqueue(ai.TextContent{Text: rc.ToolCallID})
		idsMu.Lock()
		ids = append(ids, id)
		idsMu.Unlock()
		return "worked", err
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "done" || len(ids) != 2 || ids[0] == ids[1] {
		t.Fatalf("unexpected concurrent enqueue result=%+v ids=%v err=%v", result, ids, err)
	}
}

func TestEnqueueValidation(t *testing.T) {
	var rc ai.RunContext[deps]
	if _, err := rc.Enqueue(ai.TextContent{Text: "outside"}); err == nil {
		t.Fatal("expected enqueue outside run error")
	}

	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "validate", ToolCallID: "validate", Args: json.RawMessage(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddTool(agent, "validate", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (string, error) {
		if id, err := rc.Enqueue(); err != nil || id != "" {
			return "", errors.New("empty enqueue was not a no-op")
		}
		if _, err := rc.EnqueueWithPriority("later", ai.TextContent{Text: "bad"}); err == nil {
			return "", errors.New("invalid priority was accepted")
		}
		if _, err := rc.Enqueue(ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "bad"}}}); err == nil {
			return "", errors.New("response-only enqueue was accepted")
		}
		if _, err := rc.Enqueue(
			ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "direct request"}}},
		); err != nil {
			return "", err
		}
		if _, err := rc.Enqueue(
			ai.BinaryContent{Data: []byte("data"), MediaType: "text/plain"},
			ai.ToolReturnPart{ToolName: "synthetic", ToolCallID: "synthetic", Content: "done"},
			ai.ToolAvailabilityDeltaPart{ToolsAdded: []string{"later"}},
			ai.RetryPromptPart{Content: "retry"},
		); err != nil {
			return "", err
		}
		return "", errors.New("validation complete")
	})
	_, err := agent.Run(t.Context(), "go", deps{})
	if err == nil || err.Error() != `ai: tool "validate": validation complete` {
		t.Fatalf("unexpected enqueue validation result: %v", err)
	}
}

func TestEnqueuedMessagesEventCanStopStream(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "work", Args: json.RawMessage(`{}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "unreachable"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddTool(agent, "work", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (string, error) {
		_, err := rc.Enqueue(ai.TextContent{Text: "stop"})
		return "worked", err
	})
	stream := agent.RunStream(t.Context(), "go", deps{})
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(ai.EnqueuedMessagesEvent); ok {
			break
		}
	}
	if stream.Result() != nil || request != 1 {
		t.Fatalf("stopped enqueue stream should have no result: result=%+v requests=%d", stream.Result(), request)
	}
}
