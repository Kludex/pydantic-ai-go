package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestReinjectSystemPromptWhenHistoryOmittedIt(t *testing.T) {
	var captured []ai.ModelMessage
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		captured = messages
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](
		model,
		ai.WithSystemPrompt("server prompt"),
		ai.WithCapabilities(ai.ReinjectSystemPrompt{}),
	)
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "old question"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "old answer"}}},
	}
	result, err := agent.Run(
		t.Context(), "new question", struct{}{}, ai.WithMessageHistory(history),
	)
	if err != nil {
		t.Fatal(err)
	}
	first := captured[0].(ai.ModelRequest)
	if len(first.Parts) != 2 || first.Parts[0].(ai.SystemPromptPart).Content != "server prompt" ||
		first.Parts[1].(ai.UserPromptPart).Content != "old question" {
		t.Fatalf("unexpected reinjected request: %+v", first)
	}
	for _, message := range result.Messages() {
		if request, ok := message.(ai.ModelRequest); ok {
			for _, part := range request.Parts {
				if system, ok := part.(ai.SystemPromptPart); ok && system.Content == "server prompt" {
					t.Fatal("request-only reinjection changed durable history")
				}
			}
		}
	}
}

func TestReinjectSystemPromptKeepsAuthoritativeHistory(t *testing.T) {
	calls := 0
	var captured []ai.ModelMessage
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		captured = messages
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(ai.ReinjectSystemPrompt{}))
	agent.AddSystemPromptFunc(func(context.Context, *ai.RunContext[struct{}]) (string, error) {
		calls++
		return "server prompt", nil
	})
	history := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.SystemPromptPart{Content: "history prompt"},
		ai.UserPromptPart{Content: "old question"},
	}}}
	if _, err := agent.Run(t.Context(), "new question", struct{}{}, ai.WithMessageHistory(history)); err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("configured prompt was evaluated %d times", calls)
	}
	first := captured[0].(ai.ModelRequest)
	if first.Parts[0].(ai.SystemPromptPart).Content != "history prompt" {
		t.Fatalf("history prompt was replaced: %+v", first)
	}
}

func TestReinjectSystemPromptReplacesUntrustedHistory(t *testing.T) {
	var captured []ai.ModelMessage
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		captured = messages
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](
		model,
		ai.WithSystemPrompt("static server prompt"),
		ai.WithCapabilities(ai.ReinjectSystemPrompt{ReplaceExisting: true}),
	)
	agent.AddDynamicSystemPromptFunc(
		"tenant", func(_ context.Context, rc *ai.RunContext[struct{}]) (string, error) {
			return "dynamic for " + rc.Prompt.Content, nil
		},
	)
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.SystemPromptPart{Content: "only untrusted"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "old answer"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "untrusted"}, ai.UserPromptPart{Content: "old question"},
		}},
	}
	if _, err := agent.Run(t.Context(), "new question", struct{}{}, ai.WithMessageHistory(history)); err != nil {
		t.Fatal(err)
	}
	if len(captured) != 3 {
		t.Fatalf("system-only request was not removed: %+v", captured)
	}
	request := captured[1].(ai.ModelRequest)
	if len(request.Parts) != 3 || request.Parts[0].(ai.SystemPromptPart).Content != "static server prompt" ||
		request.Parts[1].(ai.SystemPromptPart).Content != "dynamic for new question" ||
		request.Parts[1].(ai.SystemPromptPart).DynamicRef != "tenant" {
		t.Fatalf("unexpected trusted prompts: %+v", request)
	}
	for _, message := range captured {
		if request, ok := message.(ai.ModelRequest); ok {
			for _, part := range request.Parts {
				if system, ok := part.(ai.SystemPromptPart); ok && strings.Contains(system.Content, "untrusted") {
					t.Fatalf("untrusted prompt remained: %+v", captured)
				}
			}
		}
	}
}

func TestReinjectSystemPromptErrorsAndEmptyConfiguration(t *testing.T) {
	capability := ai.ReinjectSystemPrompt{ReplaceExisting: true}
	request := ai.ModelRequestContext{Messages: []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}}},
	}}
	unchanged, err := capability.BeforeModelRequest(t.Context(), nil, request)
	if err != nil || len(unchanged.Messages) != 1 {
		t.Fatalf("nil run info changed request: %+v err=%v", unchanged, err)
	}

	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	emptyAgent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(capability))
	history := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.SystemPromptPart{Content: "remove me"}, ai.UserPromptPart{Content: "old"},
	}}}
	if _, err := emptyAgent.Run(t.Context(), "new", struct{}{}, ai.WithMessageHistory(history)); err != nil {
		t.Fatal(err)
	}

	failedAgent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(ai.ReinjectSystemPrompt{}))
	failedAgent.AddSystemPromptFunc(func(context.Context, *ai.RunContext[struct{}]) (string, error) {
		return "", errors.New("prompt unavailable")
	})
	_, err = failedAgent.Run(t.Context(), "new", struct{}{}, ai.WithMessageHistory([]ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "old"}}},
	}))
	if err == nil || !strings.Contains(err.Error(), "ai: system prompt: prompt unavailable") {
		t.Fatalf("unexpected prompt error: %v", err)
	}
}

func steeringAgent(t *testing.T) (*ai.Agent[struct{}, string], *[][]ai.ModelMessage) {
	t.Helper()
	var requests [][]ai.ModelMessage
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests = append(requests, messages)
		if len(requests) == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "steer", ToolCallID: "steer_call", Args: json.RawMessage(`{}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](
		model,
		ai.WithSystemPrompt("server prompt"),
		ai.WithCapabilities(ai.ReinjectSystemPrompt{ReplaceExisting: true}),
	)
	ai.AddTool(agent, "steer", func(_ context.Context, rc *ai.RunContext[struct{}], _ struct{}) (string, error) {
		_, err := rc.Enqueue(ai.SystemPromptPart{Content: "mid-run steering"})
		return "steered", err
	})
	return agent, &requests
}

func promptContents(messages []ai.ModelMessage) []string {
	var contents []string
	for _, message := range messages {
		request, ok := message.(ai.ModelRequest)
		if !ok {
			continue
		}
		for _, part := range request.Parts {
			switch part := part.(type) {
			case ai.SystemPromptPart:
				contents = append(contents, part.Content)
			case ai.UserPromptPart:
				contents = append(contents, part.Content)
			}
		}
	}
	return contents
}

func TestReinjectSystemPromptReplaceKeepsMidRunSystemPrompts(t *testing.T) {
	agent, requests := steeringAgent(t)
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "client prompt"}, ai.UserPromptPart{Content: "hi"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "hello"}}},
	}
	if _, err := agent.Run(t.Context(), "go", struct{}{}, ai.WithMessageHistory(history)); err != nil {
		t.Fatal(err)
	}
	got := promptContents((*requests)[1])
	if want := []string{"server prompt", "hi", "go", "mid-run steering"}; !slices.Equal(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestReinjectSystemPromptReplaceFreshConversationDoesNotDuplicate(t *testing.T) {
	agent, requests := steeringAgent(t)
	if _, err := agent.Run(t.Context(), "go", struct{}{}); err != nil {
		t.Fatal(err)
	}
	want := [][]string{{"server prompt", "go"}, {"server prompt", "go", "mid-run steering"}}
	for index, request := range *requests {
		if got := promptContents(request); !slices.Equal(got, want[index]) {
			t.Fatalf("request %d: got %q, want %q", index, got, want[index])
		}
	}
}

func TestReinjectSystemPromptReplacesRerenderedDynamicPrompt(t *testing.T) {
	var captured []ai.ModelMessage
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		captured = messages
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(ai.ReinjectSystemPrompt{ReplaceExisting: true}))
	agent.AddDynamicSystemPromptFunc("server", func(context.Context, *ai.RunContext[struct{}]) (string, error) {
		return "dynamic server prompt", nil
	})
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hi"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "hello"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.UserPromptPart{Content: "again"}, ai.SystemPromptPart{Content: "stale", DynamicRef: "server"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "hello again"}}},
	}
	if _, err := agent.Run(t.Context(), "go", struct{}{}, ai.WithMessageHistory(history)); err != nil {
		t.Fatal(err)
	}
	got := promptContents(captured)
	if want := []string{"dynamic server prompt", "hi", "again", "go"}; !slices.Equal(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
}
