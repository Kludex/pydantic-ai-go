package ai_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestDynamicSystemPromptReevaluatesResumedHistory(t *testing.T) {
	var requests [][]ai.ModelMessage
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests = append(requests, messages)
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	type promptDeps struct{ Value string }
	agent := ai.NewAgent[promptDeps, string](model, ai.WithSystemPrompt("Static system prompt."))
	staticCalls := 0
	dynamicCalls := 0
	agent.AddSystemPromptFunc(func(_ context.Context, rc *ai.RunContext[promptDeps]) (string, error) {
		staticCalls++
		return "Initial " + rc.Deps.Value, nil
	})
	agent.AddDynamicSystemPromptFunc("tenant-policy", func(
		_ context.Context, rc *ai.RunContext[promptDeps],
	) (string, error) {
		dynamicCalls++
		if rc.Model == nil {
			t.Fatal("system prompt ran before model selection")
		}
		return "Policy " + rc.Deps.Value, nil
	})

	first, err := agent.Run(t.Context(), "first", promptDeps{Value: "one"})
	if err != nil {
		t.Fatal(err)
	}
	firstRequest := requests[0][0].(ai.ModelRequest)
	if len(firstRequest.Parts) != 4 {
		t.Fatalf("unexpected initial request parts: %+v", firstRequest.Parts)
	}
	static := firstRequest.Parts[0].(ai.SystemPromptPart)
	initial := firstRequest.Parts[1].(ai.SystemPromptPart)
	dynamic := firstRequest.Parts[2].(ai.SystemPromptPart)
	user := firstRequest.Parts[3].(ai.UserPromptPart)
	if static.Content != "Static system prompt." || initial.Content != "Initial one" ||
		dynamic.Content != "Policy one" || dynamic.DynamicRef != "tenant-policy" || dynamic.Timestamp.IsZero() ||
		user.Timestamp.IsZero() {
		t.Fatalf("unexpected initial system prompts: %+v", firstRequest.Parts)
	}

	second, err := agent.Run(
		t.Context(), "second", promptDeps{Value: "two"}, ai.WithMessageHistory(first.Messages()),
	)
	if err != nil {
		t.Fatal(err)
	}
	resumed := requests[1][0].(ai.ModelRequest)
	updated := resumed.Parts[2].(ai.SystemPromptPart)
	if updated.Content != "Policy two" || updated.DynamicRef != "tenant-policy" ||
		!updated.Timestamp.After(dynamic.Timestamp) {
		t.Fatalf("dynamic system prompt was not reevaluated: before=%+v after=%+v", dynamic, updated)
	}
	latest := requests[1][len(requests[1])-1].(ai.ModelRequest)
	for _, part := range latest.Parts {
		if _, ok := part.(ai.SystemPromptPart); ok {
			t.Fatalf("resumed run duplicated system prompts: %+v", latest.Parts)
		}
	}
	if staticCalls != 1 || dynamicCalls != 2 || len(second.NewMessages()) != 2 {
		t.Fatalf("unexpected callback counts/history: static=%d dynamic=%d new=%+v", staticCalls, dynamicCalls, second.NewMessages())
	}
}

func TestDynamicSystemPromptResumeEdgeCases(t *testing.T) {
	t.Run("unknown ID is preserved", func(t *testing.T) {
		model := fakes.NewFunctionModel(func(
			_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			part := messages[0].(ai.ModelRequest).Parts[0].(ai.SystemPromptPart)
			if part.Content != "old" || part.DynamicRef != "unknown" {
				t.Fatalf("unknown dynamic prompt changed: %+v", part)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		})
		history := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "old", DynamicRef: "unknown"},
		}}}
		if _, err := ai.NewAgent[deps, string](model).Run(
			t.Context(), "go", deps{}, ai.WithMessageHistory(history),
		); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("reevaluation error", func(t *testing.T) {
		sentinel := errors.New("unavailable")
		agent := ai.NewAgent[deps, string](fakes.NewTestModel())
		agent.AddDynamicSystemPromptFunc("policy", func(context.Context, *ai.RunContext[deps]) (string, error) {
			return "", sentinel
		})
		history := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "old", DynamicRef: "policy"},
		}}}
		_, err := agent.Run(t.Context(), "go", deps{}, ai.WithMessageHistory(history))
		if !errors.Is(err, sentinel) || !strings.Contains(err.Error(), `dynamic system prompt "policy"`) {
			t.Fatalf("unexpected reevaluation error: %v", err)
		}
	})
}

func TestStaticSystemPromptFunctionBehavior(t *testing.T) {
	t.Run("empty prompt is omitted", func(t *testing.T) {
		model := fakes.NewFunctionModel(func(
			_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			parts := messages[0].(ai.ModelRequest).Parts
			if len(parts) != 1 {
				t.Fatalf("empty system prompt was retained: %+v", parts)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		})
		agent := ai.NewAgent[deps, string](model)
		agent.AddSystemPromptFunc(func(context.Context, *ai.RunContext[deps]) (string, error) {
			return "", nil
		})
		if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("error aborts run", func(t *testing.T) {
		sentinel := errors.New("unavailable")
		agent := ai.NewAgent[deps, string](fakes.NewTestModel())
		agent.AddSystemPromptFunc(func(context.Context, *ai.RunContext[deps]) (string, error) {
			return "", sentinel
		})
		_, err := agent.Run(t.Context(), "go", deps{})
		if !errors.Is(err, sentinel) || !strings.Contains(err.Error(), "ai: system prompt") {
			t.Fatalf("unexpected system prompt error: %v", err)
		}
	})
}

func TestDynamicSystemPromptErrors(t *testing.T) {
	sentinel := errors.New("unavailable")
	model := fakes.NewTestModel()
	agent := ai.NewAgent[deps, string](model)
	agent.AddDynamicSystemPromptFunc("policy", func(context.Context, *ai.RunContext[deps]) (string, error) {
		return "", sentinel
	})
	_, err := agent.Run(t.Context(), "go", deps{})
	if !errors.Is(err, sentinel) || !strings.Contains(err.Error(), `dynamic system prompt "policy"`) {
		t.Fatalf("unexpected dynamic system prompt error: %v", err)
	}
}

func TestSystemPromptRegistrationValidation(t *testing.T) {
	assertSystemPromptPanic(t, "ai: dynamic system prompt ID must not be empty", func() {
		ai.NewAgent[deps, string](nil).AddDynamicSystemPromptFunc("", nil)
	})
	agent := ai.NewAgent[deps, string](nil)
	agent.AddDynamicSystemPromptFunc("policy", func(context.Context, *ai.RunContext[deps]) (string, error) {
		return "", nil
	})
	assertSystemPromptPanic(t, `ai: duplicate dynamic system prompt ID "policy"`, func() {
		agent.AddDynamicSystemPromptFunc("policy", nil)
	})
}

func assertSystemPromptPanic(t *testing.T, want string, fn func()) {
	t.Helper()
	defer func() {
		value := recover()
		if value != want {
			t.Fatalf("unexpected panic: got %v want %q", value, want)
		}
	}()
	fn()
}
