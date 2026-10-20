package ai_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
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

	history := first.Messages()
	second, err := agent.Run(
		t.Context(), "second", promptDeps{Value: "two"}, ai.WithMessageHistory(history),
	)
	if err != nil {
		t.Fatal(err)
	}
	if kept := history[0].(ai.ModelRequest).Parts[2].(ai.SystemPromptPart); kept.Content != "Policy one" {
		t.Fatalf("reevaluation mutated caller history: %+v", kept)
	}
	if fresh := second.Messages()[0].(ai.ModelRequest).Parts[2].(ai.SystemPromptPart); fresh.Content != "Policy two" {
		t.Fatalf("second run kept the stale prompt: %+v", fresh)
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

func TestSystemPromptMetadataSurvivesMergeAndContinuation(t *testing.T) {
	origin := map[string]any{"origin": "my_app.note"}
	var bodies []string
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		body, _ := io.ReadAll(request.Body)
		bodies = append(bodies, string(body))
		reply := `{"choices":[{"message":{"content":"done"},"finish_reason":"stop"}]}`
		if len(bodies) == 1 {
			reply = `{"choices":[{"message":{"tool_calls":[{"id":"c1","type":"function",` +
				`"function":{"name":"note","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`
		}
		_, _ = io.WriteString(response, reply)
	}))
	defer server.Close()
	model := openai.NewModel("gpt-5", openai.WithAPIKey("key"), openai.WithBaseURL(server.URL),
		openai.WithHTTPClient(server.Client()))
	agent := ai.NewAgent[struct{}, string](model)
	ai.AddTool(agent, "note", func(_ context.Context, rc *ai.RunContext[struct{}], _ struct{}) (string, error) {
		_, err := rc.Enqueue(ai.ModelRequest{
			Parts:    []ai.RequestPart{ai.SystemPromptPart{Content: "Keep it short.", Metadata: origin}},
			Metadata: map[string]any{"lost": "on merge"},
		})
		return "noted", err
	})
	result, err := agent.Run(t.Context(), "go", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	origin["origin"] = "mutated"
	data, err := ai.MarshalMessages(result.Messages())
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(strings.Join(bodies, ""), "my_app.note") {
		t.Fatalf("system prompt metadata reached the model: %s", bodies)
	}
	restored, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Run(t.Context(), "again", struct{}{}, ai.WithMessageHistory(restored)); err != nil {
		t.Fatal(err)
	}
	var marked int
	for _, message := range restored {
		request, ok := message.(ai.ModelRequest)
		if !ok {
			continue
		}
		for _, part := range request.Parts {
			if prompt, ok := part.(ai.SystemPromptPart); ok && prompt.Metadata["origin"] == "my_app.note" {
				marked++
			}
		}
	}
	if marked != 1 || strings.Contains(strings.Join(bodies, ""), "my_app.note") {
		t.Fatalf("metadata was lost or sent: marked=%d bodies=%s", marked, bodies)
	}
}
