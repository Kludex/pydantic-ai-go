package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type deps struct {
	Location string
}

type nilModel struct{}

func (nilModel) Name() string { return "nil-model" }

func (nilModel) Request(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
	return nil, nil
}

type weatherArgs struct {
	City string `json:"city" jsonschema:"description=City name"`
	Unit string `json:"unit,omitempty" jsonschema:"enum=celsius,enum=fahrenheit"`
}

func TestAgentRejectsNilModelResponse(t *testing.T) {
	agent := ai.NewAgent[deps, string](nilModel{})
	_, err := agent.Run(t.Context(), "go", deps{})
	if err == nil || err.Error() != "ai: unexpected model behavior: model returned no response" {
		t.Fatalf("unexpected nil response error: %v", err)
	}
}

func TestRunPlainText(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	result, err := agent.Run(t.Context(), "hello", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "success (no tool calls)" {
		t.Fatalf("unexpected output %q", result.Output)
	}
	if result.Usage().Requests != 1 {
		t.Fatalf("unexpected usage %+v", result.Usage())
	}
	if len(result.Messages()) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result.Messages()))
	}
}

func TestRunCallsTools(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(),
		ai.WithInstructions("You are a weather assistant."),
	)
	var gotArgs weatherArgs
	ai.AddTool(agent, "get_weather", func(_ context.Context, rc *ai.RunContext[deps], args weatherArgs) (string, error) {
		gotArgs = args
		return "sunny in " + rc.Deps.Location, nil
	}, ai.WithDescription("Get current weather"))

	result, err := agent.Run(t.Context(), "weather?", deps{Location: "SF"})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "success" {
		t.Fatalf("unexpected output %q", result.Output)
	}
	if gotArgs.City != "a" || gotArgs.Unit != "celsius" {
		t.Fatalf("unexpected args %+v", gotArgs)
	}
	if result.Usage().Requests != 2 {
		t.Fatalf("unexpected usage %+v", result.Usage())
	}
}

type weather struct {
	City    string  `json:"city"`
	TempC   float64 `json:"temp_c"`
	Summary string  `json:"summary,omitempty"`
}

func TestRunStructuredOutput(t *testing.T) {
	model := fakes.NewTestModel()
	model.CustomOutputArgs = json.RawMessage(`{"city": "SF", "temp_c": 18.5}`)
	agent := ai.NewAgent[deps, weather](model)

	result, err := agent.Run(t.Context(), "weather?", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.City != "SF" || result.Output.TempC != 18.5 {
		t.Fatalf("unexpected output %+v", result.Output)
	}
}

func TestSimpleTool(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	ai.AddSimpleTool(agent, "now", func(context.Context, struct{}) (string, error) {
		return "2026-01-01", nil
	})
	if _, err := agent.Run(t.Context(), "time?", deps{}); err != nil {
		t.Fatal(err)
	}
}

func TestRawTool(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	agent.AddRawTool(ai.ToolDefinition{Name: "raw", Schema: map[string]any{"type": "object"}},
		func(_ context.Context, rawArgs json.RawMessage) (any, error) {
			return string(rawArgs), nil
		})
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
}

func TestToolRetryThenSuccess(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewFunctionModel(scriptedModel(t)))
	calls := 0
	ai.AddTool(agent, "flaky", func(context.Context, *ai.RunContext[deps], struct{}) (string, error) {
		calls++
		if calls == 1 {
			return "", ai.Retryf("try again")
		}
		return "ok", nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("expected 2 tool calls, got %d", calls)
	}
	if result.Output != "done" {
		t.Fatalf("unexpected output %q", result.Output)
	}
}

// scriptedModel keeps calling "flaky" until it returns a tool result, then answers "done".
func scriptedModel(t *testing.T) func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
	t.Helper()
	return func(_ context.Context, msgs []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
		last := msgs[len(msgs)-1]
		if req, ok := last.(ai.ModelRequest); ok {
			if _, ok := req.Parts[0].(ai.ToolReturnPart); ok {
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
			}
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "flaky", Args: json.RawMessage(`{}`), ToolCallID: "c1"},
		}}, nil
	}
}

func TestMaxRetriesExceeded(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewFunctionModel(scriptedModel(t)))
	ai.AddTool(agent, "flaky", func(context.Context, *ai.RunContext[deps], struct{}) (string, error) {
		return "", ai.Retryf("never works")
	})
	_, err := agent.Run(t.Context(), "go", deps{})
	if !errors.Is(err, ai.ErrMaxRetriesExceeded) {
		t.Fatalf("expected ErrMaxRetriesExceeded, got %v", err)
	}
}

func TestToolErrorAbortsRun(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewFunctionModel(scriptedModel(t)))
	ai.AddTool(agent, "flaky", func(context.Context, *ai.RunContext[deps], struct{}) (string, error) {
		return "", errors.New("boom")
	})
	_, err := agent.Run(t.Context(), "go", deps{})
	if err == nil || !errors.Is(err, err) || err.Error() != `ai: tool "flaky": boom` {
		t.Fatalf("unexpected error %v", err)
	}
}

func TestInvalidToolArgsRetries(t *testing.T) {
	first := true
	model := fakes.NewFunctionModel(func(_ context.Context, msgs []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
		if first {
			first = false
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "get_weather", Args: json.RawMessage(`{"city": 42}`), ToolCallID: "c1"},
			}}, nil
		}
		last := msgs[len(msgs)-1].(ai.ModelRequest)
		if _, ok := last.Parts[0].(ai.RetryPromptPart); !ok {
			return nil, fmt.Errorf("expected retry prompt, got %T", last.Parts[0])
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "recovered"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddTool(agent, "get_weather", func(context.Context, *ai.RunContext[deps], weatherArgs) (string, error) {
		return "", nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "recovered" {
		t.Fatalf("unexpected output %q", result.Output)
	}
}

func TestUnknownToolCall(t *testing.T) {
	var retry ai.RetryPromptPart
	model := fakes.NewFunctionModel(func(_ context.Context, msgs []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
		if len(msgs) == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "nope", ToolCallID: "bad", Args: json.RawMessage(`{}`)},
			}}, nil
		}
		retry = msgs[len(msgs)-1].(ai.ModelRequest).Parts[0].(ai.RetryPromptPart)
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "recovered"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "known", func(context.Context, struct{}) (string, error) { return "", nil })
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "recovered" {
		t.Fatalf("unexpected output %q", result.Output)
	}
	if retry.Content != "Unknown tool name: 'nope'. Available tools: 'known'" || retry.ToolCallID != "bad" {
		t.Fatalf("unexpected retry %+v", retry)
	}
}

func TestUsageLimits(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(),
		ai.WithUsageLimits(ai.UsageLimits{RequestLimit: 1}),
	)
	ai.AddSimpleTool(agent, "noop", func(context.Context, struct{}) (string, error) { return "", nil })
	_, err := agent.Run(t.Context(), "go", deps{})
	if !errors.Is(err, ai.ErrUsageLimitExceeded) {
		t.Fatalf("expected ErrUsageLimitExceeded, got %v", err)
	}
}

func TestOutputValidatorRetry(t *testing.T) {
	model := fakes.NewTestModel()
	model.CustomOutputText = "bad"
	calls := 0
	agent := ai.NewAgent[deps, string](model, ai.WithMaxRetries(2))
	agent.AddOutputValidator(func(_ context.Context, _ *ai.RunContext[deps], out string) error {
		calls++
		if calls == 1 {
			return ai.Retryf("not good enough")
		}
		return nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 || result.Output != "bad" {
		t.Fatalf("calls=%d output=%q", calls, result.Output)
	}
}

func TestDynamicInstructions(t *testing.T) {
	var seen string
	model := fakes.NewFunctionModel(func(_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
		seen = params.Instructions
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "ok"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithInstructions("static"))
	agent.AddInstructionsFunc(func(_ context.Context, rc *ai.RunContext[deps]) (string, error) {
		return "user is in " + rc.Deps.Location, nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{Location: "SF"}); err != nil {
		t.Fatal(err)
	}
	if seen != "static\n\nuser is in SF" {
		t.Fatalf("unexpected instructions %q", seen)
	}
}

func TestMessageHistory(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	first, err := agent.Run(t.Context(), "hello", deps{})
	if err != nil {
		t.Fatal(err)
	}
	second, err := agent.Run(t.Context(), "again", deps{}, ai.WithMessageHistory(first.Messages()))
	if err != nil {
		t.Fatal(err)
	}
	if len(second.Messages()) != len(first.Messages())+2 {
		t.Fatalf("expected history to grow by 2, got %d -> %d", len(first.Messages()), len(second.Messages()))
	}
	if len(second.NewMessages()) != 2 {
		t.Fatalf("expected 2 new messages, got %d", len(second.NewMessages()))
	}
}

func TestModifyAfterRunPanics(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	ai.AddSimpleTool(agent, "late", func(context.Context, struct{}) (string, error) { return "", nil })
}

func TestModelErrorPropagates(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return nil, errors.New("transport down")
	})
	agent := ai.NewAgent[deps, string](model)
	if _, err := agent.Run(t.Context(), "go", deps{}); err == nil {
		t.Fatal("expected error")
	}
}
