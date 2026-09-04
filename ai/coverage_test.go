package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestRunContextExposesUsageAndMessages(t *testing.T) {
	var empty ai.RunContext[deps]
	if empty.RevealedTools() != nil {
		t.Fatal("zero run context has revealed tools")
	}
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	var usage ai.Usage
	var msgCount int
	ai.AddTool(agent, "peek", func(_ context.Context, rc *ai.RunContext[deps], _ struct{}) (string, error) {
		usage = rc.Usage()
		msgCount = len(rc.Messages())
		if rc.RunID == "" || rc.ToolCallID == "" {
			t.Error("expected run ID and tool call ID")
		}
		return "", nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if usage.Requests != 1 || msgCount != 2 {
		t.Fatalf("usage=%+v msgCount=%d", usage, msgCount)
	}
}

func TestModelSettingsPassedThrough(t *testing.T) {
	var got ai.ModelSettings
	model := fakes.NewFunctionModel(func(_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
		got = params.Settings
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "ok"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithModelSettings(ai.ModelSettings{MaxTokens: 42}))
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if got.MaxTokens != 42 {
		t.Fatalf("settings not passed through: %+v", got)
	}
}

func TestTokenLimits(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "ok"}},
			Usage: ai.Usage{Requests: 1, InputTokens: 100, OutputTokens: 100},
		}, nil
	})
	for _, limits := range []ai.UsageLimits{
		{InputTokenLimit: 50},
		{OutputTokenLimit: 50},
		{TotalTokenLimit: 150},
	} {
		agent := ai.NewAgent[deps, string](model, ai.WithUsageLimits(limits))
		if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, ai.ErrUsageLimitExceeded) {
			t.Fatalf("limits %+v: expected ErrUsageLimitExceeded, got %v", limits, err)
		}
	}
}

func TestErrorMessages(t *testing.T) {
	retry := ai.Retryf("try %s", "again")
	if retry.Error() != "ai: model retry: try again" {
		t.Fatalf("unexpected retry message %q", retry.Error())
	}
	ube := &ai.UnexpectedModelBehaviorError{Message: "weird"}
	if ube.Error() != "ai: unexpected model behavior: weird" {
		t.Fatalf("unexpected message %q", ube.Error())
	}
}

func TestStructuredOutputInvalidThenValid(t *testing.T) {
	first := true
	model := fakes.NewFunctionModel(func(_ context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
		args := json.RawMessage(`{"city": "SF", "temp_c": 20}`)
		if first {
			first = false
			args = json.RawMessage(`not json`)
		} else {
			last := msgs[len(msgs)-1].(ai.ModelRequest)
			if _, ok := last.Parts[0].(ai.RetryPromptPart); !ok {
				t.Errorf("expected retry prompt, got %T", last.Parts[0])
			}
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "final_result", Args: args, ToolCallID: "c1"},
		}}, nil
	})
	agent := ai.NewAgent[deps, weather](model)
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.City != "SF" {
		t.Fatalf("unexpected output %+v", result.Output)
	}
}

func TestStructuredOutputTextOnlyTriggersRetry(t *testing.T) {
	first := true
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		if first {
			first = false
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "chatty"}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "final_result", Args: json.RawMessage(`{"city":"SF","temp_c":1}`)},
		}}, nil
	})
	agent := ai.NewAgent[deps, weather](model)
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.City != "SF" {
		t.Fatalf("unexpected output %+v", result.Output)
	}
}

func TestStructuredOutputValidator(t *testing.T) {
	calls := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "final_result", Args: json.RawMessage(`{"city":"SF","temp_c":1}`), ToolCallID: "c1"},
		}}, nil
	})
	agent := ai.NewAgent[deps, weather](model, ai.WithMaxRetries(2))
	agent.AddOutputValidator(func(_ context.Context, _ *ai.RunContext[deps], out weather) error {
		calls++
		if calls == 1 {
			return ai.Retryf("too cold")
		}
		return nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("expected 2 validator calls, got %d", calls)
	}
}

func TestOutputValidatorHardErrorAborts(t *testing.T) {
	model := fakes.NewTestModel()
	agent := ai.NewAgent[deps, string](model)
	agent.AddOutputValidator(func(context.Context, *ai.RunContext[deps], string) error {
		return errors.New("broken")
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err == nil {
		t.Fatal("expected error")
	}
}

func TestOutputValidatorRetriesExhausted(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	agent.AddOutputValidator(func(context.Context, *ai.RunContext[deps], string) error {
		return ai.Retryf("never")
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, ai.ErrMaxRetriesExceeded) {
		t.Fatalf("expected ErrMaxRetriesExceeded, got %v", err)
	}
}

func TestStructuredOutputInvalidRetriesExhausted(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "final_result", Args: json.RawMessage(`bad`)},
		}}, nil
	})
	agent := ai.NewAgent[deps, weather](model)
	if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, ai.ErrMaxRetriesExceeded) {
		t.Fatalf("expected ErrMaxRetriesExceeded, got %v", err)
	}
}

func TestStructuredOutputTextRetriesExhausted(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "chatty"}}}, nil
	})
	agent := ai.NewAgent[deps, weather](model)
	if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, ai.ErrMaxRetriesExceeded) {
		t.Fatalf("expected ErrMaxRetriesExceeded, got %v", err)
	}
}

func TestStructuredOutputValidatorFinalRetry(t *testing.T) {
	// A validator retry on the output tool sends the retry prompt back
	// alongside other pending tool parts.
	calls := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "final_result", Args: json.RawMessage(`{"city":"SF","temp_c":1}`), ToolCallID: "c1"},
		}}, nil
	})
	agent := ai.NewAgent[deps, weather](model, ai.WithMaxRetries(3))
	agent.AddOutputValidator(func(context.Context, *ai.RunContext[deps], weather) error {
		calls++
		if calls < 3 {
			return ai.Retryf("no")
		}
		return nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
}

func TestStructuredOutputValidatorHardError(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "final_result", Args: json.RawMessage(`{"city":"SF","temp_c":1}`)},
		}}, nil
	})
	agent := ai.NewAgent[deps, weather](model)
	agent.AddOutputValidator(func(context.Context, *ai.RunContext[deps], weather) error {
		return errors.New("hard failure")
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err == nil {
		t.Fatal("expected error")
	}
}

func TestUnsupportedOutputType(t *testing.T) {
	agent := ai.NewAgent[deps, chan int](fakes.NewTestModel())
	if _, err := agent.Run(t.Context(), "go", deps{}); err == nil {
		t.Fatal("expected error for unsupported output type")
	}
}

func TestInstructionsFuncError(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	agent.AddInstructionsFunc(func(context.Context, *ai.RunContext[deps]) (string, error) {
		return "", errors.New("no instructions today")
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err == nil {
		t.Fatal("expected error")
	}
}

func TestInstructionsFuncEmptySkipped(t *testing.T) {
	var seen string
	model := fakes.NewFunctionModel(func(_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
		seen = params.Instructions
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "ok"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddInstructionsFunc(func(context.Context, *ai.RunContext[deps]) (string, error) {
		return "", nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if seen != "" {
		t.Fatalf("expected empty instructions, got %q", seen)
	}
}

func TestAddValidatorAfterRunPanics(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	agent.AddOutputValidator(func(context.Context, *ai.RunContext[deps], string) error { return nil })
}

func TestAddToolWithUnsupportedArgsPanics(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	ai.AddSimpleTool(agent, "bad", func(context.Context, chan int) (string, error) { return "", nil })
}

func TestRunParts(t *testing.T) {
	var gotPart ai.UserPromptPart
	model := fakes.NewFunctionModel(func(_ context.Context, msgs []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
		gotPart = msgs[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart)
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "a cat"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	result, err := agent.RunParts(t.Context(), []ai.UserContent{
		ai.TextContent{Text: "what is this?"},
		ai.ImageURL{URL: "https://example.com/cat.png"},
		ai.BinaryContent{Data: []byte("bytes"), MediaType: "image/png"},
	}, deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "a cat" || len(gotPart.Contents) != 3 {
		t.Fatalf("output=%q contents=%+v", result.Output, gotPart.Contents)
	}
}

func TestRunStreamParts(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	stream := agent.RunStreamParts(t.Context(), []ai.UserContent{ai.TextContent{Text: "hi"}}, deps{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	if stream.Result() == nil {
		t.Fatal("expected result")
	}
}

func TestNativeOutputMode(t *testing.T) {
	var gotSchema map[string]any
	responses := []string{`not json`, `{"city":"SF","temp_c":18}`}
	i := 0
	model := fakes.NewFunctionModel(func(_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
		gotSchema = params.OutputSchema
		if params.OutputTool != nil {
			t.Error("output tool should not be set in native mode")
		}
		resp := responses[i]
		i++
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: resp}}}, nil
	})
	agent := ai.NewAgent[deps, weather](model, ai.WithOutputMode(ai.OutputModeNative))
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.City != "SF" {
		t.Fatalf("unexpected output %+v", result.Output)
	}
	if gotSchema == nil {
		t.Fatal("output schema not sent to the model")
	}
	if i != 2 {
		t.Fatalf("expected an invalid-JSON retry, got %d requests", i)
	}
}

func TestNativeOutputModeRetriesExhausted(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "never json"}}}, nil
	})
	agent := ai.NewAgent[deps, weather](model, ai.WithOutputMode(ai.OutputModeNative))
	if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, ai.ErrMaxRetriesExceeded) {
		t.Fatalf("expected ErrMaxRetriesExceeded, got %v", err)
	}
}
