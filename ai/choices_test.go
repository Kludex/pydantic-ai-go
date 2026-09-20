package ai_test

import (
	"context"
	"encoding/json"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestChoices(t *testing.T) {
	calls := 0
	choices := ai.NewChoices("intent", "Choose the intent.",
		ai.Choice[int]{Key: "value", Description: "Return a value.", Value: 1},
		ai.Choice[int]{Key: "action", Action: func(context.Context) (int, error) { calls++; return 2, nil }},
	)
	if schema := choices.Schema(); schema["description"] != "Choose the intent." {
		t.Fatalf("unexpected choices schema: %#v", schema)
	}
	model := fakes.NewFunctionModel(func(_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
		if params.OutputTool.Name != "intent" {
			t.Fatalf("unexpected output tool: %+v", params.OutputTool)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "intent", ToolCallID: "choice", Args: json.RawMessage(`"action"`),
		}}}, nil
	})
	result, err := ai.NewChoicesAgent[struct{}](model, choices).Run(t.Context(), "choose", struct{}{})
	if err != nil || result.Output != 2 || calls != 1 {
		t.Fatalf("unexpected choice result=%+v calls=%d err=%v", result, calls, err)
	}

	strings := ai.NewStringChoices("", "", "first", "second")
	if len(strings.Schema()) == 0 {
		t.Fatal("missing string choices schema")
	}
	attempt := 0
	retrying := fakes.NewFunctionModel(func(_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
		attempt++
		arguments := []byte(`not-json`)
		switch attempt {
		case 2:
			arguments = []byte(`"unknown"`)
		case 3:
			arguments = []byte(`"first"`)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "choices", ToolCallID: "choice", Args: arguments}}}, nil
	})
	value, err := ai.NewChoicesAgent[struct{}](retrying, strings, ai.WithRetryLimits(ai.RetryLimits{Tools: 1, Output: 3})).Run(t.Context(), "choose", struct{}{})
	if err != nil || value.Output != "first" {
		t.Fatalf("unexpected retried choice: %+v %v", value, err)
	}
}

func TestChoicesValidation(t *testing.T) {
	for _, test := range []func(){
		func() { ai.NewChoices[int]("", "") },
		func() { ai.NewChoices("", "", ai.Choice[int]{}) },
		func() { ai.NewChoices("", "", ai.Choice[int]{Key: "same"}, ai.Choice[int]{Key: "same"}) },
		func() { ai.NewChoicesAgent[struct{}](fakes.NewTestModel(), ai.Choices[int]{}) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			test()
		}()
	}
}
