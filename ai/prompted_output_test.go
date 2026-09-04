package ai_test

import (
	"context"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type promptedResult struct {
	City string `json:"city"`
}

func TestPromptedOutputRetriesAndPersistsSchemaInstructions(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if params.OutputMode != ai.OutputModePrompted || params.OutputTool != nil ||
			params.OutputSchema["type"] != "object" || !params.AllowText ||
			!strings.Contains(params.Instructions, "Always respond with a JSON object") ||
			!strings.Contains(params.Instructions, `"city":{"type":"string"}`) ||
			!strings.HasSuffix(params.Instructions, "Don't include any text or Markdown fencing before or after.") {
			t.Fatalf("unexpected prompted output parameters: %+v", params)
		}
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"city":1}`}}}, nil
		}
		last := messages[len(messages)-1].(ai.ModelRequest)
		retry := last.Parts[0].(ai.RetryPromptPart)
		if retry.Content != "" || len(retry.Errors) == 0 ||
			!strings.Contains(last.Instructions, "Always respond with a JSON object") {
			t.Fatalf("unexpected prompted retry history: %+v", last)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"city":"Paris"}`}}}, nil
	})
	agent := ai.NewAgent[deps, promptedResult](
		model, ai.WithInstructions("Be concise."), ai.WithOutputMode(ai.OutputModePrompted),
	)
	result, err := agent.Run(t.Context(), "city", deps{})
	if err != nil || result.Output.City != "Paris" || request != 2 {
		t.Fatalf("unexpected prompted result=%+v requests=%d err=%v", result, request, err)
	}
}

func TestPromptedOutputTemplatesCanBeOverriddenPerRun(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		want := "Agent schema:"
		if request == 1 {
			want = "Run schema:"
		}
		if !strings.Contains(params.Instructions, want+"\n\n{") {
			t.Fatalf("schema was not appended to custom template: %q", params.Instructions)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"city":"Rome"}`}}}, nil
	})
	agent := ai.NewAgent[deps, promptedResult](
		model,
		ai.WithOutputMode(ai.OutputModePrompted),
		ai.WithPromptedOutputTemplate("Agent schema:"),
	)
	if _, err := agent.Run(
		t.Context(), "run", deps{}, ai.WithRunPromptedOutputTemplate("Run schema:"),
	); err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Run(t.Context(), "agent", deps{}); err != nil {
		t.Fatal(err)
	}
}

func TestPromptedOutputEndStrategies(t *testing.T) {
	for _, strategy := range []ai.EndStrategy{
		ai.EndStrategyEarly, ai.EndStrategyGraceful, ai.EndStrategyExhaustive,
	} {
		t.Run(string(strategy), func(t *testing.T) {
			request := 0
			toolCalls := 0
			model := fakes.NewFunctionModel(func(
				_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				request++
				if request == 1 {
					return &ai.ModelResponse{Parts: []ai.ResponsePart{
						ai.TextPart{Content: `{"city":"first"}`},
						ai.ToolCallPart{ToolName: "observe", ToolCallID: "observe", Args: []byte(`{}`)},
					}}, nil
				}
				return &ai.ModelResponse{Parts: []ai.ResponsePart{
					ai.TextPart{Content: `{"city":"second"}`},
				}}, nil
			})
			agent := ai.NewAgent[deps, promptedResult](
				model, ai.WithOutputMode(ai.OutputModePrompted), ai.WithEndStrategy(strategy),
			)
			ai.AddSimpleTool(agent, "observe", func(context.Context, struct{}) (string, error) {
				toolCalls++
				return "observed", nil
			})
			result, err := agent.Run(t.Context(), "go", deps{})
			if err != nil {
				t.Fatal(err)
			}
			if strategy == ai.EndStrategyEarly {
				if result.Output.City != "first" || toolCalls != 0 || request != 1 {
					t.Fatalf("early prompted output did not preempt: result=%+v calls=%d requests=%d", result, toolCalls, request)
				}
			} else if result.Output.City != "second" || toolCalls != 1 || request != 2 {
				t.Fatalf("prompted strategy mismatch: result=%+v calls=%d requests=%d", result, toolCalls, request)
			}
		})
	}
}

func TestPromptedOutputStreaming(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"city":"Oslo"}`}}}, nil
	})
	agent := ai.NewAgent[deps, promptedResult](model, ai.WithOutputMode(ai.OutputModePrompted))
	stream := agent.RunStream(t.Context(), "go", deps{})
	var outputs []promptedResult
	for output, err := range stream.Outputs() {
		if err != nil {
			t.Fatal(err)
		}
		outputs = append(outputs, output)
	}
	if stream.Result() == nil || stream.Result().Output.City != "Oslo" ||
		len(outputs) == 0 || outputs[len(outputs)-1].City != "Oslo" {
		t.Fatalf("unexpected prompted stream: outputs=%+v result=%+v", outputs, stream.Result())
	}
}

func TestPromptedOutputTemplateCannotBeEmpty(t *testing.T) {
	for name, configure := range map[string]func(){
		"agent": func() { ai.WithPromptedOutputTemplate("") },
		"run":   func() { ai.WithRunPromptedOutputTemplate("") },
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected empty template panic")
				}
			}()
			configure()
		})
	}
}
