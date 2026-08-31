package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type unionAnswer interface{ answer() string }

type cityAnswer struct {
	City string `json:"city"`
}

func (answer cityAnswer) answer() string { return answer.City }

type refusalAnswer struct {
	Reason string `json:"reason"`
}

func (answer refusalAnswer) answer() string { return answer.Reason }

type recursiveAnswer struct {
	Name string           `json:"name"`
	Next *recursiveAnswer `json:"next,omitempty"`
}

func newAnswerUnion() ai.UnionOutput[unionAnswer] {
	return ai.NewUnionOutput(
		ai.NewOutputAlternative("city", func(answer cityAnswer) unionAnswer { return answer }),
		ai.NewOutputAlternative("refusal", func(answer refusalAnswer) unionAnswer { return answer }),
	)
}

func TestUnionOutputAcrossStructuredModes(t *testing.T) {
	for name, mode := range map[string]ai.OutputMode{
		"tool": ai.OutputModeTool, "native": ai.OutputModeNative, "prompted": ai.OutputModePrompted,
	} {
		t.Run(name, func(t *testing.T) {
			model := fakes.NewFunctionModel(func(
				_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				assertUnionSchema(t, outputSchema(params))
				raw := []byte(`{"result":{"kind":"city","data":{"city":"Paris"}}}`)
				if mode == ai.OutputModeTool {
					return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
						ToolName: params.OutputTool.Name, ToolCallID: "output", Args: raw,
					}}}, nil
				}
				if mode == ai.OutputModePrompted && !strings.Contains(params.OutputPrompt, `"kind"`) {
					t.Fatalf("union schema missing from output prompt: %s", params.OutputPrompt)
				}
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: string(raw)}}}, nil
			})
			agent := ai.NewUnionAgent[struct{}](model, newAnswerUnion(), ai.WithOutputMode(mode))
			result, err := agent.Run(t.Context(), "answer", struct{}{})
			if err != nil {
				t.Fatal(err)
			}
			if result.Output.answer() != "Paris" {
				t.Fatalf("unexpected union output: %#v", result.Output)
			}
		})
	}
}

func TestUnionOutputRetriesAndRunsValidators(t *testing.T) {
	request := 0
	validated := false
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{
				Content: `{"result":{"kind":"unknown","data":{}}}`,
			}}}, nil
		}
		if len(messages) < 2 {
			t.Fatal("validation retry was not persisted")
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{
			Content: `{"result":{"kind":"refusal","data":{"reason":"unsafe"}}}`,
		}}}, nil
	})
	agent := ai.NewUnionAgent[struct{}](model, newAnswerUnion(), ai.WithOutputMode(ai.OutputModeNative))
	agent.AddOutputValidator(func(_ context.Context, _ *ai.RunContext[struct{}], output unionAnswer) error {
		validated = output.answer() == "unsafe"
		return nil
	})
	result, err := agent.Run(t.Context(), "answer", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.answer() != "unsafe" || request != 2 || !validated {
		t.Fatalf("unexpected result=%#v requests=%d validated=%v", result.Output, request, validated)
	}
}

func TestUnionOutputStreams(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{
			Content: `{"result":{"kind":"city","data":{"city":"Oslo"}}}`,
		}}}, nil
	})
	stream := ai.NewUnionAgent[struct{}](
		model, newAnswerUnion(), ai.WithOutputMode(ai.OutputModeNative),
	).RunStream(t.Context(), "answer", struct{}{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	if stream.Result() == nil || stream.Result().Output.answer() != "Oslo" {
		t.Fatalf("unexpected streamed union: %+v", stream.Result())
	}
}

func TestUnionOutputSchemaPrefixesRecursiveReferences(t *testing.T) {
	union := ai.NewUnionOutput(
		ai.NewOutputAlternative("recursive", func(answer recursiveAnswer) unionAnswer {
			return cityAnswer{City: answer.Name}
		}),
		ai.NewOutputAlternative("city", func(answer cityAnswer) unionAnswer { return answer }),
	)
	encoded := mustJSON(union.Schema())
	if strings.Contains(encoded, `"$ref":"#"`) ||
		!strings.Contains(encoded, `"$ref":"#/properties/result/anyOf/0/properties/data"`) {
		t.Fatalf("recursive reference was not prefixed: %s", encoded)
	}
}

func TestUnionOutputSchemaRewritesNestedDefinitions(t *testing.T) {
	rawSchema := map[string]any{
		"$defs": map[string]any{
			"place/name": map[string]any{
				"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string"}},
				"required": []any{"name"}, "additionalProperties": false,
			},
		},
		"$ref": "#/$defs/place~1name",
	}
	union := ai.NewUnionOutput(
		ai.NewRawOutputAlternative("nested", rawSchema, func(raw json.RawMessage) (unionAnswer, error) {
			var answer cityAnswer
			return answer, json.Unmarshal(raw, &answer)
		}),
		ai.NewOutputAlternative("city", func(answer cityAnswer) unionAnswer { return answer }),
	)
	schema := union.Schema()
	definitions := schema["$defs"].(map[string]any)
	if len(definitions) == 0 {
		t.Fatalf("nested definitions missing: %+v", schema)
	}
	encoded := mustJSON(schema)
	if strings.Contains(encoded, `"#/$defs/place~1name"`) ||
		!strings.Contains(encoded, `"#/$defs/alternative_0_place~1name"`) {
		t.Fatalf("nested references were not rewritten: %s", encoded)
	}
	definitions["mutated"] = true
	if _, exists := union.Schema()["$defs"].(map[string]any)["mutated"]; exists {
		t.Fatal("Schema returned shared definitions")
	}
}

func TestUnionOutputDecode(t *testing.T) {
	union := newAnswerUnion()
	decoded, err := union.Decode([]byte(`{"result":{"kind":"city","data":{"city":"Rome"}}}`))
	if err != nil || decoded.answer() != "Rome" {
		t.Fatalf("unexpected decoded output: %#v %v", decoded, err)
	}
	for name, raw := range map[string]string{
		"JSON":  `{`,
		"kind":  `{"result":{"kind":"missing","data":{}}}`,
		"value": `{"result":{"kind":"city","data":{"city":1}}}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := union.Decode([]byte(raw)); err == nil {
				t.Fatal("expected decode error")
			}
		})
	}
}

func TestUnionOutputConfigurationValidation(t *testing.T) {
	city := ai.NewOutputAlternative("city", func(answer cityAnswer) unionAnswer { return answer })
	tests := map[string]func(){
		"empty kind": func() {
			ai.NewOutputAlternative("", func(answer cityAnswer) unionAnswer { return answer })
		},
		"nil converter": func() {
			ai.NewOutputAlternative[unionAnswer, cityAnswer]("city", nil)
		},
		"unsupported value": func() {
			ai.NewOutputAlternative("channel", func(chan int) unionAnswer { return cityAnswer{} })
		},
		"raw empty kind": func() {
			ai.NewRawOutputAlternative("", map[string]any{}, func(json.RawMessage) (unionAnswer, error) {
				return cityAnswer{}, nil
			})
		},
		"raw nil schema": func() {
			ai.NewRawOutputAlternative("raw", nil, func(json.RawMessage) (unionAnswer, error) {
				return cityAnswer{}, nil
			})
		},
		"raw nil decoder": func() {
			ai.NewRawOutputAlternative[unionAnswer]("raw", map[string]any{}, nil)
		},
		"one alternative": func() { ai.NewUnionOutput(city) },
		"duplicate kind":  func() { ai.NewUnionOutput(city, city) },
		"zero alternative": func() {
			ai.NewUnionOutput(city, ai.OutputAlternative[unionAnswer]{})
		},
		"zero union": func() {
			ai.NewUnionAgent[struct{}](fakes.NewTestModel(), ai.UnionOutput[unionAnswer]{})
		},
	}
	for name, operation := range tests {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			operation()
		})
	}
}

func TestUnionAgentRejectsPerRunOutputSpecialization(t *testing.T) {
	agent := ai.NewUnionAgent[struct{}](fakes.NewTestModel(), newAnswerUnion())
	if _, err := ai.RunAs[cityAnswer](t.Context(), agent, "answer", struct{}{}); !errors.Is(err, ai.ErrOutputTypeOverrideWithUnion) {
		t.Fatalf("unexpected RunAs error: %v", err)
	}
	stream := ai.RunStreamAs[cityAnswer](t.Context(), agent, "answer", struct{}{})
	var streamErr error
	for _, err := range stream.Events() {
		streamErr = err
	}
	if !errors.Is(streamErr, ai.ErrOutputTypeOverrideWithUnion) {
		t.Fatalf("unexpected RunStreamAs error: %v", streamErr)
	}
}

func outputSchema(params ai.ModelRequestParams) map[string]any {
	if params.OutputTool != nil {
		return params.OutputTool.Schema
	}
	return params.OutputSchema
}

func assertUnionSchema(t *testing.T, schema map[string]any) {
	t.Helper()
	result := schema["properties"].(map[string]any)["result"].(map[string]any)
	if variants := result["anyOf"].([]any); len(variants) != 2 {
		t.Fatalf("unexpected union variants: %+v", variants)
	}
}

func mustJSON(value any) string {
	data, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	return string(data)
}
