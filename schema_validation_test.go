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

type constrainedOutput struct {
	State string `json:"state" jsonschema:"enum=ready,enum=done"`
}

type customValidatedOutput struct {
	State string `json:"state"`
}

func (o *customValidatedOutput) UnmarshalJSON(data []byte) error {
	type plain customValidatedOutput
	var value plain
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	if value.State != "ready" {
		return errors.New("state is not ready")
	}
	*o = customValidatedOutput(value)
	return nil
}

func TestOutputToolEnforcesJSONSchema(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 2 {
			retry := messages[len(messages)-1].(ai.ModelRequest).Parts[0].(ai.RetryPromptPart)
			if retry.ToolName != params.OutputTool.Name || len(retry.Errors) != 1 ||
				retry.Errors[0].Type != "enum" || retry.Errors[0].Location[0] != "state" ||
				retry.Errors[0].Input != "invalid" || retry.Timestamp.IsZero() ||
				!strings.Contains(retry.ModelResponse(), "1 validation error") {
				t.Fatalf("unexpected schema retry: %+v", retry)
			}
		}
		args := []byte(`{"state":"invalid"}`)
		if request == 2 {
			args = []byte(`{"state":"ready"}`)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: params.OutputTool.Name, ToolCallID: "result", Args: args,
		}}}, nil
	})
	result, err := ai.NewAgent[deps, constrainedOutput](model).Run(t.Context(), "go", deps{})
	if err != nil || result.Output.State != "ready" || request != 2 {
		t.Fatalf("schema-constrained output failed: result=%+v err=%v requests=%d", result, err, request)
	}
}

func TestToolValidationErrorsPreserveArrayLocations(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 3 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		}
		if request == 2 {
			retry := messages[len(messages)-1].(ai.ModelRequest).Parts[0].(ai.RetryPromptPart)
			if len(retry.Errors) != 1 || len(retry.Errors[0].Location) != 2 ||
				retry.Errors[0].Location[0] != "items" || retry.Errors[0].Location[1] != 0 ||
				retry.Errors[0].Input != "invalid" {
				t.Fatalf("array validation location was not preserved: %+v", retry)
			}
		}
		args := `{"items":["invalid"]}`
		if request == 2 {
			args = `{"items":[1]}`
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "batch", ToolCallID: "call", Args: []byte(args),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddRawTool(ai.ToolDefinition{Name: "batch", Schema: map[string]any{
		"type": "object",
		"properties": map[string]any{"items": map[string]any{
			"type": "array", "items": map[string]any{"type": "integer"},
		}},
		"required": []any{"items"},
	}}, func(context.Context, json.RawMessage) (any, error) {
		return "done", nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "done" || request != 3 {
		t.Fatalf("array validation retry failed: result=%+v err=%v requests=%d", result, err, request)
	}
	toolReturn := result.Messages()[4].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart)
	if toolReturn.Timestamp.IsZero() {
		t.Fatalf("generated tool return has no timestamp: %+v", toolReturn)
	}
}

func TestOutputToolRunsGoDecodingAfterSchemaValidation(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		state := "invalid"
		if request == 2 {
			state = "ready"
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: params.OutputTool.Name, ToolCallID: "result",
			Args: []byte(`{"state":"` + state + `"}`),
		}}}, nil
	})
	result, err := ai.NewAgent[deps, customValidatedOutput](model).Run(t.Context(), "go", deps{})
	if err != nil || result.Output.State != "ready" || request != 2 {
		t.Fatalf("custom output decoding failed: result=%+v err=%v requests=%d", result, err, request)
	}
}

func TestCustomOutputDecodingConsumesRetryBudget(t *testing.T) {
	for name, mode := range map[string]ai.OutputMode{"tool": ai.OutputModeTool, "native": ai.OutputModeNative} {
		t.Run(name, func(t *testing.T) {
			model := fakes.NewFunctionModel(func(
				_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				if mode == ai.OutputModeNative {
					return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{
						Content: `{"state":"invalid"}`,
					}}}, nil
				}
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
					ToolName: params.OutputTool.Name, ToolCallID: "result", Args: []byte(`{"state":"invalid"}`),
				}}}, nil
			})
			agent := ai.NewAgent[deps, customValidatedOutput](
				model, ai.WithOutputMode(mode), ai.WithRetryLimits(ai.RetryLimits{Output: 0}),
			)
			_, err := agent.Run(t.Context(), "go", deps{})
			if err == nil || !errors.Is(err, ai.ErrMaxRetriesExceeded) {
				t.Fatalf("custom decoding did not exhaust output budget: %v", err)
			}
		})
	}
}

func TestNativeOutputEnforcesJSONSchema(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if params.OutputSchema == nil {
			t.Fatal("native output schema missing")
		}
		content := `{"state":"invalid"}`
		if request == 2 {
			content = `{"state":"done"}`
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: content}}}, nil
	})
	agent := ai.NewAgent[deps, constrainedOutput](model, ai.WithOutputMode(ai.OutputModeNative))
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output.State != "done" || request != 2 {
		t.Fatalf("native schema validation failed: result=%+v err=%v requests=%d", result, err, request)
	}
}

func TestNativeOutputRunsGoDecodingAfterSchemaValidation(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		state := "invalid"
		if request == 2 {
			state = "ready"
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{
			Content: `{"state":"` + state + `"}`,
		}}}, nil
	})
	agent := ai.NewAgent[deps, customValidatedOutput](model, ai.WithOutputMode(ai.OutputModeNative))
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output.State != "ready" || request != 2 {
		t.Fatalf("custom native decoding failed: result=%+v err=%v requests=%d", result, err, request)
	}
}

func TestFunctionToolEnforcesPreparedJSONSchema(t *testing.T) {
	request := 0
	calls := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		args := []byte(`{"count":1}`)
		if request == 2 {
			args = []byte(`{"count":2}`)
		}
		if request == 3 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "bounded", ToolCallID: "call", Args: args,
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddRawTool(ai.ToolDefinition{
		Name: "bounded",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"count": map[string]any{"type": "integer", "minimum": 2},
			},
			"required":             []string{"count"},
			"additionalProperties": false,
		},
	}, func(context.Context, json.RawMessage) (any, error) {
		calls++
		return "ok", nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "done" || calls != 1 || request != 3 {
		t.Fatalf("tool schema validation failed: result=%+v err=%v calls=%d requests=%d", result, err, calls, request)
	}
}

func TestFunctionToolSchemaValidationConsumesRetryBudget(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "bounded", ToolCallID: "call", Args: []byte(`{"count":1}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	agent.AddRawTool(ai.ToolDefinition{
		Name: "bounded",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"count": map[string]any{"type": "integer", "minimum": 2},
			},
		},
	}, func(context.Context, json.RawMessage) (any, error) { return nil, nil }, ai.WithToolMaxRetries(0))
	_, err := agent.Run(t.Context(), "go", deps{})
	if err == nil || !errors.Is(err, ai.ErrMaxRetriesExceeded) {
		t.Fatalf("schema retry did not exhaust tool budget: %v", err)
	}
}

func TestTypedToolRunsGoDecodingAfterSchemaValidation(t *testing.T) {
	type args struct {
		Count int `json:"count"`
	}
	request := 0
	calls := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		if request == 3 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		}
		raw := []byte(`{"count":"wrong"}`)
		if request == 2 {
			raw = []byte(`{"count":2}`)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "decode", ToolCallID: "call", Args: raw,
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddPreparedTool(agent, "decode", func(context.Context, *ai.RunContext[deps], args) (string, error) {
		calls++
		return "ok", nil
	}, func(
		_ context.Context, _ *ai.RunContext[deps], tool ai.ToolDefinition,
	) (*ai.ToolDefinition, error) {
		tool.Schema = map[string]any{}
		return &tool, nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output != "done" || calls != 1 || request != 3 {
		t.Fatalf("typed decoding failed: result=%+v err=%v calls=%d requests=%d", result, err, calls, request)
	}
}

func TestPartialOutputEnforcesJSONSchema(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.ToolCallStartEvent{PartID: "output", ToolName: "final_result", ToolCallID: "result"},
			ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `{"state":"rea`},
			ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `dy"}`},
			ai.FinishEvent{},
		}
	})
	stream := ai.NewAgent[deps, constrainedOutput](model).RunStream(t.Context(), "go", deps{})
	var outputs []constrainedOutput
	for output, err := range stream.Outputs() {
		if err != nil {
			t.Fatal(err)
		}
		outputs = append(outputs, output)
	}
	if len(outputs) != 2 || outputs[0].State != "ready" || outputs[1].State != "ready" {
		t.Fatalf("invalid enum partial was emitted: %+v", outputs)
	}
}

func TestPartialOutputRunsGoDecodingAfterSchemaValidation(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.ToolCallStartEvent{PartID: "output", ToolName: "final_result", ToolCallID: "result"},
			ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `{"state":"rea`},
			ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `dy"}`},
			ai.FinishEvent{},
		}
	})
	stream := ai.NewAgent[deps, customValidatedOutput](model).RunStream(t.Context(), "go", deps{})
	var outputs []customValidatedOutput
	for output, err := range stream.Outputs() {
		if err != nil {
			t.Fatal(err)
		}
		outputs = append(outputs, output)
	}
	if len(outputs) != 2 || outputs[0].State != "ready" || outputs[1].State != "ready" {
		t.Fatalf("Go-invalid partial was emitted: %+v", outputs)
	}
}

func TestStreamedOutputRunsGoDecodingAfterSchemaValidation(t *testing.T) {
	tests := map[string]struct {
		mode   ai.OutputMode
		events []ai.ModelStreamEvent
	}{
		"tool": {
			events: []ai.ModelStreamEvent{
				ai.ToolCallStartEvent{PartID: "output", ToolName: "final_result", ToolCallID: "result"},
				ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `{"state":"invalid"}`}, ai.FinishEvent{},
			},
		},
		"native": {
			mode: ai.OutputModeNative,
			events: []ai.ModelStreamEvent{
				ai.TextDeltaEvent{PartID: "output", Delta: `{"state":"invalid"}`}, ai.FinishEvent{},
			},
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent { return test.events })
			stream := ai.NewAgent[deps, customValidatedOutput](model, ai.WithOutputMode(test.mode)).RunStream(
				t.Context(), "go", deps{},
			)
			var got error
			for _, err := range stream.Events() {
				if err != nil {
					got = err
				}
			}
			if got == nil || !strings.Contains(
				got.Error(), "Output validation failed during streaming, and retries are not supported.",
			) {
				t.Fatalf("unexpected streamed Go validation error: %v", got)
			}
		})
	}
}

func TestPreparedSchemasRejectCompilationErrors(t *testing.T) {
	t.Run("output", func(t *testing.T) {
		agent := ai.NewAgent[deps, constrainedOutput](fakes.NewTestModel())
		agent.AddOutputToolPrepareFunc(func(
			_ context.Context, _ *ai.RunContext[deps], tool ai.ToolDefinition,
		) (*ai.ToolDefinition, error) {
			tool.Schema = map[string]any{"type": "invalid"}
			return &tool, nil
		})
		_, err := agent.Run(t.Context(), "go", deps{})
		if err == nil || !strings.Contains(err.Error(), "ai: output schema: compile schema") {
			t.Fatalf("unexpected output schema compilation error: %v", err)
		}
	})

	t.Run("tool", func(t *testing.T) {
		agent := ai.NewAgent[deps, string](fakes.NewTestModel())
		agent.AddRawTool(ai.ToolDefinition{Name: "broken", Schema: map[string]any{"type": "invalid"}},
			func(context.Context, json.RawMessage) (any, error) { return nil, nil })
		_, err := agent.Run(t.Context(), "go", deps{})
		if err == nil || !strings.Contains(err.Error(), `ai: tool "broken" schema: compile schema`) {
			t.Fatalf("unexpected tool schema compilation error: %v", err)
		}
	})
}
