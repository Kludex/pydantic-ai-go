// Package fakes provides Model implementations for tests.
package fakes

import (
	"context"
	"encoding/json"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go"
)

// FunctionModel calls a user-supplied function for every request.
type FunctionModel struct {
	fn func(ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error)
}

// NewFunctionModel creates a model backed by fn.
func NewFunctionModel(fn func(ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error)) *FunctionModel {
	return &FunctionModel{fn: fn}
}

// Name returns the stable fake model name.
func (m *FunctionModel) Name() string { return "function-model" }

// SupportsNativeTool lets FunctionModel inspect every provider-neutral native tool.
func (*FunctionModel) SupportsNativeTool(tool ai.NativeTool) bool {
	return ai.ValidateNativeTools([]ai.NativeTool{tool}) == nil
}

// Request delegates one detached request to the configured function.
func (m *FunctionModel) Request(ctx context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
	resp, err := m.fn(ctx, msgs, params)
	if err != nil {
		return nil, err
	}
	if resp.Usage.IsZero() {
		resp.Usage = ai.Usage{Requests: 1}
	}
	return resp, nil
}

// TestModel calls every registered tool once with arguments generated from
// each tool's schema, then produces a final output that conforms to the
// output schema (or fixed text for string outputs). No network involved.
type TestModel struct {
	// CustomOutputText overrides the default final text output.
	CustomOutputText string
	// CustomOutputArgs overrides the generated output-tool arguments.
	CustomOutputArgs json.RawMessage
}

// NewTestModel creates a TestModel with default behavior.
func NewTestModel() *TestModel { return &TestModel{} }

// Name returns the stable test model name.
func (m *TestModel) Name() string { return "test-model" }

// Request calls the next uncalled function tool or returns generated output.
func (m *TestModel) Request(_ context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
	called := calledTools(msgs)
	for _, tool := range params.Tools {
		if !called[tool.Name] {
			return &ai.ModelResponse{
				Usage: ai.Usage{Requests: 1},
				Parts: []ai.ResponsePart{ai.ToolCallPart{
					ToolName:   tool.Name,
					Args:       argsFromSchema(tool.Schema),
					ToolCallID: fmt.Sprintf("call_%s", tool.Name),
				}},
			}, nil
		}
	}
	if params.OutputTool != nil {
		args := m.CustomOutputArgs
		if args == nil {
			args = argsFromSchema(params.OutputTool.Schema)
		}
		return &ai.ModelResponse{
			Usage: ai.Usage{Requests: 1},
			Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName:   params.OutputTool.Name,
				Args:       args,
				ToolCallID: "call_final",
			}},
		}, nil
	}
	text := m.CustomOutputText
	if text == "" {
		text = "success (no tool calls)"
		if len(called) > 0 {
			text = "success"
		}
	}
	return &ai.ModelResponse{
		Usage: ai.Usage{Requests: 1},
		Parts: []ai.ResponsePart{ai.TextPart{Content: text}},
	}, nil
}

func calledTools(msgs []ai.ModelMessage) map[string]bool {
	called := map[string]bool{}
	for _, m := range msgs {
		resp, ok := m.(ai.ModelResponse)
		if !ok {
			continue
		}
		for _, call := range resp.ToolCalls() {
			called[call.ToolName] = true
		}
	}
	return called
}

func argsFromSchema(s map[string]any) json.RawMessage {
	args := map[string]any{}
	properties, _ := s["properties"].(map[string]any)
	for name, prop := range properties {
		args[name] = valueFromSchema(prop)
	}
	b, _ := json.Marshal(args) // generated values are always marshallable
	return b
}

func valueFromSchema(prop any) any {
	p, ok := prop.(map[string]any)
	if !ok {
		return "a"
	}
	if enum, ok := p["enum"].([]string); ok && len(enum) > 0 {
		return enum[0]
	}
	switch p["type"] {
	case "string":
		return "a"
	case "integer":
		return 0
	case "number":
		return 0.0
	case "boolean":
		return false
	case "array":
		return []any{}
	case "object":
		if nested, ok := p["properties"].(map[string]any); ok {
			out := map[string]any{}
			for name, np := range nested {
				out[name] = valueFromSchema(np)
			}
			return out
		}
		return map[string]any{}
	default:
		return nil
	}
}
