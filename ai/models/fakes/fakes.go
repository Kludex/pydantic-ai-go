// Package fakes provides Model implementations for tests.
package fakes

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
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

// ModelProfile keeps provider-neutral tool availability parts visible to tests.
func (*FunctionModel) ModelProfile() ai.ModelProfile {
	return ai.ModelProfile{DefaultOutputMode: ai.OutputModeTool, SupportsToolAvailabilityDelta: true}
}

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

// ModelProfile keeps provider-neutral tool availability parts visible to tests.
func (*TestModel) ModelProfile() ai.ModelProfile {
	return ai.ModelProfile{DefaultOutputMode: ai.OutputModeTool, SupportsToolAvailabilityDelta: true}
}

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

func argsFromSchema(schema map[string]any) json.RawMessage {
	args := map[string]any{}
	properties, _ := schema["properties"].(map[string]any)
	for name, property := range properties {
		args[name] = valueFromSchema(property, schema, map[string]bool{})
	}
	data, _ := json.Marshal(args)
	return data
}

func valueFromSchema(value any, root map[string]any, resolving map[string]bool) any {
	if allowed, ok := value.(bool); ok {
		if allowed {
			return "a"
		}
		return nil
	}
	schema, ok := value.(map[string]any)
	if !ok {
		return "a"
	}
	if constant, exists := schema["const"]; exists {
		return constant
	}
	switch values := schema["enum"].(type) {
	case []string:
		if len(values) > 0 {
			return values[0]
		}
	case []any:
		if len(values) > 0 {
			return values[0]
		}
	}
	if ref, _ := schema["$ref"].(string); ref != "" {
		if resolving[ref] {
			return map[string]any{}
		}
		if resolved := resolveLocalRef(root, ref); resolved != nil {
			resolving[ref] = true
			result := valueFromSchema(resolved, root, resolving)
			delete(resolving, ref)
			return result
		}
	}
	for _, key := range []string{"anyOf", "oneOf"} {
		if alternatives, ok := schema[key].([]any); ok && len(alternatives) > 0 {
			return valueFromSchema(alternatives[0], root, resolving)
		}
	}
	switch schema["type"] {
	case "string":
		if schema["format"] == "date-time" {
			return "2000-01-01T00:00:00Z"
		}
		if schema["contentEncoding"] == "base64" {
			return "YQ=="
		}
		return "a"
	case "integer":
		return 0
	case "number":
		return 0.0
	case "boolean":
		return false
	case "array":
		prefix, _ := schema["prefixItems"].([]any)
		items := schema["items"]
		if legacy, ok := items.([]any); ok {
			if len(prefix) == 0 {
				prefix = legacy
			}
			items = nil
		}
		maximum, hasMaximum := schemaLimit(schema["maxItems"])
		minimum, _ := schemaLimit(schema["minItems"])
		result := make([]any, 0, len(prefix))
		for index, item := range prefix {
			if hasMaximum && float64(index) >= maximum && float64(index) >= minimum {
				break
			}
			result = append(result, valueFromSchema(item, root, resolving))
		}
		for float64(len(result)) < minimum {
			result = append(result, valueFromSchema(items, root, resolving))
		}
		if items != nil && len(prefix) == 0 && (!hasMaximum || maximum > 0) && len(result) == 0 {
			result = append(result, valueFromSchema(items, root, resolving))
		}
		return result
	case "object":
		result := map[string]any{}
		if properties, ok := schema["properties"].(map[string]any); ok {
			for name, property := range properties {
				result[name] = valueFromSchema(property, root, resolving)
			}
		}
		return result
	default:
		return nil
	}
}

func schemaLimit(value any) (float64, bool) {
	switch value := value.(type) {
	case int:
		return float64(value), true
	case float64:
		return value, true
	default:
		return 0, false
	}
}

func resolveLocalRef(root map[string]any, ref string) any {
	if ref == "#" {
		return root
	}
	if !strings.HasPrefix(ref, "#/") {
		return nil
	}
	var current any = root
	for _, raw := range strings.Split(strings.TrimPrefix(ref, "#/"), "/") {
		token := strings.ReplaceAll(strings.ReplaceAll(raw, "~1", "/"), "~0", "~")
		switch value := current.(type) {
		case map[string]any:
			current = value[token]
		case []any:
			index, err := strconv.Atoi(token)
			if err != nil || strconv.Itoa(index) != token || index < 0 || index >= len(value) {
				return nil
			}
			current = value[index]
		default:
			return nil
		}
	}
	return current
}
