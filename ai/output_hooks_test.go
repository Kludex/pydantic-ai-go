package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type hookedOutput struct {
	Value int `json:"value"`
}

func decodeHookedRaw(raw any, output *hookedOutput) error {
	switch raw := raw.(type) {
	case json.RawMessage:
		return json.Unmarshal(raw, output)
	case []byte:
		return json.Unmarshal(raw, output)
	case string:
		return json.Unmarshal([]byte(raw), output)
	default:
		return fmt.Errorf("unexpected raw output type %T", raw)
	}
}

type fullOutputCapability struct {
	name string
	log  *[]string
}

func (*fullOutputCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (capability *fullOutputCapability) BeforeOutputValidation(
	_ context.Context, _ *ai.RunInfo, hook ai.OutputHookContext, raw any,
) (any, error) {
	*capability.log = append(*capability.log, capability.name+":before-validation")
	if capability.name == "outer" {
		if hook.Mode != ai.OutputHookModeTool || hook.OutputType != reflect.TypeFor[hookedOutput]() ||
			hook.ToolCall == nil || hook.ToolCall.ToolCallID != "output" || hook.ToolDefinition == nil ||
			hook.Schema["type"] != "object" || !hook.Structured || hook.Partial {
			return nil, fmt.Errorf("unexpected output hook context: %+v", hook)
		}
		cloned := hook.Clone()
		cloned.ToolCall.Args[0] = '['
		cloned.Schema["type"] = "array"
		cloned.ToolDefinition.Schema["type"] = "array"
		if string(hook.ToolCall.Args) != `{"value":1}` || hook.Schema["type"] != "object" ||
			hook.ToolDefinition.Schema["type"] != "object" {
			return nil, errors.New("output hook context clone mutated its source")
		}
	}
	var output hookedOutput
	if err := decodeHookedRaw(raw, &output); err != nil {
		return nil, err
	}
	output.Value++
	encoded, err := json.Marshal(output)
	return json.RawMessage(encoded), err
}

func (capability *fullOutputCapability) WrapOutputValidation(
	ctx context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, raw any, next ai.OutputValidationFunc,
) (any, error) {
	*capability.log = append(*capability.log, capability.name+":validation-wrapper-before")
	var output hookedOutput
	if err := decodeHookedRaw(raw, &output); err != nil {
		return nil, err
	}
	output.Value++
	raw, err := json.Marshal(output)
	if err != nil {
		return nil, err
	}
	validated, err := next(ctx, raw)
	*capability.log = append(*capability.log, capability.name+":validation-wrapper-after")
	return validated, err
}

func (capability *fullOutputCapability) AfterOutputValidation(
	_ context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, output any,
) (any, error) {
	*capability.log = append(*capability.log, capability.name+":after-validation")
	value := output.(hookedOutput)
	value.Value++
	return value, nil
}

func (capability *fullOutputCapability) BeforeOutputProcessing(
	_ context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, output any,
) (any, error) {
	*capability.log = append(*capability.log, capability.name+":before-processing")
	value := output.(hookedOutput)
	value.Value++
	return value, nil
}

func (capability *fullOutputCapability) WrapOutputProcessing(
	ctx context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, output any, next ai.OutputProcessingFunc,
) (any, error) {
	*capability.log = append(*capability.log, capability.name+":processing-wrapper-before")
	value := output.(hookedOutput)
	value.Value++
	processed, err := next(ctx, value)
	*capability.log = append(*capability.log, capability.name+":processing-wrapper-after")
	return processed, err
}

func (capability *fullOutputCapability) AfterOutputProcessing(
	_ context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, output any,
) (any, error) {
	*capability.log = append(*capability.log, capability.name+":after-processing")
	value := output.(hookedOutput)
	value.Value++
	return value, nil
}

func TestOutputValidationAndProcessingHooksUseMiddlewareOrder(t *testing.T) {
	var log []string
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "final_result", ToolCallID: "output", Args: json.RawMessage(`{"value":1}`),
		}}}, nil
	})
	outer := &fullOutputCapability{name: "outer", log: &log}
	inner := &fullOutputCapability{name: "inner", log: &log}
	agent := ai.NewAgent[deps, hookedOutput](model, ai.WithCapabilities(outer, inner))
	agent.AddOutputValidator(func(_ context.Context, _ *ai.RunContext[deps], output hookedOutput) error {
		log = append(log, fmt.Sprintf("validator:%d", output.Value))
		return nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output.Value != 13 {
		t.Fatalf("unexpected output hook result=%+v err=%v", result, err)
	}
	want := []string{
		"outer:before-validation", "inner:before-validation",
		"outer:validation-wrapper-before", "inner:validation-wrapper-before",
		"inner:validation-wrapper-after", "outer:validation-wrapper-after",
		"inner:after-validation", "outer:after-validation",
		"outer:before-processing", "inner:before-processing",
		"outer:processing-wrapper-before", "inner:processing-wrapper-before", "validator:11",
		"inner:processing-wrapper-after", "outer:processing-wrapper-after",
		"inner:after-processing", "outer:after-processing",
	}
	if !slices.Equal(log, want) {
		t.Fatalf("unexpected output hook order:\n got %v\nwant %v", log, want)
	}
}

func TestOutputValidationErrorHooksRecoverInsideOut(t *testing.T) {
	var calls []string
	inner := ai.OutputValidationErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, _ any, err error,
	) (any, error) {
		calls = append(calls, "inner: "+err.Error())
		return nil, fmt.Errorf("inner: %w", err)
	})
	outer := ai.OutputValidationErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, raw any, err error,
	) (any, error) {
		calls = append(calls, "outer: "+err.Error())
		if string(raw.(json.RawMessage)) != `{"value":"bad"}` {
			return nil, errors.New("raw output changed")
		}
		return hookedOutput{Value: 7}, nil
	})
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "final_result", ToolCallID: "output", Args: json.RawMessage(`{"value":"bad"}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, hookedOutput](model, ai.WithCapabilities(outer, inner))
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output.Value != 7 || len(calls) != 2 ||
		!strings.HasPrefix(calls[0], "inner:") || !strings.HasPrefix(calls[1], "outer: inner:") {
		t.Fatalf("unexpected output recovery result=%+v calls=%v err=%v", result, calls, err)
	}
}

func TestOutputProcessingErrorHooksRecoverInsideOut(t *testing.T) {
	wrapper := &failingOutputProcessingWrapper{}
	inner := ai.OutputProcessingErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, _ any, err error,
	) (any, error) {
		return nil, fmt.Errorf("inner: %w", err)
	})
	outer := ai.OutputProcessingErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, _ any, err error,
	) (any, error) {
		if err.Error() != "inner: processing failed" {
			return nil, fmt.Errorf("unexpected error: %w", err)
		}
		return hookedOutput{Value: 9}, nil
	})
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "final_result", ToolCallID: "output", Args: json.RawMessage(`{"value":1}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, hookedOutput](model, ai.WithCapabilities(outer, wrapper, inner))
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output.Value != 9 {
		t.Fatalf("unexpected processing recovery result=%+v err=%v", result, err)
	}
}

type failingOutputProcessingWrapper struct{}

func (*failingOutputProcessingWrapper) Setup(*ai.CapabilityRegistry) error { return nil }

func (*failingOutputProcessingWrapper) WrapOutputProcessing(
	context.Context, *ai.RunInfo, ai.OutputHookContext, any, ai.OutputProcessingFunc,
) (any, error) {
	return nil, errors.New("processing failed")
}
