package ai_test

import (
	"context"
	"errors"
	"iter"
	"reflect"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type outputFunctionInput struct {
	City string `json:"city"`
}

type rejectingOutputFunctionInput struct {
	City string `json:"city"`
}

func (*rejectingOutputFunctionInput) UnmarshalJSON([]byte) error {
	return errors.New("reject decoded value")
}

func TestTextOutputFunctionRetriesAndValidatesFinalValue(t *testing.T) {
	requests := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if !params.AllowText || params.OutputSchema != nil || params.OutputTool != nil {
			t.Fatalf("text output function received structured parameters: %+v", params)
		}
		requests++
		text := "retry"
		if requests == 2 {
			text = "hello world"
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: text}}}, nil
	})
	var hookContext ai.OutputHookContext
	before := ai.BeforeOutputProcessingFunc(func(
		_ context.Context, _ *ai.RunInfo, hook ai.OutputHookContext, value any,
	) (any, error) {
		hookContext = hook
		return value, nil
	})
	output := ai.NewTextOutputFunction("word_count", func(
		_ context.Context, rc *ai.RunContext[struct{}], text string,
	) (int, error) {
		if rc.PartialOutput {
			t.Fatal("ordinary text output was marked partial")
		}
		if text == "retry" {
			return 0, ai.Retryf("write more words")
		}
		return len(strings.Fields(text)), nil
	})
	if output.Name() != "word_count" {
		t.Fatalf("unexpected text output function name %q", output.Name())
	}
	overrideAgent := ai.NewTextOutputFunctionAgent(model, output)
	if _, err := ai.RunAs[string](t.Context(), overrideAgent, "count", struct{}{}); !errors.Is(err, ai.ErrOutputTypeOverrideWithCustomOutput) {
		t.Fatalf("unexpected text output override error: %v", err)
	}
	agent := ai.NewTextOutputFunctionAgent(model, output, ai.WithCapabilities(before))
	validatorCalls := 0
	agent.AddOutputValidator(func(_ context.Context, _ *ai.RunContext[struct{}], value int) error {
		validatorCalls++
		if value != 2 {
			t.Fatalf("text output validator saw %d", value)
		}
		return nil
	})
	result, err := agent.Run(t.Context(), "count", struct{}{})
	if err != nil || result.Output != 2 || requests != 2 || validatorCalls != 1 {
		t.Fatalf("unexpected text function result: %+v requests=%d validators=%d err=%v", result, requests, validatorCalls, err)
	}
	if hookContext.Mode != ai.OutputHookModeText || !hookContext.HasFunction ||
		hookContext.FunctionName != "word_count" || hookContext.OutputType != reflect.TypeFor[string]() {
		t.Fatalf("unexpected text output hook context: %+v", hookContext)
	}
}

func TestTextOutputFunctionStreamsPartialValues(t *testing.T) {
	model := &outputFunctionStreamingModel{events: []ai.ModelStreamEvent{
		ai.TextDeltaEvent{PartID: "text", Delta: "hello"},
		ai.TextDeltaEvent{PartID: "text", Delta: " world"},
		ai.FinishEvent{},
	}}
	partialCalls := 0
	output := ai.NewTextOutputFunction("length", func(
		_ context.Context, rc *ai.RunContext[struct{}], text string,
	) (int, error) {
		if rc.PartialOutput {
			partialCalls++
		}
		return len(text), nil
	})
	stream := ai.NewTextOutputFunctionAgent(model, output).RunStream(t.Context(), "length", struct{}{})
	values := make([]int, 0)
	for value, err := range stream.Outputs() {
		if err != nil {
			t.Fatal(err)
		}
		values = append(values, value)
	}
	result := stream.Result()
	if result == nil || len(values) == 0 || values[len(values)-1] != 11 || result.Output != 11 || partialCalls == 0 {
		t.Fatalf("unexpected streamed text function output: values=%v result=%+v partial=%d", values, result, partialCalls)
	}
}

func TestTextOutputFunctionValidation(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "text"}}}, nil
	})
	assertOutputFunctionPanics(t, func() {
		ai.NewTextOutputFunction("", func(context.Context, *ai.RunContext[struct{}], string) (int, error) {
			return 0, nil
		})
	})
	assertOutputFunctionPanics(t, func() {
		ai.NewTextOutputFunction[struct{}, int]("text", nil)
	})
	assertOutputFunctionPanics(t, func() {
		var invalid ai.TextOutputFunction[struct{}, int]
		ai.NewTextOutputFunctionAgent(model, invalid)
	})

	output := ai.NewTextOutputFunction("text", func(
		context.Context, *ai.RunContext[struct{}], string,
	) (int, error) {
		return 0, nil
	})
	agent := ai.NewTextOutputFunctionAgent(model, output, ai.WithCapabilities(ai.BeforeOutputProcessingFunc(func(
		context.Context, *ai.RunInfo, ai.OutputHookContext, any,
	) (any, error) {
		return 42, nil
	})))
	if _, err := agent.Run(t.Context(), "text", struct{}{}); err == nil || !strings.Contains(err.Error(), "expected string") {
		t.Fatalf("unexpected text output function type error: %v", err)
	}
}

func TestOutputFunctionSpecificationIsDetached(t *testing.T) {
	output := ai.NewOutputFunction("city_label", func(
		context.Context, *ai.RunContext[struct{}], outputFunctionInput,
	) (string, error) {
		return "done", nil
	})
	if output.Name() != "city_label" {
		t.Fatalf("unexpected output function name %q", output.Name())
	}
	value := output.Schema()
	value["mutated"] = true
	if _, exists := output.Schema()["mutated"]; exists {
		t.Fatal("output function schema aliases internal state")
	}
}

func TestOutputFunctionAcrossStructuredModes(t *testing.T) {
	for _, mode := range []ai.OutputMode{ai.OutputModeTool, ai.OutputModeNative, ai.OutputModePrompted} {
		t.Run(map[ai.OutputMode]string{
			ai.OutputModeTool: "tool", ai.OutputModeNative: "native", ai.OutputModePrompted: "prompted",
		}[mode], func(t *testing.T) {
			model := fakes.NewFunctionModel(func(
				_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				if mode == ai.OutputModeTool {
					if params.OutputTool == nil || params.OutputTool.Name != "city_label" {
						t.Fatalf("output function did not name its tool: %+v", params.OutputTool)
					}
					return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
						ToolName: "city_label", ToolCallID: "final", Args: []byte(`{"city":"Paris"}`),
					}}}, nil
				}
				if params.OutputSchema == nil || params.OutputMode != mode {
					t.Fatalf("missing output schema for mode %v: %+v", mode, params)
				}
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"city":"Paris"}`}}}, nil
			})
			called := 0
			output := ai.NewOutputFunction("city_label", func(
				_ context.Context, rc *ai.RunContext[struct{}], value outputFunctionInput,
			) (string, error) {
				called++
				if rc.RunStep != 1 || rc.PartialOutput {
					t.Fatalf("unexpected output function context: %+v", rc)
				}
				return strings.ToUpper(value.City), nil
			})
			agent := ai.NewOutputFunctionAgent(model, output, ai.WithOutputMode(mode))
			validatorCalls := 0
			agent.AddOutputValidator(func(_ context.Context, _ *ai.RunContext[struct{}], value string) error {
				validatorCalls++
				if value != "PARIS" {
					t.Fatalf("validator saw pre-function output %q", value)
				}
				return nil
			})
			result, err := agent.Run(t.Context(), "city", struct{}{})
			if err != nil {
				t.Fatal(err)
			}
			if result.Output != "PARIS" || called != 1 || validatorCalls != 1 {
				t.Fatalf("unexpected output-function result: %+v called=%d validators=%d", result, called, validatorCalls)
			}
		})
	}
}

func TestOutputFunctionRetriesAndHooksSeeInput(t *testing.T) {
	requests := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		city := "retry"
		if requests == 2 {
			city = "Paris"
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "city_label", ToolCallID: "call", Args: []byte(`{"city":"` + city + `"}`),
		}}}, nil
	})
	var hookContext ai.OutputHookContext
	before := ai.BeforeOutputProcessingFunc(func(
		_ context.Context, _ *ai.RunInfo, hook ai.OutputHookContext, value any,
	) (any, error) {
		hookContext = hook
		if _, ok := value.(outputFunctionInput); !ok {
			t.Fatalf("processing hook saw %T, expected outputFunctionInput", value)
		}
		return value, nil
	})
	output := ai.NewOutputFunction("city_label", func(
		_ context.Context, _ *ai.RunContext[struct{}], value outputFunctionInput,
	) (string, error) {
		if value.City == "retry" {
			return "", ai.Retryf("choose a real city")
		}
		return value.City, nil
	})
	agent := ai.NewOutputFunctionAgent(model, output, ai.WithCapabilities(before))
	result, err := agent.Run(t.Context(), "city", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "Paris" || requests != 2 {
		t.Fatalf("unexpected retried function output: %+v requests=%d", result, requests)
	}
	if !hookContext.HasFunction || hookContext.FunctionName != "city_label" ||
		hookContext.OutputType != reflect.TypeFor[outputFunctionInput]() {
		t.Fatalf("unexpected output hook context: %+v", hookContext)
	}
	foundRetry := false
	for _, message := range result.Messages() {
		request, ok := message.(ai.ModelRequest)
		if !ok {
			continue
		}
		for _, part := range request.Parts {
			if retry, ok := part.(ai.RetryPromptPart); ok && strings.Contains(retry.Content, "choose a real city") {
				foundRetry = true
			}
		}
	}
	if !foundRetry {
		t.Fatalf("output function retry missing from history: %+v", result.Messages())
	}
	if _, err := ai.RunAs[int](t.Context(), agent, "city", struct{}{}); !errors.Is(err, ai.ErrOutputTypeOverrideWithCustomOutput) || errors.Is(err, ai.ErrOutputTypeOverrideWithUnion) {
		t.Fatalf("unexpected output override error: %v", err)
	}
}

func TestOutputFunctionStreamsPartialValues(t *testing.T) {
	model := &outputFunctionStreamingModel{events: []ai.ModelStreamEvent{
		ai.ToolCallStartEvent{PartID: "output", ToolName: "city_label", ToolCallID: "call"},
		ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `{"city":"Par`},
		ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `is"}`},
		ai.FinishEvent{},
	}}
	partialCalls := 0
	output := ai.NewOutputFunction("city_label", func(
		_ context.Context, rc *ai.RunContext[struct{}], value outputFunctionInput,
	) (string, error) {
		if rc.PartialOutput {
			partialCalls++
		}
		return strings.ToUpper(value.City), nil
	})
	agent := ai.NewOutputFunctionAgent(model, output)
	stream := agent.RunStream(t.Context(), "city", struct{}{})
	values := make([]string, 0)
	for value, err := range stream.Outputs() {
		if err != nil {
			t.Fatal(err)
		}
		values = append(values, value)
	}
	result := stream.Result()
	if result == nil {
		t.Fatal("stream result is nil")
	}
	if len(values) == 0 || values[len(values)-1] != "PARIS" || result.Output != "PARIS" || partialCalls == 0 {
		t.Fatalf("unexpected streamed function output: values=%v result=%+v partial=%d", values, result, partialCalls)
	}
}

func TestOutputFunctionFailuresAndValidation(t *testing.T) {
	boom := errors.New("conversion failed")
	output := ai.NewOutputFunction("convert", func(
		context.Context, *ai.RunContext[struct{}], outputFunctionInput,
	) (string, error) {
		return "", boom
	})
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: params.OutputTool.Name, Args: []byte(`{"city":"Paris"}`),
		}}}, nil
	})
	agent := ai.NewOutputFunctionAgent(model, output, ai.WithOutputTool(ai.OutputToolConfig{Name: "custom"}))
	if _, err := agent.Run(t.Context(), "city", struct{}{}); !errors.Is(err, boom) {
		t.Fatalf("unexpected function error: %v", err)
	}

	wrongType := ai.NewOutputFunction("convert", func(
		context.Context, *ai.RunContext[struct{}], outputFunctionInput,
	) (string, error) {
		return "unused", nil
	})
	wrongTypeAgent := ai.NewOutputFunctionAgent(model, wrongType, ai.WithCapabilities(ai.BeforeOutputProcessingFunc(func(
		context.Context, *ai.RunInfo, ai.OutputHookContext, any,
	) (any, error) {
		return "wrong", nil
	})))
	if _, err := wrongTypeAgent.Run(t.Context(), "city", struct{}{}); err == nil ||
		!strings.Contains(err.Error(), "expected ai_test.outputFunctionInput") {
		t.Fatalf("unexpected output function input error: %v", err)
	}

	rejecting := ai.NewOutputFunction("reject", func(
		context.Context, *ai.RunContext[struct{}], rejectingOutputFunctionInput,
	) (string, error) {
		return "unused", nil
	})
	rejectingModel := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "reject", Args: []byte(`{"city":"Paris"}`),
		}}}, nil
	})
	if _, err := ai.NewOutputFunctionAgent(rejectingModel, rejecting, ai.WithRetryLimits(ai.RetryLimits{})).Run(
		t.Context(), "city", struct{}{},
	); !errors.Is(err, ai.ErrMaxRetriesExceeded) {
		t.Fatalf("unexpected custom decode error: %v", err)
	}

	assertOutputFunctionPanics(t, func() {
		ai.NewOutputFunction("", func(context.Context, *ai.RunContext[struct{}], outputFunctionInput) (string, error) {
			return "", nil
		})
	})
	assertOutputFunctionPanics(t, func() {
		ai.NewOutputFunction[struct{}, outputFunctionInput, string]("invalid", nil)
	})
	assertOutputFunctionPanics(t, func() {
		ai.NewOutputFunction("invalid", func(context.Context, *ai.RunContext[struct{}], chan int) (string, error) {
			return "", nil
		})
	})
	assertOutputFunctionPanics(t, func() {
		var invalid ai.OutputFunction[struct{}, string]
		ai.NewOutputFunctionAgent(model, invalid)
	})
}

type outputFunctionStreamingModel struct {
	events []ai.ModelStreamEvent
}

func (*outputFunctionStreamingModel) Name() string { return "output-function-stream" }

func (*outputFunctionStreamingModel) Request(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return nil, errors.New("unexpected non-streaming request")
}

func (model *outputFunctionStreamingModel) StreamRequest(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		for _, event := range model.events {
			if !yield(event, nil) {
				return
			}
		}
	}, nil
}

func assertOutputFunctionPanics(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	fn()
}
