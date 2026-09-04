package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func structuredOutputModel(raw string) ai.Model {
	return fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "final_result", ToolCallID: "output", Args: json.RawMessage(raw),
		}}}, nil
	})
}

func TestOutputHookFunctionAdaptersTransformPlainTextAndPartials(t *testing.T) {
	beforeValidationCalls := 0
	beforeValidation := ai.BeforeOutputValidationFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, raw any,
	) (any, error) {
		beforeValidationCalls++
		return raw, nil
	})
	afterValidationCalls := 0
	afterValidation := ai.AfterOutputValidationFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, output any,
	) (any, error) {
		afterValidationCalls++
		return output, nil
	})
	var partial []bool
	beforeProcessing := ai.BeforeOutputProcessingFunc(func(
		_ context.Context, _ *ai.RunInfo, hook ai.OutputHookContext, output any,
	) (any, error) {
		partial = append(partial, hook.Partial)
		if hook.Mode != ai.OutputHookModeText || hook.Structured || hook.ToolCall != nil ||
			hook.ToolDefinition != nil {
			return nil, fmt.Errorf("unexpected text hook context: %+v", hook)
		}
		return output.(string) + "P", nil
	})
	afterProcessing := ai.AfterOutputProcessingFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, output any,
	) (any, error) {
		return output.(string) + "A", nil
	})
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.TextDeltaEvent{PartID: "text", Delta: "Hel"},
			ai.TextDeltaEvent{PartID: "text", Delta: "lo"},
			ai.FinishEvent{},
		}
	})
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(
		beforeValidation, afterValidation, beforeProcessing, afterProcessing,
	))
	stream := agent.RunStream(t.Context(), "go", deps{})
	var outputs []string
	for output, err := range stream.Outputs() {
		if err != nil {
			t.Fatal(err)
		}
		outputs = append(outputs, output)
	}
	if !slices.Equal(outputs, []string{"HelPA", "HelloPA", "HelloPA"}) ||
		!slices.Equal(partial, []bool{true, true, false}) || beforeValidationCalls != 0 || afterValidationCalls != 0 {
		t.Fatalf(
			"unexpected text hooks outputs=%v partial=%v before=%d after=%d",
			outputs, partial, beforeValidationCalls, afterValidationCalls,
		)
	}
}

func TestStructuredOutputHookRetryPaths(t *testing.T) {
	for name, capability := range map[string]ai.Capability{
		"before validation": ai.BeforeOutputValidationFunc(func(
			_ context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, raw any,
		) (any, error) {
			return raw, ai.Retryf("before validation retry")
		}),
		"after validation": ai.AfterOutputValidationFunc(func(
			_ context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, output any,
		) (any, error) {
			return output, ai.Retryf("after validation retry")
		}),
		"before processing": ai.BeforeOutputProcessingFunc(func(
			_ context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, output any,
		) (any, error) {
			return output, ai.Retryf("before processing retry")
		}),
		"after processing": ai.AfterOutputProcessingFunc(func(
			_ context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, output any,
		) (any, error) {
			return output, ai.Retryf("after processing retry")
		}),
	} {
		t.Run(name, func(t *testing.T) {
			agent := ai.NewAgent[deps, hookedOutput](
				structuredOutputModel(`{"value":1}`), ai.WithCapabilities(capability),
			)
			_, err := agent.Run(t.Context(), "go", deps{})
			if !errors.Is(err, ai.ErrMaxRetriesExceeded) {
				t.Fatalf("unexpected output hook retry error: %v", err)
			}
		})
	}
}

func TestOutputHooksRejectInvalidTransformedTypesAndRawValues(t *testing.T) {
	for name, capability := range map[string]ai.Capability{
		"raw": ai.BeforeOutputValidationFunc(func(
			context.Context, *ai.RunInfo, ai.OutputHookContext, any,
		) (any, error) {
			return 42, nil
		}),
		"validated": ai.AfterOutputValidationFunc(func(
			context.Context, *ai.RunInfo, ai.OutputHookContext, any,
		) (any, error) {
			return "wrong", nil
		}),
		"processing input": ai.BeforeOutputProcessingFunc(func(
			context.Context, *ai.RunInfo, ai.OutputHookContext, any,
		) (any, error) {
			return "wrong", nil
		}),
		"processed": ai.AfterOutputProcessingFunc(func(
			context.Context, *ai.RunInfo, ai.OutputHookContext, any,
		) (any, error) {
			return "wrong", nil
		}),
	} {
		t.Run(name, func(t *testing.T) {
			agent := ai.NewAgent[deps, hookedOutput](
				structuredOutputModel(`{"value":1}`), ai.WithCapabilities(capability),
			)
			_, err := agent.Run(t.Context(), "go", deps{})
			if err == nil {
				t.Fatal("expected transformed output type error")
			}
		})
	}
}

func TestOutputProcessingErrorHookRecoversSemanticValidator(t *testing.T) {
	hook := ai.OutputProcessingErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, output any, err error,
	) (any, error) {
		if output.(hookedOutput).Value != 1 || err.Error() != "semantic output rejected" {
			return nil, fmt.Errorf("unexpected semantic validation failure: %w", err)
		}
		return hookedOutput{Value: 7}, nil
	})
	agent := ai.NewAgent[deps, hookedOutput](
		structuredOutputModel(`{"value":1}`), ai.WithCapabilities(hook),
	)
	agent.AddOutputValidator(func(context.Context, *ai.RunContext[deps], hookedOutput) error {
		return errors.New("semantic output rejected")
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || result.Output.Value != 7 {
		t.Fatalf("unexpected semantic output recovery result=%+v err=%v", result, err)
	}
}

type retryOutputProcessingWrapper struct{}

func (*retryOutputProcessingWrapper) Setup(*ai.CapabilityRegistry) error { return nil }

func (*retryOutputProcessingWrapper) WrapOutputProcessing(
	context.Context, *ai.RunInfo, ai.OutputHookContext, any, ai.OutputProcessingFunc,
) (any, error) {
	return nil, ai.Retryf("processing retry")
}

func TestOutputProcessingRetryBypassesErrorHook(t *testing.T) {
	errorHookCalls := 0
	errorHook := ai.OutputProcessingErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, output any, err error,
	) (any, error) {
		errorHookCalls++
		return output, err
	})
	agent := ai.NewAgent[deps, hookedOutput](
		structuredOutputModel(`{"value":1}`),
		ai.WithCapabilities(&retryOutputProcessingWrapper{}, errorHook),
	)
	_, err := agent.Run(t.Context(), "go", deps{})
	if !errors.Is(err, ai.ErrMaxRetriesExceeded) || errorHookCalls != 0 {
		t.Fatalf("unexpected processing retry error=%v error_hooks=%d", err, errorHookCalls)
	}
}

func TestPartialValidationErrorsSkipRecoveryHooks(t *testing.T) {
	errorHookCalls := 0
	errorHook := ai.OutputValidationErrorFunc(func(
		_ context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, _ any, err error,
	) (any, error) {
		errorHookCalls++
		return nil, err
	})
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.ToolCallStartEvent{PartID: "output", ToolName: "final_result", ToolCallID: "result"},
			ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `{"value":`},
			ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `1}`},
			ai.FinishEvent{},
		}
	})
	stream := ai.NewAgent[deps, hookedOutput](
		model, ai.WithCapabilities(errorHook),
	).RunStream(t.Context(), "go", deps{})
	for _, err := range stream.Outputs() {
		if err != nil {
			t.Fatal(err)
		}
	}
	if errorHookCalls != 0 {
		t.Fatalf("partial validation invoked %d recovery hooks", errorHookCalls)
	}
}

func TestUnrecoveredOutputProcessingErrorPropagates(t *testing.T) {
	agent := ai.NewAgent[deps, hookedOutput](
		structuredOutputModel(`{"value":1}`), ai.WithCapabilities(&failingOutputProcessingWrapper{}),
	)
	_, err := agent.Run(t.Context(), "go", deps{})
	if err == nil || err.Error() != "ai: output processing: processing failed" {
		t.Fatalf("unexpected processing error: %v", err)
	}
}

func TestPartialOutputHookErrorsIdentifyLifecycle(t *testing.T) {
	for _, test := range []struct {
		name       string
		capability ai.Capability
		want       string
	}{
		{
			name: "validation",
			capability: ai.BeforeOutputValidationFunc(func(
				_ context.Context, _ *ai.RunInfo, hook ai.OutputHookContext, raw any,
			) (any, error) {
				if hook.Partial {
					return 42, nil
				}
				return raw, nil
			}),
			want: "ai: partial output validation: raw structured output has type int, expected string, JSON bytes, or map[string]any",
		},
		{
			name: "processing",
			capability: ai.BeforeOutputProcessingFunc(func(
				_ context.Context, _ *ai.RunInfo, hook ai.OutputHookContext, output any,
			) (any, error) {
				if hook.Partial {
					return nil, errors.New("partial processing failed")
				}
				return output, nil
			}),
			want: "ai: partial output processing: partial processing failed",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
				return []ai.ModelStreamEvent{
					ai.ToolCallStartEvent{
						PartID: "output", ToolName: "final_result", ToolCallID: "result",
					},
					ai.ToolCallDeltaEvent{PartID: "output", ArgsDelta: `{"value":1}`},
					ai.FinishEvent{},
				}
			})
			stream := ai.NewAgent[deps, hookedOutput](
				model, ai.WithCapabilities(test.capability),
			).RunStream(t.Context(), "go", deps{})
			var got error
			for _, err := range stream.Outputs() {
				if err != nil {
					got = err
					break
				}
			}
			if got == nil || got.Error() != test.want {
				t.Fatalf("unexpected partial %s error: %v", test.name, got)
			}
		})
	}
}

func TestOutputValidationErrorHookInvalidTypeIsRejected(t *testing.T) {
	hook := ai.OutputValidationErrorFunc(func(
		context.Context, *ai.RunInfo, ai.OutputHookContext, any, error,
	) (any, error) {
		return "wrong", nil
	})
	agent := ai.NewAgent[deps, hookedOutput](
		structuredOutputModel(`{"value":"bad"}`), ai.WithCapabilities(hook),
	)
	_, err := agent.Run(t.Context(), "go", deps{})
	if err == nil || !strings.Contains(err.Error(), "validated output has type") {
		t.Fatalf("unexpected validation recovery type error: %v", err)
	}
}

func TestOutputHooksReceiveNativeAndPromptedModes(t *testing.T) {
	for name, test := range map[string]struct {
		mode     ai.OutputMode
		hookMode ai.OutputHookMode
	}{
		"native":   {mode: ai.OutputModeNative, hookMode: ai.OutputHookModeNative},
		"prompted": {mode: ai.OutputModePrompted, hookMode: ai.OutputHookModePrompted},
	} {
		t.Run(name, func(t *testing.T) {
			seen := false
			before := ai.BeforeOutputValidationFunc(func(
				_ context.Context, _ *ai.RunInfo, hook ai.OutputHookContext, raw any,
			) (any, error) {
				seen = true
				if hook.Mode != test.hookMode || !hook.Structured || hook.ToolCall != nil || raw != `{"value":1}` {
					return nil, fmt.Errorf("unexpected %s context: %+v raw=%v", name, hook, raw)
				}
				return map[string]any{"value": 1}, nil
			})
			after := ai.AfterOutputValidationFunc(func(
				_ context.Context, _ *ai.RunInfo, _ ai.OutputHookContext, output any,
			) (any, error) {
				value := output.(hookedOutput)
				value.Value++
				return value, nil
			})
			model := fakes.NewFunctionModel(func(
				context.Context, []ai.ModelMessage, ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"value":1}`}}}, nil
			})
			agent := ai.NewAgent[deps, hookedOutput](
				model, ai.WithOutputMode(test.mode), ai.WithCapabilities(before, after),
			)
			result, err := agent.Run(t.Context(), "go", deps{})
			if err != nil || result.Output.Value != 2 || !seen {
				t.Fatalf("unexpected %s output result=%+v seen=%v err=%v", name, result, seen, err)
			}
		})
	}
}
