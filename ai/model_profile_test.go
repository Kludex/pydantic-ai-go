package ai_test

import (
	"context"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type profileOutput struct {
	Value string `json:"value"`
}

func TestModelProfileSelectsAutomaticOutputMode(t *testing.T) {
	tests := map[string]struct {
		profile ai.ModelProfile
		check   func(*testing.T, ai.ModelRequestParams)
		parts   []ai.ResponsePart
	}{
		"tool default": {
			profile: ai.ModelProfile{DefaultOutputMode: ai.OutputModeTool},
			check: func(t *testing.T, params ai.ModelRequestParams) {
				if params.OutputMode != ai.OutputModeTool || params.OutputTool == nil || params.OutputSchema != nil ||
					params.AllowText {
					t.Fatalf("unexpected tool output parameters: %+v", params)
				}
			},
			parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "final_result", ToolCallID: "output", Args: []byte(`{"value":"tool"}`),
			}},
		},
		"native default": {
			profile: ai.ModelProfile{DefaultOutputMode: ai.OutputModeNative},
			check: func(t *testing.T, params ai.ModelRequestParams) {
				if params.OutputMode != ai.OutputModeNative || params.OutputTool != nil || params.OutputSchema == nil ||
					!params.AllowText || params.OutputPrompt != "" {
					t.Fatalf("unexpected native output parameters: %+v", params)
				}
			},
			parts: []ai.ResponsePart{ai.TextPart{Content: `{"value":"native"}`}},
		},
		"prompted default": {
			profile: ai.ModelProfile{
				DefaultOutputMode: ai.OutputModePrompted, PromptedOutputTemplate: "Profile schema: {schema}",
			},
			check: func(t *testing.T, params ai.ModelRequestParams) {
				if params.OutputMode != ai.OutputModePrompted || params.OutputTool != nil || params.OutputSchema == nil ||
					!strings.Contains(params.OutputPrompt, "Profile schema:") ||
					!strings.Contains(params.Instructions, params.OutputPrompt) {
					t.Fatalf("unexpected prompted output parameters: %+v", params)
				}
			},
			parts: []ai.ResponsePart{ai.TextPart{Content: `{"value":"prompted"}`}},
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			base := fakes.NewFunctionModel(func(
				_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				test.check(t, params)
				return &ai.ModelResponse{Parts: test.parts}, nil
			})
			model := ai.NewProfiledModel(base, test.profile)
			result, err := ai.NewAgent[struct{}, profileOutput](model).Run(t.Context(), "go", struct{}{})
			if err != nil {
				t.Fatal(err)
			}
			if result.Output.Value == "" {
				t.Fatal("empty output")
			}
		})
	}
}

func TestExplicitOutputModeOverridesProfile(t *testing.T) {
	base := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if params.OutputMode != ai.OutputModeTool || params.OutputTool == nil {
			t.Fatalf("explicit mode did not win: %+v", params)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: params.OutputTool.Name, ToolCallID: "output", Args: []byte(`{"value":"ok"}`),
		}}}, nil
	})
	profiled := ai.NewProfiledModel(base, ai.ModelProfile{DefaultOutputMode: ai.OutputModeNative})
	agent := ai.NewAgent[struct{}, profileOutput](profiled, ai.WithOutputMode(ai.OutputModeTool))
	if _, err := agent.Run(t.Context(), "go", struct{}{}); err != nil {
		t.Fatal(err)
	}
}

func TestRunOutputModeOverridesAutomaticProfile(t *testing.T) {
	base := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if params.OutputMode != ai.OutputModePrompted || !strings.Contains(params.OutputPrompt, "Run schema:") {
			t.Fatalf("run mode or template did not win: %+v", params)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"value":"ok"}`}}}, nil
	})
	profiled := ai.NewProfiledModel(base, ai.ModelProfile{DefaultOutputMode: ai.OutputModeNative})
	agent := ai.NewAgent[struct{}, profileOutput](profiled)
	_, err := agent.Run(t.Context(), "go", struct{}{},
		ai.WithRunOutputMode(ai.OutputModePrompted), ai.WithRunPromptedOutputTemplate("Run schema:"))
	if err != nil {
		t.Fatal(err)
	}
}

func TestNativeProfileCanRequirePromptedSchema(t *testing.T) {
	base := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if params.OutputMode != ai.OutputModeNative || !strings.Contains(params.OutputPrompt, "Native schema:") ||
			!strings.Contains(params.Instructions, params.OutputPrompt) {
			t.Fatalf("native schema prompt missing: %+v", params)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"value":"ok"}`}}}, nil
	})
	profiled := ai.NewProfiledModel(base, ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeNative, PromptedOutputTemplate: "Native schema:",
		NativeOutputRequiresPrompt: true,
	})
	if _, err := ai.NewAgent[struct{}, profileOutput](profiled).Run(t.Context(), "go", struct{}{}); err != nil {
		t.Fatal(err)
	}
}

func TestAutomaticOutputTracksAdaptiveModels(t *testing.T) {
	var requests int
	base := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			if params.OutputMode != ai.OutputModeNative {
				t.Fatalf("first mode = %v", params.OutputMode)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"value":1}`}}}, nil
		}
		if params.OutputMode != ai.OutputModePrompted || !strings.Contains(params.OutputPrompt, "Fallback schema:") {
			t.Fatalf("second mode = %+v", params)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"value":"done"}`}}}, nil
	})
	native := ai.NewProfiledModel(base, ai.ModelProfile{DefaultOutputMode: ai.OutputModeNative})
	prompted := ai.NewProfiledModel(base, ai.ModelProfile{
		DefaultOutputMode: ai.OutputModePrompted, PromptedOutputTemplate: "Fallback schema:",
	})
	agent := ai.NewAgent[struct{}, profileOutput](native, ai.WithRetryLimits(ai.RetryLimits{Output: 1}))
	agent.AddModelSelector(func(_ context.Context, selection ai.ModelSelectionContext[struct{}]) (ai.ModelSelection, error) {
		if selection.Step == 2 {
			return ai.ModelSelection{Model: prompted}, nil
		}
		return ai.ModelSelection{}, nil
	})
	result, err := agent.Run(t.Context(), "go", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.Value != "done" || requests != 2 {
		t.Fatalf("unexpected result: %+v requests=%d", result, requests)
	}
}

func TestFallbackDispatchesAutomaticProfilesPerModel(t *testing.T) {
	primary := ai.NewProfiledModel(fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if params.OutputMode != ai.OutputModeNative || params.OutputSchema == nil || params.OutputTool != nil {
			t.Fatalf("unexpected primary parameters: %+v", params)
		}
		return nil, profileFallbackError{}
	}), ai.ModelProfile{DefaultOutputMode: ai.OutputModeNative})
	backup := ai.NewProfiledModel(fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if params.OutputMode != ai.OutputModePrompted || params.OutputSchema == nil || params.OutputTool != nil ||
			!strings.Contains(params.OutputPrompt, "Backup schema:") {
			t.Fatalf("unexpected backup parameters: %+v", params)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"value":"backup"}`}}}, nil
	}), ai.ModelProfile{
		DefaultOutputMode: ai.OutputModePrompted, PromptedOutputTemplate: "Backup schema:",
	})
	fallback := ai.WrapModel(ai.NewFallbackModel(primary, ai.WithFallbackModels(backup)))
	result, err := ai.NewAgent[struct{}, profileOutput](fallback).Run(t.Context(), "go", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.Value != "backup" {
		t.Fatalf("unexpected fallback output: %+v", result.Output)
	}
}

func TestTransparentWrapperUsesDefaultProfile(t *testing.T) {
	base := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if params.OutputMode != ai.OutputModeTool || params.OutputTool == nil {
			t.Fatalf("default profile lost: %+v", params)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: params.OutputTool.Name, ToolCallID: "output", Args: []byte(`{"value":"ok"}`),
		}}}, nil
	})
	if _, err := ai.NewAgent[struct{}, profileOutput](ai.WrapModel(base)).Run(t.Context(), "go", struct{}{}); err != nil {
		t.Fatal(err)
	}
}

func TestModelProfileWorksThroughTransparentWrapper(t *testing.T) {
	base := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if params.OutputMode != ai.OutputModeNative {
			t.Fatalf("wrapped profile lost: %+v", params)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"value":"ok"}`}}}, nil
	})
	model := ai.WrapModel(ai.NewProfiledModel(base, ai.ModelProfile{DefaultOutputMode: ai.OutputModeNative}))
	if _, err := ai.NewAgent[struct{}, profileOutput](model).Run(t.Context(), "go", struct{}{}); err != nil {
		t.Fatal(err)
	}
}

func TestProfileValidation(t *testing.T) {
	tests := map[string]func(){
		"nil model": func() { ai.NewProfiledModel(nil, ai.ModelProfile{DefaultOutputMode: ai.OutputModeTool}) },
		"auto default": func() {
			ai.NewProfiledModel(fakes.NewTestModel(), ai.ModelProfile{DefaultOutputMode: ai.OutputModeAuto})
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

func TestInvalidAutomaticProfileFromCustomModel(t *testing.T) {
	model := invalidProfileModel{Model: fakes.NewTestModel()}
	agent := ai.NewAgent[struct{}, profileOutput](model)
	_, err := agent.Run(t.Context(), "go", struct{}{})
	if err == nil || !strings.Contains(err.Error(), "invalid default output mode") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestModelRequestHookProfileAndSchemaErrors(t *testing.T) {
	base := fakes.NewTestModel()
	invalid := invalidProfileModel{Model: base}
	invalidProfileHook := ai.BeforeModelRequestFunc(func(
		_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
	) (ai.ModelRequestContext, error) {
		request.Model = invalid
		return request, nil
	})
	agent := ai.NewAgent[struct{}, profileOutput](
		ai.NewProfiledModel(base, ai.ModelProfile{DefaultOutputMode: ai.OutputModeNative}),
		ai.WithCapabilities(invalidProfileHook),
	)
	if _, err := agent.Run(t.Context(), "go", struct{}{}); err == nil ||
		!strings.Contains(err.Error(), "invalid default output mode") {
		t.Fatalf("unexpected profile error: %v", err)
	}

	invalidSchemaHook := ai.BeforeModelRequestFunc(func(
		_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
	) (ai.ModelRequestContext, error) {
		request.Params.OutputSchema = map[string]any{"type": "not-a-json-schema-type"}
		return request, nil
	})
	agent = ai.NewAgent[struct{}, profileOutput](
		ai.NewProfiledModel(base, ai.ModelProfile{DefaultOutputMode: ai.OutputModeNative}),
		ai.WithCapabilities(invalidSchemaHook),
	)
	if _, err := agent.Run(t.Context(), "go", struct{}{}); err == nil ||
		!strings.Contains(err.Error(), "output schema") {
		t.Fatalf("unexpected schema error: %v", err)
	}
}

type profileFallbackError struct{}

func (profileFallbackError) Error() string         { return "fallback" }
func (profileFallbackError) IsModelAPIError() bool { return true }

type invalidProfileModel struct{ ai.Model }

func (invalidProfileModel) ModelProfile() ai.ModelProfile {
	return ai.ModelProfile{DefaultOutputMode: ai.OutputMode(99)}
}

func TestModelRequestHookCanSwitchAutomaticOutputProfile(t *testing.T) {
	base := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if params.OutputMode != ai.OutputModePrompted || !strings.Contains(params.OutputPrompt, "Hook schema:") ||
			params.OutputSchema["description"] != "hook schema" {
			t.Fatalf("hook-selected profile was not applied: %+v", params)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: `{"value":"ok"}`}}}, nil
	})
	initial := ai.NewProfiledModel(base, ai.ModelProfile{
		DefaultOutputMode: ai.OutputModePrompted, PromptedOutputTemplate: "Initial schema:",
	})
	prompted := ai.NewProfiledModel(base, ai.ModelProfile{
		DefaultOutputMode: ai.OutputModePrompted, PromptedOutputTemplate: "Hook schema:",
	})
	hook := ai.BeforeModelRequestFunc(func(
		_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
	) (ai.ModelRequestContext, error) {
		request.Model = prompted
		request.Params.OutputSchema["description"] = "hook schema"
		return request, nil
	})
	agent := ai.NewAgent[struct{}, profileOutput](
		initial, ai.WithInstructions("Keep this instruction."), ai.WithCapabilities(hook),
	)
	if _, err := agent.Run(t.Context(), "go", struct{}{}); err != nil {
		t.Fatal(err)
	}
}
