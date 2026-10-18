package ai_test

import (
	"context"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type profileOutput struct {
	Value string `json:"value"`
}

func TestPromptCacheOutlook(t *testing.T) {
	now := time.Date(2026, 10, 8, 12, 0, 0, 0, time.UTC)
	profile := ai.ModelProfile{DefaultCacheRetention: 5 * time.Minute}
	history := func(age time.Duration, ttl ai.CachePointTTL) []ai.ModelMessage {
		contents := []ai.UserContent{ai.TextContent{Text: "hello"}}
		if ttl != "" {
			contents = append(contents, ai.CachePoint{TTL: ttl})
		}
		return []ai.ModelMessage{
			ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: contents}}},
			ai.ModelResponse{Timestamp: now.Add(-age), Parts: []ai.ResponsePart{ai.TextPart{Content: "hi"}}},
		}
	}
	if got := ai.PromptCacheOutlook(history(5*time.Minute, ""), &profile, nil, now); got != ai.CacheOutlookWarm {
		t.Fatalf("boundary outlook = %q", got)
	}
	if got := ai.PromptCacheOutlook(history(6*time.Minute, ""), &profile, nil, now); got != ai.CacheOutlookCold {
		t.Fatalf("cold outlook = %q", got)
	}
	if got := ai.PromptCacheOutlook(history(30*time.Minute, ai.CachePointTTL1Hour), &profile, nil, now); got != ai.CacheOutlookWarm {
		t.Fatalf("cache point outlook = %q", got)
	}
	hour := time.Hour
	if got := ai.PromptCacheOutlook(history(30*time.Minute, ""), &profile, &hour, now); got != ai.CacheOutlookWarm {
		t.Fatalf("explicit retention outlook = %q", got)
	}
	recent := history(time.Second, "")
	recent[1] = ai.ModelResponse{Timestamp: time.Now().UTC(), Parts: []ai.ResponsePart{ai.TextPart{Content: "hi"}}}
	if got := ai.PromptCacheOutlook(recent, &profile, nil, time.Time{}); got != ai.CacheOutlookWarm {
		t.Fatalf("current-time outlook = %q", got)
	}
	if got := ai.PromptCacheOutlook(nil, &profile, nil, now); got != ai.CacheOutlookUnknown {
		t.Fatalf("empty history outlook = %q", got)
	}
	unknown := ai.ModelProfile{}
	if got := ai.PromptCacheOutlook(history(time.Minute, ""), &unknown, nil, now); got != ai.CacheOutlookUnknown {
		t.Fatalf("unknown retention outlook = %q", got)
	}
	withUnsentPoint := append(history(30*time.Minute, ""), ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Contents: []ai.UserContent{ai.CachePoint{TTL: ai.CachePointTTL1Hour}}},
	}})
	if got := ai.PromptCacheOutlook(withUnsentPoint, &profile, nil, now); got != ai.CacheOutlookCold {
		t.Fatalf("unsent cache point outlook = %q", got)
	}
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
		"negative tool call ID length": func() {
			ai.NewProfiledModel(fakes.NewTestModel(), ai.ModelProfile{
				DefaultOutputMode: ai.OutputModeTool, ToolCallIDLength: -1,
			})
		},
		"negative cache retention": func() {
			ai.NewProfiledModel(fakes.NewTestModel(), ai.ModelProfile{
				DefaultOutputMode: ai.OutputModeTool, DefaultCacheRetention: -time.Second,
			})
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

func TestRequestScopedSystemPromptsReachOnlyTheirRequest(t *testing.T) {
	reminder := func(content string) ai.SystemPromptPart {
		return ai.SystemPromptPart{Content: content, Scope: ai.SystemPromptScopeRequest}
	}
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "first"}, reminder("old")}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{reminder("dropped request")}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "again"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{reminder("current"), ai.UserPromptPart{Content: "earlier"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{}},
	}
	var seen []ai.ModelMessage
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		seen = messages
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "ok"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model)
	if _, err := agent.Run(t.Context(), "now", struct{}{}, ai.WithMessageHistory(history)); err != nil {
		t.Fatal(err)
	}
	var scoped []string
	for _, message := range seen {
		if request, ok := message.(ai.ModelRequest); ok {
			for _, part := range request.Parts {
				if system, ok := part.(ai.SystemPromptPart); ok {
					scoped = append(scoped, system.Content)
				}
			}
		}
	}
	if strings.Join(scoped, ",") != "current" || len(seen) != 5 {
		t.Fatalf("superseded prompts reached the model: %#v", seen)
	}
	last := seen[len(seen)-1].(ai.ModelRequest).Parts
	if system, ok := last[len(last)-1].(ai.SystemPromptPart); !ok || system.Content != "current" {
		t.Fatalf("current prompt not moved to the end: %#v", last)
	}

	native := ai.NewProfiledModel(model, ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool, SupportsRequestScopedSystemPrompts: true,
	})
	if _, err := ai.NewAgent[struct{}, string](native).Run(
		t.Context(), "now", struct{}{}, ai.WithMessageHistory(history),
	); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 6 {
		t.Fatalf("native model did not receive every prompt unchanged: %#v", seen)
	}
}

func TestRetryWithoutResponseDropsRequestScopedSystemPrompts(t *testing.T) {
	for also, want := range map[string]int{"": 5, "also": 6} {
		var requests int
		model := fakes.NewFunctionModel(func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			requests++
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "ok"}}}, nil
		})
		native := ai.NewProfiledModel(model, ai.ModelProfile{
			DefaultOutputMode: ai.OutputModeTool, SupportsRequestScopedSystemPrompts: true,
		})
		retried := false
		retry := ai.BeforeModelRequestFunc(func(
			_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
		) (ai.ModelRequestContext, error) {
			if !retried {
				retried = true
				return request, ai.Retryf("again")
			}
			return request, nil
		})
		history := []ai.ModelMessage{
			ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hi"}}},
			ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "hello"}}},
			ai.ModelRequest{Parts: []ai.RequestPart{
				ai.SystemPromptPart{Content: "reminder", Scope: ai.SystemPromptScopeRequest},
			}},
		}
		if also != "" {
			last := history[2].(ai.ModelRequest)
			last.Parts = append(last.Parts, ai.UserPromptPart{Content: also})
			history[2] = last
		}
		result, err := ai.NewAgent[struct{}, string](native, ai.WithCapabilities(retry)).Run(
			t.Context(), "kept", struct{}{}, ai.WithMessageHistory(history),
		)
		if err != nil {
			t.Fatal(err)
		}
		for _, message := range result.Messages() {
			if request, ok := message.(ai.ModelRequest); ok {
				for _, part := range request.Parts {
					if system, ok := part.(ai.SystemPromptPart); ok {
						t.Fatalf("unanswered request-scoped prompt survived the retry: %#v", system)
					}
				}
			}
		}
		if requests != 1 || len(result.Messages()) != want {
			t.Fatalf("%q: requests = %d, messages = %#v", also, requests, result.Messages())
		}
	}
}

func TestPrepareModelMessagesKeepsRequestScopedPromptsOfASuspendedResponse(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{}, nil
	})
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "current", Scope: ai.SystemPromptScopeRequest},
			ai.UserPromptPart{Content: "go"},
		}},
		ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "partial"}}, State: ai.ModelResponseStateSuspended,
		},
	}
	prepared, err := ai.PrepareModelMessages(model, history)
	if err != nil {
		t.Fatal(err)
	}
	parts := prepared[0].(ai.ModelRequest).Parts
	if len(prepared) != 2 || len(parts) != 2 || parts[1].(ai.SystemPromptPart).Content != "current" {
		t.Fatalf("the continued request lost its request-scoped prompt: %#v", prepared)
	}
}
