package ai_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestXSearchToolValidationAndDetachment(t *testing.T) {
	from := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
	to := from.Add(time.Hour)
	tool := ai.XSearchTool{
		AllowedXHandles: []string{"pydantic"}, FromDate: &from, ToDate: &to,
		EnableImageUnderstanding: true, EnableVideoUnderstanding: true, IncludeOutput: true, Optional: true,
	}
	if tool.Kind() != "x_search" || tool.UniqueID() != "x_search" || !tool.IsOptional() {
		t.Fatalf("unexpected X-search identity: %#v", tool)
	}
	clone := tool.CloneNativeTool().(ai.XSearchTool)
	clone.AllowedXHandles[0] = "changed"
	*clone.FromDate = from.Add(-time.Hour)
	*clone.ToDate = to.Add(time.Hour)
	if tool.AllowedXHandles[0] != "pydantic" || !tool.FromDate.Equal(from) || !tool.ToDate.Equal(to) {
		t.Fatalf("X-search clone shares mutable state: %#v", tool)
	}
	if err := ai.ValidateNativeTools([]ai.NativeTool{&tool}); err != nil {
		t.Fatal(err)
	}
	invalidPointer := &ai.XSearchTool{AllowedXHandles: []string{}, ExcludedXHandles: []string{}}
	if err := ai.ValidateNativeTools([]ai.NativeTool{invalidPointer}); err == nil ||
		!strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("unexpected X-search pointer validation error: %v", err)
	}

	tooMany := make([]string, 21)
	for _, test := range []struct {
		name string
		tool ai.XSearchTool
		want string
	}{
		{name: "both filters", tool: ai.XSearchTool{
			AllowedXHandles: []string{}, ExcludedXHandles: []string{},
		}, want: "mutually exclusive"},
		{name: "allowed limit", tool: ai.XSearchTool{AllowedXHandles: tooMany}, want: "allowed handles"},
		{name: "excluded limit", tool: ai.XSearchTool{ExcludedXHandles: tooMany}, want: "excluded handles"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := ai.ValidateNativeTools([]ai.NativeTool{test.tool}); err == nil ||
				!strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected X-search validation error: %v", err)
			}
		})
	}
}

func TestXSearchCapabilitySelectionAndConstraints(t *testing.T) {
	localCalls := 0
	local := ai.NewFunctionToolset(webCapabilityTool(t, &localCalls))
	model := &selectiveNativeModel{supported: map[string]bool{"x_search": true}}
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.NewXSearchCapability(ai.XSearchCapabilityConfig[struct{}]{
			Native: ai.XSearchTool{EnableImageUnderstanding: true}, Local: local,
		}),
	))
	if _, err := agent.Run(t.Context(), "search X", struct{}{}); err != nil || localCalls != 0 ||
		len(model.requests) != 1 || len(model.requests[0].NativeTools) != 1 || len(model.requests[0].Tools) != 0 {
		t.Fatalf("native X search was not selected: local=%d requests=%#v err=%v", localCalls, model.requests, err)
	}

	model = &selectiveNativeModel{supported: map[string]bool{}, callLocal: true}
	agent = ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.NewXSearchCapability(ai.XSearchCapabilityConfig[struct{}]{Native: ai.XSearchTool{}, Local: local}),
	))
	if _, err := agent.Run(t.Context(), "search X", struct{}{}); err != nil || localCalls != 1 ||
		len(model.requests) != 2 || len(model.requests[0].NativeTools) != 0 || len(model.requests[0].Tools) != 1 {
		t.Fatalf("local X search was not selected: local=%d requests=%#v err=%v", localCalls, model.requests, err)
	}

	for _, native := range []ai.XSearchTool{
		{AllowedXHandles: []string{"pydantic"}},
		{ExcludedXHandles: []string{"spam"}},
	} {
		model = &selectiveNativeModel{supported: map[string]bool{}}
		agent = ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
			ai.NewXSearchCapability(ai.XSearchCapabilityConfig[struct{}]{Native: native, Local: local}),
		))
		if _, err := agent.Run(t.Context(), "search X", struct{}{}); err == nil ||
			!strings.Contains(err.Error(), "have no local fallback") || len(model.requests) != 0 {
			t.Fatalf("native-only X constraint degraded locally: requests=%d err=%v", len(model.requests), err)
		}
	}

	model = &selectiveNativeModel{supported: map[string]bool{}}
	agent = ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.NewXSearchCapability(ai.XSearchCapabilityConfig[struct{}]{}),
	))
	if _, err := agent.Run(t.Context(), "search X", struct{}{}); err == nil ||
		!strings.Contains(err.Error(), "have no local fallback") {
		t.Fatalf("X search without a path succeeded: %v", err)
	}
}

func TestDynamicXSearchCapability(t *testing.T) {
	resolveCalls := 0
	model := &selectiveNativeModel{supported: map[string]bool{"x_search": true}}
	capability := ai.NewDynamicXSearchCapability(
		func(_ context.Context, rc *ai.RunContext[nativeOrLocalDeps]) (ai.XSearchTool, error) {
			resolveCalls++
			return ai.XSearchTool{EnableVideoUnderstanding: rc.Deps.Location == "video"}, nil
		},
		ai.NewFunctionToolset(nativeOrLocalSearchTool(t, new(int))),
	)
	agent := ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(capability))
	if _, err := agent.Run(t.Context(), "search X", nativeOrLocalDeps{Location: "video"}); err != nil ||
		resolveCalls != 1 {
		t.Fatalf("dynamic X search failed: resolves=%d err=%v", resolveCalls, err)
	}
	if tool := model.requests[0].NativeTools[0].(ai.XSearchTool); !tool.EnableVideoUnderstanding {
		t.Fatalf("unexpected dynamic X-search tool: %#v", tool)
	}

	model = &selectiveNativeModel{supported: map[string]bool{}}
	local := ai.NewFunctionToolset(nativeOrLocalSearchTool(t, new(int)))
	agent = ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(
		ai.NewDynamicXSearchCapability(func(
			context.Context, *ai.RunContext[nativeOrLocalDeps],
		) (ai.XSearchTool, error) {
			return ai.XSearchTool{AllowedXHandles: []string{"pydantic"}}, nil
		}, local),
	))
	if _, err := agent.Run(t.Context(), "search X", nativeOrLocalDeps{}); err == nil ||
		!strings.Contains(err.Error(), "native-only constraint") {
		t.Fatalf("dynamic X-search constraint degraded locally: %v", err)
	}

	resolveErr := errors.New("resolve X search")
	model = &selectiveNativeModel{supported: map[string]bool{"x_search": true}}
	agent = ai.NewAgent[nativeOrLocalDeps, string](model, ai.WithCapabilities(
		ai.NewDynamicXSearchCapability(func(
			context.Context, *ai.RunContext[nativeOrLocalDeps],
		) (ai.XSearchTool, error) {
			return ai.XSearchTool{}, resolveErr
		}, local),
	))
	if _, err := agent.Run(t.Context(), "search X", nativeOrLocalDeps{}); !errors.Is(err, resolveErr) {
		t.Fatalf("unexpected X-search resolver error: %v", err)
	}

	assertNativeOrLocalPanic(t, "dynamic X-search resolver must not be nil", func() {
		ai.NewDynamicXSearchCapability[nativeOrLocalDeps](nil, nil)
	})
}

func TestXSearchSubagentValidationAndDynamicModel(t *testing.T) {
	assertNativeOrLocalPanic(t, "requires exactly one model", func() {
		ai.NewXSearchSubagentTool(ai.XSearchSubagentConfig[struct{}]{})
	})
	assertNativeOrLocalPanic(t, "requires exactly one model", func() {
		ai.NewXSearchSubagentTool(ai.XSearchSubagentConfig[struct{}]{
			Model: fakes.NewTestModel(),
			ResolveModel: func(context.Context, *ai.RunContext[struct{}]) (ai.Model, error) {
				return fakes.NewTestModel(), nil
			},
		})
	})

	resolveCalls := 0
	inner := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if !strings.Contains(params.Instructions, "Search X/Twitter") {
			t.Fatalf("default X-search instructions missing: %q", params.Instructions)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "summary"}}}, nil
	})
	tool := ai.NewXSearchSubagentTool(ai.XSearchSubagentConfig[nativeOrLocalDeps]{
		ResolveModel: func(
			_ context.Context, rc *ai.RunContext[nativeOrLocalDeps],
		) (ai.Model, error) {
			resolveCalls++
			if rc.Deps.Location != "Mexico" || rc.ToolName != "x_search" {
				t.Fatalf("unexpected X-search resolver context: %#v", rc)
			}
			return inner, nil
		},
	})
	outerCalls := 0
	outer := &noNativeFunctionModel{fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		outerCalls++
		if outerCalls == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "x_search", ToolCallID: "x-1", Args: []byte(`{"query":"Pydantic"}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})}
	agent := ai.NewAgent[nativeOrLocalDeps, string](outer)
	agent.AddTool(tool)
	if _, err := agent.Run(t.Context(), "search", nativeOrLocalDeps{Location: "Mexico"}); err != nil ||
		resolveCalls != 1 {
		t.Fatalf("dynamic X-search subagent failed: resolves=%d err=%v", resolveCalls, err)
	}
}

func TestXSearchSubagentResolverErrors(t *testing.T) {
	for _, test := range []struct {
		name    string
		resolve ai.XSearchFallbackModelFunc[struct{}]
		want    string
	}{
		{name: "error", want: "resolve X model", resolve: func(
			context.Context, *ai.RunContext[struct{}],
		) (ai.Model, error) {
			return nil, errors.New("resolve X model")
		}},
		{name: "nil", want: "resolver returned nil", resolve: func(
			context.Context, *ai.RunContext[struct{}],
		) (ai.Model, error) {
			return nil, nil
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			outer := &noNativeFunctionModel{fakes.NewFunctionModel(func(
				context.Context, []ai.ModelMessage, ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
					ToolName: "x_search", ToolCallID: "x-1", Args: []byte(`{"query":"Pydantic"}`),
				}}}, nil
			})}
			agent := ai.NewAgent[struct{}, string](outer)
			agent.AddTool(ai.NewXSearchSubagentTool(ai.XSearchSubagentConfig[struct{}]{ResolveModel: test.resolve}))
			if _, err := agent.Run(t.Context(), "search", struct{}{}); err == nil ||
				!strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected X-search resolver error: %v", err)
			}
		})
	}
}

func TestXSearchSubagentFailures(t *testing.T) {
	providerErr := errors.New("X provider failed")
	for _, test := range []struct {
		name      string
		inner     ai.Model
		wantError error
		wantRetry bool
	}{
		{name: "provider error", wantError: providerErr, inner: fakes.NewFunctionModel(func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			return nil, providerErr
		})},
		{name: "unexpected behavior", wantRetry: true, inner: fakes.NewFunctionModel(func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			return nil, &ai.UnexpectedModelBehaviorError{Message: "bad X response"}
		})},
		{name: "empty summary", wantRetry: true, inner: fakes.NewFunctionModel(func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: ""}}}, nil
		})},
	} {
		t.Run(test.name, func(t *testing.T) {
			outerCalls := 0
			outer := &noNativeFunctionModel{fakes.NewFunctionModel(func(
				_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				outerCalls++
				if outerCalls == 1 {
					return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
						ToolName: "x_search", ToolCallID: "x-1", Args: []byte(`{"query":"Pydantic"}`),
					}}}, nil
				}
				if _, ok := messages[len(messages)-1].(ai.ModelRequest).Parts[0].(ai.RetryPromptPart); !ok {
					t.Fatalf("X-search failure was not returned as a retry: %#v", messages)
				}
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "gave up"}}}, nil
			})}
			agent := ai.NewAgent[struct{}, string](outer)
			agent.AddTool(ai.NewXSearchSubagentTool(ai.XSearchSubagentConfig[struct{}]{Model: test.inner}))
			result, err := agent.Run(t.Context(), "search", struct{}{})
			if test.wantError != nil {
				if !errors.Is(err, test.wantError) {
					t.Fatalf("X-search provider error was not preserved: %v", err)
				}
				return
			}
			if err != nil || !test.wantRetry || result.Output != "gave up" || outerCalls != 2 {
				t.Fatalf("unexpected X-search retry result=%#v calls=%d err=%v", result, outerCalls, err)
			}
		})
	}
}

func TestXSearchSubagentFallback(t *testing.T) {
	innerRequests := 0
	inner := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		innerRequests++
		if len(params.NativeTools) != 1 || !strings.Contains(params.Instructions, "Search X now") {
			t.Fatalf("unexpected X-search subagent params: %#v", params)
		}
		native := params.NativeTools[0].(ai.XSearchTool)
		if len(native.AllowedXHandles) != 1 || native.AllowedXHandles[0] != "pydantic" || !native.IncludeOutput {
			t.Fatalf("X-search settings were not forwarded: %#v", native)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "X summary"}}}, nil
	})
	outerRequests := 0
	outer := &noNativeFunctionModel{fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		outerRequests++
		if outerRequests == 1 {
			if len(params.NativeTools) != 0 || len(params.Tools) != 1 || params.Tools[0].Name != "x_search" {
				t.Fatalf("unexpected outer X-search tools: native=%#v local=%#v", params.NativeTools, params.Tools)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "x_search", ToolCallID: "x-1", Args: []byte(`{"query":"Pydantic"}`),
			}}}, nil
		}
		request := messages[len(messages)-1].(ai.ModelRequest)
		if request.Parts[0].(ai.ToolReturnPart).Content != "X summary" {
			t.Fatalf("unexpected X-search return: %#v", request)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})}
	allowed := []string{"pydantic"}
	capability := ai.NewXSearchCapabilityWithFallback(ai.XSearchSubagentConfig[struct{}]{
		Model: inner,
		Native: ai.XSearchTool{
			AllowedXHandles: allowed, IncludeOutput: true,
		},
		Instructions: "Search X now and summarize.",
	})
	allowed[0] = "mutated"
	result, err := ai.NewAgent[struct{}, string](outer, ai.WithCapabilities(capability)).Run(
		t.Context(), "search X", struct{}{},
	)
	if err != nil || result.Output != "done" || innerRequests != 1 || outerRequests != 2 {
		t.Fatalf("unexpected X-search fallback result=%#v inner=%d outer=%d err=%v",
			result, innerRequests, outerRequests, err)
	}
}
