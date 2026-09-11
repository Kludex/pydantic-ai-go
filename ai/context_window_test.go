package ai_test

import (
	"context"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/anthropic"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
	"github.com/Kludex/pydantic-ai-go/ai/models/google"
	"github.com/Kludex/pydantic-ai-go/ai/models/openai"
)

func TestRunContextReportsLatestContextWindowUsage(t *testing.T) {
	requests := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			return &ai.ModelResponse{
				Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "record", ToolCallID: "record", Args: []byte(`{}`)}},
				Usage: ai.Usage{InputTokens: 175, OutputTokens: 25, CacheReadTokens: 50},
			}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	profiled := ai.NewProfiledModel(model, ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool,
		ContextWindow:     1000,
	})
	agent := ai.NewAgent[deps, string](profiled)
	var used float64
	var known bool
	ai.AddTool(agent, "record", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (string, error) {
		used, known = rc.ContextWindowUsed()
		return "recorded", nil
	})

	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if !known || used != 0.2 {
		t.Fatalf("unexpected context-window usage: used=%v known=%v", used, known)
	}
}

type profileOnlyModel struct {
	*fakes.FunctionModel
	profile ai.ModelProfile
}

func (model *profileOnlyModel) ModelProfile() ai.ModelProfile { return model.profile }

type contextWindowCapability struct {
	values []float64
	known  []bool
}

func (*contextWindowCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (capability *contextWindowCapability) BeforeModelRequest(
	_ context.Context, info *ai.RunInfo, request ai.ModelRequestContext,
) (ai.ModelRequestContext, error) {
	used, known := info.ContextWindowUsed()
	capability.values = append(capability.values, used)
	capability.known = append(capability.known, known)
	return request, nil
}

func TestContextWindowUnknownAndFallbackMinimum(t *testing.T) {
	empty := &ai.RunContext[deps]{}
	if used, known := empty.ContextWindowUsed(); known || used != 0 || empty.Messages() != nil {
		t.Fatalf("empty run context reported state: used=%v known=%v messages=%v", used, known, empty.Messages())
	}
	first := ai.NewProfiledModel(fakes.NewTestModel(), ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool,
		ContextWindow:     1000,
	})
	second := ai.NewProfiledModel(fakes.NewTestModel(), ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool,
		ContextWindow:     500,
	})
	fallback := ai.NewFallbackModel(first, ai.WithFallbackModels(second, fakes.NewTestModel()))
	if fallback.ContextWindow() != 500 {
		t.Fatalf("unexpected fallback context window %d", fallback.ContextWindow())
	}
	if ai.NewFallbackModel(fakes.NewTestModel()).ContextWindow() != 0 {
		t.Fatal("windowless fallback reported a context window")
	}
	if ai.WrapModel(first).ContextWindow() != 1000 {
		t.Fatal("model wrapper did not delegate the context window")
	}
	profileOnly := &profileOnlyModel{
		FunctionModel: fakes.NewFunctionModel(func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		}),
		profile: ai.ModelProfile{DefaultOutputMode: ai.OutputModeTool, ContextWindow: 2048},
	}
	if ai.WrapModel(profileOnly).ContextWindow() != 2048 {
		t.Fatal("model profile did not supply the context window")
	}
}

func TestContextWindowUsageIsUnknownWithoutReportedTokens(t *testing.T) {
	model := ai.NewProfiledModel(fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "record", ToolCallID: "record", Args: []byte(`{}`),
		}}}, nil
	}), ai.ModelProfile{DefaultOutputMode: ai.OutputModeTool, ContextWindow: 1000})
	agent := ai.NewAgent[deps, string](model)
	ai.AddTool(agent, "record", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (string, error) {
		if used, known := rc.ContextWindowUsed(); known || used != 0 {
			t.Fatalf("unreported usage produced a ratio: used=%v known=%v", used, known)
		}
		rc.Cancel()
		return "", nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err == nil {
		t.Fatal("expected cancellation")
	}
}

func TestRunInfoReportsContextWindowUsage(t *testing.T) {
	requests := 0
	model := ai.NewProfiledModel(fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			return &ai.ModelResponse{
				Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "work", ToolCallID: "work", Args: []byte(`{}`)}},
				Usage: ai.Usage{InputTokens: 300, OutputTokens: 100},
			}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	}), ai.ModelProfile{DefaultOutputMode: ai.OutputModeTool, ContextWindow: 1000})
	capability := &contextWindowCapability{}
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(capability))
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) { return "done", nil })
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if len(capability.values) != 2 || capability.known[0] || !capability.known[1] || capability.values[1] != 0.4 {
		t.Fatalf("unexpected capability context-window usage: values=%v known=%v", capability.values, capability.known)
	}
}

func TestBundledModelContextWindows(t *testing.T) {
	models := []struct {
		model ai.Model
		want  int
	}{
		{model: openai.NewModel("gpt-5"), want: 400_000},
		{model: openai.NewResponsesModel("gpt-5.4"), want: 1_050_000},
		{model: anthropic.NewModel("claude-sonnet-4-5"), want: 200_000},
		{model: google.NewModel("gemini-2.5-flash"), want: 1_048_576},
	}
	for _, test := range models {
		profile := test.model.(ai.ModelProfiler).ModelProfile()
		if profile.ContextWindow != test.want {
			t.Fatalf("unexpected %s context window: got %d want %d", test.model.Name(), profile.ContextWindow, test.want)
		}
		if window := test.model.(ai.ModelContextWindow).ContextWindow(); window != test.want {
			t.Fatalf("unexpected %s direct context window: got %d want %d", test.model.Name(), window, test.want)
		}
	}

	compatible := openai.NewModel("gpt-5", openai.WithBaseURL("https://example.com/v1"))
	if profile := compatible.ModelProfile(); profile.ContextWindow != 400_000 {
		t.Fatalf("provider fallback reported context window %d", profile.ContextWindow)
	}
	unknown := openai.NewModel("unknown-model")
	if profile := unknown.ModelProfile(); profile.ContextWindow != 0 {
		t.Fatalf("unknown model reported context window %d", profile.ContextWindow)
	}
	fallback := ai.NewFallbackModel(models[0].model, ai.WithFallbackModels(models[2].model, unknown))
	if fallback.ContextWindow() != 200_000 {
		t.Fatalf("unexpected bundled fallback context window %d", fallback.ContextWindow())
	}
}

func TestBundledContextWindowUsage(t *testing.T) {
	for _, model := range []ai.Model{
		openai.NewModel("gpt-5"),
		anthropic.NewModel("claude-sonnet-4-5"),
		google.NewModel("gemini-2.5-flash"),
	} {
		scripted := &profileResponseModel{Model: model}
		agent := ai.NewAgent[deps, string](scripted)
		var used float64
		var known bool
		ai.AddTool(agent, "record", func(
			_ context.Context, rc *ai.RunContext[deps], _ struct{},
		) (string, error) {
			used, known = rc.ContextWindowUsed()
			return "recorded", nil
		})
		if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
			t.Fatal(err)
		}
		want := 200.0 / float64(model.(ai.ModelProfiler).ModelProfile().ContextWindow)
		if !known || used != want {
			t.Fatalf("unexpected %s context-window usage: got %v, %v want %v", model.Name(), used, known, want)
		}
	}
}

type profileResponseModel struct {
	ai.Model
	requests int
}

func (model *profileResponseModel) ModelProfile() ai.ModelProfile {
	return model.Model.(ai.ModelProfiler).ModelProfile()
}

func (model *profileResponseModel) Request(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	model.requests++
	if model.requests == 1 {
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "record", ToolCallID: "record", Args: []byte(`{}`)}},
			Usage: ai.Usage{InputTokens: 175, OutputTokens: 25},
		}, nil
	}
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
}

func TestModelProfileRejectsNegativeContextWindow(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected negative context-window panic")
		}
	}()
	ai.NewProfiledModel(fakes.NewTestModel(), ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool,
		ContextWindow:     -1,
	})
}
