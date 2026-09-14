package ai_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type noNativeFunctionModel struct{ *fakes.FunctionModel }

func (*noNativeFunctionModel) SupportsNativeTool(ai.NativeTool) bool { return false }

type namedModel struct{ name string }

func (model namedModel) Name() string { return model.name }
func (namedModel) Request(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
	return nil, errors.New("unexpected request")
}

func TestRepeatedImageCapabilitiesRebuildFallbackFromMergedConfig(t *testing.T) {
	inner := imageOutputModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		tool := params.NativeTools[0].(ai.ImageGenerationTool)
		if tool.Quality != ai.ImageGenerationQualityHigh || tool.Size != ai.ImageGenerationSize2K {
			t.Fatalf("merged image settings did not reach rebuilt fallback: %#v", tool)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.FilePart{Content: ai.BinaryContent{
			Data: []byte("image"), MediaType: "image/png",
		}}}}, nil
	})
	outerCalls := 0
	outer := &noNativeFunctionModel{fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		outerCalls++
		if outerCalls == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: params.Tools[0].Name, ToolCallID: "image", Args: []byte(`{"prompt":"draw"}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})}
	agent := ai.NewAgent[struct{}, string](outer, ai.WithCapabilities(
		ai.NewImageGenerationCapabilityWithFallback(ai.ImageGenerationSubagentConfig[struct{}]{
			Model: inner, Native: ai.ImageGenerationTool{Quality: ai.ImageGenerationQualityHigh},
		}),
		ai.NewImageGenerationCapabilityWithFallback(ai.ImageGenerationSubagentConfig[struct{}]{
			Model: inner, Native: ai.ImageGenerationTool{Size: ai.ImageGenerationSize2K},
		}),
	))
	if result, err := agent.Run(t.Context(), "draw", struct{}{}); err != nil || result.Output != "done" {
		t.Fatalf("merged image fallback failed: result=%+v err=%v", result, err)
	}
}

func TestDynamicImageGenerationFallbackValidation(t *testing.T) {
	defer func() {
		if recovered := fmt.Sprint(recover()); !strings.Contains(recovered, "resolver must not be nil") {
			t.Fatalf("unexpected panic: %s", recovered)
		}
	}()
	ai.NewDynamicImageGenerationCapabilityWithFallback[struct{}](nil, ai.ImageGenerationSubagentConfig[struct{}]{
		Model: imageOutputModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
			return nil, errors.New("unused")
		}),
	})
}

func TestImageGenerationSubagentNativeResolverError(t *testing.T) {
	sentinel := errors.New("native settings")
	tool := ai.NewImageGenerationSubagentTool(ai.ImageGenerationSubagentConfig[struct{}]{
		Model: imageOutputModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
			return nil, errors.New("unused")
		}),
		ResolveNative: func(context.Context, *ai.RunContext[struct{}]) (ai.ImageGenerationTool, error) {
			return ai.ImageGenerationTool{}, sentinel
		},
	})
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	agent.AddTool(tool)
	if _, err := agent.Run(t.Context(), "draw", struct{}{}); !errors.Is(err, sentinel) {
		t.Fatalf("unexpected native resolver error: %v", err)
	}
}

func TestDynamicImageGenerationFallbackPreservesNativeConfig(t *testing.T) {
	inner := imageOutputModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		tool := params.NativeTools[0].(ai.ImageGenerationTool)
		if tool.Quality != ai.ImageGenerationQualityHigh || tool.Size != ai.ImageGenerationSize2K {
			t.Fatalf("dynamic image settings did not reach fallback: %#v", tool)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.FilePart{Content: ai.BinaryContent{
			Data: []byte("image"), MediaType: "image/png",
		}}}}, nil
	})
	outerCalls := 0
	outer := &noNativeFunctionModel{fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		outerCalls++
		if outerCalls == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: params.Tools[0].Name, ToolCallID: "image", Args: []byte(`{"prompt":"draw"}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})}
	resolve := func(context.Context, *ai.RunContext[string]) (ai.ImageGenerationTool, error) {
		return ai.ImageGenerationTool{Quality: ai.ImageGenerationQualityHigh, Size: ai.ImageGenerationSize2K}, nil
	}
	agent := ai.NewAgent[string, string](outer, ai.WithCapabilities(
		ai.NewDynamicImageGenerationCapabilityWithFallback(resolve, ai.ImageGenerationSubagentConfig[string]{Model: inner}),
	))
	if result, err := agent.Run(t.Context(), "draw", "tenant"); err != nil || result.Output != "done" {
		t.Fatalf("dynamic image fallback failed: result=%+v err=%v", result, err)
	}
}

func TestImageGenerationSubagentFallback(t *testing.T) {
	innerRequests := 0
	inner := imageOutputModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		innerRequests++
		if !params.AllowImageOutput || len(params.NativeTools) != 1 ||
			!strings.Contains(params.Instructions, "Create the requested image") {
			t.Fatalf("unexpected image subagent request: %#v", params)
		}
		native := params.NativeTools[0].(ai.ImageGenerationTool)
		if native.Quality != ai.ImageGenerationQualityHigh {
			t.Fatalf("image settings were not forwarded: %#v", native)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.FilePart{Content: ai.BinaryContent{
			Data: []byte("generated"), MediaType: "image/png",
		}}}}, nil
	})
	outerRequests := 0
	outer := &noNativeFunctionModel{fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		outerRequests++
		if outerRequests == 1 {
			if len(params.NativeTools) != 0 || len(params.Tools) != 1 || params.Tools[0].Name != "generate_image" {
				t.Fatalf("unexpected outer image tools: native=%#v local=%#v", params.NativeTools, params.Tools)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "generate_image", ToolCallID: "image-1", Args: []byte(`{"prompt":"A gopher"}`),
			}}}, nil
		}
		request := messages[len(messages)-1].(ai.ModelRequest)
		if len(request.Parts) != 2 {
			t.Fatalf("generated image was not returned as rich content: %#v", request)
		}
		toolReturn := request.Parts[0].(ai.ToolReturnPart)
		image := toolReturn.Content.(ai.BinaryContent)
		extra := request.Parts[1].(ai.UserPromptPart).Contents
		if len(extra) != 3 {
			t.Fatalf("unexpected generated image framing: %#v", request)
		}
		imageContent := extra[1].(ai.BinaryContent)
		if string(image.Data) != "generated" || string(imageContent.Data) != "generated" {
			t.Fatalf("unexpected generated image return: %#v", request)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})}
	capability := ai.NewImageGenerationCapabilityWithFallback(ai.ImageGenerationSubagentConfig[struct{}]{
		Model: inner,
		Native: ai.ImageGenerationTool{
			Quality: ai.ImageGenerationQualityHigh,
		},
		Instructions: "Create the requested image immediately.",
	})
	agent := ai.NewAgent[struct{}, string](outer, ai.WithCapabilities(capability))
	result, err := agent.Run(t.Context(), "draw", struct{}{})
	if err != nil || result.Output != "done" || innerRequests != 1 || outerRequests != 2 {
		t.Fatalf("unexpected image fallback result=%#v inner=%d outer=%d err=%v",
			result, innerRequests, outerRequests, err)
	}
}

func TestDynamicImageGenerationSubagentModel(t *testing.T) {
	resolveCalls := 0
	inner := imageOutputModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.FilePart{Content: ai.BinaryContent{
			Data: []byte("generated"), MediaType: "image/png",
		}}}}, nil
	})
	tool := ai.NewImageGenerationSubagentTool(ai.ImageGenerationSubagentConfig[nativeOrLocalDeps]{
		ResolveModel: func(
			_ context.Context, rc *ai.RunContext[nativeOrLocalDeps],
		) (ai.Model, error) {
			resolveCalls++
			if rc.Deps.Location != "Mexico" || rc.ToolName != "generate_image" {
				t.Fatalf("unexpected fallback resolver context: %#v", rc)
			}
			return inner, nil
		},
	})
	outerRequests := 0
	outer := &noNativeFunctionModel{fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		outerRequests++
		if outerRequests == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "generate_image", ToolCallID: "image-1", Args: []byte(`{"prompt":"A gopher"}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})}
	agent := ai.NewAgent[nativeOrLocalDeps, string](outer)
	agent.AddTool(tool)
	if _, err := agent.Run(t.Context(), "draw", nativeOrLocalDeps{Location: "Mexico"}); err != nil ||
		resolveCalls != 1 {
		t.Fatalf("dynamic image fallback failed: resolves=%d err=%v", resolveCalls, err)
	}
}

func TestImageGenerationSubagentErrors(t *testing.T) {
	assertNativeOrLocalPanic(t, "requires exactly one model", func() {
		ai.NewImageGenerationSubagentTool(ai.ImageGenerationSubagentConfig[struct{}]{})
	})
	assertNativeOrLocalPanic(t, "requires exactly one model", func() {
		ai.NewImageGenerationSubagentTool(ai.ImageGenerationSubagentConfig[struct{}]{
			Model: namedModel{name: "model"},
			ResolveModel: func(context.Context, *ai.RunContext[struct{}]) (ai.Model, error) {
				return namedModel{name: "model"}, nil
			},
		})
	})
	for _, name := range []string{"openai-responses:gpt-image-1", "gpt-image-2"} {
		assertNativeOrLocalPanic(t, "dedicated image-generation model", func() {
			ai.NewImageGenerationSubagentTool(ai.ImageGenerationSubagentConfig[struct{}]{
				Model: namedModel{name: name},
			})
		})
	}

	for _, test := range []struct {
		name    string
		resolve ai.ImageGenerationFallbackModelFunc[struct{}]
		want    string
	}{
		{name: "resolver error", want: "resolve model", resolve: func(
			context.Context, *ai.RunContext[struct{}],
		) (ai.Model, error) {
			return nil, errors.New("resolve model")
		}},
		{name: "nil model", want: "resolver returned nil", resolve: func(
			context.Context, *ai.RunContext[struct{}],
		) (ai.Model, error) {
			return nil, nil
		}},
		{name: "image-only model", want: "dedicated image-generation model", resolve: func(
			context.Context, *ai.RunContext[struct{}],
		) (ai.Model, error) {
			return namedModel{name: "google/imagen-3.0-generate-002"}, nil
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			tool := ai.NewImageGenerationSubagentTool(ai.ImageGenerationSubagentConfig[struct{}]{
				ResolveModel: test.resolve,
			})
			outerCalls := 0
			outer := &noNativeFunctionModel{fakes.NewFunctionModel(func(
				context.Context, []ai.ModelMessage, ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				outerCalls++
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
					ToolName: "generate_image", ToolCallID: "image-1", Args: []byte(`{"prompt":"A gopher"}`),
				}}}, nil
			})}
			agent := ai.NewAgent[struct{}, string](outer)
			agent.AddTool(tool)
			if _, err := agent.Run(t.Context(), "draw", struct{}{}); err == nil ||
				!strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected image subagent error: %v", err)
			}
		})
	}
}

func TestImageGenerationSubagentModelError(t *testing.T) {
	modelErr := errors.New("image provider failed")
	inner := imageOutputModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, modelErr
	})
	outer := &noNativeFunctionModel{fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "generate_image", ToolCallID: "image-1", Args: []byte(`{"prompt":"A gopher"}`),
		}}}, nil
	})}
	agent := ai.NewAgent[struct{}, string](outer, ai.WithCapabilities(
		ai.NewImageGenerationCapabilityWithFallback(ai.ImageGenerationSubagentConfig[struct{}]{Model: inner}),
	))
	if _, err := agent.Run(t.Context(), "draw", struct{}{}); !errors.Is(err, modelErr) {
		t.Fatalf("image model error was not preserved: %v", err)
	}
}

func TestBinaryPointerToolReturnIsRich(t *testing.T) {
	requests := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "binary", ToolCallID: "binary-1", Args: []byte(`{}`),
			}}}, nil
		}
		request := messages[len(messages)-1].(ai.ModelRequest)
		if len(request.Parts) != 2 {
			t.Fatalf("binary pointer was not returned as rich content: %#v", request)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model)
	ai.AddSimpleTool(agent, "binary", func(context.Context, struct{}) (*ai.BinaryContent, error) {
		return &ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"}, nil
	})
	if _, err := agent.Run(t.Context(), "run", struct{}{}); err != nil {
		t.Fatal(err)
	}
}

func TestImageGenerationSubagentFailureBecomesRetry(t *testing.T) {
	innerRequests := 0
	inner := imageOutputModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		innerRequests++
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "no image"}}}, nil
	})
	outerRequests := 0
	outer := &noNativeFunctionModel{fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		outerRequests++
		if outerRequests == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "generate_image", ToolCallID: "image-1", Args: []byte(`{"prompt":"A gopher"}`),
			}}}, nil
		}
		request := messages[len(messages)-1].(ai.ModelRequest)
		if _, ok := request.Parts[0].(ai.RetryPromptPart); !ok {
			t.Fatalf("subagent failure was not a retry: %#v", request)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "gave up"}}}, nil
	})}
	capability := ai.NewImageGenerationCapabilityWithFallback(ai.ImageGenerationSubagentConfig[struct{}]{
		Model: inner,
	})
	result, err := ai.NewAgent[struct{}, string](outer, ai.WithCapabilities(capability)).Run(
		t.Context(), "draw", struct{}{},
	)
	if err != nil || result.Output != "gave up" || innerRequests != 2 || outerRequests != 2 {
		t.Fatalf("unexpected failed image fallback result=%#v inner=%d outer=%d err=%v",
			result, innerRequests, outerRequests, err)
	}
}
