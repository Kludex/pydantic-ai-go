package ai_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func imageOutputModel(fn func(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (*ai.ModelResponse, error)) ai.Model {
	return ai.NewProfiledModel(fakes.NewFunctionModel(fn), ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool, SupportsImageOutput: true,
	})
}

func TestImageOutputAgentReturnsDetachedImage(t *testing.T) {
	data := []byte("image")
	metadata := map[string]any{"source": "model"}
	model := imageOutputModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if !params.AllowImageOutput || params.AllowText || params.OutputSchema != nil || params.OutputTool != nil {
			t.Fatalf("unexpected image output params: %#v", params)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.FilePart{Content: ai.BinaryContent{
			Data: data, MediaType: "image/png", VendorMetadata: metadata,
		}}}}, nil
	})
	agent := ai.NewImageOutputAgent[struct{}](model)
	result, err := agent.Run(t.Context(), "draw", struct{}{})
	if err != nil || string(result.Output.Data) != "image" || result.Output.MediaType != "image/png" {
		t.Fatalf("unexpected image output: %#v err=%v", result, err)
	}
	data[0] = 'X'
	metadata["source"] = "mutated"
	result.Output.Data[1] = 'Y'
	result.Output.VendorMetadata["source"] = "consumer"
	history := result.Messages()
	file := history[1].(ai.ModelResponse).Parts[0].(ai.FilePart)
	if string(file.Content.Data) != "image" || file.Content.VendorMetadata["source"] != "model" {
		t.Fatalf("image output was not detached from history: %#v", file)
	}
}

func TestImageOutputRetriesMissingAndRejectedImages(t *testing.T) {
	requests := 0
	model := imageOutputModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		switch requests {
		case 1:
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.FilePart{Content: ai.BinaryContent{Data: []byte("audio"), MediaType: "audio/wav"}},
				ai.TextPart{Content: "not an image"},
			}}, nil
		case 2:
			request := messages[len(messages)-1].(ai.ModelRequest)
			if request.Parts[0].(ai.RetryPromptPart).Content != "Please return an image." {
				t.Fatalf("unexpected missing-image retry: %#v", request)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.FilePart{Content: ai.BinaryContent{
				Data: []byte("bad"), MediaType: "image/png",
			}}}}, nil
		default:
			request := messages[len(messages)-1].(ai.ModelRequest)
			if request.Parts[0].(ai.RetryPromptPart).Content != "reject generated image" {
				t.Fatalf("unexpected validator retry: %#v", request)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.FilePart{Content: ai.BinaryContent{
				Data: []byte("good"), MediaType: "image/png",
			}}}}, nil
		}
	})
	agent := ai.NewImageOutputAgent[struct{}](model, ai.WithRetryLimits(ai.RetryLimits{Output: 2}))
	agent.AddOutputValidator(func(_ context.Context, _ *ai.RunContext[struct{}], image ai.BinaryContent) error {
		if string(image.Data) == "bad" {
			return ai.Retryf("reject generated image")
		}
		return nil
	})
	result, err := agent.Run(t.Context(), "draw", struct{}{})
	if err != nil || string(result.Output.Data) != "good" || requests != 3 {
		t.Fatalf("unexpected retried image output: %#v requests=%d err=%v", result, requests, err)
	}
}

func TestImageOutputProcessingHooks(t *testing.T) {
	validationCalls := 0
	processingCalls := 0
	model := imageOutputModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.FilePart{Content: ai.BinaryContent{
			Data: []byte("original"), MediaType: "image/png",
		}}}}, nil
	})
	beforeValidation := ai.BeforeOutputValidationFunc(func(
		context.Context, *ai.RunInfo, ai.OutputHookContext, any,
	) (any, error) {
		validationCalls++
		return nil, errors.New("image validation hook must not run")
	})
	afterProcessing := ai.AfterOutputProcessingFunc(func(
		_ context.Context, _ *ai.RunInfo, hook ai.OutputHookContext, output any,
	) (any, error) {
		processingCalls++
		if hook.Mode != ai.OutputHookModeImage || !hook.AllowsImage || hook.AllowsText || hook.Structured {
			t.Fatalf("unexpected image hook context: %#v", hook)
		}
		image := output.(ai.BinaryContent)
		image.Data = []byte("transformed")
		return image, nil
	})
	agent := ai.NewImageOutputAgent[struct{}](model, ai.WithCapabilities(beforeValidation, afterProcessing))
	result, err := agent.Run(t.Context(), "draw", struct{}{})
	if err != nil || string(result.Output.Data) != "transformed" || validationCalls != 0 || processingCalls != 1 {
		t.Fatalf("unexpected processed image: %#v validation=%d processing=%d err=%v",
			result, validationCalls, processingCalls, err)
	}
}

func TestImageOutputProfileAndProcessingValidation(t *testing.T) {
	requests := 0
	unsupported := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		return nil, nil
	})
	if _, err := ai.NewImageOutputAgent[struct{}](unsupported).Run(t.Context(), "draw", struct{}{}); err == nil ||
		!strings.Contains(err.Error(), "image output is not supported") || requests != 0 {
		t.Fatalf("unsupported image model was called: requests=%d err=%v", requests, err)
	}

	model := imageOutputModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.FilePart{Content: ai.BinaryContent{
			Data: []byte("image"), MediaType: "image/png",
		}}}}, nil
	})
	invalid := ai.AfterOutputProcessingFunc(func(
		context.Context, *ai.RunInfo, ai.OutputHookContext, any,
	) (any, error) {
		return ai.BinaryContent{Data: []byte("audio"), MediaType: "audio/wav"}, nil
	})
	if _, err := ai.NewImageOutputAgent[struct{}](model, ai.WithCapabilities(invalid)).Run(
		t.Context(), "draw", struct{}{},
	); err == nil || !strings.Contains(err.Error(), "processed image output must have an image media type") {
		t.Fatalf("invalid processed image succeeded: %v", err)
	}

	agent := ai.NewImageOutputAgent[struct{}](model)
	if _, err := ai.RunAs[string](t.Context(), agent, "draw", struct{}{}); !errors.Is(
		err, ai.ErrOutputTypeOverrideWithImageOutput,
	) {
		t.Fatalf("image output override succeeded: %v", err)
	}
}

func TestImageOutputRetryExhaustion(t *testing.T) {
	for _, test := range []struct {
		name     string
		response ai.ModelResponse
		validate bool
	}{
		{name: "missing", response: ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "text"}}}},
		{name: "rejected", validate: true, response: ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.FilePart{Content: ai.BinaryContent{Data: []byte("bad"), MediaType: "image/png"}},
		}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := imageOutputModel(func(
				context.Context, []ai.ModelMessage, ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				response := test.response
				return &response, nil
			})
			agent := ai.NewImageOutputAgent[struct{}](model, ai.WithRetryLimits(ai.RetryLimits{}))
			if test.validate {
				agent.AddOutputValidator(func(context.Context, *ai.RunContext[struct{}], ai.BinaryContent) error {
					return ai.Retryf("reject")
				})
			}
			if _, err := agent.Run(t.Context(), "draw", struct{}{}); err == nil ||
				!strings.Contains(err.Error(), "output exceeded 0 retries") {
				t.Fatalf("image retry limit was ignored: %v", err)
			}
		})
	}
}

func TestEarlyImageOutputRetryAndError(t *testing.T) {
	requests := 0
	toolCalls := 0
	model := imageOutputModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.FilePart{Content: ai.BinaryContent{Data: []byte("bad"), MediaType: "image/png"}},
				ai.ToolCallPart{ToolName: "record", ToolCallID: "call", Args: []byte(`{}`)},
			}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.FilePart{Content: ai.BinaryContent{
			Data: []byte("good"), MediaType: "image/png",
		}}}}, nil
	})
	agent := ai.NewImageOutputAgent[struct{}](model, ai.WithEndStrategy(ai.EndStrategyEarly))
	agent.AddOutputValidator(func(_ context.Context, _ *ai.RunContext[struct{}], image ai.BinaryContent) error {
		if string(image.Data) == "bad" {
			return ai.Retryf("try another image")
		}
		return nil
	})
	ai.AddSimpleTool(agent, "record", func(context.Context, struct{}) (string, error) {
		toolCalls++
		return "recorded", nil
	})
	result, err := agent.Run(t.Context(), "draw", struct{}{})
	if err != nil || string(result.Output.Data) != "good" || toolCalls != 1 {
		t.Fatalf("early image retry did not continue: result=%#v calls=%d err=%v", result, toolCalls, err)
	}

	processingErr := errors.New("process image")
	failing := ai.NewImageOutputAgent[struct{}](model, ai.WithEndStrategy(ai.EndStrategyEarly), ai.WithCapabilities(
		ai.BeforeOutputProcessingFunc(func(
			context.Context, *ai.RunInfo, ai.OutputHookContext, any,
		) (any, error) {
			return nil, processingErr
		}),
	))
	ai.AddSimpleTool(failing, "record", func(context.Context, struct{}) (string, error) { return "", nil })
	requests = 0
	if _, err := failing.Run(t.Context(), "draw", struct{}{}); !errors.Is(err, processingErr) {
		t.Fatalf("early image processing error was ignored: %v", err)
	}
}

func TestImageOutputStreaming(t *testing.T) {
	model := imageOutputModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.FilePart{Content: ai.BinaryContent{
			Data: []byte("streamed"), MediaType: "image/webp",
		}}}}, nil
	})
	stream := ai.NewImageOutputAgent[struct{}](model).RunStream(t.Context(), "draw", struct{}{})
	events := 0
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		events++
	}
	result := stream.Result()
	if result == nil || string(result.Output.Data) != "streamed" || result.Output.MediaType != "image/webp" ||
		events == 0 {
		t.Fatalf("unexpected streamed image result=%#v events=%d", result, events)
	}
}

func TestImageOutputEndStrategies(t *testing.T) {
	for _, test := range []struct {
		name      string
		strategy  ai.EndStrategy
		wantImage string
		wantCalls int
	}{
		{name: "early", strategy: ai.EndStrategyEarly, wantImage: "first"},
		{name: "graceful", strategy: ai.EndStrategyGraceful, wantImage: "second", wantCalls: 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			requests := 0
			toolCalls := 0
			model := imageOutputModel(func(
				context.Context, []ai.ModelMessage, ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				requests++
				if requests == 1 {
					return &ai.ModelResponse{Parts: []ai.ResponsePart{
						ai.FilePart{Content: ai.BinaryContent{Data: []byte("first"), MediaType: "image/png"}},
						ai.ToolCallPart{ToolName: "record", ToolCallID: "call", Args: []byte(`{}`)},
					}}, nil
				}
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.FilePart{Content: ai.BinaryContent{
					Data: []byte("second"), MediaType: "image/png",
				}}}}, nil
			})
			agent := ai.NewImageOutputAgent[struct{}](model, ai.WithEndStrategy(test.strategy))
			ai.AddSimpleTool(agent, "record", func(context.Context, struct{}) (string, error) {
				toolCalls++
				return "recorded", nil
			})
			result, err := agent.Run(t.Context(), "draw", struct{}{})
			if err != nil || string(result.Output.Data) != test.wantImage || toolCalls != test.wantCalls {
				t.Fatalf("unexpected image strategy result=%#v calls=%d err=%v", result, toolCalls, err)
			}
		})
	}
}
