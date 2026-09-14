package ai

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

const defaultImageGenerationInstructions = "Generate an image based on the user prompt. Do not ask clarifying questions."

// ImageGenerationArgs is the model-facing input for the default local image-generation tool.
type ImageGenerationArgs struct {
	// Prompt describes the image to generate.
	Prompt string `json:"prompt" jsonschema_description:"A description of the image to generate."`
}

// ImageGenerationFallbackModelFunc resolves the subagent model for one tool call.
// It may run concurrently and must return a concurrency-safe model.
type ImageGenerationFallbackModelFunc[Deps any] func(
	ctx context.Context, rc *RunContext[Deps],
) (Model, error)

// ImageGenerationSubagentConfig configures the default local image-generation subagent.
type ImageGenerationSubagentConfig[Deps any] struct {
	// Model is the fixed image-capable subagent model.
	Model Model
	// ResolveModel selects an image-capable model for each call.
	ResolveModel ImageGenerationFallbackModelFunc[Deps]
	// Native configures the subagent's provider-native image tool.
	Native ImageGenerationTool
	// ResolveNative resolves image settings from the outer tool-call context.
	ResolveNative ImageGenerationFunc[Deps]
	// Instructions overrides the default image-generation prompt.
	Instructions string
}

// NewImageGenerationSubagentTool creates a generate_image tool backed by an image-output subagent.
func NewImageGenerationSubagentTool[Deps any](config ImageGenerationSubagentConfig[Deps]) Tool[Deps] {
	if modelIsNil(config.Model) == (config.ResolveModel == nil) {
		panic("ai: image-generation subagent requires exactly one model or model resolver")
	}
	if !modelIsNil(config.Model) {
		validateImageGenerationFallbackModel(config.Model)
	}
	native := cloneNativeTool(config.Native).(ImageGenerationTool)
	instructions := config.Instructions
	if instructions == "" {
		instructions = defaultImageGenerationInstructions
	}
	return NewTool("generate_image", func(
		ctx context.Context, rc *RunContext[Deps], args ImageGenerationArgs,
	) (BinaryContent, error) {
		model := config.Model
		if config.ResolveModel != nil {
			var err error
			model, err = config.ResolveModel(ctx, rc.clone())
			if err != nil {
				return BinaryContent{}, err
			}
			if modelIsNil(model) {
				return BinaryContent{}, errors.New("ai: image-generation model resolver returned nil")
			}
			if err := imageGenerationFallbackModelError(model); err != nil {
				return BinaryContent{}, err
			}
		}
		resolvedNative := native
		if config.ResolveNative != nil {
			var err error
			resolvedNative, err = config.ResolveNative(ctx, rc.clone())
			if err != nil {
				return BinaryContent{}, err
			}
		}
		agent := NewImageOutputAgent[struct{}](
			model,
			WithInstructions(instructions),
			WithCapabilities(NewImageGenerationCapability(ImageGenerationCapabilityConfig[struct{}]{Native: resolvedNative})),
		)
		result, err := agent.Run(ctx, args.Prompt, struct{}{})
		if err != nil {
			var unexpected *UnexpectedModelBehaviorError
			if errors.As(err, &unexpected) || errors.Is(err, ErrMaxRetriesExceeded) {
				return BinaryContent{}, Retryf("%v", err)
			}
			return BinaryContent{}, err
		}
		return cloneBinaryContent(result.Output), nil
	}, WithDescription("Generate an image based on the given prompt."))
}

func validateImageGenerationFallbackModel(model Model) {
	if err := imageGenerationFallbackModelError(model); err != nil {
		panic(err.Error())
	}
}

func imageGenerationFallbackModelError(model Model) error {
	name := model.Name()
	if index := strings.LastIndexAny(name, ":/"); index >= 0 {
		name = name[index+1:]
	}
	var suggestion string
	switch name {
	case "gpt-image-2", "gpt-image-1.5":
		suggestion = "openai.NewResponsesModel(\"gpt-5.5\")"
	case "gpt-image-1", "gpt-image-1-mini", "dall-e-3", "dall-e-2":
		suggestion = "openai.NewResponsesModel(\"gpt-5.4\")"
	case "imagen-3.0-generate-002", "imagen-3.0-fast-generate-001":
		suggestion = "google.NewModel(\"gemini-3-pro-image\")"
	default:
		return nil
	}
	return fmt.Errorf(
		"ai: %q is a dedicated image-generation model that cannot run the fallback subagent; use %s or another conversational image model",
		name, suggestion,
	)
}
