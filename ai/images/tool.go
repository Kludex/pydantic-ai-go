package images

import (
	"context"
	"errors"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// ToolConfig configures a local generate_image tool backed by a direct Generator.
type ToolConfig struct {
	// Settings are applied to each direct image request.
	Settings Settings
	// Action rejects edit-only native requests because the fallback receives no reference images.
	Action ai.ImageGenerationAction
	// Name overrides the default generate_image tool name.
	Name string
	// Description overrides the default model-facing description.
	Description string
}

// NewGenerationTool creates a local agent tool backed by a direct image generator.
// Add it to ImageGenerationCapabilityConfig.Local when native generation needs a fallback.
func NewGenerationTool[Deps any](generator *Generator, config ToolConfig) ai.Tool[Deps] {
	if generator == nil {
		panic("images: generator must not be nil")
	}
	name := config.Name
	if name == "" {
		name = "generate_image"
	}
	description := config.Description
	if description == "" {
		description = "Generate an image based on the given prompt."
	}
	settings := config.Settings.Clone()
	return ai.NewTool(name, func(
		ctx context.Context, _ *ai.RunContext[Deps], args struct {
			Prompt string `json:"prompt" jsonschema_description:"A description of the image to generate."`
		},
	) (ai.BinaryContent, error) {
		if config.Action == ai.ImageGenerationActionEdit {
			return ai.BinaryContent{}, fmt.Errorf(
				"images: direct image generation fallback cannot edit without reference images; call Generator.Generate with inputs",
			)
		}
		result, err := generator.Generate(ctx, args.Prompt, nil, settings)
		if err != nil {
			var filtered *ai.ContentFilterError
			if errors.As(err, &filtered) {
				return ai.BinaryContent{}, ai.Retryf("%v", err)
			}
			return ai.BinaryContent{}, err
		}
		if len(result.Images) != 1 {
			return ai.BinaryContent{}, &ai.UnexpectedModelBehaviorError{Message: fmt.Sprintf(
				"direct image generation fallback returned %d images; expected exactly one", len(result.Images),
			)}
		}
		return result.Image(), nil
	}, ai.WithDescription(description))
}
