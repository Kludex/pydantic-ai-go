package images

import (
	"context"
	"fmt"
	"reflect"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// CapabilityConfig configures native image generation with a direct or custom local fallback.
type CapabilityConfig[Deps any] struct {
	// Native configures provider-hosted image generation.
	Native ai.ImageGenerationTool
	// ResolveNative resolves provider-hosted settings before each model request.
	ResolveNative ai.ImageGenerationFunc[Deps]
	// Local supplies an application-owned fallback toolset.
	Local ai.Toolset[Deps]
	// Generator supplies a configured direct image generator as the fallback.
	Generator *Generator
	// FallbackModel supplies a direct image model as the fallback.
	FallbackModel Model
	// Settings applies to direct fallback calls. Portable aspect ratios also apply to the native tool.
	Settings Settings
}

// ImageGenerationCapability combines repeated direct image-generation declarations.
type ImageGenerationCapability[Deps any] struct {
	native   []imageNativeDeclaration[Deps]
	fallback imageFallback[Deps]
	settings Settings
}

type imageNativeDeclaration[Deps any] struct {
	native  ai.ImageGenerationTool
	resolve ai.ImageGenerationFunc[Deps]
}

type imageFallback[Deps any] struct {
	local     ai.Toolset[Deps]
	generator *Generator
	stated    bool
}

// NewImageGenerationCapability creates native-first image generation with an optional direct fallback.
func NewImageGenerationCapability[Deps any](config CapabilityConfig[Deps]) *ImageGenerationCapability[Deps] {
	fallbacks := 0
	if !toolsetIsNil(config.Local) {
		fallbacks++
	}
	if config.Generator != nil {
		fallbacks++
	}
	if !modelIsNil(config.FallbackModel) {
		fallbacks++
	}
	if fallbacks > 1 {
		panic("images: configure only one of Local, Generator, or FallbackModel")
	}
	if err := config.Settings.Validate(); err != nil {
		panic(err.Error())
	}
	generator := config.Generator
	if !modelIsNil(config.FallbackModel) {
		generator = New(config.FallbackModel)
	}
	declaration := imageNativeDeclaration[Deps]{
		native: config.Native.CloneNativeTool().(ai.ImageGenerationTool), resolve: config.ResolveNative,
	}
	if declaration.resolve != nil {
		declaration.native = ai.ImageGenerationTool{}
	}
	return &ImageGenerationCapability[Deps]{
		native: []imageNativeDeclaration[Deps]{declaration},
		fallback: imageFallback[Deps]{
			local: config.Local, generator: generator, stated: !toolsetIsNil(config.Local) || generator != nil,
		},
		settings: config.Settings.Clone(),
	}
}

// CapabilityID returns the shared native image-generation identity.
func (*ImageGenerationCapability[Deps]) CapabilityID() string { return "image_generation" }

// Setup registers the combined native tool and fallback.
func (capability *ImageGenerationCapability[Deps]) Setup(registry *ai.CapabilityRegistry) error {
	if capability == nil {
		return fmt.Errorf("images: image-generation capability must not be nil")
	}
	config := ai.ImageGenerationCapabilityConfig[Deps]{Local: capability.fallback.local}
	if capability.hasDynamicNative() {
		config.ResolveNative = func(
			ctx context.Context, rc *ai.RunContext[Deps],
		) (ai.ImageGenerationTool, error) {
			var native ai.ImageGenerationTool
			for _, declaration := range capability.native {
				resolved := declaration.native
				if declaration.resolve != nil {
					var err error
					resolved, err = declaration.resolve(ctx, rc)
					if err != nil {
						return ai.ImageGenerationTool{}, err
					}
				}
				native = mergeNativeImageSettings(native, resolved)
			}
			return applyNativeImageSettings(native, capability.settings), nil
		}
	} else {
		var native ai.ImageGenerationTool
		for _, declaration := range capability.native {
			native = mergeNativeImageSettings(native, declaration.native)
		}
		config.Native = applyNativeImageSettings(native, capability.settings)
	}
	if capability.fallback.generator != nil {
		settings := capability.settings.Clone()
		generator := capability.fallback.generator
		config.LocalForNative = func(native ai.ImageGenerationTool) ai.Tool[Deps] {
			callSettings := settings.Clone()
			if callSettings.Dimensions == nil && callSettings.AspectRatio == "" && native.AspectRatio != "" {
				callSettings.AspectRatio = AspectRatio(native.AspectRatio)
			}
			return NewGenerationTool[Deps](generator, ToolConfig{Settings: callSettings, Action: native.Action})
		}
	}
	return ai.NewImageGenerationCapability(config).Setup(registry)
}

func toolsetIsNil[Deps any](toolset ai.Toolset[Deps]) bool {
	if toolset == nil {
		return true
	}
	value := reflect.ValueOf(toolset)
	return value.Kind() == reflect.Pointer && value.IsNil()
}
