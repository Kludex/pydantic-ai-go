package images

import (
	"context"
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

// NewImageGenerationCapability creates native-first image generation with an optional direct fallback.
func NewImageGenerationCapability[Deps any](config CapabilityConfig[Deps]) *ai.NativeOrLocalTool[Deps] {
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
	capability := ai.ImageGenerationCapabilityConfig[Deps]{Local: config.Local}
	if config.ResolveNative == nil {
		capability.Native = config.Native.CloneNativeTool().(ai.ImageGenerationTool)
		if ratio, ok := nativeAspectRatio(config.Settings.AspectRatio); ok {
			capability.Native.AspectRatio = ratio
		}
	} else {
		capability.ResolveNative = func(
			ctx context.Context, rc *ai.RunContext[Deps],
		) (ai.ImageGenerationTool, error) {
			native, err := config.ResolveNative(ctx, rc)
			if err != nil {
				return ai.ImageGenerationTool{}, err
			}
			if ratio, ok := nativeAspectRatio(config.Settings.AspectRatio); ok {
				native.AspectRatio = ratio
			}
			return native, nil
		}
	}
	if generator != nil {
		settings := config.Settings.Clone()
		capability.LocalForNative = func(native ai.ImageGenerationTool) ai.Tool[Deps] {
			callSettings := settings.Clone()
			if callSettings.Dimensions == nil && callSettings.AspectRatio == "" && native.AspectRatio != "" {
				callSettings.AspectRatio = AspectRatio(native.AspectRatio)
			}
			return NewGenerationTool[Deps](generator, ToolConfig{Settings: callSettings, Action: native.Action})
		}
	}
	return ai.NewImageGenerationCapability(capability)
}

func nativeAspectRatio(ratio AspectRatio) (ai.ImageAspectRatio, bool) {
	switch ratio {
	case AspectRatio21To9, AspectRatio16To9, AspectRatio4To3, AspectRatio3To2, AspectRatio1To1,
		AspectRatio9To16, AspectRatio3To4, AspectRatio2To3, AspectRatio5To4, AspectRatio4To5:
		return ai.ImageAspectRatio(ratio), true
	default:
		return "", false
	}
}

func toolsetIsNil[Deps any](toolset ai.Toolset[Deps]) bool {
	if toolset == nil {
		return true
	}
	value := reflect.ValueOf(toolset)
	return value.Kind() == reflect.Pointer && value.IsNil()
}
