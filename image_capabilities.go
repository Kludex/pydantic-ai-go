package ai

import "context"

// ImageGenerationCapabilityConfig configures native image generation and an optional local fallback.
type ImageGenerationCapabilityConfig[Deps any] struct {
	Native ImageGenerationTool
	Local  Toolset[Deps]
}

// NewImageGenerationCapability creates native-first image generation.
// A nil Local requires native support.
func NewImageGenerationCapability[Deps any](
	config ImageGenerationCapabilityConfig[Deps],
) *NativeOrLocalTool[Deps] {
	var options []NativeOrLocalOption
	if toolsetIsNil(config.Local) {
		options = nativeRequirementOption("no local image-generation fallback was configured")
	}
	return NewNativeOrLocalToolset(config.Native, config.Local, options...)
}

// ImageGenerationFunc resolves native image-generation settings before a model request.
// It may run concurrently and must not return shared mutable state.
type ImageGenerationFunc[Deps any] func(
	ctx context.Context, rc *RunContext[Deps],
) (ImageGenerationTool, error)

// NewDynamicImageGenerationCapability creates dependency-aware native-first image generation.
func NewDynamicImageGenerationCapability[Deps any](
	resolve ImageGenerationFunc[Deps], local Toolset[Deps], options ...NativeOrLocalOption,
) *NativeOrLocalTool[Deps] {
	if resolve == nil {
		panic("ai: dynamic image-generation resolver must not be nil")
	}
	options = nativeRequiredWithoutLocal(local, "no local image-generation fallback was configured", options)
	return NewDynamicNativeOrLocalToolset(
		"image_generation",
		func(ctx context.Context, rc *RunContext[Deps]) (NativeTool, error) {
			return resolve(ctx, rc)
		},
		local,
		options...,
	)
}
