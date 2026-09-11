package ai

import "context"

// ImageGenerationCapabilityConfig configures native image generation and an optional local fallback.
type ImageGenerationCapabilityConfig[Deps any] struct {
	// Native configures provider-hosted image generation.
	Native ImageGenerationTool
	// ResolveNative configures provider-hosted image generation before each model request.
	// When set, it replaces Native.
	ResolveNative ImageGenerationFunc[Deps]
	// Local is the lifecycle-aware fallback toolset.
	Local Toolset[Deps]
	// LocalForNative derives a local fallback from the native settings resolved for the same request.
	LocalForNative ImageGenerationLocalFunc[Deps]
}

// ImageGenerationLocalFunc derives one local fallback tool from the native settings resolved for a request.
type ImageGenerationLocalFunc[Deps any] func(ImageGenerationTool) Tool[Deps]

// NewImageGenerationCapability creates native-first image generation.
// A nil local fallback requires native support.
func NewImageGenerationCapability[Deps any](
	config ImageGenerationCapabilityConfig[Deps],
) *NativeOrLocalTool[Deps] {
	if !toolsetIsNil(config.Local) && config.LocalForNative != nil {
		panic("ai: image-generation capability accepts either Local or LocalForNative, not both")
	}
	var options []NativeOrLocalOption
	if toolsetIsNil(config.Local) && config.LocalForNative == nil {
		options = nativeFallbackRequirementOption("no local image-generation fallback was configured")
	}
	var factory func(NativeTool) Toolset[Deps]
	if config.LocalForNative != nil {
		factory = func(tool NativeTool) Toolset[Deps] {
			return NewFunctionToolset(config.LocalForNative(tool.(ImageGenerationTool)))
		}
	}
	if config.ResolveNative == nil {
		capability := NewNativeOrLocalToolset(config.Native, config.Local, options...)
		capability.registration.localFactory = factory
		return capability
	}
	return newDynamicNativeOrLocalToolset(
		"image_generation",
		func(ctx context.Context, rc *RunContext[Deps]) (NativeTool, error) {
			return config.ResolveNative(ctx, rc)
		},
		config.Local,
		factory,
		options...,
	)
}

// NewImageGenerationCapabilityWithFallback creates native-first image generation
// with a generate_image tool backed by an image-output subagent.
func NewImageGenerationCapabilityWithFallback[Deps any](
	config ImageGenerationSubagentConfig[Deps],
) *NativeOrLocalTool[Deps] {
	capability := NewImageGenerationCapability(ImageGenerationCapabilityConfig[Deps]{
		Native: config.Native,
		Local:  NewFunctionToolset(NewImageGenerationSubagentTool(config)),
	})
	capability.registration.rebuildLocal = func(native NativeTool) Toolset[Deps] {
		updated := config
		updated.Native = native.(ImageGenerationTool)
		return NewFunctionToolset(NewImageGenerationSubagentTool(updated))
	}
	return capability
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

// NewDynamicImageGenerationCapabilityWithFallback creates dynamic native-first
// image generation with a subagent that resolves the same settings when used.
func NewDynamicImageGenerationCapabilityWithFallback[Deps any](
	resolve ImageGenerationFunc[Deps], config ImageGenerationSubagentConfig[Deps], options ...NativeOrLocalOption,
) *NativeOrLocalTool[Deps] {
	if resolve == nil {
		panic("ai: dynamic image-generation resolver must not be nil")
	}
	config.ResolveNative = resolve
	return NewDynamicImageGenerationCapability(
		resolve, NewFunctionToolset(NewImageGenerationSubagentTool(config)), options...,
	)
}
