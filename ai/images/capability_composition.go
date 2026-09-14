package images

import (
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// CombineCapabilities merges repeated declarations in registration order.
func (*ImageGenerationCapability[Deps]) CombineCapabilities(capabilities []ai.Capability) (ai.Capability, error) {
	if len(capabilities) == 0 {
		return nil, fmt.Errorf("images: cannot combine an empty image-generation capability collection")
	}
	combined := &ImageGenerationCapability[Deps]{}
	for _, capability := range capabilities {
		value, ok := capability.(*ImageGenerationCapability[Deps])
		if !ok || value == nil {
			return nil, fmt.Errorf("images: image-generation capability has incompatible type %T", capability)
		}
		for _, declaration := range value.native {
			combined.native = append(combined.native, imageNativeDeclaration[Deps]{
				native: declaration.native.CloneNativeTool().(ai.ImageGenerationTool), resolve: declaration.resolve,
			})
		}
		combined.settings = MergeSettings(combined.settings, value.settings)
		if value.fallback.stated {
			combined.fallback = value.fallback
		}
	}
	if err := combined.settings.Validate(); err != nil {
		return nil, err
	}
	return combined, nil
}

func (capability *ImageGenerationCapability[Deps]) hasDynamicNative() bool {
	for _, declaration := range capability.native {
		if declaration.resolve != nil {
			return true
		}
	}
	return false
}

func mergeNativeImageSettings(base, override ai.ImageGenerationTool) ai.ImageGenerationTool {
	if override.Action != "" {
		base.Action = override.Action
	}
	if override.Background != "" {
		base.Background = override.Background
	}
	if override.InputFidelity != "" {
		base.InputFidelity = override.InputFidelity
	}
	if override.Moderation != "" {
		base.Moderation = override.Moderation
	}
	if override.Model != "" {
		base.Model = override.Model
	}
	if override.OutputCompression != nil {
		compression := *override.OutputCompression
		base.OutputCompression = &compression
	}
	if override.OutputFormat != "" {
		base.OutputFormat = override.OutputFormat
	}
	if override.PartialImages != 0 {
		base.PartialImages = override.PartialImages
	}
	if override.Quality != "" {
		base.Quality = override.Quality
	}
	if override.Size != "" {
		base.Size = override.Size
	}
	if override.AspectRatio != "" {
		base.AspectRatio = override.AspectRatio
	}
	if override.Optional {
		base.Optional = true
	}
	return base
}

func applyNativeImageSettings(native ai.ImageGenerationTool, settings Settings) ai.ImageGenerationTool {
	if ratio, ok := nativeAspectRatio(settings.AspectRatio); ok {
		native.AspectRatio = ratio
	}
	return native
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
