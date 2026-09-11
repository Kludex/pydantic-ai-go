package google

import (
	"fmt"
	"strings"

	"github.com/Kludex/pydantic-ai-go/ai/images"
)

type geometryProfile struct {
	dimensions  map[images.AspectRatio]map[ImageSize]images.Dimensions
	defaultSize ImageSize
}

var gemini25Dimensions = map[images.AspectRatio]images.Dimensions{
	images.AspectRatio1To1: {Width: 1024, Height: 1024}, images.AspectRatio2To3: {Width: 832, Height: 1248},
	images.AspectRatio3To2: {Width: 1248, Height: 832}, images.AspectRatio3To4: {Width: 864, Height: 1184},
	images.AspectRatio4To3: {Width: 1184, Height: 864}, images.AspectRatio4To5: {Width: 896, Height: 1152},
	images.AspectRatio5To4: {Width: 1152, Height: 896}, images.AspectRatio9To16: {Width: 768, Height: 1344},
	images.AspectRatio16To9: {Width: 1344, Height: 768}, images.AspectRatio21To9: {Width: 1536, Height: 672},
}

var gemini31Base = map[images.AspectRatio]images.Dimensions{
	images.AspectRatio1To1: {Width: 512, Height: 512}, images.AspectRatio2To3: {Width: 424, Height: 632},
	images.AspectRatio3To2: {Width: 632, Height: 424}, images.AspectRatio3To4: {Width: 448, Height: 600},
	images.AspectRatio4To3: {Width: 600, Height: 448}, images.AspectRatio4To5: {Width: 464, Height: 576},
	images.AspectRatio5To4: {Width: 576, Height: 464}, images.AspectRatio9To16: {Width: 384, Height: 688},
	images.AspectRatio16To9: {Width: 688, Height: 384}, images.AspectRatio21To9: {Width: 784, Height: 336},
}

var extendedDimensions = map[images.AspectRatio]map[ImageSize]images.Dimensions{
	images.AspectRatio21To9: {
		ImageSize512: {Width: 784, Height: 336}, ImageSize1K: {Width: 1584, Height: 672},
		ImageSize2K: {Width: 3168, Height: 1344}, ImageSize4K: {Width: 6336, Height: 2688},
	},
	images.AspectRatio1To4: {
		ImageSize512: {Width: 256, Height: 1024}, ImageSize1K: {Width: 512, Height: 2064},
		ImageSize2K: {Width: 1024, Height: 4128}, ImageSize4K: {Width: 2048, Height: 8256},
	},
	images.AspectRatio1To8: {
		ImageSize512: {Width: 176, Height: 1456}, ImageSize1K: {Width: 352, Height: 2928},
		ImageSize2K: {Width: 704, Height: 5856}, ImageSize4K: {Width: 1408, Height: 11712},
	},
	images.AspectRatio4To1: {
		ImageSize512: {Width: 1024, Height: 256}, ImageSize1K: {Width: 2064, Height: 512},
		ImageSize2K: {Width: 4128, Height: 1024}, ImageSize4K: {Width: 8256, Height: 2048},
	},
	images.AspectRatio8To1: {
		ImageSize512: {Width: 1456, Height: 176}, ImageSize1K: {Width: 2928, Height: 352},
		ImageSize2K: {Width: 5856, Height: 704}, ImageSize4K: {Width: 11712, Height: 1408},
	},
}

func profile(name string) *geometryProfile {
	if strings.Contains(name, "gemini-2.5-flash-image") {
		values := map[images.AspectRatio]map[ImageSize]images.Dimensions{}
		for ratio, dimensions := range gemini25Dimensions {
			values[ratio] = map[ImageSize]images.Dimensions{"": dimensions}
		}
		return &geometryProfile{dimensions: values}
	}
	if !strings.Contains(name, "gemini-3.1-flash-image") &&
		!strings.Contains(name, "gemini-3.1-flash-lite-image") && !strings.Contains(name, "gemini-3-pro-image") {
		return nil
	}
	values := map[images.AspectRatio]map[ImageSize]images.Dimensions{}
	for ratio, base := range gemini31Base {
		values[ratio] = map[ImageSize]images.Dimensions{
			ImageSize512: base, ImageSize1K: scale(base, 2), ImageSize2K: scale(base, 4), ImageSize4K: scale(base, 8),
		}
	}
	for ratio, dimensions := range extendedDimensions {
		values[ratio] = dimensions
	}
	if strings.Contains(name, "gemini-3-pro-image") {
		for ratio, tiers := range values {
			delete(tiers, ImageSize512)
			if ratio == images.AspectRatio1To4 || ratio == images.AspectRatio1To8 ||
				ratio == images.AspectRatio4To1 || ratio == images.AspectRatio8To1 {
				delete(values, ratio)
			}
		}
	} else if strings.Contains(name, "gemini-3.1-flash-lite-image") {
		for ratio, tiers := range values {
			values[ratio] = map[ImageSize]images.Dimensions{ImageSize1K: tiers[ImageSize1K]}
		}
	}
	return &geometryProfile{dimensions: values, defaultSize: ImageSize1K}
}

func resolveGeometry(
	name string, settings images.Settings, provider providerSettings,
) (string, ImageSize, []string, error) {
	aspectRatio, imageSize := provider.aspectRatio, provider.imageSize
	conflicts := []string{}
	if settings.Dimensions != nil {
		mappedRatio, mappedSize, err := dimensionsGeometry(name, *settings.Dimensions)
		if err != nil {
			return "", "", nil, err
		}
		if aspectRatio == "" {
			aspectRatio = string(mappedRatio)
		} else if aspectRatio != string(mappedRatio) {
			conflicts = append(conflicts, "dimensions")
		}
		if imageSize == "" {
			imageSize = mappedSize
		} else if imageSize != mappedSize {
			conflicts = append(conflicts, "dimensions")
		}
	} else if settings.AspectRatio != "" {
		if aspectRatio == "" {
			aspectRatio = string(settings.AspectRatio)
		} else if aspectRatio != string(settings.AspectRatio) {
			conflicts = append(conflicts, "aspect_ratio")
		}
		if imageSize == "" {
			if selected := profile(name); selected != nil {
				imageSize = selected.defaultSize
			}
		}
	}
	return aspectRatio, imageSize, unique(conflicts), nil
}

func dimensionsGeometry(name string, dimensions images.Dimensions) (images.AspectRatio, ImageSize, error) {
	if selected := profile(name); selected != nil {
		for ratio, tiers := range selected.dimensions {
			for size, supported := range tiers {
				if dimensions == supported {
					return ratio, size, nil
				}
			}
		}
	}
	return "", "", fmt.Errorf("google images: model %q does not support dimensions %+v", name, dimensions)
}

func scale(value images.Dimensions, multiplier int) images.Dimensions {
	return images.Dimensions{Width: value.Width * multiplier, Height: value.Height * multiplier}
}

func unique(values []string) []string {
	seen := map[string]bool{}
	result := []string{}
	for _, value := range values {
		if !seen[value] {
			seen[value] = true
			result = append(result, value)
		}
	}
	return result
}
