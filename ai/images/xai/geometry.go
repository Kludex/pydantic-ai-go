package xai

import (
	"fmt"

	"github.com/Kludex/pydantic-ai-go/ai/images"
)

var geometries = map[images.AspectRatio]map[Resolution]images.Dimensions{
	images.AspectRatio1To1:    {Resolution1K: {Width: 1024, Height: 1024}, Resolution2K: {Width: 2048, Height: 2048}},
	images.AspectRatio3To4:    {Resolution1K: {Width: 864, Height: 1152}, Resolution2K: {Width: 1776, Height: 2368}},
	images.AspectRatio4To3:    {Resolution1K: {Width: 1152, Height: 864}, Resolution2K: {Width: 2368, Height: 1776}},
	images.AspectRatio9To16:   {Resolution1K: {Width: 720, Height: 1280}, Resolution2K: {Width: 1584, Height: 2816}},
	images.AspectRatio16To9:   {Resolution1K: {Width: 1280, Height: 720}, Resolution2K: {Width: 2816, Height: 1584}},
	images.AspectRatio2To3:    {Resolution1K: {Width: 832, Height: 1248}, Resolution2K: {Width: 1664, Height: 2496}},
	images.AspectRatio3To2:    {Resolution1K: {Width: 1248, Height: 832}, Resolution2K: {Width: 2496, Height: 1664}},
	images.AspectRatio9To19_5: {Resolution1K: {Width: 576, Height: 1248}, Resolution2K: {Width: 1344, Height: 2912}},
	images.AspectRatio19_5To9: {Resolution1K: {Width: 1248, Height: 576}, Resolution2K: {Width: 2912, Height: 1344}},
	images.AspectRatio9To20:   {Resolution1K: {Width: 576, Height: 1280}, Resolution2K: {Width: 1440, Height: 3200}},
	images.AspectRatio20To9:   {Resolution1K: {Width: 1280, Height: 576}, Resolution2K: {Width: 3200, Height: 1440}},
	images.AspectRatio1To2:    {Resolution1K: {Width: 704, Height: 1408}, Resolution2K: {Width: 1456, Height: 2912}},
	images.AspectRatio2To1:    {Resolution1K: {Width: 1408, Height: 704}, Resolution2K: {Width: 2912, Height: 1456}},
}

var geometryModels = map[string]bool{
	"grok-imagine-image": true, "grok-imagine-image-2026-03-02": true,
	"grok-imagine-image-quality": true, "grok-imagine-image-quality-20260403": true,
	"grok-imagine-image-quality-latest": true, "grok-imagine-image-pro": true,
}

func resolveGeometry(name string, settings images.Settings, provider providerSettings) (
	images.AspectRatio, Resolution, []string, error,
) {
	conflicts := []string{}
	if settings.Dimensions != nil {
		ratio, resolution, err := dimensionsGeometry(name, *settings.Dimensions)
		if err != nil {
			return "", "", nil, err
		}
		if provider.aspectRatio != "" {
			if provider.aspectRatio != ratio {
				conflicts = append(conflicts, "dimensions")
			}
			ratio = provider.aspectRatio
		}
		if provider.resolution != "" {
			if provider.resolution != resolution {
				conflicts = append(conflicts, "dimensions")
			}
			resolution = provider.resolution
		}
		return ratio, resolution, unique(conflicts), nil
	}
	ratio := provider.aspectRatio
	if ratio == "" && settings.AspectRatio != "" {
		if _, ok := geometries[settings.AspectRatio]; !ok {
			return "", "", nil, fmt.Errorf(
				"xai images: aspect ratio %q cannot be represented by the xAI API", settings.AspectRatio,
			)
		}
		ratio = settings.AspectRatio
	} else if ratio != "" && settings.AspectRatio != "" && ratio != settings.AspectRatio {
		conflicts = append(conflicts, "aspect_ratio")
	}
	resolution := provider.resolution
	if resolution == "" && settings.AspectRatio != "" {
		resolution = Resolution1K
	}
	return ratio, resolution, conflicts, nil
}

func dimensionsGeometry(name string, dimensions images.Dimensions) (images.AspectRatio, Resolution, error) {
	if !geometryModels[name] {
		return "", "", fmt.Errorf("xai images: model %q has no known exact-dimensions mapping", name)
	}
	for ratio, resolutions := range geometries {
		for resolution, supported := range resolutions {
			if dimensions == supported {
				return ratio, resolution, nil
			}
		}
	}
	return "", "", fmt.Errorf("xai images: model %q does not support dimensions %+v", name, dimensions)
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
