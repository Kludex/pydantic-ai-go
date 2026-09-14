package openai

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/Kludex/pydantic-ai-go/ai/images"
)

var legacyAspectSizes = map[images.AspectRatio]string{
	images.AspectRatio1To1: "1024x1024", images.AspectRatio2To3: "1024x1536", images.AspectRatio3To2: "1536x1024",
}

var image2AspectSizes = map[images.AspectRatio]string{
	images.AspectRatio1To1: "1024x1024", images.AspectRatio1To2: "704x1408",
	images.AspectRatio2To1: "1408x704", images.AspectRatio2To3: "832x1248",
	images.AspectRatio3To2: "1248x832", images.AspectRatio3To4: "864x1152",
	images.AspectRatio4To3: "1152x864", images.AspectRatio4To5: "896x1120",
	images.AspectRatio5To4: "1120x896", images.AspectRatio9To16: "720x1280",
	images.AspectRatio9To19_5: "672x1456", images.AspectRatio9To20: "720x1600",
	images.AspectRatio16To9: "1280x720", images.AspectRatio19_5To9: "1456x672",
	images.AspectRatio20To9: "1600x720", images.AspectRatio21To9: "1568x672",
}

func resolveGeometry(name string, settings images.Settings, providerSize *string) (string, []string, error) {
	var mappedDimensions string
	if settings.Dimensions != nil {
		var err error
		mappedDimensions, err = resolveDimensions(name, *settings.Dimensions)
		if err != nil {
			return "", nil, err
		}
	}
	if providerSize != nil {
		conflicts := []string{}
		if settings.Dimensions != nil && *providerSize != mappedDimensions {
			conflicts = append(conflicts, "dimensions")
		}
		if settings.AspectRatio != "" && !matchesRatio(*providerSize, settings.AspectRatio) {
			conflicts = append(conflicts, "aspect_ratio")
		}
		return *providerSize, conflicts, nil
	}
	if mappedDimensions != "" {
		return mappedDimensions, nil, nil
	}
	if settings.AspectRatio != "" {
		mapped, err := resolveAspectRatio(name, settings.AspectRatio)
		return mapped, nil, err
	}
	return "", nil, nil
}

func resolveDimensions(name string, dimensions images.Dimensions) (string, error) {
	size := fmt.Sprintf("%dx%d", dimensions.Width, dimensions.Height)
	if isImage2(name) {
		problems := []string{}
		if dimensions.Width%16 != 0 || dimensions.Height%16 != 0 {
			problems = append(problems, "width and height must be multiples of 16")
		}
		if max(dimensions.Width, dimensions.Height) > 3840 {
			problems = append(problems, "the longest edge must be at most 3840 pixels")
		}
		if max(dimensions.Width, dimensions.Height) > 3*min(dimensions.Width, dimensions.Height) {
			problems = append(problems, "the aspect ratio must not exceed 3:1")
		}
		pixels := dimensions.Width * dimensions.Height
		if pixels < 655360 || pixels > 8294400 {
			problems = append(problems, "the total pixel count must be between 655360 and 8294400")
		}
		if len(problems) > 0 {
			return "", fmt.Errorf(
				"openai images: model %q does not support dimensions %+v: %s",
				name, dimensions, strings.Join(problems, "; "),
			)
		}
	}
	if isImage1(name) && size != "1024x1024" && size != "1024x1536" && size != "1536x1024" {
		return "", fmt.Errorf("openai images: model %q does not support dimensions %+v", name, dimensions)
	}
	return size, nil
}

func resolveAspectRatio(name string, ratio images.AspectRatio) (string, error) {
	mapping := legacyAspectSizes
	if isImage2(name) {
		mapping = image2AspectSizes
	} else if !isImage1(name) {
		return "", fmt.Errorf("openai images: no aspect ratio mapping is known for model %q", name)
	}
	if size := mapping[ratio]; size != "" {
		return size, nil
	}
	return "", fmt.Errorf("openai images: model %q does not support aspect ratio %q", name, ratio)
}

func isImage1(name string) bool {
	return name == "gpt-image-1" || strings.HasPrefix(name, "gpt-image-1-") ||
		name == "gpt-image-1.5" || strings.HasPrefix(name, "gpt-image-1.5-")
}

func isImage2(name string) bool {
	return name == "gpt-image-2" || strings.HasPrefix(name, "gpt-image-2-")
}

func matchesRatio(size string, ratio images.AspectRatio) bool {
	parts := strings.Split(size, "x")
	ratioParts := strings.Split(string(ratio), ":")
	if len(parts) != 2 || len(ratioParts) != 2 {
		return false
	}
	width, widthErr := strconv.Atoi(parts[0])
	height, heightErr := strconv.Atoi(parts[1])
	ratioWidth, ratioWidthErr := strconv.ParseFloat(ratioParts[0], 64)
	ratioHeight, ratioHeightErr := strconv.ParseFloat(ratioParts[1], 64)
	if widthErr != nil || heightErr != nil || ratioWidthErr != nil || ratioHeightErr != nil || width <= 0 || height <= 0 {
		return false
	}
	return float64(width)*ratioHeight == float64(height)*ratioWidth
}
