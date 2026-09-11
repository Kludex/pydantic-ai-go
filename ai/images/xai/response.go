package xai

import (
	"encoding/base64"
	"fmt"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/images"
	"github.com/Kludex/pydantic-ai-go/ai/images/xai/internal/xaiapi"
)

func (model *Model) parseResponse(prompt string, response *xaiapi.ImageResponse, count int) (*images.Result, error) {
	if response == nil || len(response.Images) == 0 {
		return nil, &ai.UnexpectedModelBehaviorError{Message: "xAI image generation response contained no images"}
	}
	if len(response.Images) < count {
		return nil, &ai.UnexpectedModelBehaviorError{Message: fmt.Sprintf(
			"xAI image generation response contained %d images, expected %d", len(response.Images), count,
		)}
	}
	items := response.Images[:count]
	generated := make([]images.GeneratedImage, 0, len(items))
	moderated := []int{}
	for index, item := range items {
		if !item.RespectModeration {
			moderated = append(moderated, index)
			continue
		}
		content, err := decodeImage(item.GetBase64())
		if err != nil {
			return nil, &ai.UnexpectedModelBehaviorError{Message: "xAI image generation response contained invalid base64 image data"}
		}
		generated = append(generated, images.GeneratedImage{
			Content: content, OutputFormat: images.OutputFormat(content.MediaType),
			ProviderDetails: map[string]any{"respect_moderation": true},
		})
	}
	if len(generated) == 0 {
		return nil, &ai.ContentFilterError{Message: "xAI flagged all generated images for content moderation"}
	}
	providerDetails := map[string]any{}
	if len(moderated) > 0 {
		providerDetails["moderated_image_indices"] = moderated
	}
	if response.Usage != nil && response.Usage.CostInUsdTicks != nil {
		providerDetails["cost_in_usd_ticks"] = *response.Usage.CostInUsdTicks
		providerDetails["cost_usd"] = float64(*response.Usage.CostInUsdTicks) / 10_000_000_000
	}
	if len(providerDetails) == 0 {
		providerDetails = nil
	}
	modelName := response.Model
	if modelName == "" {
		modelName = model.name
	}
	return &images.Result{
		Images: generated, Prompt: prompt, ModelName: modelName, ProviderName: model.ProviderName(),
		ProviderURL: model.providerURL, Timestamp: time.Now().UTC(), Usage: xaiUsage(response.Usage),
		ProviderDetails: providerDetails,
	}, nil
}

func decodeImage(value string) (ai.BinaryContent, error) {
	header, encoded, ok := strings.Cut(value, ",")
	if !ok || !strings.HasPrefix(header, "data:image/") || !strings.HasSuffix(header, ";base64") {
		return ai.BinaryContent{}, fmt.Errorf("invalid image data URL")
	}
	mediaType := strings.TrimSuffix(strings.TrimPrefix(header, "data:"), ";base64")
	data, err := base64.StdEncoding.Strict().DecodeString(encoded)
	if err != nil || len(data) == 0 {
		return ai.BinaryContent{}, fmt.Errorf("invalid base64 image")
	}
	return ai.BinaryContent{Data: data, MediaType: mediaType}, nil
}
