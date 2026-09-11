package xai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/images"
)

type imageItem struct {
	Base64            string         `json:"base64"`
	Base64JSON        string         `json:"b64_json"`
	Image             string         `json:"image"`
	Model             string         `json:"model"`
	RespectModeration *bool          `json:"respect_moderation"`
	ProviderDetails   map[string]any `json:"provider_details"`
}

type imageResponse struct {
	ID      string         `json:"id"`
	Model   string         `json:"model"`
	Data    []imageItem    `json:"data"`
	Images  []imageItem    `json:"images"`
	CostUSD *float64       `json:"cost_usd"`
	Usage   *usageResponse `json:"usage"`
}

func (model *Model) parseResponse(prompt string, body []byte) (*images.Result, error) {
	var response imageResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("xai images: decode response: %w", err)
	}
	items := response.Data
	if items == nil {
		items = response.Images
	}
	if len(items) == 0 {
		return nil, &ai.UnexpectedModelBehaviorError{Message: "xAI image response contained no images"}
	}
	generated := []images.GeneratedImage{}
	moderated := []int{}
	for index, item := range items {
		if item.RespectModeration != nil && !*item.RespectModeration {
			moderated = append(moderated, index)
			continue
		}
		encoded := item.Base64
		if encoded == "" {
			encoded = item.Base64JSON
		}
		if encoded == "" {
			encoded = item.Image
		}
		content, err := decodeImage(encoded)
		if err != nil {
			return nil, &ai.UnexpectedModelBehaviorError{Message: "xAI image response contained invalid base64 image data"}
		}
		details := item.ProviderDetails
		if details == nil {
			details = map[string]any{}
		}
		if item.RespectModeration != nil {
			details["respect_moderation"] = *item.RespectModeration
		}
		if len(details) == 0 {
			details = nil
		}
		generated = append(generated, images.GeneratedImage{
			Content: content, OutputFormat: images.OutputFormat(content.MediaType), ProviderDetails: details,
		})
	}
	if len(generated) == 0 {
		return nil, &ai.ContentFilterError{Message: "xAI flagged all generated images for content moderation"}
	}
	providerDetails := map[string]any{}
	if len(moderated) > 0 {
		providerDetails["moderated_image_indices"] = moderated
	}
	if response.Usage != nil && response.Usage.CostTicks != nil {
		providerDetails["cost_in_usd_ticks"] = *response.Usage.CostTicks
	}
	cost := response.CostUSD
	if cost == nil && response.Usage != nil {
		cost = response.Usage.CostUSD
	}
	if cost != nil {
		providerDetails["cost_usd"] = *cost
	}
	if len(providerDetails) == 0 {
		providerDetails = nil
	}
	modelName := response.Model
	if modelName == "" {
		for _, item := range items {
			if item.Model != "" {
				modelName = item.Model
				break
			}
		}
	}
	if modelName == "" {
		modelName = model.name
	}
	return &images.Result{
		Images: generated, Prompt: prompt, ModelName: modelName, ProviderName: model.providerName,
		ProviderURL: model.baseURL, ProviderResponseID: response.ID, Timestamp: time.Now().UTC(),
		Usage: xaiUsage(response.Usage), ProviderDetails: providerDetails,
	}, nil
}

func decodeImage(value string) (ai.BinaryContent, error) {
	mediaType := ""
	encoded := value
	if strings.HasPrefix(value, "data:") {
		header, payload, ok := strings.Cut(value, ",")
		if !ok || !strings.HasPrefix(header, "data:image/") || !strings.HasSuffix(header, ";base64") {
			return ai.BinaryContent{}, fmt.Errorf("invalid image data URL")
		}
		mediaType = strings.TrimSuffix(strings.TrimPrefix(header, "data:"), ";base64")
		encoded = payload
	}
	data, err := base64.StdEncoding.Strict().DecodeString(encoded)
	if err != nil || len(data) == 0 {
		return ai.BinaryContent{}, fmt.Errorf("invalid base64 image")
	}
	if mediaType == "" {
		mediaType = images.MediaTypeFromBytes(data)
	}
	if mediaType == "" {
		return ai.BinaryContent{}, fmt.Errorf("unrecognized image format")
	}
	return ai.BinaryContent{Data: data, MediaType: mediaType}, nil
}
