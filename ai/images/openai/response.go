package openai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/images"
)

type imageResponse struct {
	ID         string `json:"id"`
	Created    int64  `json:"created"`
	Model      string `json:"model"`
	Size       string `json:"size"`
	Quality    string `json:"quality"`
	Background string `json:"background"`
	Data       []struct {
		Base64        string         `json:"b64_json"`
		RevisedPrompt string         `json:"revised_prompt"`
		Details       map[string]any `json:"provider_details"`
	} `json:"data"`
	Usage *struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
		TotalTokens  int `json:"total_tokens"`
		InputDetails *struct {
			TextTokens  int `json:"text_tokens"`
			ImageTokens int `json:"image_tokens"`
		} `json:"input_tokens_details"`
		OutputDetails *struct {
			TextTokens  int `json:"text_tokens"`
			ImageTokens int `json:"image_tokens"`
		} `json:"output_tokens_details"`
	} `json:"usage"`
}

func (model *Model) parseResponse(prompt string, body []byte) (*images.Result, error) {
	var response imageResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("openai images: decode response: %w", err)
	}
	if len(response.Data) == 0 {
		return nil, &ai.UnexpectedModelBehaviorError{Message: "OpenAI image response contained no images"}
	}
	generated := make([]images.GeneratedImage, len(response.Data))
	for index, item := range response.Data {
		if item.Base64 == "" {
			return nil, &ai.UnexpectedModelBehaviorError{Message: "OpenAI image response omitted base64 image data"}
		}
		data, err := base64.StdEncoding.Strict().DecodeString(item.Base64)
		if err != nil {
			return nil, &ai.UnexpectedModelBehaviorError{Message: "OpenAI image response contained invalid base64 data"}
		}
		mediaType := images.MediaTypeFromBytes(data)
		if mediaType == "" {
			return nil, &ai.UnexpectedModelBehaviorError{Message: "OpenAI image response contained an unrecognized image format"}
		}
		generated[index] = images.GeneratedImage{
			Content: ai.BinaryContent{Data: data, MediaType: mediaType}, RevisedPrompt: item.RevisedPrompt,
			OutputFormat: images.OutputFormat(mediaType), ProviderDetails: item.Details,
		}
	}
	modelName := response.Model
	if modelName == "" {
		modelName = model.name
	}
	providerDetails := map[string]any{}
	if response.Created != 0 {
		providerDetails["created"] = response.Created
	}
	if response.Size != "" {
		providerDetails["size"] = response.Size
	}
	if response.Quality != "" {
		providerDetails["quality"] = response.Quality
	}
	if response.Background != "" {
		providerDetails["background"] = response.Background
	}
	if len(providerDetails) == 0 {
		providerDetails = nil
	}
	return &images.Result{
		Images: generated, Prompt: prompt, ModelName: modelName, ProviderName: model.providerName,
		ProviderURL: model.baseURL, ProviderResponseID: response.ID, Timestamp: time.Now().UTC(),
		Usage: mapUsage(response.Usage), ProviderDetails: providerDetails,
	}, nil
}

func mapUsage(usage *struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
	InputDetails *struct {
		TextTokens  int `json:"text_tokens"`
		ImageTokens int `json:"image_tokens"`
	} `json:"input_tokens_details"`
	OutputDetails *struct {
		TextTokens  int `json:"text_tokens"`
		ImageTokens int `json:"image_tokens"`
	} `json:"output_tokens_details"`
}) ai.Usage {
	mapped := ai.Usage{Requests: 1}
	if usage == nil {
		return mapped
	}
	mapped.InputTokens = usage.InputTokens
	mapped.OutputTokens = usage.OutputTokens
	details := map[string]int{}
	if usage.InputDetails != nil {
		details["input_text_tokens"] = usage.InputDetails.TextTokens
		details["input_image_tokens"] = usage.InputDetails.ImageTokens
	}
	if usage.OutputDetails != nil {
		details["output_text_tokens"] = usage.OutputDetails.TextTokens
		details["output_image_tokens"] = usage.OutputDetails.ImageTokens
	}
	if len(details) > 0 {
		mapped.Details = details
	}
	return mapped
}
