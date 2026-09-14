package google

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/images"
)

type generateResponse struct {
	ResponseID   string `json:"responseId"`
	ModelVersion string `json:"modelVersion"`
	CreateTime   string `json:"createTime"`
	Candidates   []struct {
		FinishReason  string           `json:"finishReason"`
		SafetyRatings []map[string]any `json:"safetyRatings"`
		Content       *struct {
			Parts []struct {
				Thought          bool   `json:"thought"`
				ThoughtSignature string `json:"thoughtSignature"`
				InlineData       *struct {
					MIMEType string `json:"mimeType"`
					Data     string `json:"data"`
				} `json:"inlineData"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	PromptFeedback *struct {
		BlockReason        string           `json:"blockReason"`
		BlockReasonMessage string           `json:"blockReasonMessage"`
		SafetyRatings      []map[string]any `json:"safetyRatings"`
	} `json:"promptFeedback"`
	UsageMetadata *usageMetadata `json:"usageMetadata"`
}

type usageMetadata struct {
	PromptTokens              int           `json:"promptTokenCount"`
	CandidateTokens           int           `json:"candidatesTokenCount"`
	ThoughtsTokens            int           `json:"thoughtsTokenCount"`
	CachedTokens              int           `json:"cachedContentTokenCount"`
	ToolUsePromptTokens       int           `json:"toolUsePromptTokenCount"`
	TotalTokens               int           `json:"totalTokenCount"`
	TrafficType               string        `json:"trafficType"`
	PromptTokenDetails        []tokenDetail `json:"promptTokensDetails"`
	CacheTokenDetails         []tokenDetail `json:"cacheTokensDetails"`
	CandidateTokenDetails     []tokenDetail `json:"candidatesTokensDetails"`
	ToolUsePromptTokenDetails []tokenDetail `json:"toolUsePromptTokensDetails"`
}

type tokenDetail struct {
	Modality   string `json:"modality"`
	TokenCount int    `json:"tokenCount"`
}

var filteredReasons = map[string]bool{
	"SAFETY": true, "RECITATION": true, "BLOCKLIST": true, "PROHIBITED_CONTENT": true,
	"SPII": true, "IMAGE_SAFETY": true, "IMAGE_PROHIBITED_CONTENT": true,
	"IMAGE_RECITATION": true, "MODEL_ARMOR": true,
}

func (model *Model) parseResponse(prompt string, body []byte) (*images.Result, error) {
	var response generateResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("google images: decode response: %w", err)
	}
	generated := []images.GeneratedImage{}
	for _, candidate := range response.Candidates {
		if candidate.Content == nil {
			continue
		}
		for _, item := range candidate.Content.Parts {
			if item.Thought || item.InlineData == nil || item.InlineData.Data == "" {
				continue
			}
			data, err := base64.StdEncoding.Strict().DecodeString(item.InlineData.Data)
			if err != nil {
				return nil, &ai.UnexpectedModelBehaviorError{Message: "Google image response contained invalid base64 data"}
			}
			mediaType := item.InlineData.MIMEType
			if mediaType == "" {
				mediaType = images.MediaTypeFromBytes(data)
			}
			if mediaType == "" {
				mediaType = "image/png"
			}
			details := map[string]any(nil)
			if item.ThoughtSignature != "" {
				details = map[string]any{"has_thought_signature": true}
			}
			generated = append(generated, images.GeneratedImage{
				Content:      ai.BinaryContent{Data: data, MediaType: mediaType},
				OutputFormat: images.OutputFormat(mediaType), ProviderDetails: details,
			})
		}
	}
	providerDetails := googleProviderDetails(response)
	if len(generated) == 0 {
		reason, _ := providerDetails["finish_reason"].(string)
		block, _ := providerDetails["block_reason"].(string)
		if block != "" || filteredReasons[reason] {
			return nil, &ai.ContentFilterError{Message: "Google image generation was blocked for content moderation"}
		}
		message := "Google image response contained no images"
		if reason != "" {
			message += " (finish_reason: " + reason + ")"
		}
		return nil, &ai.UnexpectedModelBehaviorError{Message: message}
	}
	modelName := response.ModelVersion
	if modelName == "" {
		modelName = model.name
	}
	return &images.Result{
		Images: generated, Prompt: prompt, ModelName: modelName, ProviderName: model.providerName,
		ProviderURL: model.baseURL, ProviderResponseID: response.ResponseID, Timestamp: time.Now().UTC(),
		Usage: googleUsage(response.UsageMetadata), ProviderDetails: providerDetails,
	}, nil
}

func googleUsage(metadata *usageMetadata) ai.Usage {
	usage := ai.Usage{Requests: 1}
	if metadata == nil {
		return usage
	}
	usage.InputTokens = metadata.PromptTokens + metadata.ToolUsePromptTokens
	usage.OutputTokens = metadata.CandidateTokens + metadata.ThoughtsTokens
	usage.ReasoningTokens = metadata.ThoughtsTokens
	usage.CacheReadTokens = metadata.CachedTokens
	usage.Details = map[string]int{}
	if metadata.CachedTokens != 0 {
		usage.Details["cached_content_tokens"] = metadata.CachedTokens
	}
	if metadata.ThoughtsTokens != 0 {
		usage.Details["thoughts_tokens"] = metadata.ThoughtsTokens
	}
	if metadata.ToolUsePromptTokens != 0 {
		usage.Details["tool_use_prompt_tokens"] = metadata.ToolUsePromptTokens
	}
	addTokenDetails(usage.Details, metadata.PromptTokenDetails, "prompt")
	addTokenDetails(usage.Details, metadata.CacheTokenDetails, "cache")
	addTokenDetails(usage.Details, metadata.CandidateTokenDetails, "candidates")
	addTokenDetails(usage.Details, metadata.ToolUsePromptTokenDetails, "tool_use_prompt")
	for _, detail := range metadata.PromptTokenDetails {
		if detail.Modality == "AUDIO" {
			usage.InputAudioTokens += detail.TokenCount
		}
	}
	for _, detail := range metadata.CacheTokenDetails {
		if detail.Modality == "AUDIO" {
			usage.CacheAudioReadTokens += detail.TokenCount
		}
	}
	for _, detail := range metadata.CandidateTokenDetails {
		if detail.Modality == "AUDIO" {
			usage.OutputAudioTokens += detail.TokenCount
		}
	}
	if len(usage.Details) == 0 {
		usage.Details = nil
	}
	return usage
}

func addTokenDetails(details map[string]int, values []tokenDetail, suffix string) {
	for _, detail := range values {
		if detail.Modality != "" && detail.TokenCount != 0 {
			details[strings.ToLower(detail.Modality)+"_"+suffix+"_tokens"] += detail.TokenCount
		}
	}
}

func googleProviderDetails(response generateResponse) map[string]any {
	details := map[string]any{}
	if len(response.Candidates) > 0 {
		candidate := response.Candidates[0]
		if candidate.FinishReason != "" {
			details["finish_reason"] = candidate.FinishReason
		}
		if len(candidate.SafetyRatings) > 0 {
			details["safety_ratings"] = candidate.SafetyRatings
		}
	}
	if response.PromptFeedback != nil {
		if response.PromptFeedback.BlockReason != "" {
			details["block_reason"] = response.PromptFeedback.BlockReason
		}
		if response.PromptFeedback.BlockReasonMessage != "" {
			details["block_reason_message"] = response.PromptFeedback.BlockReasonMessage
		}
		if len(response.PromptFeedback.SafetyRatings) > 0 {
			details["safety_ratings"] = response.PromptFeedback.SafetyRatings
		}
	}
	if response.CreateTime != "" {
		details["timestamp"] = response.CreateTime
	}
	if response.UsageMetadata != nil && response.UsageMetadata.TrafficType != "" {
		details["traffic_type"] = response.UsageMetadata.TrafficType
	}
	if len(details) == 0 {
		return nil
	}
	return details
}
