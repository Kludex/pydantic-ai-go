package xai

import (
	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/images/xai/internal/xaiapi"
)

func xaiUsage(usage *xaiapi.SamplingUsage) ai.Usage {
	mapped := ai.Usage{Requests: 1}
	if usage == nil {
		return mapped
	}
	mapped.InputTokens = int(usage.PromptTokens)
	mapped.OutputTokens = int(usage.CompletionTokens)
	mapped.ReasoningTokens = int(usage.ReasoningTokens)
	mapped.CacheReadTokens = int(usage.CachedPromptTextTokens)
	details := map[string]int{}
	if usage.PromptTextTokens != 0 {
		details["input_text_tokens"] = int(usage.PromptTextTokens)
	}
	if usage.PromptImageTokens != 0 {
		details["input_image_tokens"] = int(usage.PromptImageTokens)
	}
	if len(details) > 0 {
		mapped.Details = details
	}
	return mapped
}
