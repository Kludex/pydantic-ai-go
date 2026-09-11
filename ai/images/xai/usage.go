package xai

import ai "github.com/Kludex/pydantic-ai-go/ai"

type usageResponse struct {
	PromptTokens           int      `json:"prompt_tokens"`
	CompletionTokens       int      `json:"completion_tokens"`
	ReasoningTokens        int      `json:"reasoning_tokens"`
	PromptTextTokens       int      `json:"prompt_text_tokens"`
	PromptImageTokens      int      `json:"prompt_image_tokens"`
	CachedPromptTextTokens int      `json:"cached_prompt_text_tokens"`
	CostUSD                *float64 `json:"cost_usd"`
	CostTicks              *int     `json:"cost_in_usd_ticks"`
}

func xaiUsage(usage *usageResponse) ai.Usage {
	mapped := ai.Usage{Requests: 1}
	if usage == nil {
		return mapped
	}
	mapped.InputTokens = usage.PromptTokens
	mapped.OutputTokens = usage.CompletionTokens
	mapped.ReasoningTokens = usage.ReasoningTokens
	mapped.CacheReadTokens = usage.CachedPromptTextTokens
	details := map[string]int{}
	if usage.PromptTextTokens != 0 {
		details["input_text_tokens"] = usage.PromptTextTokens
	}
	if usage.PromptImageTokens != 0 {
		details["input_image_tokens"] = usage.PromptImageTokens
	}
	if len(details) > 0 {
		mapped.Details = details
	}
	return mapped
}
