package ai

import (
	"errors"

	genaiprices "github.com/pydantic/genai-prices/packages/go"
)

// PriceCalculation contains the best matching input, output, and total prices in US dollars.
type PriceCalculation = genaiprices.PriceCalculation

// Price calculates this response's price using the bundled genai-prices snapshot.
// It returns lookup and usage errors to callers that need explicit pricing diagnostics.
func (response ModelResponse) Price() (PriceCalculation, error) {
	request := genaiprices.PriceRequest{
		Usage: usageForPricing(response.Usage), Model: response.ModelName, Timestamp: response.Timestamp,
	}
	if response.ProviderURL != "" {
		request.ProviderAPIURL = response.ProviderURL
		calculation, err := genaiprices.Calculate(request)
		if err == nil {
			return calculation, nil
		}
		if !errors.Is(err, genaiprices.ErrProviderNotFound) && !errors.Is(err, genaiprices.ErrModelNotFound) {
			return PriceCalculation{}, err
		}
	}
	request.ProviderAPIURL = ""
	request.ProviderID = response.ProviderName
	return genaiprices.Calculate(request)
}

func fillResponseCost(response *ModelResponse) {
	if response == nil || response.Usage.CostUSD != nil || response.ModelName == "" {
		return
	}
	calculation, err := response.Price()
	if err != nil {
		return
	}
	response.Usage.CostUSD = &calculation.TotalPrice
}

func usageForPricing(usage Usage) genaiprices.Usage {
	priced := make(genaiprices.Usage, len(usage.Details)+8)
	for key, value := range usage.Details {
		priced[pricingUsageKey(key)] = float64(value)
	}
	setPricingUsage(priced, genaiprices.UsageInputTokens, usage.InputTokens)
	setPricingUsage(priced, genaiprices.UsageCacheWriteTokens, usage.CacheWriteTokens)
	setPricingUsage(priced, genaiprices.UsageCacheReadTokens, usage.CacheReadTokens)
	setPricingUsage(priced, genaiprices.UsageOutputTokens, usage.OutputTokens)
	setPricingUsage(priced, genaiprices.UsageInputAudioTokens, usage.InputAudioTokens)
	setPricingUsage(priced, genaiprices.UsageCacheAudioReadTokens, usage.CacheAudioReadTokens)
	setPricingUsage(priced, genaiprices.UsageOutputAudioTokens, usage.OutputAudioTokens)
	setPricingUsage(priced, genaiprices.UsageOutputReasoningTokens, usage.ReasoningTokens)
	return priced
}

func setPricingUsage(priced genaiprices.Usage, key genaiprices.UsageKey, value int) {
	if value != 0 {
		priced[key] = float64(value)
	}
}

var pricingUsageAliases = map[string]genaiprices.UsageKey{
	"reasoning_tokens":             genaiprices.UsageOutputReasoningTokens,
	"thoughts_tokens":              genaiprices.UsageOutputReasoningTokens,
	"cached_content_tokens":        genaiprices.UsageCacheReadTokens,
	"cache_read_input_tokens":      genaiprices.UsageCacheReadTokens,
	"cache_creation_input_tokens":  genaiprices.UsageCacheWriteTokens,
	"prompt_tokens":                genaiprices.UsageInputTokens,
	"completion_tokens":            genaiprices.UsageOutputTokens,
	"tool_use_prompt_tokens":       genaiprices.UsageInputToolTokens,
	"text_prompt_tokens":           genaiprices.UsageInputTextTokens,
	"audio_prompt_tokens":          genaiprices.UsageInputAudioTokens,
	"image_prompt_tokens":          genaiprices.UsageInputImageTokens,
	"video_prompt_tokens":          genaiprices.UsageInputVideoTokens,
	"text_cache_tokens":            genaiprices.UsageCacheTextReadTokens,
	"audio_cache_tokens":           genaiprices.UsageCacheAudioReadTokens,
	"image_cache_tokens":           genaiprices.UsageCacheImageReadTokens,
	"video_cache_tokens":           genaiprices.UsageCacheVideoReadTokens,
	"text_candidates_tokens":       genaiprices.UsageOutputTextTokens,
	"audio_candidates_tokens":      genaiprices.UsageOutputAudioTokens,
	"image_candidates_tokens":      genaiprices.UsageOutputImageTokens,
	"video_candidates_tokens":      genaiprices.UsageOutputVideoTokens,
	"text_tool_use_prompt_tokens":  genaiprices.UsageInputTextToolTokens,
	"audio_tool_use_prompt_tokens": genaiprices.UsageInputAudioToolTokens,
	"image_tool_use_prompt_tokens": genaiprices.UsageInputImageToolTokens,
	"video_tool_use_prompt_tokens": genaiprices.UsageInputVideoToolTokens,
}

func pricingUsageKey(key string) genaiprices.UsageKey {
	if normalized, ok := pricingUsageAliases[key]; ok {
		return normalized
	}
	return genaiprices.UsageKey(key)
}
