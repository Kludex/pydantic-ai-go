package ai

import (
	"context"
	"errors"
	"time"

	genaiprices "github.com/pydantic/genai-prices/packages/go"
)

// PriceCalculation contains the best matching input, output, and total prices in US dollars.
type PriceCalculation = genaiprices.PriceCalculation

// PricingDiagnosticKind classifies a best-effort pricing diagnostic.
type PricingDiagnosticKind string

const (
	// PricingDiagnosticUnavailable means pricing data or compatible usage was unavailable.
	PricingDiagnosticUnavailable PricingDiagnosticKind = "unavailable"
	// PricingDiagnosticWarning reports a non-fatal warning from a successful calculation.
	PricingDiagnosticWarning PricingDiagnosticKind = "warning"
	// PricingDiagnosticFailed reports an unexpected calculation failure.
	PricingDiagnosticFailed PricingDiagnosticKind = "failed"
)

// PricingDiagnostic describes why automatic pricing was incomplete or noteworthy.
type PricingDiagnostic struct {
	// Kind classifies the diagnostic.
	Kind PricingDiagnosticKind
	// ModelName identifies the model used for lookup.
	ModelName string
	// ProviderName identifies the provider used for lookup.
	ProviderName string
	// ProviderURL is the provider endpoint used for lookup.
	ProviderURL string
	// Message summarizes the diagnostic.
	Message string
	// Err contains the underlying pricing error when available.
	Err error
}

// PricingDiagnosticSink receives optional best-effort pricing diagnostics.
type PricingDiagnosticSink func(ctx context.Context, diagnostic PricingDiagnostic)

type pricingDiagnosticContextKey struct{}

type pricingDiagnosticReporter struct {
	sink PricingDiagnosticSink
}

// WithPricingDiagnosticSink returns a context that reports automatic-pricing diagnostics.
// A nil sink disables diagnostics without changing pricing behavior.
func WithPricingDiagnosticSink(ctx context.Context, sink PricingDiagnosticSink) context.Context {
	if sink == nil {
		return ctx
	}
	return context.WithValue(ctx, pricingDiagnosticContextKey{}, pricingDiagnosticReporter{sink: sink})
}

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

func priceProspectiveUsage(ctx context.Context, model Model, usage Usage) Usage {
	response := &ModelResponse{ModelName: model.Name(), Timestamp: time.Now().UTC(), Usage: usage.Clone()}
	if identity, ok := model.(ModelProviderIdentity); ok {
		response.ProviderName = identity.ProviderName()
		response.ProviderURL = identity.ProviderURL()
	}
	fillResponseCost(ctx, response)
	return response.Usage
}

func fillResponseCost(ctx context.Context, response *ModelResponse) {
	if response == nil || response.Usage.CostUSD != nil || response.pricingAttempted {
		return
	}
	response.pricingAttempted = true
	if response.ModelName == "" {
		reportPricingDiagnostic(ctx, response, PricingDiagnostic{
			Kind: PricingDiagnosticUnavailable, Err: genaiprices.ErrModelNotFound,
			Message: "model name is unavailable",
		})
		return
	}
	calculation, err := response.Price()
	if err != nil {
		kind := PricingDiagnosticFailed
		if errors.Is(err, genaiprices.ErrProviderNotFound) || errors.Is(err, genaiprices.ErrModelNotFound) ||
			errors.Is(err, genaiprices.ErrExtractorNotFound) || errors.Is(err, genaiprices.ErrInvalidUsage) {
			kind = PricingDiagnosticUnavailable
		}
		reportPricingDiagnostic(ctx, response, PricingDiagnostic{Kind: kind, Err: err, Message: err.Error()})
		return
	}
	response.Usage.CostUSD = &calculation.TotalPrice
	for _, warning := range calculation.Warnings {
		reportPricingDiagnostic(ctx, response, PricingDiagnostic{
			Kind: PricingDiagnosticWarning, Message: warning,
		})
	}
}

func reportPricingDiagnostic(ctx context.Context, response *ModelResponse, diagnostic PricingDiagnostic) {
	reporter, ok := ctx.Value(pricingDiagnosticContextKey{}).(pricingDiagnosticReporter)
	if !ok {
		return
	}
	diagnostic.ModelName = response.ModelName
	diagnostic.ProviderName = response.ProviderName
	diagnostic.ProviderURL = response.ProviderURL
	reporter.sink(ctx, diagnostic)
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
	"web_search_requests":          genaiprices.UsageWebSearches,
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
