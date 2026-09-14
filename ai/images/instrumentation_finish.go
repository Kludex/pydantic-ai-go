package images

import (
	"context"
	"strconv"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
)

func (model *InstrumentedModel) finish(ctx context.Context, span trace.Span, result *Result) {
	responseModel := result.ModelName
	if responseModel == "" {
		responseModel = model.Name()
	}
	attributes := []attribute.KeyValue{
		attribute.String("gen_ai.response.model", responseModel),
		attribute.Int("image_count", len(result.Images)),
	}
	if result.ProviderResponseID != "" {
		attributes = append(attributes, attribute.String("gen_ai.response.id", result.ProviderResponseID))
	}
	if result.Usage.InputTokens != 0 {
		attributes = append(attributes, attribute.Int("gen_ai.usage.input_tokens", result.Usage.InputTokens))
	}
	if result.Usage.OutputTokens != 0 {
		attributes = append(attributes, attribute.Int("gen_ai.usage.output_tokens", result.Usage.OutputTokens))
	}
	for name, value := range result.Usage.Details {
		attributes = append(attributes, attribute.Int("gen_ai.usage.details."+name, value))
	}
	for index, image := range result.Images {
		prefix := "image." + strconv.Itoa(index)
		attributes = append(attributes, attribute.String(prefix+".media_type", image.Content.MediaType))
		if image.OutputFormat != "" {
			attributes = append(attributes, attribute.String(prefix+".output_format", image.OutputFormat))
		}
	}
	cost, hasCost := imageCost(result)
	if hasCost {
		attributes = append(attributes, attribute.Float64("operation.cost", cost))
	}
	span.SetAttributes(attributes...)
	span.End()

	metricAttributes := []attribute.KeyValue{
		attribute.String("gen_ai.provider.name", model.ProviderName()),
		attribute.String("gen_ai.operation.name", "image_generation"),
		attribute.String("gen_ai.request.model", model.Name()),
		attribute.String("gen_ai.response.model", responseModel),
	}
	for tokenType, count := range map[string]int{"input": result.Usage.InputTokens, "output": result.Usage.OutputTokens} {
		if count != 0 {
			model.tokenHistogram.Record(ctx, int64(count), metric.WithAttributes(append(
				metricAttributes, attribute.String("gen_ai.token.type", tokenType),
			)...))
		}
	}
	if hasCost {
		model.costHistogram.Record(ctx, cost, metric.WithAttributes(metricAttributes...))
	}
}

func imageCost(result *Result) (float64, bool) {
	if result.Usage.CostUSD != nil {
		return *result.Usage.CostUSD, true
	}
	calculation, err := result.Price()
	if err != nil {
		return 0, false
	}
	return calculation.TotalPrice, true
}
