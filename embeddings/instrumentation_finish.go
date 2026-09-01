package embeddings

import (
	"context"
	"encoding/json"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
)

func (model *InstrumentedModel) finish(ctx context.Context, span trace.Span, result *Result) {
	responseModel := result.ModelName
	if responseModel == "" {
		responseModel = model.Name()
	}
	attributes := []attribute.KeyValue{attribute.String("gen_ai.response.model", responseModel)}
	if result.Usage.InputTokens != 0 {
		attributes = append(attributes, attribute.Int("gen_ai.usage.input_tokens", result.Usage.InputTokens))
	}
	for name, value := range result.Usage.Details {
		attributes = append(attributes, attribute.Int("gen_ai.usage.details."+name, value))
	}
	if result.ProviderResponseID != "" {
		attributes = append(attributes, attribute.String("gen_ai.response.id", result.ProviderResponseID))
	}
	if len(result.Embeddings) > 0 {
		attributes = append(attributes, attribute.Int("gen_ai.embeddings.dimension.count", len(result.Embeddings[0])))
		if model.includeContent {
			attributes = append(attributes, attribute.String("embeddings", telemetryJSON(result.Embeddings)))
		}
	}
	cost, hasCost := embeddingCost(result)
	if hasCost {
		attributes = append(attributes, attribute.Float64("operation.cost", cost))
	}
	span.SetAttributes(attributes...)
	span.End()
	metricAttributes := []attribute.KeyValue{
		attribute.String("gen_ai.provider.name", model.ProviderName()),
		attribute.String("gen_ai.operation.name", "embeddings"),
		attribute.String("gen_ai.request.model", model.Name()),
		attribute.String("gen_ai.response.model", responseModel),
	}
	if result.Usage.InputTokens != 0 {
		model.tokenHistogram.Record(ctx, int64(result.Usage.InputTokens), metric.WithAttributes(append(
			metricAttributes, attribute.String("gen_ai.token.type", "input"),
		)...))
	}
	if hasCost {
		model.costHistogram.Record(ctx, cost, metric.WithAttributes(metricAttributes...))
	}
}

func embeddingCost(result *Result) (float64, bool) {
	if result.Usage.CostUSD != nil {
		return *result.Usage.CostUSD, true
	}
	calculation, err := result.Price()
	if err != nil {
		return 0, false
	}
	return calculation.TotalPrice, true
}

func telemetrySettings(settings Settings) string {
	value := map[string]any{
		"dimensions": settings.Dimensions, "truncate": settings.Truncate,
		"extra_headers": settings.ExtraHeaders, "extra_body": settings.ExtraBody,
	}
	return telemetryJSON(value)
}

func telemetryJSON(value any) string {
	encoded, err := json.Marshal(value)
	if err != nil {
		return `"unable to serialize"`
	}
	return string(encoded)
}

func embeddingLogfireSchema(includeContent bool) string {
	properties := map[string]any{
		"input_type":         map[string]any{"type": "string"},
		"inputs_count":       map[string]any{"type": "integer"},
		"embedding_settings": map[string]any{"type": "object"},
	}
	if includeContent {
		properties["inputs"] = map[string]any{"type": []string{"array"}}
		properties["embeddings"] = map[string]any{"type": "array"}
	}
	return telemetryJSON(map[string]any{"type": "object", "properties": properties})
}
