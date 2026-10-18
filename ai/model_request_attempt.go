package ai

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

type modelRequestAttemptRecorderKey struct{}

type modelRequestAttemptRecorder struct {
	tracer         trace.Tracer
	parent         trace.Span
	includeContent bool
}

func recordModelRequestAttempt(
	ctx context.Context,
	attempt ModelRequestAttempt,
	requestErr error,
	response *ModelResponse,
	index int,
) {
	recorder, _ := ctx.Value(modelRequestAttemptRecorderKey{}).(*modelRequestAttemptRecorder)
	if recorder == nil || !recorder.parent.IsRecording() {
		return
	}
	attributes := []attribute.KeyValue{
		attribute.String("gen_ai.request.model", attempt.ModelName),
		attribute.Int("pydantic_ai.model_request.attempt", index),
	}
	if attempt.ProviderName != "" {
		attributes = append(attributes,
			attribute.String("gen_ai.provider.name", attempt.ProviderName),
			attribute.String("gen_ai.system", attempt.ProviderName),
		)
	}
	if response != nil {
		attributes = append(attributes, responseTelemetryAttributes(response)...)
	}
	_, span := recorder.tracer.Start(
		trace.ContextWithSpan(ctx, recorder.parent), "model request attempt "+attempt.ModelName,
		trace.WithTimestamp(attempt.Timestamp), trace.WithAttributes(attributes...),
	)
	if requestErr != nil {
		recordSpanError(span, requestErr, recorder.includeContent)
	} else {
		span.SetStatus(codes.Error, "response rejected by fallback response predicate")
	}
	span.End(trace.WithTimestamp(attempt.Timestamp.Add(attempt.Duration)))
}
