package ai

import (
	"context"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// otelScope is the instrumentation scope name, matching the Python package
// so traces from both libraries look the same in Logfire.
const otelScope = "pydantic-ai"

type runSpanContextKey struct{}
type modelRequestSpanContextKey struct{}
type toolSpanContextKey struct{}

func runSpanActive(ctx context.Context) bool {
	active, _ := ctx.Value(runSpanContextKey{}).(bool)
	return active
}

func modelRequestSpanActive(ctx context.Context) bool {
	active, _ := ctx.Value(modelRequestSpanContextKey{}).(bool)
	return active
}

func toolSpanActive(ctx context.Context) bool {
	active, _ := ctx.Value(toolSpanContextKey{}).(bool)
	return active
}

// tracer returns the package tracer from the global provider. When no
// provider is configured this is a no-op tracer, so instrumentation costs
// nothing unless the user opts in via otel.SetTracerProvider.
func tracer() trace.Tracer {
	return otel.GetTracerProvider().Tracer(otelScope)
}

func startRunSpan(ctx context.Context, modelName string, enabled ...bool) (context.Context, trace.Span) {
	if len(enabled) > 0 && !enabled[0] {
		return ctx, nil
	}
	ctx, span := tracer().Start(ctx, "agent run",
		trace.WithAttributes(
			attribute.String("gen_ai.operation.name", "invoke_agent"),
			attribute.String("gen_ai.request.model", modelName),
		),
	)
	return context.WithValue(ctx, runSpanContextKey{}, true), span
}

func recordRunModel(span trace.Span, modelName string) {
	if span != nil {
		span.SetAttributes(attribute.String("gen_ai.request.model", modelName))
	}
}

func startRequestSpan(ctx context.Context, modelName string) (context.Context, trace.Span) {
	ctx, span := tracer().Start(ctx, "chat "+modelName,
		trace.WithAttributes(
			attribute.String("gen_ai.operation.name", "chat"),
			attribute.String("gen_ai.request.model", modelName),
		),
	)
	return context.WithValue(ctx, modelRequestSpanContextKey{}, true), span
}

func startToolSpan(ctx context.Context, toolName, toolCallID string, enabled ...bool) (context.Context, trace.Span) {
	if len(enabled) > 0 && !enabled[0] {
		return ctx, nil
	}
	ctx, span := tracer().Start(ctx, "running tool: "+toolName,
		trace.WithAttributes(
			attribute.String("gen_ai.operation.name", "execute_tool"),
			attribute.String("gen_ai.tool.name", toolName),
			attribute.String("gen_ai.tool.call.id", toolCallID),
		),
	)
	return context.WithValue(ctx, toolSpanContextKey{}, true), span
}

func endSpan(span trace.Span, err error) {
	if span == nil {
		return
	}
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		span.RecordError(err)
	}
	span.End()
}

func recordUsage(span trace.Span, usage Usage) {
	if span == nil {
		return
	}
	attributes := []attribute.KeyValue{
		attribute.Int("gen_ai.usage.input_tokens", usage.InputTokens),
		attribute.Int("gen_ai.usage.output_tokens", usage.OutputTokens),
	}
	if usage.CostUSD != nil {
		attributes = append(attributes, attribute.Float64("operation.cost", *usage.CostUSD))
	}
	span.SetAttributes(attributes...)
}
