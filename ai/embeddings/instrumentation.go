package embeddings

import (
	"context"
	"net/url"
	"strconv"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
)

const instrumentationScope = "github.com/Kludex/pydantic-ai-go/ai"

// InstrumentationOption configures embedding telemetry.
type InstrumentationOption func(*instrumentationConfig)

type instrumentationConfig struct {
	tracerProvider trace.TracerProvider
	meterProvider  metric.MeterProvider
	includeContent bool
}

// WithInstrumentationTracerProvider selects a tracer provider instead of the global provider.
func WithInstrumentationTracerProvider(provider trace.TracerProvider) InstrumentationOption {
	return func(config *instrumentationConfig) { config.tracerProvider = provider }
}

// WithInstrumentationMeterProvider selects a meter provider instead of the global provider.
func WithInstrumentationMeterProvider(provider metric.MeterProvider) InstrumentationOption {
	return func(config *instrumentationConfig) { config.meterProvider = provider }
}

// WithInstrumentationContent controls input and embedding content attributes.
func WithInstrumentationContent(include bool) InstrumentationOption {
	return func(config *instrumentationConfig) { config.includeContent = include }
}

// InstrumentedModel emits OpenTelemetry spans and metrics around embedding requests.
type InstrumentedModel struct {
	*Wrapper
	tracer         trace.Tracer
	tokenHistogram metric.Int64Histogram
	costHistogram  metric.Float64Histogram
	includeContent bool
}

// NewInstrumentedModel creates a transparent embedding model decorator. Content is included by default.
func NewInstrumentedModel(model Model, options ...InstrumentationOption) *InstrumentedModel {
	if embeddingModelIsNil(model) {
		panic("embeddings: cannot instrument a nil model")
	}
	config := instrumentationConfig{
		tracerProvider: otel.GetTracerProvider(), meterProvider: otel.GetMeterProvider(), includeContent: true,
	}
	for _, option := range options {
		option(&config)
	}
	if config.tracerProvider == nil {
		config.tracerProvider = otel.GetTracerProvider()
	}
	if config.meterProvider == nil {
		config.meterProvider = otel.GetMeterProvider()
	}
	meter := config.meterProvider.Meter(instrumentationScope)
	tokens, _ := meter.Int64Histogram(
		"gen_ai.client.token.usage", metric.WithUnit("{token}"),
		metric.WithDescription("Measures number of input and output tokens used"),
	)
	cost, _ := meter.Float64Histogram(
		"operation.cost", metric.WithUnit("{USD}"), metric.WithDescription("Monetary cost"),
	)
	return &InstrumentedModel{
		Wrapper: WrapModel(model), tracer: config.tracerProvider.Tracer(instrumentationScope),
		tokenHistogram: tokens, costHistogram: cost, includeContent: config.includeContent,
	}
}

// InstrumentModel wraps model unless it is already instrumented.
func InstrumentModel(model Model, options ...InstrumentationOption) Model {
	if _, ok := model.(*InstrumentedModel); ok {
		return model
	}
	return NewInstrumentedModel(model, options...)
}

// Embed instruments one embedding request.
func (model *InstrumentedModel) Embed(
	ctx context.Context, inputs []string, inputType InputType, settings Settings,
) (*Result, error) {
	attributes := []attribute.KeyValue{
		attribute.String("gen_ai.operation.name", "embeddings"),
		attribute.String("gen_ai.provider.name", model.ProviderName()),
		attribute.String("gen_ai.request.model", model.Name()),
		attribute.String("input_type", string(inputType)),
		attribute.Int("inputs_count", len(inputs)),
		attribute.String("logfire.json_schema", embeddingLogfireSchema(model.includeContent)),
	}
	if settings.Dimensions != nil || settings.Truncate != nil || len(settings.ExtraHeaders) > 0 ||
		len(settings.ExtraBody) > 0 {
		attributes = append(attributes, attribute.String("embedding_settings", telemetrySettings(settings)))
	}
	if parsed, err := url.Parse(model.ProviderURL()); err == nil && parsed.Hostname() != "" {
		attributes = append(attributes, attribute.String("server.address", parsed.Hostname()))
		if port, err := strconv.Atoi(parsed.Port()); err == nil && port != 0 {
			attributes = append(attributes, attribute.Int("server.port", port))
		}
	}
	if model.includeContent {
		attributes = append(attributes, attribute.String("inputs", telemetryJSON(inputs)))
	}
	spanCtx, span := model.tracer.Start(
		ctx, "embeddings "+model.Name(), trace.WithSpanKind(trace.SpanKindClient), trace.WithAttributes(attributes...),
	)
	result, err := model.UnwrapModel().Embed(spanCtx, inputs, inputType, settings)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		span.RecordError(err)
		span.End()
		return nil, err
	}
	if result == nil {
		span.End()
		return nil, nil
	}
	model.finish(spanCtx, span, result)
	return result, nil
}
