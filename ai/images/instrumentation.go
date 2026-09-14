package images

import (
	"context"
	"encoding/json"
	"net/url"
	"strconv"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
)

const instrumentationScope = "github.com/Kludex/pydantic-ai-go/ai"

// InstrumentationOption configures image generation telemetry.
type InstrumentationOption func(*instrumentationConfig)

type instrumentationConfig struct {
	tracerProvider  trace.TracerProvider
	meterProvider   metric.MeterProvider
	includeContent  bool
	includeSettings bool
}

// WithInstrumentationTracerProvider selects a tracer provider instead of the global provider.
func WithInstrumentationTracerProvider(provider trace.TracerProvider) InstrumentationOption {
	return func(config *instrumentationConfig) { config.tracerProvider = provider }
}

// WithInstrumentationMeterProvider selects a meter provider instead of the global provider.
func WithInstrumentationMeterProvider(provider metric.MeterProvider) InstrumentationOption {
	return func(config *instrumentationConfig) { config.meterProvider = provider }
}

// WithInstrumentationContent controls prompt content attributes.
func WithInstrumentationContent(include bool) InstrumentationOption {
	return func(config *instrumentationConfig) { config.includeContent = include }
}

// WithInstrumentationRequestParameters controls normalized settings attributes.
func WithInstrumentationRequestParameters(include bool) InstrumentationOption {
	return func(config *instrumentationConfig) { config.includeSettings = include }
}

// InstrumentedModel emits OpenTelemetry spans and metrics around image requests.
type InstrumentedModel struct {
	*Wrapper
	tracer          trace.Tracer
	tokenHistogram  metric.Int64Histogram
	costHistogram   metric.Float64Histogram
	includeContent  bool
	includeSettings bool
}

// NewInstrumentedModel creates a transparent image model decorator.
func NewInstrumentedModel(model Model, options ...InstrumentationOption) *InstrumentedModel {
	if modelIsNil(model) {
		panic("images: cannot instrument a nil model")
	}
	config := instrumentationConfig{
		tracerProvider: otel.GetTracerProvider(), meterProvider: otel.GetMeterProvider(),
		includeContent: true, includeSettings: true,
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
		includeSettings: config.includeSettings,
	}
}

// InstrumentModel wraps model unless it is already instrumented.
func InstrumentModel(model Model, options ...InstrumentationOption) Model {
	if _, ok := model.(*InstrumentedModel); ok {
		return model
	}
	return NewInstrumentedModel(model, options...)
}

// Generate instruments one image generation request.
func (model *InstrumentedModel) Generate(
	ctx context.Context, prompt string, inputs []Input, settings Settings,
) (*Result, error) {
	settings = MergeSettings(model.DefaultSettings(), settings)
	attributes := []attribute.KeyValue{
		attribute.String("gen_ai.operation.name", "image_generation"),
		attribute.String("gen_ai.output.type", "image"),
		attribute.String("gen_ai.provider.name", model.ProviderName()),
		attribute.String("gen_ai.request.model", model.Name()),
		attribute.Int("prompt_length", len(prompt)),
		attribute.Int("input_image_count", len(inputs)),
		attribute.String("logfire.json_schema", imageLogfireSchema(model.includeContent, model.includeSettings)),
	}
	if parsed, err := url.Parse(model.ProviderURL()); err == nil && parsed.Hostname() != "" {
		attributes = append(attributes, attribute.String("server.address", parsed.Hostname()))
		if port, err := strconv.Atoi(parsed.Port()); err == nil && port != 0 {
			attributes = append(attributes, attribute.Int("server.port", port))
		}
	}
	if model.includeSettings &&
		(settings.Dimensions != nil || settings.AspectRatio != "" || len(settings.ProviderSettings) > 0) {
		attributes = append(attributes, attribute.String("image_generation_settings", telemetrySettings(settings)))
	}
	if model.includeContent {
		attributes = append(attributes, attribute.String("prompt", prompt))
	}
	spanCtx, span := model.tracer.Start(
		ctx, "image_generation "+model.Name(),
		trace.WithSpanKind(trace.SpanKindClient), trace.WithAttributes(attributes...),
	)
	result, err := model.UnwrapModel().Generate(spanCtx, prompt, inputs, settings)
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

func telemetrySettings(settings Settings) string {
	value := map[string]any{
		"dimensions": settings.Dimensions, "aspect_ratio": settings.AspectRatio,
		"provider_settings": settings.ProviderSettings,
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return `"unable to serialize"`
	}
	return string(encoded)
}

func imageLogfireSchema(includeContent, includeSettings bool) string {
	properties := map[string]any{
		"prompt_length":     map[string]any{"type": "integer"},
		"input_image_count": map[string]any{"type": "integer"},
		"image_count":       map[string]any{"type": "integer"},
	}
	if includeContent {
		properties["prompt"] = map[string]any{"type": "string"}
	}
	if includeSettings {
		properties["image_generation_settings"] = map[string]any{"type": "object"}
	}
	encoded, _ := json.Marshal(map[string]any{"type": "object", "properties": properties})
	return string(encoded)
}
