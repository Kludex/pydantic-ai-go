package images_test

import (
	"errors"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/images"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

func TestInstrumentation(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	tracerProvider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	reader := sdkmetric.NewManualReader()
	meterProvider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	t.Cleanup(func() {
		_ = tracerProvider.Shutdown(t.Context())
		_ = meterProvider.Shutdown(t.Context())
	})
	cost := 0.25
	base := &model{
		name: "request-model", provider: "provider", url: "https://example.com:8443/v1",
		result: &images.Result{
			Images: []images.GeneratedImage{{
				Content: ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"}, OutputFormat: "png",
			}},
			ModelName: "response-model", ProviderResponseID: "response-id",
			Usage: ai.Usage{InputTokens: 7, OutputTokens: 8, Details: map[string]int{"image_tokens": 6}, CostUSD: &cost},
		},
	}
	instrumented := images.NewInstrumentedModel(
		base, images.WithInstrumentationTracerProvider(tracerProvider),
		images.WithInstrumentationMeterProvider(meterProvider),
	)
	result, err := instrumented.Generate(t.Context(), "draw", nil, images.Settings{AspectRatio: images.AspectRatio1To1})
	if err != nil || result == nil || instrumented.UnwrapModel() != base {
		t.Fatalf("unexpected result: %#v %v", result, err)
	}
	spans := exporter.GetSpans()
	if len(spans) != 1 || spans[0].Name != "image_generation request-model" || spans[0].SpanKind != trace.SpanKindClient {
		t.Fatalf("unexpected spans: %#v", spans)
	}
	attributes := imageSpanAttributes(spans[0].Attributes)
	for key, want := range map[string]any{
		"gen_ai.operation.name": "image_generation", "gen_ai.output.type": "image",
		"gen_ai.provider.name": "provider", "gen_ai.request.model": "request-model",
		"gen_ai.response.model": "response-model", "gen_ai.response.id": "response-id",
		"gen_ai.usage.input_tokens": int64(7), "gen_ai.usage.output_tokens": int64(8),
		"gen_ai.usage.details.image_tokens": int64(6), "image_count": int64(1),
		"image.0.media_type": "image/png", "image.0.output_format": "png",
		"server.address": "example.com", "server.port": int64(8443), "operation.cost": 0.25,
	} {
		if attributes[key] != want {
			t.Fatalf("attribute %q = %#v, want %#v", key, attributes[key], want)
		}
	}
	var metrics metricdata.ResourceMetrics
	if err := reader.Collect(t.Context(), &metrics); err != nil {
		t.Fatal(err)
	}
	if len(metrics.ScopeMetrics) == 0 {
		t.Fatal("instrumentation emitted no metrics")
	}
	if images.InstrumentModel(instrumented) != instrumented {
		t.Fatal("instrumentation was not idempotent")
	}
	priced := images.NewInstrumentedModel(&model{
		name: "gpt-image-1", provider: "openai", result: &images.Result{
			Images:    []images.GeneratedImage{{Content: ai.BinaryContent{MediaType: "image/png"}}},
			ModelName: "gpt-image-1", ProviderName: "openai", Usage: ai.Usage{InputTokens: 1, OutputTokens: 1},
		},
	}, images.WithInstrumentationTracerProvider(tracerProvider), images.WithInstrumentationMeterProvider(meterProvider))
	if _, err := priced.Generate(t.Context(), "draw", nil, images.Settings{}); err != nil {
		t.Fatal(err)
	}
}

func TestInstrumentationErrorsPrivacyAndDefaults(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
	requestErr := errors.New("request failed")
	failed := &model{name: "failed", provider: "test", err: requestErr}
	instrumented := images.NewInstrumentedModel(
		failed, images.WithInstrumentationTracerProvider(provider),
		images.WithInstrumentationMeterProvider(nil), images.WithInstrumentationContent(false),
		images.WithInstrumentationRequestParameters(false),
	)
	if _, err := instrumented.Generate(t.Context(), "secret", nil, images.Settings{}); !errors.Is(err, requestErr) {
		t.Fatalf("unexpected error: %v", err)
	}
	spans := exporter.GetSpans()
	if len(spans) != 1 || spans[0].Status.Code != codes.Error || len(spans[0].Events) != 1 {
		t.Fatalf("unexpected failed span: %#v", spans)
	}
	attributes := imageSpanAttributes(spans[0].Attributes)
	if _, exists := attributes["prompt"]; exists {
		t.Fatal("privacy-disabled span included prompt")
	}
	if _, exists := attributes["image_generation_settings"]; exists {
		t.Fatal("disabled span included settings")
	}

	success := &model{name: "success", provider: "test", result: &images.Result{
		Images: []images.GeneratedImage{{Content: ai.BinaryContent{Data: []byte("x"), MediaType: "image/png"}}},
	}}
	restore := images.InstrumentAll(images.WithInstrumentationTracerProvider(provider))
	generator := images.New(success)
	if _, err := generator.Generate(t.Context(), "draw", nil); err != nil {
		t.Fatal(err)
	}
	restore()
	if len(exporter.GetSpans()) != 2 {
		t.Fatal("default instrumentation did not emit a span")
	}

	restore = images.InstrumentAll(images.WithInstrumentationTracerProvider(provider))
	disable := images.DisableInstrumentation()
	if _, err := generator.Generate(t.Context(), "draw", nil); err != nil {
		t.Fatal(err)
	}
	disable()
	if _, err := generator.Generate(t.Context(), "draw", nil); err != nil {
		t.Fatal(err)
	}
	restore()
	if len(exporter.GetSpans()) != 3 {
		t.Fatalf("unexpected default spans: %d", len(exporter.GetSpans()))
	}

	explicitDisabled := images.New(success, images.WithoutInstrumentation())
	if _, err := explicitDisabled.Generate(t.Context(), "draw", nil); err != nil {
		t.Fatal(err)
	}
	if len(exporter.GetSpans()) != 3 {
		t.Fatal("disabled generator emitted a span")
	}
	explicit := images.New(success, images.WithInstrumentation(images.WithInstrumentationTracerProvider(provider)))
	if _, err := explicit.Generate(t.Context(), "draw", nil); err != nil {
		t.Fatal(err)
	}
	if len(exporter.GetSpans()) != 4 {
		t.Fatal("explicit instrumentation emitted no span")
	}

	nilResult := images.NewInstrumentedModel(&model{name: "nil", provider: "test"},
		images.WithInstrumentationTracerProvider(provider), images.WithInstrumentationTracerProvider(nil))
	if result, err := nilResult.Generate(t.Context(), "draw", nil, images.Settings{}); err != nil || result != nil {
		t.Fatalf("unexpected nil result: %#v %v", result, err)
	}
	var typedNil *model
	if capturePanic(func() { images.NewInstrumentedModel(typedNil) }) == nil {
		t.Fatal("typed nil was instrumented")
	}
}

func imageSpanAttributes(values []attribute.KeyValue) map[string]any {
	attributes := make(map[string]any, len(values))
	for _, value := range values {
		attributes[string(value.Key)] = value.Value.AsInterface()
	}
	return attributes
}
