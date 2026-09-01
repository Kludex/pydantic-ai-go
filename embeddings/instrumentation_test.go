package embeddings_test

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/embeddings"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

func TestEmbedderInstrumentationDefaults(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	tracerProvider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = tracerProvider.Shutdown(t.Context()) })
	base := &embeddingModel{
		name: "model", provider: "provider", providerURL: "https://example.com",
		embed: func(
			_ context.Context, inputs []string, inputType embeddings.InputType, _ embeddings.Settings,
		) (*embeddings.Result, error) {
			vectors := make([][]float64, len(inputs))
			for index := range vectors {
				vectors[index] = []float64{float64(index)}
			}
			return &embeddings.Result{
				Embeddings: vectors, Inputs: inputs, InputType: inputType, ModelName: "model", ProviderName: "provider",
			}, nil
		},
	}
	embedder := embeddings.New(base)
	restore := embeddings.InstrumentAll(embeddings.WithInstrumentationTracerProvider(tracerProvider))
	t.Cleanup(restore)
	if _, err := embedder.EmbedQuery(t.Context(), "global"); err != nil {
		t.Fatal(err)
	}
	if len(exporter.GetSpans()) != 1 {
		t.Fatalf("global instrumentation emitted %d spans", len(exporter.GetSpans()))
	}

	disabled := embeddings.New(base, embeddings.WithoutInstrumentation())
	if _, err := disabled.EmbedQuery(t.Context(), "disabled"); err != nil {
		t.Fatal(err)
	}
	if len(exporter.GetSpans()) != 1 {
		t.Fatal("explicitly disabled embedder emitted a span")
	}

	restore()
	if _, err := embedder.EmbedQuery(t.Context(), "restored"); err != nil {
		t.Fatal(err)
	}
	if len(exporter.GetSpans()) != 1 {
		t.Fatal("restored global default emitted a span")
	}

	explicit := embeddings.New(base, embeddings.WithInstrumentation(
		embeddings.WithInstrumentationTracerProvider(tracerProvider),
	))
	ctx := embeddings.WithModel(t.Context(), base)
	if _, err := explicit.EmbedQuery(ctx, "explicit"); err != nil {
		t.Fatal(err)
	}
	if len(exporter.GetSpans()) != 2 {
		t.Fatal("explicit or overridden model was not instrumented")
	}

	restoreEnabled := embeddings.InstrumentAll(embeddings.WithInstrumentationTracerProvider(tracerProvider))
	restoreDisabled := embeddings.DisableInstrumentation()
	if _, err := embedder.EmbedQuery(t.Context(), "globally disabled"); err != nil {
		t.Fatal(err)
	}
	if len(exporter.GetSpans()) != 2 {
		t.Fatal("disabled global default emitted a span")
	}
	restoreDisabled()
	if _, err := embedder.EmbedQuery(t.Context(), "nested restore"); err != nil {
		t.Fatal(err)
	}
	if len(exporter.GetSpans()) != 3 {
		t.Fatal("nested default restoration did not restore instrumentation")
	}
	restoreEnabled()
}

func TestInstrumentedEmbedding(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	tracerProvider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	reader := sdkmetric.NewManualReader()
	meterProvider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	t.Cleanup(func() {
		_ = tracerProvider.Shutdown(t.Context())
		_ = meterProvider.Shutdown(t.Context())
	})

	cost := 0.25
	base := &embeddingModel{
		name: "request-model", provider: "provider", providerURL: "https://example.com:8443/v1",
		embed: func(ctx context.Context, inputs []string, inputType embeddings.InputType, settings embeddings.Settings) (
			*embeddings.Result, error,
		) {
			if !trace.SpanFromContext(ctx).SpanContext().IsValid() {
				t.Fatal("embedding request did not receive a span context")
			}
			if !reflect.DeepEqual(inputs, []string{"one", "two"}) || inputType != embeddings.InputTypeDocument ||
				settings.ExtraBody["provider"] != "value" {
				t.Fatalf("unexpected request: %#v %q %#v", inputs, inputType, settings)
			}
			return &embeddings.Result{
				Embeddings: [][]float64{{1, 2}, {3, 4}}, Inputs: inputs, InputType: inputType,
				ModelName: "response-model", ProviderName: "provider", ProviderURL: "https://example.com:8443/v1",
				ProviderResponseID: "response-id",
				Usage:              ai.Usage{InputTokens: 7, Details: map[string]int{"search_units": 2}, CostUSD: &cost},
			}, nil
		},
	}
	model := embeddings.NewInstrumentedModel(
		base,
		embeddings.WithInstrumentationTracerProvider(tracerProvider),
		embeddings.WithInstrumentationMeterProvider(meterProvider),
	)
	dimensions := 2
	result, err := model.Embed(t.Context(), []string{"one", "two"}, embeddings.InputTypeDocument, embeddings.Settings{
		Dimensions: &dimensions, ExtraBody: map[string]any{"provider": "value"},
	})
	if err != nil || len(result.Embeddings) != 2 || model.UnwrapModel() != base {
		t.Fatalf("unexpected result: %#v %v", result, err)
	}

	spans := exporter.GetSpans()
	if len(spans) != 1 || spans[0].Name != "embeddings request-model" || spans[0].SpanKind != trace.SpanKindClient {
		t.Fatalf("unexpected spans: %#v", spans)
	}
	attributes := spanAttributes(spans[0].Attributes)
	for key, want := range map[string]any{
		"gen_ai.operation.name": "embeddings", "gen_ai.provider.name": "provider",
		"gen_ai.request.model": "request-model", "gen_ai.response.model": "response-model",
		"gen_ai.response.id": "response-id", "gen_ai.usage.input_tokens": int64(7),
		"gen_ai.usage.details.search_units": int64(2), "gen_ai.embeddings.dimension.count": int64(2),
		"input_type": "document", "inputs_count": int64(2), "server.address": "example.com",
		"server.port": int64(8443), "operation.cost": 0.25,
	} {
		if got := attributes[key]; got != want {
			t.Fatalf("attribute %q = %#v, want %#v", key, got, want)
		}
	}
	for key, fragment := range map[string]string{
		"inputs": `"one"`, "embeddings": `[[1,2],[3,4]]`,
		"embedding_settings": `"dimensions":2`, "logfire.json_schema": `"embeddings"`,
	} {
		value, _ := attributes[key].(string)
		if !strings.Contains(value, fragment) {
			t.Fatalf("attribute %q omitted %q: %s", key, fragment, value)
		}
	}

	var metrics metricdata.ResourceMetrics
	if err := reader.Collect(t.Context(), &metrics); err != nil {
		t.Fatal(err)
	}
	metricNames := map[string]bool{}
	for _, scope := range metrics.ScopeMetrics {
		for _, instrument := range scope.Metrics {
			metricNames[instrument.Name] = true
		}
	}
	for _, name := range []string{"gen_ai.client.token.usage", "operation.cost"} {
		if !metricNames[name] {
			t.Fatalf("missing metric %q: %#v", name, metricNames)
		}
	}
}

func TestInstrumentationPrivacyAndCalculatedCost(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	tracerProvider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	reader := sdkmetric.NewManualReader()
	meterProvider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	t.Cleanup(func() {
		_ = tracerProvider.Shutdown(t.Context())
		_ = meterProvider.Shutdown(t.Context())
	})
	base := &embeddingModel{
		name: "text-embedding-3-small", provider: "openai", providerURL: "https://api.openai.com/v1",
		embed: func(context.Context, []string, embeddings.InputType, embeddings.Settings) (*embeddings.Result, error) {
			return &embeddings.Result{
				Embeddings: [][]float64{{1}}, ModelName: "text-embedding-3-small", ProviderName: "openai",
				Usage: ai.Usage{InputTokens: 1000},
			}, nil
		},
	}
	model := embeddings.NewInstrumentedModel(
		base,
		embeddings.WithInstrumentationTracerProvider(tracerProvider),
		embeddings.WithInstrumentationMeterProvider(meterProvider),
		embeddings.WithInstrumentationContent(false),
	)
	if _, err := model.Embed(t.Context(), []string{"secret"}, embeddings.InputTypeQuery, embeddings.Settings{
		ExtraBody: map[string]any{"unsupported": make(chan int)},
	}); err != nil {
		t.Fatal(err)
	}
	attributes := spanAttributes(exporter.GetSpans()[0].Attributes)
	if _, exists := attributes["inputs"]; exists {
		t.Fatal("privacy-disabled span included inputs")
	}
	if _, exists := attributes["embeddings"]; exists {
		t.Fatal("privacy-disabled span included embeddings")
	}
	if schema, _ := attributes["logfire.json_schema"].(string); strings.Contains(schema, `"inputs"`) {
		t.Fatalf("privacy-disabled schema included content: %s", schema)
	}
	if attributes["embedding_settings"] != `"unable to serialize"` {
		t.Fatalf("unexpected serialization fallback: %#v", attributes["embedding_settings"])
	}
	if _, exists := attributes["operation.cost"]; !exists {
		t.Fatal("known model price was not calculated")
	}
}

func TestInstrumentationErrorsAndIdempotency(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
	requestErr := errors.New("request failed")
	base := &embeddingModel{
		name: "failed", provider: "provider", providerURL: "://invalid",
		embed: func(context.Context, []string, embeddings.InputType, embeddings.Settings) (*embeddings.Result, error) {
			return nil, requestErr
		},
	}
	model := embeddings.NewInstrumentedModel(
		base,
		embeddings.WithInstrumentationTracerProvider(provider),
		embeddings.WithInstrumentationMeterProvider(nil),
	)
	if _, err := model.Embed(t.Context(), []string{"text"}, embeddings.InputTypeQuery, embeddings.Settings{}); !errors.Is(err, requestErr) {
		t.Fatalf("unexpected request error: %v", err)
	}
	spans := exporter.GetSpans()
	if len(spans) != 1 || spans[0].Status.Code != codes.Error || len(spans[0].Events) != 1 {
		t.Fatalf("unexpected failed span: %#v", spans)
	}
	if embeddings.InstrumentModel(model) != model {
		t.Fatal("instrumentation was not idempotent")
	}
	wrapped := embeddings.InstrumentModel(base, embeddings.WithInstrumentationTracerProvider(provider))
	if _, ok := wrapped.(*embeddings.InstrumentedModel); !ok {
		t.Fatalf("unexpected wrapper: %T", wrapped)
	}

	nilModel := &embeddingModel{}
	_ = embeddings.InstrumentModel(
		nilModel,
		embeddings.WithInstrumentationTracerProvider(nil),
		embeddings.WithInstrumentationMeterProvider(nil),
	)
	var typedNil *embeddingModel
	for _, function := range []func(){
		func() { embeddings.NewInstrumentedModel(nil) },
		func() { embeddings.NewInstrumentedModel(typedNil) },
		func() { embeddings.WrapModel(typedNil) },
		func() { embeddings.New(typedNil) },
	} {
		if panicValue := capturePanic(function); panicValue == nil {
			t.Fatal("nil model did not panic")
		}
	}
}

func TestInstrumentationNilResult(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
	model := embeddings.NewInstrumentedModel(&embeddingModel{
		name: "nil", provider: "provider",
		embed: func(context.Context, []string, embeddings.InputType, embeddings.Settings) (*embeddings.Result, error) {
			return nil, nil
		},
	}, embeddings.WithInstrumentationTracerProvider(provider))
	result, err := model.Embed(t.Context(), []string{"text"}, embeddings.InputTypeQuery, embeddings.Settings{})
	if err != nil || result != nil || len(exporter.GetSpans()) != 1 {
		t.Fatalf("unexpected nil result: %#v %v", result, err)
	}
	unknown := embeddings.NewInstrumentedModel(&embeddingModel{
		name: "unknown", provider: "unknown",
		embed: func(context.Context, []string, embeddings.InputType, embeddings.Settings) (*embeddings.Result, error) {
			return &embeddings.Result{}, nil
		},
	}, embeddings.WithInstrumentationTracerProvider(provider))
	if _, err := unknown.Embed(t.Context(), []string{"text"}, embeddings.InputTypeQuery, embeddings.Settings{}); err != nil {
		t.Fatal(err)
	}
	if spans := exporter.GetSpans(); len(spans) != 2 ||
		spanAttributes(spans[1].Attributes)["gen_ai.response.model"] != "unknown" {
		t.Fatalf("unexpected unknown-model span: %#v", spans)
	}
}

func spanAttributes(values []attribute.KeyValue) map[string]any {
	attributes := make(map[string]any, len(values))
	for _, value := range values {
		attributes[string(value.Key)] = value.Value.AsInterface()
	}
	return attributes
}

type embeddingModel struct {
	name        string
	provider    string
	providerURL string
	embed       func(context.Context, []string, embeddings.InputType, embeddings.Settings) (*embeddings.Result, error)
}

func (model *embeddingModel) Embed(
	ctx context.Context, inputs []string, inputType embeddings.InputType, settings embeddings.Settings,
) (*embeddings.Result, error) {
	return model.embed(ctx, inputs, inputType, settings)
}

func (model *embeddingModel) Name() string         { return model.name }
func (model *embeddingModel) ProviderName() string { return model.provider }
func (model *embeddingModel) ProviderURL() string  { return model.providerURL }

func capturePanic(function func()) (value any) {
	defer func() { value = recover() }()
	function()
	return nil
}
