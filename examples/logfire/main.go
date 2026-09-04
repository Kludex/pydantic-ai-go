package main

import (
	"context"
	"fmt"
	"log"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/openai"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetrichttp"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

func main() {
	ctx := context.Background()
	traceExporter, err := otlptracehttp.New(ctx)
	if err != nil {
		log.Printf("create trace exporter: %v", err)
		return
	}
	tracerProvider := sdktrace.NewTracerProvider(sdktrace.WithBatcher(traceExporter))
	defer func() {
		if err := tracerProvider.Shutdown(context.Background()); err != nil {
			log.Printf("flush traces: %v", err)
		}
	}()

	metricExporter, err := otlpmetrichttp.New(ctx)
	if err != nil {
		log.Printf("create metric exporter: %v", err)
		return
	}
	meterProvider := sdkmetric.NewMeterProvider(
		sdkmetric.WithReader(sdkmetric.NewPeriodicReader(metricExporter)),
	)
	defer func() {
		if err := meterProvider.Shutdown(context.Background()); err != nil {
			log.Printf("flush metrics: %v", err)
		}
	}()
	otel.SetTracerProvider(tracerProvider)
	otel.SetMeterProvider(meterProvider)

	agent := ai.NewAgent[struct{}, string](
		openai.NewModel("gpt-5-mini"),
		ai.WithAgentName("logfire-example"),
		ai.WithCapabilities(ai.NewInstrumentation(
			ai.WithInstrumentationContent(false),
			ai.WithInstrumentationBinaryContent(false),
			ai.WithInstrumentationModelRequestParameters(false),
		)),
	)
	result, err := agent.Run(ctx, "What is 2 + 2?", struct{}{})
	if err != nil {
		log.Printf("run agent: %v", err)
		return
	}
	fmt.Println(result.Output)
}
