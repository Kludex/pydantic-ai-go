package ai_test

import (
	"context"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestRunEmitsSpans(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	t.Cleanup(func() { otel.SetTracerProvider(prev) })

	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	ai.AddSimpleTool(agent, "noop", func(context.Context, struct{}) (string, error) { return "", nil })
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}

	spans := exporter.GetSpans()
	names := map[string]int{}
	for _, s := range spans {
		names[s.Name]++
	}
	if names["agent run"] != 1 {
		t.Fatalf("expected 1 agent run span, got %v", names)
	}
	if names["chat test-model"] != 2 {
		t.Fatalf("expected 2 request spans, got %v", names)
	}
	if names["running tool: noop"] != 1 {
		t.Fatalf("expected 1 tool span, got %v", names)
	}
}

func TestFailedRunRecordsError(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	t.Cleanup(func() { otel.SetTracerProvider(prev) })

	agent := ai.NewAgent[deps, string](fakes.NewTestModel(),
		ai.WithUsageLimits(ai.UsageLimits{RequestLimit: 1}),
	)
	ai.AddSimpleTool(agent, "noop", func(context.Context, struct{}) (string, error) { return "", nil })
	if _, err := agent.Run(t.Context(), "go", deps{}); err == nil {
		t.Fatal("expected error")
	}

	for _, s := range exporter.GetSpans() {
		if s.Name == "agent run" && s.Status.Description != "" {
			return
		}
	}
	t.Fatal("agent run span did not record the error")
}
