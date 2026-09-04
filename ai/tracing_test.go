package ai_test

import (
	"context"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
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

func TestRunSpanUsesSelectedModel(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	t.Cleanup(func() { otel.SetTracerProvider(prev) })

	agent := ai.NewAgent[deps, string](nil)
	agent.AddModelSelector(func(
		context.Context, ai.ModelSelectionContext[deps],
	) (ai.ModelSelection, error) {
		return ai.ModelSelection{Model: textModel("selected", "done", nil)}, nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	for _, span := range exporter.GetSpans() {
		if span.Name != "agent run" {
			continue
		}
		for _, attr := range span.Attributes {
			if string(attr.Key) == "gen_ai.request.model" && attr.Value.AsString() == "selected" {
				return
			}
		}
	}
	t.Fatal("agent run span did not record the selected model")
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

func TestRequestSpanRecordsCalculatedCost(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	previous := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	t.Cleanup(func() { otel.SetTracerProvider(previous) })

	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{
			ModelName: "gpt-5", ProviderName: "openai", Usage: ai.Usage{InputTokens: 1},
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}},
		}, nil
	})
	if _, err := ai.NewAgent[deps, string](model).Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	for _, span := range exporter.GetSpans() {
		for _, attr := range span.Attributes {
			if string(attr.Key) == "operation.cost" && attr.Value.AsFloat64() > 0 {
				return
			}
		}
	}
	t.Fatal("request span did not record the calculated cost")
}
