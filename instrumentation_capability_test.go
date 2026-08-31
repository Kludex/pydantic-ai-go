package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
	"go.opentelemetry.io/otel/baggage"
	"go.opentelemetry.io/otel/codes"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestInstrumentationCapabilityRecordsRunRequestsAndTools(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	tracerProvider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	reader := sdkmetric.NewManualReader()
	meterProvider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	t.Cleanup(func() {
		_ = tracerProvider.Shutdown(t.Context())
		_ = meterProvider.Shutdown(t.Context())
	})
	instrumentation := ai.NewInstrumentation(
		ai.WithInstrumentationTracerProvider(tracerProvider),
		ai.WithInstrumentationMeterProvider(meterProvider),
		ai.WithInstrumentationAgentName("support"),
		ai.WithInstrumentationBinaryContent(false),
	)
	request := 0
	model := fakes.NewFunctionModel(func(
		ctx context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		bag := baggage.FromContext(ctx)
		if bag.Member("gen_ai.agent.name").Value() != "support" ||
			bag.Member("gen_ai.agent.call.id").Value() == "" ||
			bag.Member("gen_ai.conversation.id").Value() == "" {
			t.Fatalf("agent baggage missing in model request: %v", bag)
		}
		cost := 0.01
		if request == 1 {
			return &ai.ModelResponse{
				Parts: []ai.ResponsePart{ai.ToolCallPart{
					ToolName: "lookup", ToolCallID: "call-1", Args: json.RawMessage(`{"query":"go"}`),
				}},
				ModelName: "gpt-5", ProviderName: "openai",
				Usage: ai.Usage{Requests: 1, InputTokens: 3, Details: map[string]int{"reasoning_tokens": 1}, CostUSD: &cost},
			}, nil
		}
		return &ai.ModelResponse{
			Parts:     []ai.ResponsePart{ai.TextPart{Content: "done"}},
			ModelName: "gpt-5", ProviderName: "openai",
			Usage: ai.Usage{Requests: 1, InputTokens: 2, OutputTokens: 1, CostUSD: &cost},
		}, nil
	})
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		instrumentation,
		ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(tracerProvider)),
	))
	ai.AddSimpleTool(agent, "lookup", func(ctx context.Context, _ struct {
		Query string `json:"query"`
	}) (ai.ToolReturn, error) {
		if baggage.FromContext(ctx).Member("gen_ai.agent.name").Value() != "support" {
			t.Fatal("agent baggage missing in tool call")
		}
		return ai.ToolReturn{
			ReturnValue: map[string]any{
				"label": "result",
				"items": []any{ai.BinaryContent{Data: []byte("secret"), MediaType: "image/png"}},
			},
			Content: []ai.UserContent{
				ai.TextContent{Text: "attachment"},
				ai.BinaryContent{Data: []byte("secret"), MediaType: "audio/wav"},
			},
		}, nil
	})
	result, err := agent.Run(t.Context(), "go", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "done" || request != 2 {
		t.Fatalf("unexpected result: result=%+v requests=%d", result, request)
	}

	spans := exporter.GetSpans()
	if len(spans) != 4 {
		t.Fatalf("expected run, two request, and tool spans: %+v", spans)
	}
	counts := map[string]int{}
	for _, span := range spans {
		counts[span.Name]++
	}
	for name, want := range map[string]int{
		"support run": 1, "chat function-model": 2, "running tool: lookup": 1,
	} {
		if counts[name] != want {
			t.Fatalf("span %q count = %d, want %d: %+v", name, counts[name], want, counts)
		}
	}
	var runAttributes map[string]any
	var toolAttributes map[string]any
	for _, span := range spans {
		switch span.Name {
		case "support run":
			runAttributes = instrumentationSpanAttributes(span.Attributes)
		case "running tool: lookup":
			toolAttributes = instrumentationSpanAttributes(span.Attributes)
		}
	}
	for key, want := range map[string]any{
		"gen_ai.agent.name": "support", "gen_ai.operation.name": "invoke_agent",
		"gen_ai.request.model": "function-model", "final_result": "done",
		"gen_ai.aggregated_usage.input_tokens":  int64(5),
		"gen_ai.aggregated_usage.output_tokens": int64(1),
		"pydantic_ai.requests":                  int64(2), "pydantic_ai.tool_calls": int64(1),
	} {
		if got := runAttributes[key]; got != want {
			t.Fatalf("run attribute %q = %#v, want %#v", key, got, want)
		}
	}
	if allMessages, _ := runAttributes["pydantic_ai.all_messages"].(string); !strings.Contains(allMessages, "done") {
		t.Fatalf("run messages missing output: %s", allMessages)
	}
	toolResult, _ := toolAttributes["gen_ai.tool.call.result"].(string)
	if !strings.Contains(toolResult, "image/png") || !strings.Contains(toolResult, "audio/wav") ||
		strings.Contains(toolResult, "c2VjcmV0") {
		t.Fatalf("tool result binary data was not redacted: %s", toolResult)
	}
	if args, _ := toolAttributes["gen_ai.tool.call.arguments"].(string); !strings.Contains(args, "go") {
		t.Fatalf("tool arguments missing: %s", args)
	}

	var metrics metricdata.ResourceMetrics
	if err := reader.Collect(t.Context(), &metrics); err != nil {
		t.Fatal(err)
	}
	if len(metrics.ScopeMetrics) == 0 {
		t.Fatal("instrumentation capability did not record model metrics")
	}
}

func TestInstrumentationCapabilityPrivacyAndNoStacking(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
	first := ai.NewInstrumentation(
		ai.WithInstrumentationTracerProvider(provider), ai.WithInstrumentationAgentName(""),
		ai.WithInstrumentationContent(false),
		ai.WithInstrumentationModelRequestParameters(false),
	)
	second := ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(provider))
	agent := ai.NewAgent[struct{}, string](
		fallbackTextModel("private-agent", "secret", nil), ai.WithCapabilities(first, second),
	)
	agent.AddRawTool(ai.ToolDefinition{Name: "visible", Schema: map[string]any{"type": "object"}}, func(
		context.Context, json.RawMessage,
	) (any, error) {
		return "unused", nil
	})
	if _, err := agent.Run(t.Context(), "secret prompt", struct{}{}); err != nil {
		t.Fatal(err)
	}
	spans := exporter.GetSpans()
	if len(spans) != 2 {
		t.Fatalf("stacked instrumentation emitted duplicate spans: %+v", spans)
	}
	for _, span := range spans {
		attributes := instrumentationSpanAttributes(span.Attributes)
		for _, key := range []string{
			"gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.system_instructions",
			"model_request_parameters", "pydantic_ai.all_messages", "final_result",
		} {
			if _, exists := attributes[key]; exists {
				t.Fatalf("private attribute %q was emitted on %q", key, span.Name)
			}
		}
		if span.Name == "chat private-agent" {
			definitions, _ := attributes["gen_ai.tool.definitions"].(string)
			if !strings.Contains(definitions, "visible") {
				t.Fatalf("tool definition missing with content disabled: %s", definitions)
			}
		}
	}
}

func TestInstrumentationCapabilityErrorsAndEmptyModel(t *testing.T) {
	if (&ai.RunInfo{}).Model() != nil {
		t.Fatal("empty run info returned a model")
	}
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
	requestErr := errors.New("request failed")
	model := requestModel{name: "failed", request: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return nil, requestErr
	}}
	agent := ai.NewAgent[struct{}, string](
		model, ai.WithCapabilities(ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(provider))),
	)
	if _, err := agent.Run(t.Context(), "go", struct{}{}); !errors.Is(err, requestErr) {
		t.Fatalf("unexpected run error: %v", err)
	}
	spans := exporter.GetSpans()
	if len(spans) != 2 {
		t.Fatalf("unexpected error spans: %+v", spans)
	}
	for _, span := range spans {
		if span.Status.Code != codes.Error {
			t.Fatalf("span %q did not record the error: %+v", span.Name, span.Status)
		}
	}

	exporter.Reset()
	instrumentation := ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(provider))
	toolErr := errors.New("tool failed")
	if _, err := instrumentation.WrapToolExecution(
		t.Context(), nil, ai.ToolHookContext{Call: ai.ToolCallPart{ToolName: "failed-tool", ToolCallID: "call"}},
		map[string]any{"input": true},
		func(context.Context, any) (any, error) { return nil, toolErr },
	); !errors.Is(err, toolErr) {
		t.Fatalf("unexpected tool error: %v", err)
	}
	toolSpans := exporter.GetSpans()
	if len(toolSpans) != 1 || toolSpans[0].Status.Code != codes.Error {
		t.Fatalf("tool error was not recorded: %+v", toolSpans)
	}

	exporter.Reset()
	if _, err := instrumentation.WrapToolExecution(
		t.Context(), nil, ai.ToolHookContext{Call: ai.ToolCallPart{ToolName: "binary-tool"}}, nil,
		func(context.Context, any) (any, error) {
			return ai.BinaryContent{Data: []byte("secret"), MediaType: "image/png"}, nil
		},
	); err != nil {
		t.Fatal(err)
	}
	binaryResult, _ := instrumentationSpanAttributes(exporter.GetSpans()[0].Attributes)["gen_ai.tool.call.result"].(string)
	if !strings.Contains(binaryResult, "c2VjcmV0") {
		t.Fatalf("binary tool result was unexpectedly redacted: %s", binaryResult)
	}

	called := false
	response, err := instrumentation.WrapModelRequest(
		t.Context(), &ai.RunInfo{}, nil, ai.ModelRequestParams{},
		func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
			called = true
			return &ai.ModelResponse{}, nil
		},
	)
	if err != nil || response == nil || !called {
		t.Fatalf("empty-model request was not delegated: response=%+v called=%v err=%v", response, called, err)
	}
}

type instrumentationResult struct {
	Answer string `json:"answer"`
}

func TestInstrumentationCapabilityStructuredFinalResult(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "final_result", Args: json.RawMessage(`{"answer":"done"}`), ToolCallID: "final",
		}}}, nil
	})
	agent := ai.NewAgent[struct{}, instrumentationResult](
		model, ai.WithCapabilities(ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(provider))),
	)
	result, err := agent.Run(t.Context(), "go", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.Answer != "done" {
		t.Fatalf("unexpected structured result: %+v", result.Output)
	}
	for _, span := range exporter.GetSpans() {
		if span.Name != "agent run" {
			continue
		}
		final, _ := instrumentationSpanAttributes(span.Attributes)["final_result"].(string)
		if !strings.Contains(final, `"answer":"done"`) {
			t.Fatalf("structured final result missing: %s", final)
		}
		return
	}
	t.Fatal("agent run span missing")
}
