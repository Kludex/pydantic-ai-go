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
	"go.opentelemetry.io/otel/trace"
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
	agent := ai.NewAgent[struct{}, string](model,
		ai.WithAgentName("application-support"),
		ai.WithMetadata(map[string]any{
			"tenant": "acme", "attachment": ai.BinaryContent{Data: []byte("secret"), MediaType: "image/png"},
		}),
		ai.WithCapabilities(
			instrumentation,
			ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(tracerProvider)),
		),
	)
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
				ai.TextContent{Text: "attachment"}, ai.CachePoint{},
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
		"invoke_agent support": 1, "chat function-model": 2, "execute_tool lookup": 1,
	} {
		if counts[name] != want {
			t.Fatalf("span %q count = %d, want %d: %+v", name, counts[name], want, counts)
		}
	}
	var runAttributes map[string]any
	var toolAttributes map[string]any
	for _, span := range spans {
		switch span.Name {
		case "invoke_agent support":
			runAttributes = instrumentationSpanAttributes(span.Attributes)
		case "execute_tool lookup":
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
	metadata, _ := runAttributes["metadata"].(string)
	if !strings.Contains(metadata, "acme") || !strings.Contains(metadata, "image/png") || strings.Contains(metadata, "secret") {
		t.Fatalf("run metadata was not recorded safely: %s", metadata)
	}
	toolResult, _ := toolAttributes["gen_ai.tool.call.result"].(string)
	if !strings.Contains(toolResult, "image/png") || !strings.Contains(toolResult, "audio/wav") ||
		strings.Contains(toolResult, "c2VjcmV0") {
		t.Fatalf("tool result binary data was not redacted: %s", toolResult)
	}
	if args, _ := toolAttributes["gen_ai.tool.call.arguments"].(string); !strings.Contains(args, "go") {
		t.Fatalf("tool arguments missing: %s", args)
	}
	if toolAttributes["logfire.msg"] != "running tool: lookup" ||
		toolAttributes["gen_ai.agent.name"] != "support" ||
		!strings.Contains(toolAttributes["logfire.json_schema"].(string), "gen_ai.tool.call.result") {
		t.Fatalf("tool Logfire attributes are incomplete: %+v", toolAttributes)
	}

	var metrics metricdata.ResourceMetrics
	if err := reader.Collect(t.Context(), &metrics); err != nil {
		t.Fatal(err)
	}
	if len(metrics.ScopeMetrics) == 0 {
		t.Fatal("instrumentation capability did not record model metrics")
	}
}

func TestInstrumentationCapabilityUsageNamesAndLegacySpans(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
	cost := 0.25
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		for _, part := range messages[len(messages)-1].(ai.ModelRequest).Parts {
			if _, ok := part.(ai.ToolReturnPart); ok {
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
			}
		}
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "lookup", ToolCallID: "call", Args: json.RawMessage(`{}`)}},
			Usage: ai.Usage{
				InputTokens: 10, OutputTokens: 5, CacheWriteTokens: 2, CacheReadTokens: 3,
				InputAudioTokens: 4, CacheAudioReadTokens: 1, OutputAudioTokens: 2, ReasoningTokens: 3,
				AcceptedPredictionTokens: 1, RejectedPredictionTokens: 2,
				Details: map[string]int{"input_tokens": 999, "provider_detail": 7}, CostUSD: &cost,
			},
		}, nil
	})
	instrumentation := ai.NewInstrumentation(
		ai.WithInstrumentationTracerProvider(provider),
		ai.WithInstrumentationVersion(2),
		ai.WithInstrumentationAggregatedUsageAttributeNames(false),
	)
	agent := ai.NewAgent[struct{}, string](
		model, ai.WithCapabilities(instrumentation), ai.WithInstructions("Use concise answers."),
	)
	ai.AddSimpleTool(agent, "lookup", func(context.Context, struct{}) (string, error) { return "found", nil })
	result, err := agent.Run(
		t.Context(), "go", struct{}{}, ai.WithMessageHistory([]ai.ModelMessage{ai.ModelRequest{
			Parts: []ai.RequestPart{ai.UserPromptPart{Content: "history"}},
		}}),
	)
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected result: %+v err=%v", result, err)
	}
	spans := exporter.GetSpans()
	counts := map[string]int{}
	var runAttributes, toolAttributes map[string]any
	for _, span := range spans {
		counts[span.Name]++
		switch span.Name {
		case "agent run":
			runAttributes = instrumentationSpanAttributes(span.Attributes)
		case "running tool":
			toolAttributes = instrumentationSpanAttributes(span.Attributes)
		}
	}
	if counts["agent run"] != 1 || counts["running tool"] != 1 {
		t.Fatalf("legacy span names missing: %+v", counts)
	}
	if _, exists := toolAttributes["gen_ai.tool.call.arguments"]; exists {
		t.Fatalf("modern tool argument attribute used in v2: %+v", toolAttributes)
	}
	if _, exists := toolAttributes["tool_arguments"]; !exists {
		t.Fatalf("legacy tool argument attribute missing: %+v", toolAttributes)
	}
	for key, want := range map[string]any{
		"gen_ai.usage.input_tokens":                       int64(10),
		"gen_ai.usage.output_tokens":                      int64(5),
		"gen_ai.usage.cache_creation.input_tokens":        int64(2),
		"gen_ai.usage.cache_read.input_tokens":            int64(3),
		"gen_ai.usage.details.input_audio_tokens":         int64(4),
		"gen_ai.usage.details.cache_audio_read_tokens":    int64(1),
		"gen_ai.usage.details.output_audio_tokens":        int64(2),
		"gen_ai.usage.details.reasoning_tokens":           int64(3),
		"gen_ai.usage.details.accepted_prediction_tokens": int64(1),
		"gen_ai.usage.details.rejected_prediction_tokens": int64(2),
		"gen_ai.usage.details.provider_detail":            int64(7),
		"pydantic_ai.new_message_index":                   int64(1),
	} {
		if got := runAttributes[key]; got != want {
			t.Fatalf("run attribute %q = %#v, want %#v", key, got, want)
		}
	}
	if _, exists := runAttributes["gen_ai.aggregated_usage.input_tokens"]; exists {
		t.Fatalf("aggregated usage was not disabled: %+v", runAttributes)
	}
	if _, exists := runAttributes["gen_ai.usage.details.input_tokens"]; exists {
		t.Fatalf("first-class input tokens were duplicated: %+v", runAttributes)
	}
	if instructions, _ := runAttributes["gen_ai.system_instructions"].(string); !strings.Contains(instructions, "Use concise answers.") {
		t.Fatalf("run instructions missing: %s", instructions)
	}
}

func TestInstrumentationCapabilityDeferredMetadata(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "queue", ToolCallID: "call", Args: json.RawMessage(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(ai.NewInstrumentation(
		ai.WithInstrumentationTracerProvider(provider), ai.WithInstrumentationBinaryContent(false),
	)))
	ai.AddSimpleTool(agent, "queue", func(context.Context, struct{}) (ai.ExternalToolRequest, error) {
		return ai.RequestExternalToolExecution(map[string]any{
			"task_id": "task-123", "attachment": ai.BinaryContent{Data: []byte("secret"), MediaType: "image/png"},
		}), nil
	}, ai.WithDynamicExternalExecution())
	result, err := agent.Run(t.Context(), "go", struct{}{})
	if err != nil || result.Deferred() == nil {
		t.Fatalf("run did not defer: result=%+v err=%v", result, err)
	}

	var runAttributes, toolAttributes map[string]any
	var toolStatus codes.Code
	for _, span := range exporter.GetSpans() {
		switch span.Name {
		case "invoke_agent agent":
			runAttributes = instrumentationSpanAttributes(span.Attributes)
		case "execute_tool queue":
			toolAttributes = instrumentationSpanAttributes(span.Attributes)
			toolStatus = span.Status.Code
		}
	}
	if toolStatus != codes.Unset || toolAttributes["pydantic_ai.tool.deferral.name"] != "CallDeferred" {
		t.Fatalf("unexpected deferred tool telemetry: status=%v attributes=%+v", toolStatus, toolAttributes)
	}
	metadata, _ := toolAttributes["pydantic_ai.tool.deferral.metadata"].(string)
	if !strings.Contains(metadata, "task-123") || !strings.Contains(metadata, "image/png") ||
		strings.Contains(metadata, "secret") {
		t.Fatalf("deferred metadata was not safely recorded: %s", metadata)
	}
	if _, exists := toolAttributes["gen_ai.tool.call.result"]; exists {
		t.Fatalf("deferral was recorded as a tool result: %+v", toolAttributes)
	}
	final, _ := runAttributes["final_result"].(string)
	if !strings.Contains(final, "queue") || !strings.Contains(final, "call") ||
		strings.Contains(final, "secret") || strings.Contains(final, "c2VjcmV0") {
		t.Fatalf("deferred final result missing or unsafe: %s", final)
	}

	exporter.Reset()
	legacy := ai.NewInstrumentation(
		ai.WithInstrumentationTracerProvider(provider), ai.WithInstrumentationVersion(2),
	)
	approval := ai.RequestToolApproval(nil)
	if _, err := legacy.WrapToolExecution(
		t.Context(), nil, ai.ToolHookContext{Call: ai.ToolCallPart{ToolName: "approve", ToolCallID: "approval"}}, nil,
		func(context.Context, any) (any, error) { return &approval, nil },
	); err != nil {
		t.Fatal(err)
	}
	span := exporter.GetSpans()[0]
	attributes := instrumentationSpanAttributes(span.Attributes)
	if span.Name != "running tool" || span.Status.Code != codes.Error ||
		attributes["pydantic_ai.tool.deferral.name"] != "ApprovalRequired" || len(span.Events) != 1 {
		t.Fatalf("unexpected legacy deferral telemetry: %+v attributes=%+v", span, attributes)
	}

	for _, deferred := range []any{ai.RequestToolApproval(nil), func() *ai.ExternalToolRequest {
		request := ai.RequestExternalToolExecution(nil)
		return &request
	}()} {
		if _, err := legacy.WrapToolExecution(
			t.Context(), nil, ai.ToolHookContext{Call: ai.ToolCallPart{ToolName: "defer"}}, nil,
			func(context.Context, any) (any, error) { return deferred, nil },
		); err != nil {
			t.Fatal(err)
		}
	}

	exporter.Reset()
	redacting := ai.NewInstrumentation(
		ai.WithInstrumentationTracerProvider(provider), ai.WithInstrumentationBinaryContent(false),
	)
	if _, err := redacting.WrapToolExecution(
		t.Context(), nil, ai.ToolHookContext{Call: ai.ToolCallPart{ToolName: "nested"}}, nil,
		func(context.Context, any) (any, error) {
			return map[string]any{
				"slice": []ai.BinaryContent{{Data: []byte("secret"), MediaType: "image/png"}},
				"array": [1]ai.BinaryContent{{Data: []byte("secret"), MediaType: "audio/wav"}},
				"bytes": []byte("plain"), "nil_slice": []string(nil),
				"nil_map": map[string]string(nil), "number_map": map[int]string{1: "one"},
			}, nil
		},
	); err != nil {
		t.Fatal(err)
	}
	nested, _ := instrumentationSpanAttributes(exporter.GetSpans()[0].Attributes)["gen_ai.tool.call.result"].(string)
	if strings.Contains(nested, "secret") || strings.Contains(nested, "c2VjcmV0") ||
		!strings.Contains(nested, "image/png") || !strings.Contains(nested, "audio/wav") ||
		!strings.Contains(nested, "cGxhaW4=") || !strings.Contains(nested, `"1":"one"`) {
		t.Fatalf("nested tool result was not redacted safely: %s", nested)
	}
}

func TestInstrumentationCapabilityToolValidationFailure(t *testing.T) {
	for _, includeContent := range []bool{true, false} {
		t.Run(map[bool]string{true: "content", false: "redacted"}[includeContent], func(t *testing.T) {
			exporter := tracetest.NewInMemoryExporter()
			provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
			t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
			requests := 0
			model := fakes.NewFunctionModel(func(
				context.Context, []ai.ModelMessage, ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				requests++
				if requests == 1 {
					return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
						ToolName: "lookup", ToolCallID: "invalid", Args: json.RawMessage(`{"query":42}`),
					}}}, nil
				}
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
			})
			agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(ai.NewInstrumentation(
				ai.WithInstrumentationTracerProvider(provider), ai.WithInstrumentationContent(includeContent),
			)))
			called := false
			ai.AddSimpleTool(agent, "lookup", func(context.Context, struct {
				Query string `json:"query"`
			}) (string, error) {
				called = true
				return "unexpected", nil
			})
			result, err := agent.Run(t.Context(), "go", struct{}{})
			if err != nil || result.Output != "done" || called {
				t.Fatalf("unexpected validation run: result=%+v called=%v err=%v", result, called, err)
			}
			var validation *tracetest.SpanStub
			for index := range exporter.GetSpans() {
				span := exporter.GetSpans()[index]
				attributes := instrumentationSpanAttributes(span.Attributes)
				if attributes["pydantic_ai.tool.failure_stage"] == "validation" {
					validation = &span
					break
				}
			}
			if validation == nil || validation.Name != "execute_tool lookup" || validation.Status.Code != codes.Error ||
				len(validation.Events) != 1 {
				t.Fatalf("validation failure span missing: %+v", exporter.GetSpans())
			}
			attributes := instrumentationSpanAttributes(validation.Attributes)
			if attributes["logfire.msg"] != "invalid tool call: lookup" ||
				!strings.Contains(attributes["logfire.json_schema"].(string), "gen_ai.tool.name") {
				t.Fatalf("validation Logfire attributes are incomplete: %+v", attributes)
			}
			arguments, argumentsPresent := attributes["gen_ai.tool.call.arguments"].(string)
			resultText, resultPresent := attributes["gen_ai.tool.call.result"].(string)
			if includeContent {
				if !argumentsPresent || !strings.Contains(arguments, `"query":42`) ||
					!resultPresent || resultText == "" {
					t.Fatalf("validation content missing: %+v", attributes)
				}
			} else if argumentsPresent || resultPresent {
				t.Fatalf("validation content leaked: span=%+v attributes=%+v", validation, attributes)
			}
		})
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
		for _, key := range []string{"gen_ai.system_instructions", "model_request_parameters", "final_result"} {
			if _, exists := attributes[key]; exists {
				t.Fatalf("private attribute %q was emitted on %q", key, span.Name)
			}
		}
		for _, key := range []string{"gen_ai.input.messages", "gen_ai.output.messages", "pydantic_ai.all_messages"} {
			if messages, _ := attributes[key].(string); strings.Contains(messages, "secret") {
				t.Fatalf("private content was emitted in %q on %q: %s", key, span.Name, messages)
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

	for _, test := range []struct {
		name string
		err  error
		want string
	}{
		{name: "retry", err: ai.Retryf("try again"), want: "try again"},
		{name: "failed", err: ai.ToolFailedf("not available"), want: "not available"},
	} {
		exporter.Reset()
		if _, err := instrumentation.WrapToolExecution(
			t.Context(), nil, ai.ToolHookContext{Call: ai.ToolCallPart{ToolName: test.name}}, nil,
			func(context.Context, any) (any, error) { return nil, test.err },
		); !errors.Is(err, test.err) {
			t.Fatalf("unexpected %s error: %v", test.name, err)
		}
		attributes := instrumentationSpanAttributes(exporter.GetSpans()[0].Attributes)
		if attributes["gen_ai.tool.call.result"] != test.want {
			t.Fatalf("%s model-visible result missing: %+v", test.name, attributes)
		}
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

func TestInstrumentationCapabilityOutputFunctionSpan(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
	instrumentation := ai.NewInstrumentation(
		ai.WithInstrumentationTracerProvider(provider), ai.WithInstrumentationAgentName("classifier"),
	)
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "normalize", ToolCallID: "final", Args: json.RawMessage(`{"city":"Paris"}`),
		}}}, nil
	})
	var activeSpan trace.SpanContext
	output := ai.NewOutputFunction("normalize", func(
		ctx context.Context, _ *ai.RunContext[struct{}], input outputFunctionInput,
	) (string, error) {
		activeSpan = trace.SpanContextFromContext(ctx)
		return strings.ToUpper(input.City), nil
	})
	agent := ai.NewOutputFunctionAgent(model, output, ai.WithCapabilities(
		instrumentation,
		ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(provider)),
	))
	result, err := agent.Run(t.Context(), "city", struct{}{})
	if err != nil || result.Output != "PARIS" {
		t.Fatalf("unexpected output function result: %+v err=%v", result, err)
	}

	spans := exporter.GetSpans()
	var runSpan, outputSpan *tracetest.SpanStub
	for index := range spans {
		span := &spans[index]
		switch span.Name {
		case "invoke_agent classifier":
			runSpan = span
		case "execute_tool normalize":
			outputSpan = span
		}
	}
	if runSpan == nil || outputSpan == nil || len(spans) != 3 {
		t.Fatalf("expected one run, request, and output function span: %+v", spans)
	}
	if outputSpan.Parent.SpanID() != runSpan.SpanContext.SpanID() || activeSpan.SpanID() != outputSpan.SpanContext.SpanID() {
		t.Fatalf("output function span hierarchy is wrong: run=%+v output=%+v active=%+v", runSpan, outputSpan, activeSpan)
	}
	attributes := instrumentationSpanAttributes(outputSpan.Attributes)
	for key, want := range map[string]any{
		"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "normalize",
		"gen_ai.tool.call.id": "final", "gen_ai.agent.name": "classifier",
		"logfire.msg": "running output function: normalize",
	} {
		if got := attributes[key]; got != want {
			t.Fatalf("output function attribute %q = %#v, want %#v", key, got, want)
		}
	}
	if arguments, _ := attributes["gen_ai.tool.call.arguments"].(string); !strings.Contains(arguments, "Paris") {
		t.Fatalf("output function arguments missing: %+v", attributes)
	}
	if value, _ := attributes["gen_ai.tool.call.result"].(string); value != `"PARIS"` {
		t.Fatalf("output function result missing: %+v", attributes)
	}
	if schema, _ := attributes["logfire.json_schema"].(string); !strings.Contains(schema, "gen_ai.tool.call.result") {
		t.Fatalf("output function log schema missing: %+v", attributes)
	}
}

func TestInstrumentationCapabilityOutputFunctionLegacyPrivacyAndErrors(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
	instrumentation := ai.NewInstrumentation(
		ai.WithInstrumentationTracerProvider(provider), ai.WithInstrumentationVersion(2),
		ai.WithInstrumentationContent(false),
	)
	hook := ai.OutputHookContext{HasFunction: true, FunctionName: "normalize"}
	boom := errors.New("output function failed")
	if _, err := instrumentation.WrapOutputProcessing(
		t.Context(), nil, hook, map[string]any{"secret": true},
		func(context.Context, any) (any, error) { return nil, boom },
	); !errors.Is(err, boom) {
		t.Fatalf("unexpected output function error: %v", err)
	}
	spans := exporter.GetSpans()
	if len(spans) != 1 || spans[0].Name != "running output function" || spans[0].Status.Code != codes.Error ||
		len(spans[0].Events) != 1 {
		t.Fatalf("unexpected legacy output function span: %+v", spans)
	}
	attributes := instrumentationSpanAttributes(spans[0].Attributes)
	if _, exists := attributes["tool_arguments"]; exists {
		t.Fatalf("private output function arguments leaked: %+v", attributes)
	}
	if _, exists := attributes["tool_response"]; exists {
		t.Fatalf("private output function result leaked: %+v", attributes)
	}
	if schema, _ := attributes["logfire.json_schema"].(string); strings.Contains(schema, "tool_arguments") {
		t.Fatalf("private output schema exposed content fields: %s", schema)
	}

	exporter.Reset()
	result, err := instrumentation.WrapOutputProcessing(
		t.Context(), nil, ai.OutputHookContext{HasFunction: true}, "input",
		func(context.Context, any) (any, error) { return "done", nil },
	)
	spans = exporter.GetSpans()
	if err != nil || result != "done" || len(spans) != 1 ||
		instrumentationSpanAttributes(spans[0].Attributes)["gen_ai.tool.name"] != "output_function" {
		t.Fatalf("generic output function span missing: result=%v err=%v spans=%+v", result, err, spans)
	}

	exporter.Reset()
	result, err = instrumentation.WrapOutputProcessing(
		t.Context(), nil, ai.OutputHookContext{}, "plain",
		func(context.Context, any) (any, error) { return "done", nil },
	)
	if err != nil || result != "done" || len(exporter.GetSpans()) != 0 {
		t.Fatalf("plain output unexpectedly emitted a span: result=%v err=%v spans=%+v", result, err, exporter.GetSpans())
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
		if span.Name != "invoke_agent agent" {
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
