package ai

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/baggage"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// Instrumentation adds configurable OpenTelemetry spans to agent runs,
// model requests, and local tool execution.
type Instrumentation struct {
	runtime   *InstrumentedModel
	agentName string
}

// NewInstrumentation creates an outermost instrumentation capability.
func NewInstrumentation(options ...InstrumentationOption) *Instrumentation {
	runtime, agentName := newInstrumentationRuntime(options)
	return &Instrumentation{runtime: runtime, agentName: agentName}
}

// Setup implements Capability.
func (*Instrumentation) Setup(*CapabilityRegistry) error { return nil }

// CapabilityOrdering keeps telemetry around all other capability middleware.
func (*Instrumentation) CapabilityOrdering() CapabilityOrdering {
	return CapabilityOrdering{Position: CapabilityOutermost}
}

func (*Instrumentation) instrumentsAgent() bool { return true }

func hasInstrumentationCapability(capabilities []Capability) bool {
	for _, capability := range capabilities {
		if instrumentation, ok := capability.(interface{ instrumentsAgent() bool }); ok && instrumentation.instrumentsAgent() {
			return true
		}
	}
	return false
}

// WrapRun records one invoke-agent span around the complete run.
func (instrumentation *Instrumentation) WrapRun(
	ctx context.Context, info *RunInfo, next RunFunc,
) (outcome RunOutcome, err error) {
	if runSpanActive(ctx) {
		return next(ctx)
	}
	name := instrumentation.agentName
	if name == "" {
		name = "agent"
	}
	modelName := modelName(info.Model())
	ctx, span := instrumentation.runtime.tracer.Start(ctx, name+" run", trace.WithAttributes(
		attribute.String("gen_ai.operation.name", "invoke_agent"),
		attribute.String("gen_ai.agent.name", name),
		attribute.String("gen_ai.agent.call.id", info.RunID),
		attribute.String("gen_ai.conversation.id", info.ConversationID),
		attribute.String("gen_ai.request.model", modelName),
	))
	ctx = context.WithValue(ctx, runSpanContextKey{}, true)
	ctx = instrumentationBaggage(ctx, name, info.RunID, info.ConversationID)
	defer func() {
		if err != nil {
			span.SetStatus(codes.Error, err.Error())
			span.RecordError(err)
		}
		usage := info.Usage()
		span.SetAttributes(aggregatedUsageAttributes(usage)...)
		if selected := info.Model(); !modelIsNil(selected) {
			span.SetAttributes(attribute.String("gen_ai.request.model", selected.Name()))
		}
		if instrumentation.runtime.includeContent {
			span.SetAttributes(attribute.String(
				"pydantic_ai.all_messages",
				telemetryMessagesJSON(info.Messages(), instrumentation.runtime.includeBinaryContent),
			))
			if err == nil && outcome.Deferred == nil {
				span.SetAttributes(attribute.String(
					"final_result",
					telemetryFinalResult(outcome.Output, instrumentation.runtime.includeBinaryContent),
				))
			}
		}
		span.End()
	}()
	return next(ctx)
}

// WrapModelRequest records a client span around one logical model request.
func (instrumentation *Instrumentation) WrapModelRequest(
	ctx context.Context,
	info *RunInfo,
	messages []ModelMessage,
	params ModelRequestParams,
	next ModelRequestFunc,
) (*ModelResponse, error) {
	if modelRequestSpanActive(ctx) {
		return next(ctx, messages, params)
	}
	model := info.Model()
	if modelIsNil(model) {
		return next(ctx, messages, params)
	}
	runtime := *instrumentation.runtime
	runtime.ModelWrapper = WrapModel(model)
	spanCtx, request := runtime.startRequest(ctx, messages, params)
	response, err := next(spanCtx, messages, params)
	request.finish(spanCtx, response, err, 0)
	return response, err
}

// WrapToolExecution records local function-tool arguments and results.
func (instrumentation *Instrumentation) WrapToolExecution(
	ctx context.Context,
	_ *RunInfo,
	hook ToolHookContext,
	args any,
	next ToolExecutionFunc,
) (result any, err error) {
	if toolSpanActive(ctx) {
		return next(ctx, args)
	}
	ctx, span := instrumentation.runtime.tracer.Start(ctx, "running tool: "+hook.Call.ToolName, trace.WithAttributes(
		attribute.String("gen_ai.operation.name", "execute_tool"),
		attribute.String("gen_ai.tool.name", hook.Call.ToolName),
		attribute.String("gen_ai.tool.call.id", hook.Call.ToolCallID),
	))
	ctx = context.WithValue(ctx, toolSpanContextKey{}, true)
	if instrumentation.runtime.includeContent {
		span.SetAttributes(attribute.String("gen_ai.tool.call.arguments", telemetryJSON(telemetryValue(args))))
	}
	defer func() {
		if err != nil {
			span.SetStatus(codes.Error, err.Error())
			span.RecordError(err)
		} else if instrumentation.runtime.includeContent {
			span.SetAttributes(attribute.String(
				"gen_ai.tool.call.result",
				telemetryJSON(telemetryOutputValue(result, instrumentation.runtime.includeBinaryContent)),
			))
		}
		span.End()
	}()
	return next(ctx, args)
}

func instrumentationBaggage(ctx context.Context, name, runID, conversationID string) context.Context {
	values := []struct{ key, value string }{
		{key: "gen_ai.agent.name", value: name},
		{key: "gen_ai.agent.call.id", value: runID},
		{key: "gen_ai.conversation.id", value: conversationID},
	}
	current := baggage.FromContext(ctx)
	for _, value := range values {
		member, _ := baggage.NewMember(value.key, value.value)
		current, _ = current.SetMember(member)
	}
	return baggage.ContextWithBaggage(ctx, current)
}

func aggregatedUsageAttributes(usage Usage) []attribute.KeyValue {
	attributes := []attribute.KeyValue{
		attribute.Int("gen_ai.aggregated_usage.input_tokens", usage.InputTokens),
		attribute.Int("gen_ai.aggregated_usage.output_tokens", usage.OutputTokens),
		attribute.Int("pydantic_ai.requests", usage.Requests),
		attribute.Int("pydantic_ai.tool_calls", usage.ToolCalls),
	}
	if usage.CostUSD != nil {
		attributes = append(attributes, attribute.Float64("operation.cost", *usage.CostUSD))
	}
	for key, value := range usage.Details {
		attributes = append(attributes, attribute.Int("gen_ai.aggregated_usage.details."+key, value))
	}
	return attributes
}

func telemetryFinalResult(value any, includeBinary bool) string {
	if text, ok := value.(string); ok {
		return text
	}
	return telemetryJSON(telemetryOutputValue(value, includeBinary))
}

func telemetryOutputValue(value any, includeBinary bool) any {
	switch value := value.(type) {
	case BinaryContent:
		result := map[string]any{"media_type": value.MediaType}
		if includeBinary {
			result["data"] = value.Data
		}
		return result
	case ToolReturn:
		value.ReturnValue = telemetryOutputValue(value.ReturnValue, includeBinary)
		value.Content = cloneUserContents(value.Content)
		for index, content := range value.Content {
			value.Content[index] = telemetryOutputUserContent(content, includeBinary)
		}
		return value
	case []any:
		result := make([]any, len(value))
		for index, item := range value {
			result[index] = telemetryOutputValue(item, includeBinary)
		}
		return result
	case map[string]any:
		result := make(map[string]any, len(value))
		for key, item := range value {
			result[key] = telemetryOutputValue(item, includeBinary)
		}
		return result
	default:
		return value
	}
}

func telemetryOutputUserContent(content UserContent, includeBinary bool) UserContent {
	binary, ok := content.(BinaryContent)
	if ok && !includeBinary {
		binary.Data = nil
		return binary
	}
	return content
}

var (
	_ Capability                 = (*Instrumentation)(nil)
	_ CapabilityOrderingProvider = (*Instrumentation)(nil)
	_ RunWrapper                 = (*Instrumentation)(nil)
	_ ModelRequestWrapper        = (*Instrumentation)(nil)
	_ ToolExecutionWrapper       = (*Instrumentation)(nil)
)
