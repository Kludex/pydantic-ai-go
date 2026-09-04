package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sync"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/baggage"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// Instrumentation adds configurable OpenTelemetry spans to agent runs,
// model requests, and local tool execution.
type Instrumentation struct {
	runtime      *InstrumentedModel
	agentName    string
	agentNameSet bool
}

// NewInstrumentation creates an outermost instrumentation capability.
func NewInstrumentation(options ...InstrumentationOption) *Instrumentation {
	runtime, agentName, agentNameSet := newInstrumentationRuntime(options)
	return &Instrumentation{runtime: runtime, agentName: agentName, agentNameSet: agentNameSet}
}

// Setup implements Capability.
func (*Instrumentation) Setup(*CapabilityRegistry) error { return nil }

// CapabilityOrdering keeps telemetry around all other capability middleware.
func (*Instrumentation) CapabilityOrdering() CapabilityOrdering {
	return CapabilityOrdering{Position: CapabilityOutermost}
}

func (*Instrumentation) instrumentsAgent() bool { return true }

type instrumentationRunStateKey struct{}

type instrumentationRunState struct {
	mu               sync.Mutex
	lastInstructions string
	observed         bool
	variable         bool
}

func (state *instrumentationRunState) observe(instructions string) {
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.observed && state.lastInstructions != instructions {
		state.variable = true
	}
	state.lastInstructions = instructions
	state.observed = true
}

func (state *instrumentationRunState) snapshot() (instructions string, observed, variable bool) {
	state.mu.Lock()
	defer state.mu.Unlock()
	return state.lastInstructions, state.observed, state.variable
}

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
	name := info.AgentName()
	if instrumentation.agentNameSet {
		name = instrumentation.agentName
	}
	if name == "" {
		name = "agent"
	}
	modelName := modelName(info.Model())
	names := namesForInstrumentationVersion(instrumentation.runtime.version)
	newMessageIndex := info.newMessages
	attributes := []attribute.KeyValue{
		attribute.String("model_name", modelName),
		attribute.String("agent_name", name),
		attribute.String("gen_ai.operation.name", "invoke_agent"),
		attribute.String("gen_ai.agent.name", name),
		attribute.String("gen_ai.agent.call.id", info.RunID),
		attribute.String("gen_ai.conversation.id", info.ConversationID),
		attribute.String("gen_ai.request.model", modelName),
		attribute.String("logfire.msg", name+" run"),
	}
	if description := info.AgentDescription(); description != "" {
		attributes = append(attributes, attribute.String("gen_ai.agent.description", description))
	}
	ctx, span := instrumentation.runtime.tracer.Start(ctx, names.runSpan(name), trace.WithAttributes(attributes...))
	ctx = context.WithValue(ctx, runSpanContextKey{}, true)
	ctx = context.WithValue(ctx, instrumentationRuntimeContextKey{}, instrumentation.runtime)
	runState := &instrumentationRunState{}
	ctx = context.WithValue(ctx, instrumentationRunStateKey{}, runState)
	ctx = instrumentationBaggage(ctx, name, info.RunID, info.ConversationID)
	defer func() {
		if err != nil {
			span.SetStatus(codes.Error, err.Error())
			span.RecordError(err)
		}
		usage := info.Usage()
		span.SetAttributes(aggregatedUsageAttributes(usage, instrumentation.runtime.useAggregatedUsage)...)
		if selected := info.Model(); !modelIsNil(selected) {
			span.SetAttributes(
				attribute.String("model_name", selected.Name()),
				attribute.String("gen_ai.request.model", selected.Name()),
			)
		}
		messages := info.Messages()
		span.SetAttributes(attribute.String(
			"pydantic_ai.all_messages",
			telemetryMessagesJSON(
				messages, instrumentation.runtime.includeContent, instrumentation.runtime.includeBinaryContent,
				instrumentation.runtime.version,
			),
		))
		if newMessageIndex > 0 {
			span.SetAttributes(attribute.Int("pydantic_ai.new_message_index", newMessageIndex))
		}
		if metadata := info.Metadata(); metadata != nil {
			span.SetAttributes(attribute.String(
				"metadata", telemetryJSON(telemetryOutputValue(metadata, instrumentation.runtime.includeBinaryContent)),
			))
		}
		properties := map[string]any{
			"pydantic_ai.all_messages": map[string]any{"type": "array"},
			"final_result":             map[string]any{"type": "object"},
		}
		instructions, observedInstructions, variableInstructions := runState.snapshot()
		if !observedInstructions {
			instructions = latestTelemetryInstructions(messages)
		}
		if variableInstructions {
			span.SetAttributes(attribute.Bool("pydantic_ai.variable_instructions", true))
			properties["pydantic_ai.variable_instructions"] = map[string]any{}
		}
		if newMessageIndex > 0 {
			properties["pydantic_ai.new_message_index"] = map[string]any{}
		}
		if info.Metadata() != nil {
			properties["metadata"] = map[string]any{"type": "array"}
		}
		if instrumentation.runtime.includeContent {
			if instructions != "" {
				span.SetAttributes(attribute.String("gen_ai.system_instructions", telemetryJSON([]map[string]any{{
					"type": "text", "content": instructions,
				}})))
				properties["gen_ai.system_instructions"] = map[string]any{"type": "array"}
			}
			if err == nil {
				final := outcome.Output
				if outcome.Deferred != nil {
					final = outcome.Deferred.Clone()
				}
				span.SetAttributes(attribute.String(
					"final_result", telemetryFinalResult(final, instrumentation.runtime.includeBinaryContent),
				))
			}
		}
		span.SetAttributes(attribute.String(
			"logfire.json_schema", telemetryJSON(map[string]any{"type": "object", "properties": properties}),
		))
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
	if state, ok := ctx.Value(instrumentationRunStateKey{}).(*instrumentationRunState); ok {
		state.observe(params.Instructions)
	}
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

// OnToolValidationError records a failed tool call after inner capabilities decline recovery.
func (instrumentation *Instrumentation) OnToolValidationError(
	ctx context.Context,
	_ *RunInfo,
	hook ToolHookContext,
	rawArgs json.RawMessage,
	validationErr error,
) (any, error) {
	names := namesForInstrumentationVersion(instrumentation.runtime.version)
	attributes := instrumentation.toolSpanAttributes(ctx, hook.Call, telemetryRawJSON(rawArgs))
	attributes = append(attributes,
		attribute.String("logfire.msg", "invalid tool call: "+hook.Call.ToolName),
		attribute.String("pydantic_ai.tool.failure_stage", "validation"),
	)
	if instrumentation.runtime.includeContent {
		attributes = append(attributes, attribute.String(names.toolResult, validationErr.Error()))
	}
	_, span := instrumentation.runtime.tracer.Start(
		ctx, names.toolSpan(hook.Call.ToolName), trace.WithAttributes(attributes...),
	)
	span.SetStatus(codes.Error, validationErr.Error())
	if instrumentation.runtime.includeContent {
		span.RecordError(validationErr)
	} else {
		span.AddEvent("exception", trace.WithAttributes(
			attribute.String("exception.type", fmt.Sprintf("%T", validationErr)),
			attribute.String("exception.escaped", "true"),
		))
	}
	span.End()
	return nil, validationErr
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
	names := namesForInstrumentationVersion(instrumentation.runtime.version)
	ctx, span := instrumentation.runtime.tracer.Start(
		ctx, names.toolSpan(hook.Call.ToolName),
		trace.WithAttributes(instrumentation.toolSpanAttributes(ctx, hook.Call, telemetryValue(args))...),
	)
	ctx = context.WithValue(ctx, toolSpanContextKey{}, true)
	defer func() {
		if err != nil {
			span.SetStatus(codes.Error, err.Error())
			if instrumentation.runtime.includeContent {
				var retry *RetryError
				var failed *ToolFailedError
				switch {
				case errors.As(err, &retry):
					span.SetAttributes(attribute.String(names.toolResult, retry.Message))
				case errors.As(err, &failed):
					span.SetAttributes(attribute.String(names.toolResult, failed.Message))
				}
			}
			span.RecordError(err)
		} else if deferralName, metadata, deferred := telemetryToolDeferral(result); deferred {
			span.SetAttributes(attribute.String("pydantic_ai.tool.deferral.name", deferralName))
			if instrumentation.runtime.includeContent && metadata != nil {
				span.SetAttributes(attribute.String(
					"pydantic_ai.tool.deferral.metadata",
					telemetryJSON(telemetryOutputValue(metadata, instrumentation.runtime.includeBinaryContent)),
				))
			}
			if instrumentation.runtime.version < 5 {
				span.SetStatus(codes.Error, deferralName)
				span.AddEvent("exception", trace.WithAttributes(
					attribute.String("exception.type", deferralName),
					attribute.String("exception.escaped", "true"),
				))
			}
		} else if instrumentation.runtime.includeContent {
			span.SetAttributes(attribute.String(
				names.toolResult,
				telemetryJSON(telemetryOutputValue(result, instrumentation.runtime.includeBinaryContent)),
			))
		}
		span.End()
	}()
	return next(ctx, args)
}

func (instrumentation *Instrumentation) toolSpanAttributes(
	ctx context.Context, call ToolCallPart, arguments any,
) []attribute.KeyValue {
	names := namesForInstrumentationVersion(instrumentation.runtime.version)
	attributes := []attribute.KeyValue{
		attribute.String("gen_ai.operation.name", "execute_tool"),
		attribute.String("gen_ai.tool.name", call.ToolName),
		attribute.String("gen_ai.tool.call.id", call.ToolCallID),
		attribute.String("logfire.msg", "running tool: "+call.ToolName),
	}
	attributes = append(attributes, instrumentationBaggageAttributes(ctx)...)
	properties := map[string]any{
		"gen_ai.tool.name":    map[string]any{},
		"gen_ai.tool.call.id": map[string]any{},
	}
	if instrumentation.runtime.includeContent {
		attributes = append(attributes, attribute.String(names.toolArguments, telemetryJSON(arguments)))
		properties[names.toolArguments] = map[string]any{"type": "object"}
		properties[names.toolResult] = map[string]any{"type": "object"}
	}
	return append(attributes, attribute.String(
		"logfire.json_schema", telemetryJSON(map[string]any{"type": "object", "properties": properties}),
	))
}

// WrapOutputProcessing records user output-function arguments and results.
func (instrumentation *Instrumentation) WrapOutputProcessing(
	ctx context.Context,
	_ *RunInfo,
	hook OutputHookContext,
	output any,
	next OutputProcessingFunc,
) (result any, err error) {
	if !hook.HasFunction || outputFunctionSpanActive(ctx) {
		return next(ctx, output)
	}
	target := hook.FunctionName
	if hook.ToolCall != nil {
		target = hook.ToolCall.ToolName
	}
	if target == "" {
		target = "output_function"
	}
	names := namesForInstrumentationVersion(instrumentation.runtime.version)
	attributes := []attribute.KeyValue{
		attribute.String("gen_ai.operation.name", "execute_tool"),
		attribute.String("gen_ai.tool.name", target),
		attribute.String("logfire.msg", "running output function: "+target),
	}
	attributes = append(attributes, instrumentationBaggageAttributes(ctx)...)
	if hook.ToolCall != nil && hook.ToolCall.ToolCallID != "" {
		attributes = append(attributes, attribute.String("gen_ai.tool.call.id", hook.ToolCall.ToolCallID))
	}
	if instrumentation.runtime.includeContent {
		attributes = append(attributes, attribute.String(
			names.toolArguments,
			telemetryJSON(telemetryOutputValue(output, instrumentation.runtime.includeBinaryContent)),
		))
	}
	properties := map[string]any{"gen_ai.tool.name": map[string]any{}}
	if instrumentation.runtime.includeContent {
		properties[names.toolArguments] = map[string]any{"type": "object"}
		properties[names.toolResult] = map[string]any{"type": "object"}
	}
	if hook.ToolCall != nil && hook.ToolCall.ToolCallID != "" {
		properties["gen_ai.tool.call.id"] = map[string]any{}
	}
	attributes = append(attributes, attribute.String(
		"logfire.json_schema", telemetryJSON(map[string]any{"type": "object", "properties": properties}),
	))
	ctx, span := instrumentation.runtime.tracer.Start(
		ctx, names.outputFunctionSpan(target), trace.WithAttributes(attributes...),
	)
	ctx = context.WithValue(ctx, outputFunctionSpanContextKey{}, true)
	defer func() {
		if err != nil {
			span.SetStatus(codes.Error, err.Error())
			span.RecordError(err)
		} else if instrumentation.runtime.includeContent {
			span.SetAttributes(attribute.String(
				names.toolResult,
				telemetryJSON(telemetryOutputValue(result, instrumentation.runtime.includeBinaryContent)),
			))
		}
		span.End()
	}()
	return next(ctx, output)
}

func instrumentationBaggageAttributes(ctx context.Context) []attribute.KeyValue {
	current := baggage.FromContext(ctx)
	attributes := make([]attribute.KeyValue, 0, 3)
	for _, key := range []string{"gen_ai.agent.name", "gen_ai.agent.call.id", "gen_ai.conversation.id"} {
		if value := current.Member(key).Value(); value != "" {
			attributes = append(attributes, attribute.String(key, value))
		}
	}
	return attributes
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

func aggregatedUsageAttributes(usage Usage, aggregate bool) []attribute.KeyValue {
	prefix := "gen_ai.usage."
	if aggregate {
		prefix = "gen_ai.aggregated_usage."
	}
	attributes := append(usageTelemetryAttributes(usage, prefix),
		attribute.Int("pydantic_ai.requests", usage.Requests),
		attribute.Int("pydantic_ai.tool_calls", usage.ToolCalls),
	)
	if usage.CostUSD != nil {
		attributes = append(attributes, attribute.Float64("operation.cost", *usage.CostUSD))
	}
	return attributes
}

func latestTelemetryInstructions(messages []ModelMessage) string {
	for index := len(messages) - 1; index >= 0; index-- {
		request, ok := messages[index].(ModelRequest)
		if ok && request.Instructions != "" {
			return request.Instructions
		}
	}
	return ""
}

func telemetryToolDeferral(result any) (string, map[string]any, bool) {
	switch result := result.(type) {
	case ExternalToolRequest:
		return "CallDeferred", result.Metadata, true
	case *ExternalToolRequest:
		if result != nil {
			return "CallDeferred", result.Metadata, true
		}
	case ToolApprovalRequest:
		return "ApprovalRequired", result.Metadata, true
	case *ToolApprovalRequest:
		if result != nil {
			return "ApprovalRequired", result.Metadata, true
		}
	}
	return "", nil, false
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
		content := make([]UserContent, 0, len(value.Content))
		for _, item := range cloneUserContents(value.Content) {
			if _, ok := item.(CachePoint); ok {
				continue
			}
			content = append(content, telemetryOutputUserContent(item, includeBinary))
		}
		value.Content = content
		return value
	case DeferredToolRequests:
		return map[string]any{
			"calls": value.Calls, "approvals": value.Approvals,
			"metadata": telemetryOutputValue(value.Metadata, includeBinary),
		}
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Array, reflect.Slice:
		if reflected.Type().Elem().Kind() == reflect.Uint8 {
			return value
		}
		if reflected.Kind() == reflect.Slice && reflected.IsNil() {
			return value
		}
		result := make([]any, reflected.Len())
		for index := range reflected.Len() {
			result[index] = telemetryOutputValue(reflected.Index(index).Interface(), includeBinary)
		}
		return result
	case reflect.Map:
		if reflected.Type().Key().Kind() != reflect.String || reflected.IsNil() {
			return value
		}
		result := make(map[string]any, reflected.Len())
		iterator := reflected.MapRange()
		for iterator.Next() {
			result[iterator.Key().String()] = telemetryOutputValue(iterator.Value().Interface(), includeBinary)
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
	_ ToolValidationErrorHook    = (*Instrumentation)(nil)
	_ ToolExecutionWrapper       = (*Instrumentation)(nil)
)
