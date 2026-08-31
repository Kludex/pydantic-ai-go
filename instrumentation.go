package ai

import (
	"context"
	"iter"
	"sync"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
)

// InstrumentationOption configures model request telemetry.
type InstrumentationOption func(*instrumentationConfig)

type instrumentationConfig struct {
	tracerProvider                trace.TracerProvider
	meterProvider                 metric.MeterProvider
	includeContent                bool
	includeBinaryContent          bool
	includeModelRequestParameters bool
}

// WithInstrumentationTracerProvider selects a tracer provider instead of the global provider.
func WithInstrumentationTracerProvider(provider trace.TracerProvider) InstrumentationOption {
	return func(config *instrumentationConfig) { config.tracerProvider = provider }
}

// WithInstrumentationMeterProvider selects a meter provider instead of the global provider.
func WithInstrumentationMeterProvider(provider metric.MeterProvider) InstrumentationOption {
	return func(config *instrumentationConfig) { config.meterProvider = provider }
}

// WithInstrumentationContent controls prompt, completion, and tool content attributes.
func WithInstrumentationContent(include bool) InstrumentationOption {
	return func(config *instrumentationConfig) { config.includeContent = include }
}

// WithInstrumentationBinaryContent controls inline binary payloads when content is enabled.
func WithInstrumentationBinaryContent(include bool) InstrumentationOption {
	return func(config *instrumentationConfig) { config.includeBinaryContent = include }
}

// WithInstrumentationModelRequestParameters controls the full request-parameter attribute.
// OpenTelemetry tool definitions are emitted independently of this setting.
func WithInstrumentationModelRequestParameters(include bool) InstrumentationOption {
	return func(config *instrumentationConfig) { config.includeModelRequestParameters = include }
}

// InstrumentedModel emits OpenTelemetry spans and metrics around direct model requests.
type InstrumentedModel struct {
	*ModelWrapper
	tracer                        trace.Tracer
	tokenHistogram                metric.Int64Histogram
	costHistogram                 metric.Float64Histogram
	firstChunkHistogram           metric.Float64Histogram
	includeContent                bool
	includeBinaryContent          bool
	includeModelRequestParameters bool
}

// NewInstrumentedModel creates a transparent model decorator. Content is included by default.
func NewInstrumentedModel(model Model, options ...InstrumentationOption) *InstrumentedModel {
	config := instrumentationConfig{
		tracerProvider: otel.GetTracerProvider(), meterProvider: otel.GetMeterProvider(),
		includeContent: true, includeBinaryContent: true, includeModelRequestParameters: true,
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
	meter := config.meterProvider.Meter(otelScope)
	tokens, _ := meter.Int64Histogram(
		"gen_ai.client.token.usage", metric.WithUnit("{token}"),
		metric.WithDescription("Measures number of input and output tokens used"),
	)
	cost, _ := meter.Float64Histogram(
		"operation.cost", metric.WithUnit("{USD}"), metric.WithDescription("Monetary cost"),
	)
	firstChunk, _ := meter.Float64Histogram(
		"gen_ai.client.operation.time_to_first_chunk", metric.WithUnit("s"),
		metric.WithDescription("Time from issuing a streaming request to the first chunk"),
	)
	return &InstrumentedModel{
		ModelWrapper: WrapModel(model), tracer: config.tracerProvider.Tracer(otelScope),
		tokenHistogram: tokens, costHistogram: cost, firstChunkHistogram: firstChunk,
		includeContent: config.includeContent, includeBinaryContent: config.includeBinaryContent,
		includeModelRequestParameters: config.includeModelRequestParameters,
	}
}

// InstrumentModel wraps model unless it is already instrumented.
func InstrumentModel(model Model, options ...InstrumentationOption) Model {
	if _, ok := model.(*InstrumentedModel); ok {
		return model
	}
	return NewInstrumentedModel(model, options...)
}

// Request instruments a non-streaming request.
func (model *InstrumentedModel) Request(
	ctx context.Context, messages []ModelMessage, params ModelRequestParams,
) (*ModelResponse, error) {
	if modelRequestSpanActive(ctx) {
		return model.UnwrapModel().Request(ctx, messages, params)
	}
	spanCtx, request := model.startRequest(ctx, messages, params)
	response, err := model.UnwrapModel().Request(spanCtx, messages, params)
	request.finish(spanCtx, response, err, 0)
	return response, err
}

// StreamRequest instruments stream opening and keeps the span open through consumption.
func (model *InstrumentedModel) StreamRequest(
	ctx context.Context, messages []ModelMessage, params ModelRequestParams,
) (iter.Seq2[ModelStreamEvent, error], error) {
	if modelRequestSpanActive(ctx) {
		return model.ModelWrapper.StreamRequest(ctx, messages, params)
	}
	spanCtx, request := model.startRequest(ctx, messages, params)
	started := time.Now()
	events, err := model.ModelWrapper.StreamRequest(spanCtx, messages, params)
	if err != nil {
		request.finish(spanCtx, nil, err, 0)
		return nil, err
	}
	if events == nil {
		err := &UnexpectedModelBehaviorError{Message: "model returned no stream"}
		request.finish(spanCtx, nil, err, 0)
		return nil, err
	}
	return func(yield func(ModelStreamEvent, error) bool) {
		accumulator := newTelemetryStreamAccumulator()
		var streamErr error
		var firstChunk time.Duration
		for event, err := range events {
			if firstChunk == 0 {
				firstChunk = time.Since(started)
			}
			if err != nil {
				streamErr = err
				yield(nil, err)
				break
			}
			accumulator.observe(event)
			if !yield(event, nil) {
				break
			}
		}
		request.finish(spanCtx, accumulator.snapshot(), streamErr, firstChunk)
	}, nil
}

type instrumentedRequest struct {
	model    *InstrumentedModel
	span     trace.Span
	messages []ModelMessage
	params   ModelRequestParams
	once     sync.Once
}

func (model *InstrumentedModel) startRequest(
	ctx context.Context, messages []ModelMessage, params ModelRequestParams,
) (context.Context, *instrumentedRequest) {
	attributes := []attribute.KeyValue{
		attribute.String("gen_ai.operation.name", "chat"),
		attribute.String("gen_ai.request.model", model.Name()),
	}
	attributes = append(attributes, modelSettingAttributes(params.Settings)...)
	if definitions := telemetryToolDefinitions(params); definitions != "" {
		attributes = append(attributes, attribute.String("gen_ai.tool.definitions", definitions))
	}
	if model.includeModelRequestParameters {
		attributes = append(attributes, attribute.String("model_request_parameters", telemetryRequestParameters(params)))
	}
	spanCtx, span := model.tracer.Start(
		ctx, "chat "+model.Name(), trace.WithSpanKind(trace.SpanKindClient), trace.WithAttributes(attributes...),
	)
	spanCtx = context.WithValue(spanCtx, modelRequestSpanContextKey{}, true)
	return spanCtx, &instrumentedRequest{
		model: model, span: span, messages: cloneModelMessages(messages),
		params: ModelRequestContext{Params: params}.Clone().Params,
	}
}

func (request *instrumentedRequest) finish(
	ctx context.Context, response *ModelResponse, requestErr error, firstChunk time.Duration,
) {
	request.once.Do(func() {
		if requestErr != nil {
			request.span.SetStatus(codes.Error, requestErr.Error())
			request.span.RecordError(requestErr)
		}
		if response == nil {
			request.span.End()
			return
		}
		response = cloneModelResponse(response)
		attributes := responseTelemetryAttributes(response)
		if request.model.includeContent {
			attributes = append(attributes,
				attribute.String("gen_ai.input.messages", telemetryMessagesJSON(
					request.messages, request.model.includeBinaryContent,
				)),
				attribute.String("gen_ai.output.messages", telemetryMessagesJSON(
					[]ModelMessage{*response}, request.model.includeBinaryContent,
				)),
			)
			if request.params.Instructions != "" {
				attributes = append(attributes, attribute.String(
					"gen_ai.system_instructions", telemetryJSON([]map[string]any{{
						"type": "text", "content": request.params.Instructions,
					}}),
				))
			}
		}
		if firstChunk > 0 {
			attributes = append(attributes, attribute.Float64(
				"gen_ai.client.operation.time_to_first_chunk", firstChunk.Seconds(),
			))
		}
		request.span.SetAttributes(attributes...)
		request.span.End()
		request.model.recordMetrics(ctx, response, firstChunk)
	})
}

func hasInstrumentedModel(model Model) bool {
	seen := make([]Model, 0, 4)
	for !modelIsNil(model) {
		if _, ok := model.(*InstrumentedModel); ok {
			return true
		}
		for _, previous := range seen {
			if sameModelInstance(previous, model) {
				return false
			}
		}
		seen = append(seen, model)
		wrapper, ok := model.(ModelUnwrapper)
		if !ok {
			return false
		}
		model = wrapper.UnwrapModel()
	}
	return false
}

func (model *InstrumentedModel) recordMetrics(
	ctx context.Context, response *ModelResponse, firstChunk time.Duration,
) {
	attributes := []attribute.KeyValue{attribute.String("gen_ai.operation.name", "chat")}
	if response.ProviderName != "" {
		attributes = append(attributes,
			attribute.String("gen_ai.provider.name", response.ProviderName),
			attribute.String("gen_ai.system", response.ProviderName),
		)
	}
	attributes = append(attributes, attribute.String("gen_ai.request.model", model.Name()))
	if response.ModelName != "" {
		attributes = append(attributes, attribute.String("gen_ai.response.model", response.ModelName))
	}
	if response.Usage.InputTokens != 0 {
		model.tokenHistogram.Record(ctx, int64(response.Usage.InputTokens),
			metric.WithAttributes(append(attributes, attribute.String("gen_ai.token.type", "input"))...))
	}
	if response.Usage.OutputTokens != 0 {
		model.tokenHistogram.Record(ctx, int64(response.Usage.OutputTokens),
			metric.WithAttributes(append(attributes, attribute.String("gen_ai.token.type", "output"))...))
	}
	if response.Usage.CostUSD != nil {
		model.costHistogram.Record(ctx, *response.Usage.CostUSD, metric.WithAttributes(attributes...))
	} else if calculation, err := response.Price(); err == nil {
		model.costHistogram.Record(ctx, calculation.TotalPrice, metric.WithAttributes(attributes...))
	}
	if firstChunk > 0 {
		model.firstChunkHistogram.Record(ctx, firstChunk.Seconds(), metric.WithAttributes(attributes...))
	}
}
