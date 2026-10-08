package ai

import (
	"context"
	"iter"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type cacheHealth struct {
	ratio       float64
	established int
	previous    int
	missed      int
	reason      string
	alert       bool
	notEnabled  bool
}

type modelRequestObservationKey struct{}

type modelRequestObservation struct {
	marks        *conversationCacheMarks
	model        Model
	messages     []ModelMessage
	params       ModelRequestParams
	response     *ModelResponse
	finalSegment *ModelResponse
	started      time.Time
	firstChunk   time.Duration
}

func (observation *modelRequestObservation) record(
	model Model, messages []ModelMessage, params ModelRequestParams, previous, segment *ModelResponse,
) {
	observation.model = model
	observation.messages = cloneModelMessages(messages)
	observation.params = ModelRequestContext{Params: params}.Clone().Params
	observation.finalSegment = cloneModelResponse(segment)
	observation.response = cloneModelResponse(segment)
	if previous != nil {
		observation.response, _ = mergeModelResponses(previous, segment)
	}
	stampDirectResponse(model, observation.response)
}

func observeRequestStream(ctx context.Context, events iter.Seq2[ModelStreamEvent, error]) iter.Seq2[ModelStreamEvent, error] {
	observation, _ := ctx.Value(modelRequestObservationKey{}).(*modelRequestObservation)
	if observation == nil {
		return events
	}
	return func(yield func(ModelStreamEvent, error) bool) {
		for event, err := range events {
			if observation.firstChunk == 0 {
				observation.firstChunk = time.Since(observation.started)
			}
			if !yield(event, err) {
				return
			}
		}
	}
}

func recordCacheHealth(
	span trace.Span, marks *conversationCacheMarks, model Model, messages []ModelMessage,
	params ModelRequestParams, response, measured *ModelResponse,
) {
	profile := modelProfile(model)
	var retention *time.Duration
	if resolved, known := ResolveCacheRetention(model, &params.Settings); known {
		retention = &resolved
	}
	notEnabled := false
	if configured, ok := model.(CacheConfigurationModel); ok && measured.Usage.InputTokens >= 4096 {
		settings := params.Settings
		if defaults, ok := model.(ModelDefaultSettings); ok {
			settings = mergeModelSettings(defaults.DefaultModelSettings(), &settings)
		}
		notEnabled = configured.CachingNotEnabled(settings)
		for _, message := range messages {
			if request, ok := message.(ModelRequest); ok {
				for _, part := range request.Parts {
					if prompt, ok := part.(UserPromptPart); ok {
						for _, content := range prompt.Contents {
							if _, explicit := content.(CachePoint); explicit {
								notEnabled = false
							}
						}
					}
				}
			}
		}
	}
	health := observeCacheHealth(marks, messages, response, measured,
		expectedCacheRetention(messages, &profile, retention), notEnabled)
	if health == nil {
		return
	}
	span.SetAttributes(
		attribute.Float64("pydantic_ai.cache.hit_ratio", health.ratio),
		attribute.Int("pydantic_ai.cache.established_tokens", health.established),
	)
	if health.notEnabled {
		span.SetAttributes(attribute.Bool("pydantic_ai.cache.not_enabled", true))
		attributes := []attribute.KeyValue{attribute.Int("input_tokens", measured.Usage.InputTokens)}
		if response.ProviderName != "" {
			attributes = append(attributes, attribute.String("provider_name", response.ProviderName))
		}
		if response.ModelName != "" {
			attributes = append(attributes, attribute.String("model_name", response.ModelName))
		}
		span.AddEvent("pydantic_ai.cache.not_enabled", trace.WithAttributes(attributes...))
		return
	}
	if health.reason == "" {
		return
	}
	span.SetAttributes(
		attribute.Bool("pydantic_ai.cache.collapsed", true),
		attribute.Int("pydantic_ai.cache.missed_tokens", health.missed),
		attribute.String("pydantic_ai.cache.collapse_reason", health.reason),
	)
	if health.alert {
		attributes := []attribute.KeyValue{
			attribute.Int("established_tokens", health.previous),
			attribute.Int("cache_read_tokens", measured.Usage.CacheReadTokens),
			attribute.Int("missed_tokens", health.missed),
		}
		if response.ProviderName != "" {
			attributes = append(attributes, attribute.String("provider_name", response.ProviderName))
		}
		if response.ModelName != "" {
			attributes = append(attributes, attribute.String("model_name", response.ModelName))
		}
		span.AddEvent("pydantic_ai.cache.collapse", trace.WithAttributes(attributes...))
	}
}
