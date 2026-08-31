package ai

import (
	"context"
	"iter"
	"time"
)

// ModelUnwrapper is implemented by model decorators that expose their wrapped model.
type ModelUnwrapper interface {
	UnwrapModel() Model
}

// ModelWrapper transparently delegates the model contract and optional model capabilities.
// Embed it in a custom model and override only the methods that change behavior.
type ModelWrapper struct {
	wrapped Model
}

// WrapModel creates a transparent model decorator.
func WrapModel(model Model) *ModelWrapper {
	if modelIsNil(model) {
		panic("ai: wrapped model must not be nil")
	}
	return &ModelWrapper{wrapped: model}
}

// UnwrapModel returns the immediate wrapped model.
func (wrapper *ModelWrapper) UnwrapModel() Model { return wrapper.wrapped }

// Name returns the wrapped model name.
func (wrapper *ModelWrapper) Name() string { return wrapper.wrapped.Name() }

// Request delegates a non-streaming request.
func (wrapper *ModelWrapper) Request(
	ctx context.Context, messages []ModelMessage, params ModelRequestParams,
) (*ModelResponse, error) {
	return wrapper.wrapped.Request(ctx, messages, params)
}

// ProviderName returns the wrapped model's provider identity when present.
func (wrapper *ModelWrapper) ProviderName() string {
	if model, ok := wrapper.wrapped.(ModelProviderIdentity); ok {
		return model.ProviderName()
	}
	return ""
}

// ProviderURL returns the wrapped model's provider API URL when present.
func (wrapper *ModelWrapper) ProviderURL() string {
	if model, ok := wrapper.wrapped.(ModelProviderIdentity); ok {
		return model.ProviderURL()
	}
	return ""
}

// CountTokens delegates request token counting when supported.
func (wrapper *ModelWrapper) CountTokens(
	ctx context.Context, messages []ModelMessage, params ModelRequestParams,
) (Usage, error) {
	return CountModelTokens(ctx, wrapper.wrapped, messages, params)
}

// StreamRequest delegates streaming or replays a non-streaming response as events.
func (wrapper *ModelWrapper) StreamRequest(
	ctx context.Context, messages []ModelMessage, params ModelRequestParams,
) (iter.Seq2[ModelStreamEvent, error], error) {
	if model, ok := wrapper.wrapped.(StreamingModel); ok {
		return model.StreamRequest(ctx, messages, params)
	}
	response, err := wrapper.wrapped.Request(ctx, messages, params)
	if err != nil || response == nil {
		return nil, err
	}
	return replayAsEvents(response), nil
}

// OpenModel delegates the per-run model lifecycle when present.
func (wrapper *ModelWrapper) OpenModel(ctx context.Context) (ModelCloseFunc, error) {
	if model, ok := wrapper.wrapped.(ModelOpener); ok {
		return model.OpenModel(ctx)
	}
	return nil, nil
}

// DefaultModelSettings returns detached wrapped-model defaults when present.
func (wrapper *ModelWrapper) DefaultModelSettings() ModelSettings {
	if model, ok := wrapper.wrapped.(ModelDefaultSettings); ok {
		return model.DefaultModelSettings().Clone()
	}
	return ModelSettings{}
}

// PromptCacheRetention delegates provider cache-retention resolution when available.
func (wrapper *ModelWrapper) PromptCacheRetention(settings ModelSettings) (time.Duration, bool) {
	if model, ok := wrapper.wrapped.(PromptCacheRetentionModel); ok {
		return model.PromptCacheRetention(settings.Clone())
	}
	return 0, false
}

// SupportsToolSearchStrategy reports wrapped-model strategy support.
func (wrapper *ModelWrapper) SupportsToolSearchStrategy(strategy ToolSearchStrategy) bool {
	model, ok := wrapper.wrapped.(ToolSearchStrategyModel)
	return ok && model.SupportsToolSearchStrategy(strategy)
}

// NativeToolSearchProvider returns the wrapped model's native history identity.
func (wrapper *ModelWrapper) NativeToolSearchProvider() string {
	if model, ok := wrapper.wrapped.(NativeToolSearchHistoryModel); ok {
		return model.NativeToolSearchProvider()
	}
	return ""
}

// ContinuationDelay returns the wrapped model's continuation delay.
func (wrapper *ModelWrapper) ContinuationDelay(response ModelResponse) time.Duration {
	if model, ok := wrapper.wrapped.(ModelContinuationDelayer); ok {
		return model.ContinuationDelay(response)
	}
	return 0
}

// CancelSuspendedResponse delegates cancellation when supported.
func (wrapper *ModelWrapper) CancelSuspendedResponse(ctx context.Context, response ModelResponse) error {
	if model, ok := wrapper.wrapped.(SuspendedResponseCanceler); ok {
		return model.CancelSuspendedResponse(ctx, response)
	}
	return nil
}

// UnwrapModel removes nested model decorators. The outermost model is returned
// when a wrapper is nil or cyclic.
func UnwrapModel(model Model) Model {
	original := model
	seen := make([]Model, 0, 4)
	for !modelIsNil(model) {
		for _, previous := range seen {
			if sameModelInstance(previous, model) {
				return original
			}
		}
		seen = append(seen, model)
		wrapper, ok := model.(ModelUnwrapper)
		if !ok {
			return model
		}
		next := wrapper.UnwrapModel()
		if modelIsNil(next) {
			return original
		}
		model = next
	}
	return original
}
