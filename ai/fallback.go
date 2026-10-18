package ai

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"strings"
	"time"
)

const fallbackModelIDMetadataKey = "fallback_model_id"

// ErrFallbackExhausted reports that every model in a fallback chain failed or was rejected.
var ErrFallbackExhausted = errors.New("ai: all fallback models failed")

// FallbackErrorPredicate decides whether an error should advance to the next model.
type FallbackErrorPredicate func(ctx context.Context, err error) (bool, error)

// FallbackResponsePredicate decides whether a complete response should advance to the next model.
type FallbackResponsePredicate func(ctx context.Context, response *ModelResponse) (bool, error)

// FallbackModelOption configures a FallbackModel.
type FallbackModelOption func(*FallbackModel)

// WithFallbackModels appends models to a fallback chain.
func WithFallbackModels(models ...Model) FallbackModelOption {
	return func(fallback *FallbackModel) {
		for _, model := range models {
			if modelIsNil(model) {
				panic("ai: fallback model must not be nil")
			}
			fallback.models = append(fallback.models, model)
		}
	}
}

// WithFallbackOnError replaces the default API-error policy with an error predicate.
// Multiple predicates are combined with OR semantics in option order.
func WithFallbackOnError(predicate FallbackErrorPredicate) FallbackModelOption {
	if predicate == nil {
		panic("ai: fallback error predicate must not be nil")
	}
	return func(fallback *FallbackModel) {
		fallback.useCustomPolicy()
		fallback.errorPredicates = append(fallback.errorPredicates, predicate)
	}
}

// WithFallbackOnResponse replaces the default API-error policy with a response predicate.
// Multiple predicates are combined with OR semantics in option order.
//
// Response predicates only apply to Request. A streamed response has already
// reached the consumer by the time it could be judged, so StreamRequest accepts
// it as is and reports a warning. A run streams whenever something consumes its
// events: RunStream, StartRun, or Run with an event listener or event-stream capability.
func WithFallbackOnResponse(predicate FallbackResponsePredicate) FallbackModelOption {
	if predicate == nil {
		panic("ai: fallback response predicate must not be nil")
	}
	return func(fallback *FallbackModel) {
		fallback.useCustomPolicy()
		fallback.responsePredicates = append(fallback.responsePredicates, predicate)
	}
}

// WithFallbackWarningHandler receives warnings, such as response predicates
// skipped on a streamed request. Without a handler, warnings go to slog.Warn.
func WithFallbackWarningHandler(handler func(string)) FallbackModelOption {
	return func(fallback *FallbackModel) { fallback.warningHandler = handler }
}

// FallbackModel tries models in order when configured errors or responses are rejected.
type FallbackModel struct {
	models             []Model
	errorPredicates    []FallbackErrorPredicate
	responsePredicates []FallbackResponsePredicate
	customPolicy       bool
	warningHandler     func(string)
}

// NewFallbackModel creates a fallback chain. By default, only ModelAPIError values
// advance to the next model.
func NewFallbackModel(primary Model, options ...FallbackModelOption) *FallbackModel {
	if modelIsNil(primary) {
		panic("ai: primary fallback model must not be nil")
	}
	fallback := &FallbackModel{
		models: []Model{primary},
		errorPredicates: []FallbackErrorPredicate{func(_ context.Context, err error) (bool, error) {
			var apiError ModelAPIError
			return errors.As(err, &apiError) && apiError.IsModelAPIError(), nil
		}},
	}
	for _, option := range options {
		option(fallback)
	}
	return fallback
}

// DispatchesOutputProfile reports that each fallback candidate resolves OutputModeAuto independently.
func (*FallbackModel) DispatchesOutputProfile() bool { return true }

// DispatchesMessageProfile reports that each fallback candidate prepares message history independently.
func (*FallbackModel) DispatchesMessageProfile() bool { return true }

// Name identifies the ordered fallback chain.
func (fallback *FallbackModel) Name() string {
	names := make([]string, len(fallback.models))
	for index, model := range fallback.models {
		names[index] = model.Name()
	}
	return "fallback:" + strings.Join(names, ",")
}

// Models returns a detached copy of the ordered models.
func (fallback *FallbackModel) Models() []Model {
	return append([]Model(nil), fallback.models...)
}

// ContextWindow returns the smallest known candidate context window.
// Candidates with unknown windows do not constrain the result.
func (fallback *FallbackModel) ContextWindow() int {
	window := 0
	for _, model := range fallback.models {
		candidate := modelContextWindow(model)
		if candidate > 0 && (window == 0 || candidate < window) {
			window = candidate
		}
	}
	return window
}

// Request tries each model until one returns an accepted response.
func (fallback *FallbackModel) Request(
	ctx context.Context, messages []ModelMessage, params ModelRequestParams,
) (*ModelResponse, error) {
	failures := make([]error, 0, len(fallback.models))
	rejected := make([]*ModelResponse, 0)
	attempts := make([]ModelRequestAttempt, 0, len(fallback.models))
	rewound := false

	if model := fallback.pinnedModel(messages); model != nil {
		started := time.Now()
		response, err := fallback.requestModel(ctx, model, messages, params)
		if err == nil && response == nil {
			err = &UnexpectedModelBehaviorError{Message: "fallback model returned no response"}
		}
		if err == nil {
			fallback.stampResponse(response, model, false)
			return response, nil
		}
		shouldFallback, predicateErr := fallback.shouldFallbackError(ctx, err)
		if predicateErr != nil {
			return nil, predicateErr
		}
		if !shouldFallback {
			return response, err
		}
		fallback.cancelModel(ctx, model, messages[len(messages)-1].(ModelResponse))
		messages = rewindFallbackMessages(messages)
		rewound = true
		failures = append(failures, err)
		fallback.recordAttempt(ctx, &attempts, model, err, response, started)
	}

	for _, model := range fallback.models {
		started := time.Now()
		response, err := fallback.requestModel(ctx, model, messages, params)
		if err != nil {
			shouldFallback, predicateErr := fallback.shouldFallbackError(ctx, err)
			if predicateErr != nil {
				return nil, predicateErr
			}
			if !shouldFallback {
				return response, err
			}
			failures = append(failures, err)
			fallback.recordAttempt(ctx, &attempts, model, err, response, started)
			continue
		}
		if response == nil {
			return nil, &UnexpectedModelBehaviorError{Message: "fallback model returned no response"}
		}
		fallback.stampResponse(response, model, rewound)
		reject, predicateErr := fallback.shouldFallbackResponse(ctx, response)
		if predicateErr != nil {
			return nil, predicateErr
		}
		if reject {
			fillResponseCost(ctx, response)
			rejected = append(rejected, cloneModelResponse(response))
			fallback.recordAttempt(ctx, &attempts, model, nil, response, started)
			continue
		}
		response.FailedAttempts = append(cloneModelRequestAttempts(attempts), response.FailedAttempts...)
		return response, nil
	}
	return nil, newFallbackExhaustedError(failures, rejected, attempts)
}

// StreamRequest hands a stream over only once it has produced its first event,
// so an error the stream reports before any output, such as an overload after
// the provider accepted the request, falls back like an error while opening it.
// ResponseMetadataEvent values do not count as output. Errors after the first
// event reach the consumer and do not switch models. Range over the returned
// sequence, or break from it, to release the underlying stream.
//
// Response predicates are not applied: the response has already been streamed
// by the time it could be judged, so it is accepted as is and a warning is reported.
func (fallback *FallbackModel) StreamRequest(
	ctx context.Context, messages []ModelMessage, params ModelRequestParams,
) (iter.Seq2[ModelStreamEvent, error], error) {
	failures := make([]error, 0, len(fallback.models))
	attempts := make([]ModelRequestAttempt, 0, len(fallback.models))
	rewound := false
	if model := fallback.pinnedModel(messages); model != nil {
		started := time.Now()
		events, err := fallback.streamModel(ctx, model, messages, params)
		if err == nil {
			return fallback.stampStream(events, model, false, attempts), nil
		}
		shouldFallback, predicateErr := fallback.shouldFallbackError(ctx, err)
		if predicateErr != nil {
			return nil, predicateErr
		}
		if !shouldFallback {
			if events != nil {
				return fallback.stampStream(events, model, false, attempts), nil
			}
			return nil, err
		}
		fallback.cancelModel(ctx, model, messages[len(messages)-1].(ModelResponse))
		messages = rewindFallbackMessages(messages)
		rewound = true
		failures = append(failures, err)
		fallback.recordAttempt(ctx, &attempts, model, err, nil, started)
	}
	if len(fallback.responsePredicates) > 0 {
		message := "ai: FallbackModel response predicates are not applied to streamed requests, " +
			"so the streamed response is accepted as is. A request is streamed whenever something consumes " +
			"the run's events, such as RunStream, StartRun, or an event listener or event-stream capability."
		if fallback.warningHandler != nil {
			fallback.warningHandler(message)
		} else {
			slog.Warn(message)
		}
	}
	for _, model := range fallback.models {
		started := time.Now()
		events, err := fallback.streamModel(ctx, model, messages, params)
		if err == nil {
			return fallback.stampStream(events, model, rewound, attempts), nil
		}
		shouldFallback, predicateErr := fallback.shouldFallbackError(ctx, err)
		if predicateErr != nil {
			return nil, predicateErr
		}
		if !shouldFallback {
			if events != nil {
				return fallback.stampStream(events, model, rewound, attempts), nil
			}
			return nil, err
		}
		failures = append(failures, err)
		fallback.recordAttempt(ctx, &attempts, model, err, nil, started)
	}
	return nil, newFallbackExhaustedError(failures, nil, attempts)
}

// OpenModel opens every model and returns a reverse-order closer.
func (fallback *FallbackModel) OpenModel(ctx context.Context) (ModelCloseFunc, error) {
	closers := make([]ModelCloseFunc, 0, len(fallback.models))
	for _, model := range fallback.models {
		opener, ok := model.(ModelOpener)
		if !ok {
			continue
		}
		closeModel, err := opener.OpenModel(ctx)
		if err != nil {
			return nil, errors.Join(fmt.Errorf("ai: open fallback model %q: %w", model.Name(), err), closeFallbackModels(ctx, closers))
		}
		if closeModel != nil {
			closers = append(closers, closeModel)
		}
	}
	return func(ctx context.Context) error { return closeFallbackModels(ctx, closers) }, nil
}

// SupportsToolSearchStrategy reports true only when every model supports the strategy.
func (fallback *FallbackModel) SupportsToolSearchStrategy(strategy ToolSearchStrategy) bool {
	for _, model := range fallback.models {
		support, ok := model.(ToolSearchStrategyModel)
		if !ok || !support.SupportsToolSearchStrategy(strategy) {
			return false
		}
	}
	return true
}

// NativeToolSearchProvider returns a provider only when the whole chain shares it.
func (fallback *FallbackModel) NativeToolSearchProvider() string {
	provider := ""
	for _, model := range fallback.models {
		native, ok := model.(NativeToolSearchHistoryModel)
		if !ok || native.NativeToolSearchProvider() == "" {
			return ""
		}
		if provider == "" {
			provider = native.NativeToolSearchProvider()
		} else if provider != native.NativeToolSearchProvider() {
			return ""
		}
	}
	return provider
}

// ContinuationDelay delegates to the pinned model, or the first model with a delay.
func (fallback *FallbackModel) ContinuationDelay(response ModelResponse) time.Duration {
	if model := fallback.pinnedResponseModel(response); model != nil {
		if delayer, ok := model.(ModelContinuationDelayer); ok {
			return delayer.ContinuationDelay(response)
		}
		return 0
	}
	for _, model := range fallback.models {
		if delayer, ok := model.(ModelContinuationDelayer); ok {
			if delay := delayer.ContinuationDelay(response); delay != 0 {
				return delay
			}
		}
	}
	return 0
}

// CancelSuspendedResponse delegates to the pinned model. Without a pin, it
// attempts cancellation on every model and ignores individual failures.
func (fallback *FallbackModel) CancelSuspendedResponse(ctx context.Context, response ModelResponse) error {
	if model := fallback.pinnedResponseModel(response); model != nil {
		if canceler, ok := model.(SuspendedResponseCanceler); ok {
			return canceler.CancelSuspendedResponse(ctx, response)
		}
		return nil
	}
	for _, model := range fallback.models {
		fallback.cancelModel(ctx, model, response)
	}
	return nil
}

func (fallback *FallbackModel) useCustomPolicy() {
	if fallback.customPolicy {
		return
	}
	fallback.errorPredicates = nil
	fallback.responsePredicates = nil
	fallback.customPolicy = true
}

func (fallback *FallbackModel) requestModel(
	ctx context.Context, model Model, messages []ModelMessage, params ModelRequestParams,
) (*ModelResponse, error) {
	messages, params, err := prepareDirectRequest(model, messages, params)
	if err != nil {
		return nil, err
	}
	if err := validateModelSettings(params.Settings); err != nil {
		return nil, err
	}
	return model.Request(ctx, messages, params)
}

func (fallback *FallbackModel) streamModel(
	ctx context.Context, model Model, messages []ModelMessage, params ModelRequestParams,
) (iter.Seq2[ModelStreamEvent, error], error) {
	messages, params, err := prepareDirectRequest(model, messages, params)
	if err != nil {
		return nil, err
	}
	if err := validateModelSettings(params.Settings); err != nil {
		return nil, err
	}
	if streaming, ok := model.(StreamingModel); ok {
		events, err := streaming.StreamRequest(ctx, messages, params)
		if err != nil {
			return nil, err
		}
		return peekModelStream(events)
	}
	response, err := model.Request(ctx, messages, params)
	if err != nil {
		return nil, err
	}
	if response == nil {
		return nil, &UnexpectedModelBehaviorError{Message: "fallback model returned no response"}
	}
	fallback.stampResponse(response, model, false)
	return replayAsEvents(response), nil
}

// peekModelStream reads events up to the first one a consumer sees. When that
// read fails, it returns the error together with a replay of what was read.
func peekModelStream(events iter.Seq2[ModelStreamEvent, error]) (iter.Seq2[ModelStreamEvent, error], error) {
	next, stop := iter.Pull2(events)
	var buffered []ModelStreamEvent
	var firstErr error
	for {
		event, err, ok := next()
		if !ok || err != nil {
			stop()
			firstErr = err
			break
		}
		buffered = append(buffered, event)
		if _, metadata := event.(ResponseMetadataEvent); !metadata {
			break
		}
	}
	return func(yield func(ModelStreamEvent, error) bool) {
		defer stop()
		for _, event := range buffered {
			if !yield(event, nil) {
				return
			}
		}
		if firstErr != nil {
			yield(nil, firstErr)
			return
		}
		for {
			event, err, ok := next()
			if !ok || !yield(event, err) {
				return
			}
		}
	}, firstErr
}

func (fallback *FallbackModel) shouldFallbackError(ctx context.Context, requestErr error) (bool, error) {
	if ctx.Err() != nil {
		return false, nil
	}
	for _, predicate := range fallback.errorPredicates {
		fallbackNow, err := predicate(ctx, requestErr)
		if err != nil {
			return false, fmt.Errorf("ai: evaluate fallback error: %w", err)
		}
		if fallbackNow {
			return true, nil
		}
	}
	return false, nil
}

func (fallback *FallbackModel) shouldFallbackResponse(
	ctx context.Context, response *ModelResponse,
) (bool, error) {
	if ctx.Err() != nil {
		return false, nil
	}
	for _, predicate := range fallback.responsePredicates {
		fallbackNow, err := predicate(ctx, cloneModelResponse(response))
		if err != nil {
			return false, fmt.Errorf("ai: evaluate fallback response: %w", err)
		}
		if fallbackNow {
			return true, nil
		}
	}
	return false, nil
}

func (fallback *FallbackModel) recordAttempt(
	ctx context.Context,
	attempts *[]ModelRequestAttempt,
	model Model,
	requestErr error,
	response *ModelResponse,
	started time.Time,
) {
	if response != nil {
		*attempts = append(*attempts, cloneModelRequestAttempts(response.FailedAttempts)...)
	} else {
		var exhausted *FallbackExhaustedError
		if errors.As(requestErr, &exhausted) {
			*attempts = append(*attempts, exhausted.Attempts()...)
		}
	}
	attempt := ModelRequestAttempt{
		ModelName: model.Name(), ProviderName: fallbackProviderName(model),
		Outcome: ModelRequestAttemptError, Timestamp: started.UTC(), Duration: time.Since(started),
	}
	if requestErr != nil {
		attempt.Error = describeFallbackError(requestErr)
	} else {
		attempt.Outcome = ModelRequestAttemptRejected
		fillResponseCost(ctx, response)
		attempt.ModelName = response.ModelName
		attempt.ProviderName = response.ProviderName
		if attempt.ProviderName == "" {
			attempt.ProviderName = fallbackProviderName(model)
		}
		usage := response.Usage.Clone()
		attempt.Usage = &usage
	}
	*attempts = append(*attempts, attempt)
	recordModelRequestAttempt(ctx, attempt, requestErr, response, len(*attempts)-1)
}

func fallbackProviderName(model Model) string {
	if identified, ok := model.(ModelProviderIdentity); ok {
		return identified.ProviderName()
	}
	return ""
}

func describeFallbackError(err error) string {
	typeName := fmt.Sprintf("%T", err)
	name := strings.TrimPrefix(typeName, "*")
	if index := strings.LastIndex(name, "."); index >= 0 {
		name = name[index+1:]
	}
	if message := err.Error(); message != "" {
		return name + ": " + message
	}
	return name
}

func (fallback *FallbackModel) stampResponse(response *ModelResponse, model Model, replace bool) {
	stampDirectResponse(model, response)
	if response.State == ModelResponseStateSuspended {
		stampFallbackMetadata(response, fallbackModelIDMetadataKey, model.Name())
	}
	if replace {
		stampFallbackMetadata(response, "replace_previous_response", true)
	}
}

func (fallback *FallbackModel) stampStream(
	events iter.Seq2[ModelStreamEvent, error], model Model, replace bool, attempts []ModelRequestAttempt,
) iter.Seq2[ModelStreamEvent, error] {
	return func(yield func(ModelStreamEvent, error) bool) {
		for event, err := range events {
			if err != nil {
				yield(nil, err)
				return
			}
			switch typed := event.(type) {
			case ResponseMetadataEvent:
				if typed.ModelName == "" {
					typed.ModelName = model.Name()
				}
				if typed.State == ModelResponseStateSuspended {
					typed.Metadata = stampedFallbackMetadata(typed.Metadata, fallbackModelIDMetadataKey, model.Name())
				}
				if replace {
					typed.Metadata = stampedFallbackMetadata(typed.Metadata, "replace_previous_response", true)
				}
				typed.FailedAttempts = append(cloneModelRequestAttempts(attempts), typed.FailedAttempts...)
				event = typed
			case FinishEvent:
				if typed.ModelName == "" {
					typed.ModelName = model.Name()
				}
				if typed.State == ModelResponseStateSuspended {
					typed.Metadata = stampedFallbackMetadata(typed.Metadata, fallbackModelIDMetadataKey, model.Name())
				}
				if replace {
					typed.Metadata = stampedFallbackMetadata(typed.Metadata, "replace_previous_response", true)
				}
				typed.FailedAttempts = append(cloneModelRequestAttempts(attempts), typed.FailedAttempts...)
				event = typed
			}
			if !yield(event, nil) {
				return
			}
		}
	}
}

func (fallback *FallbackModel) pinnedModel(messages []ModelMessage) Model {
	if len(messages) == 0 {
		return nil
	}
	response, ok := messages[len(messages)-1].(ModelResponse)
	if !ok || response.State != ModelResponseStateSuspended {
		return nil
	}
	return fallback.pinnedResponseModel(response)
}

func (fallback *FallbackModel) pinnedResponseModel(response ModelResponse) Model {
	namespace, _ := response.Metadata["__pydantic_ai__"].(map[string]any)
	modelName, _ := namespace[fallbackModelIDMetadataKey].(string)
	if modelName == "" {
		return nil
	}
	for _, model := range fallback.models {
		if model.Name() == modelName {
			return model
		}
	}
	return nil
}

func (fallback *FallbackModel) cancelModel(ctx context.Context, model Model, response ModelResponse) {
	if canceler, ok := model.(SuspendedResponseCanceler); ok {
		_ = canceler.CancelSuspendedResponse(context.WithoutCancel(ctx), *cloneModelResponse(&response))
	}
}

func stampFallbackMetadata(response *ModelResponse, key string, value any) {
	response.Metadata = stampedFallbackMetadata(response.Metadata, key, value)
}

func stampedFallbackMetadata(metadata map[string]any, key string, value any) map[string]any {
	metadata = cloneSchemaMap(metadata)
	if metadata == nil {
		metadata = make(map[string]any, 1)
	}
	namespace, _ := metadata["__pydantic_ai__"].(map[string]any)
	namespace = cloneSchemaMap(namespace)
	if namespace == nil {
		namespace = make(map[string]any, 1)
	}
	namespace[key] = cloneSchemaValue(value)
	metadata["__pydantic_ai__"] = namespace
	return metadata
}

func rewindFallbackMessages(messages []ModelMessage) []ModelMessage {
	return cloneModelMessages(messages[:len(messages)-1])
}

func closeFallbackModels(ctx context.Context, closers []ModelCloseFunc) error {
	var closeErrors []error
	for index := len(closers) - 1; index >= 0; index-- {
		if err := closers[index](context.WithoutCancel(ctx)); err != nil {
			closeErrors = append(closeErrors, err)
		}
	}
	if err := errors.Join(closeErrors...); err != nil {
		return fmt.Errorf("ai: close fallback models: %w", err)
	}
	return nil
}

// FallbackExhaustedError contains failures and rejected responses from a chain.
type FallbackExhaustedError struct {
	failures []error
	rejected []*ModelResponse
	attempts []ModelRequestAttempt
}

func newFallbackExhaustedError(
	failures []error, rejected []*ModelResponse, attempts []ModelRequestAttempt,
) *FallbackExhaustedError {
	cloned := make([]*ModelResponse, len(rejected))
	for index, response := range rejected {
		cloned[index] = cloneModelResponse(response)
	}
	return &FallbackExhaustedError{
		failures: append([]error(nil), failures...), rejected: cloned,
		attempts: cloneModelRequestAttempts(attempts),
	}
}

// Error summarizes failed and rejected fallback candidates.
func (err *FallbackExhaustedError) Error() string {
	return fmt.Sprintf("%s: %d error(s), %d rejected response(s)", ErrFallbackExhausted, len(err.failures), len(err.rejected))
}

// Unwrap exposes the sentinel and individual model failures to errors.Is and errors.As.
func (err *FallbackExhaustedError) Unwrap() []error {
	return append([]error{ErrFallbackExhausted}, err.failures...)
}

// Failures returns the model errors in attempt order.
func (err *FallbackExhaustedError) Failures() []error {
	return append([]error(nil), err.failures...)
}

// Attempts returns detached failed and rejected attempts in order.
func (err *FallbackExhaustedError) Attempts() []ModelRequestAttempt {
	return cloneModelRequestAttempts(err.attempts)
}

// RejectedResponses returns detached rejected responses in attempt order.
func (err *FallbackExhaustedError) RejectedResponses() []*ModelResponse {
	responses := make([]*ModelResponse, len(err.rejected))
	for index, response := range err.rejected {
		responses[index] = cloneModelResponse(response)
	}
	return responses
}
