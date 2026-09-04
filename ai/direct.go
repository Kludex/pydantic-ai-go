package ai

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"
)

// ErrModelResponseStreamConsumed reports a second attempt to consume one direct model stream.
var ErrModelResponseStreamConsumed = errors.New("ai: model response stream can only be consumed once")

// RequestModel makes one logical model request without running an agent loop.
// Suspended provider responses are continued automatically.
func RequestModel(
	ctx context.Context, model Model, messages []ModelMessage, params ModelRequestParams,
) (response *ModelResponse, err error) {
	if modelIsNil(model) {
		return nil, ErrNoModel
	}
	messages, params, err = prepareDirectRequest(model, messages, params)
	if err != nil {
		return nil, err
	}
	if err := validateModelSettings(params.Settings); err != nil {
		return nil, err
	}
	closeModel, err := openDirectModel(ctx, model)
	if err != nil {
		return nil, err
	}
	defer func() {
		err = errors.Join(err, closeDirectModel(ctx, closeModel))
	}()
	response, err = requestModelDirect(ctx, model, messages, params, false, nil, nil)
	if response != nil {
		response = cloneModelResponse(response)
	}
	return response, err
}

// ModelResponseStream is a single-consumer direct model stream.
// Response and Usage return detached snapshots and are safe during consumption.
type ModelResponseStream struct {
	ctx      context.Context
	model    Model
	messages []ModelMessage
	params   ModelRequestParams
	setupErr error

	mu       sync.RWMutex
	response *ModelResponse
	err      error
	done     bool
	started  bool
}

// StreamModel makes one logical streaming model request without running an agent loop.
// Models without native streaming support are replayed as normalized events.
func StreamModel(
	ctx context.Context, model Model, messages []ModelMessage, params ModelRequestParams,
) *ModelResponseStream {
	messages, params, err := prepareDirectRequest(model, messages, params)
	return &ModelResponseStream{ctx: ctx, model: model, messages: messages, params: params, setupErr: err}
}

// Events returns normalized response-part events. The sequence may be consumed once.
func (stream *ModelResponseStream) Events() EventStream {
	return func(yield func(StreamEvent, error) bool) {
		stream.mu.Lock()
		if stream.started {
			stream.mu.Unlock()
			yield(nil, ErrModelResponseStreamConsumed)
			return
		}
		stream.started = true
		stream.mu.Unlock()

		if modelIsNil(stream.model) {
			stream.finish(nil, ErrNoModel)
			yield(nil, ErrNoModel)
			return
		}
		if stream.setupErr != nil {
			stream.finish(nil, stream.setupErr)
			yield(nil, stream.setupErr)
			return
		}
		if err := validateModelSettings(stream.params.Settings); err != nil {
			stream.finish(nil, err)
			yield(nil, err)
			return
		}
		closeModel, err := openDirectModel(stream.ctx, stream.model)
		if err != nil {
			stream.finish(nil, err)
			yield(nil, err)
			return
		}
		consumerStopped := false
		response, requestErr := requestModelDirect(
			stream.ctx, stream.model, stream.messages, stream.params, true,
			func(event StreamEvent) bool {
				stream.observeEvent(event)
				if !yield(event, nil) {
					consumerStopped = true
					return false
				}
				return true
			},
			stream.observe,
		)
		if consumerStopped && errors.Is(requestErr, context.Canceled) {
			requestErr = nil
		}
		terminalErr := errors.Join(requestErr, closeDirectModel(stream.ctx, closeModel))
		stream.finish(response, terminalErr)
		if terminalErr != nil && !consumerStopped {
			yield(nil, terminalErr)
		}
	}
}

// Response returns the latest accumulated response, or nil before response data arrives.
func (stream *ModelResponseStream) Response() *ModelResponse {
	stream.mu.RLock()
	defer stream.mu.RUnlock()
	if stream.response == nil {
		return nil
	}
	return cloneModelResponse(stream.response)
}

// Usage returns a detached snapshot of the accumulated response usage.
func (stream *ModelResponseStream) Usage() Usage {
	stream.mu.RLock()
	defer stream.mu.RUnlock()
	if stream.response == nil {
		return Usage{}
	}
	return stream.response.Usage.Clone()
}

// Err returns the terminal stream error after Events finishes.
func (stream *ModelResponseStream) Err() error {
	stream.mu.RLock()
	defer stream.mu.RUnlock()
	return stream.err
}

// Done reports whether Events has finished or failed.
func (stream *ModelResponseStream) Done() bool {
	stream.mu.RLock()
	defer stream.mu.RUnlock()
	return stream.done
}

func (stream *ModelResponseStream) observe(response *ModelResponse) {
	stream.mu.Lock()
	defer stream.mu.Unlock()
	cloned := cloneModelResponse(response)
	if cloned.Parts == nil && stream.response != nil {
		cloned.Parts = cloneModelResponse(stream.response).Parts
	}
	stream.response = cloned
}

func (stream *ModelResponseStream) observeEvent(event StreamEvent) {
	stream.mu.Lock()
	defer stream.mu.Unlock()
	switch event := event.(type) {
	case PartStartEvent:
		for len(stream.response.Parts) <= event.Index {
			stream.response.Parts = append(stream.response.Parts, nil)
		}
		stream.response.Parts[event.Index] = cloneResponsePart(event.Part)
	case PartDeltaEvent:
		part, _ := event.Delta.Apply(stream.response.Parts[event.Index])
		stream.response.Parts[event.Index] = part
	case PartEndEvent:
		for len(stream.response.Parts) <= event.Index {
			stream.response.Parts = append(stream.response.Parts, nil)
		}
		stream.response.Parts[event.Index] = cloneResponsePart(event.Part)
	case FinishEvent:
		stream.response.Usage = event.Usage.Clone()
		stream.response.ModelName = event.ModelName
		stream.response.Timestamp = event.Timestamp
		stream.response.ProviderName = event.ProviderName
		stream.response.ProviderURL = event.ProviderURL
		stream.response.ProviderDetails = cloneSchemaMap(event.ProviderDetails)
		stream.response.Metadata = cloneSchemaMap(event.Metadata)
		stream.response.ProviderResponseID = event.ProviderResponseID
		stream.response.FinishReason = event.FinishReason
		stream.response.State = event.State
	}
}

func (stream *ModelResponseStream) finish(response *ModelResponse, err error) {
	stream.mu.Lock()
	defer stream.mu.Unlock()
	if response != nil {
		stream.response = cloneModelResponse(response)
	}
	stream.err = err
	stream.done = true
}

func prepareDirectRequest(
	model Model, messages []ModelMessage, params ModelRequestParams,
) ([]ModelMessage, ModelRequestParams, error) {
	request := ModelRequestContext{Messages: messages, Params: params}.Clone()
	if !modelIsNil(model) {
		if defaults, ok := model.(ModelDefaultSettings); ok {
			request.Params.Settings = mergeModelSettings(defaults.DefaultModelSettings(), &request.Params.Settings)
		}
	}
	messages, params = request.Messages, request.Params
	if params.InstructionParts == nil {
		for index := len(messages) - 1; index >= 0; index-- {
			message, ok := messages[index].(ModelRequest)
			if ok && message.Instructions != "" {
				params.InstructionParts = []InstructionPart{{Content: message.Instructions}}
				break
			}
		}
	}
	if params.Instructions == "" && len(params.InstructionParts) > 0 {
		params.Instructions = joinInstructionParts(params.InstructionParts)
	}
	params.InstructionParts = cloneInstructionParts(params.InstructionParts)
	params, err := resolveModelOutputParams(model, params, OutputToolConfig{}, "")
	if err != nil {
		return nil, ModelRequestParams{}, err
	}
	params, err = ResolveNativeToolPreferences(model, params)
	if err != nil {
		return nil, ModelRequestParams{}, err
	}
	messages, err = PrepareModelMessages(model, messages)
	if err != nil {
		return nil, ModelRequestParams{}, err
	}
	return messages, params, nil
}

func openDirectModel(ctx context.Context, model Model) (ModelCloseFunc, error) {
	opener, ok := model.(ModelOpener)
	if !ok {
		return nil, nil
	}
	closeModel, err := opener.OpenModel(ctx)
	if err != nil {
		return nil, fmt.Errorf("ai: open model %q: %w", model.Name(), err)
	}
	return closeModel, nil
}

func closeDirectModel(ctx context.Context, closeModel ModelCloseFunc) error {
	if closeModel == nil {
		return nil
	}
	if err := closeModel(context.WithoutCancel(ctx)); err != nil {
		return fmt.Errorf("ai: close model: %w", err)
	}
	return nil
}

func requestModelDirect(
	ctx context.Context,
	model Model,
	messages []ModelMessage,
	params ModelRequestParams,
	streaming bool,
	emit func(StreamEvent) bool,
	observe func(*ModelResponse),
) (*ModelResponse, error) {
	baseMessages := slices.Clone(messages)
	var response *ModelResponse
	if historyEndsSuspended(baseMessages) {
		seed := baseMessages[len(baseMessages)-1].(ModelResponse)
		response = cloneModelResponse(&seed)
		fillResponseCost(ctx, response)
		baseMessages = baseMessages[:len(baseMessages)-1]
		if observe != nil {
			observe(response)
		}
	}
	lastMode := continuationAccumulate
	generationCount := 0
	pollCount := 0
	lastSegmentOffset := 0
	for {
		segmentMessages := baseMessages
		if response != nil {
			if response.State != ModelResponseStateSuspended {
				return response, nil
			}
			if err := continuationLimitError(response, lastMode, &generationCount, &pollCount); err != nil {
				cancelDirectSuspendedResponse(ctx, model, response)
				return response, err
			}
			if err := waitForDirectContinuation(ctx, model, response); err != nil {
				cancelDirectSuspendedResponse(ctx, model, response)
				return response, err
			}
			segmentMessages = append(slices.Clone(baseMessages), *cloneModelResponse(response))
		}

		eventOffset := 0
		segmentEmit := emit
		if response != nil && emit != nil {
			if background, _ := response.ProviderDetails["background"].(bool); background {
				eventOffset = lastSegmentOffset
			} else {
				eventOffset = len(response.Parts)
			}
			segmentEmit = func(event StreamEvent) bool {
				return emit(reindexContinuationEvent(event, eventOffset))
			}
		}
		segmentObserve := func(segment *ModelResponse) {
			segment = cloneModelResponse(segment)
			stampDirectResponse(model, segment)
			fillResponseCost(ctx, segment)
			current := segment
			if response != nil {
				current, _ = mergeModelResponses(response, segment)
			}
			if observe != nil {
				observe(current)
			}
		}
		segment, err := requestDirectSegment(
			ctx, model, segmentMessages, params, streaming, segmentEmit, segmentObserve,
		)
		if segment != nil {
			stampDirectResponse(model, segment)
			fillResponseCost(ctx, segment)
			if speechErr := validateResponseSpeech(segment); speechErr != nil && err == nil {
				err = speechErr
			}
		}
		if err != nil {
			partial := response
			if segment != nil {
				if partial == nil {
					partial = segment
				} else {
					partial, _ = mergeModelResponses(partial, segment)
				}
			}
			cancelDirectSuspendedResponse(ctx, model, partial)
			return partial, err
		}
		if segment == nil {
			cancelDirectSuspendedResponse(ctx, model, response)
			return response, &UnexpectedModelBehaviorError{Message: "model returned no response"}
		}
		if segment.State == "" {
			segment.State = ModelResponseStateComplete
		}
		if response == nil {
			response = segment
		} else {
			var mode continuationMergeMode
			response, mode = mergeModelResponses(response, segment)
			lastMode = mode
			switch mode {
			case continuationAccumulate, continuationReplaceSameID:
				lastSegmentOffset = eventOffset
			default:
				lastSegmentOffset = 0
			}
		}
		if observe != nil {
			observe(response)
		}
	}
}

func requestDirectSegment(
	ctx context.Context,
	model Model,
	messages []ModelMessage,
	params ModelRequestParams,
	streaming bool,
	emit func(StreamEvent) bool,
	observe func(*ModelResponse),
) (*ModelResponse, error) {
	provider := ""
	if nativeHistoryModel, ok := model.(NativeToolSearchHistoryModel); ok {
		provider = nativeHistoryModel.NativeToolSearchProvider()
	}
	messages = adaptNativeToolSearchHistory(messages, provider)
	if params.Settings.RequestTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, params.Settings.RequestTimeout)
		defer cancel()
	}
	if !streaming {
		response, err := model.Request(ctx, messages, params)
		if response != nil && observe != nil {
			observe(response)
		}
		return response, err
	}
	if streamingModel, ok := model.(StreamingModel); ok {
		events, err := streamingModel.StreamRequest(ctx, messages, params)
		if err != nil {
			return nil, err
		}
		return accumulate(events, params, emit, observe)
	}
	response, err := model.Request(ctx, messages, params)
	if err != nil || response == nil {
		return response, err
	}
	return accumulate(replayAsEvents(response), params, emit, observe)
}

func stampDirectResponse(model Model, response *ModelResponse) {
	if response.Timestamp.IsZero() {
		response.Timestamp = time.Now().UTC()
	}
	if response.ModelName == "" {
		response.ModelName = model.Name()
	}
}

func waitForDirectContinuation(ctx context.Context, model Model, response *ModelResponse) error {
	delayer, ok := model.(ModelContinuationDelayer)
	if !ok {
		return nil
	}
	delay := delayer.ContinuationDelay(*cloneModelResponse(response))
	if delay <= 0 {
		return nil
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return context.Cause(ctx)
	case <-timer.C:
		return nil
	}
}

func cancelDirectSuspendedResponse(ctx context.Context, model Model, response *ModelResponse) {
	canceler, ok := model.(SuspendedResponseCanceler)
	if !ok || response == nil || response.State != ModelResponseStateSuspended {
		return
	}
	_ = canceler.CancelSuspendedResponse(context.WithoutCancel(ctx), *cloneModelResponse(response))
}
