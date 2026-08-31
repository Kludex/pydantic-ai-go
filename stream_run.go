package ai

import (
	"context"
	"iter"
	"slices"
	"strconv"
	"time"
)

// StreamedRun is a run in progress. Range over Events to consume it;
// Result is valid once the range completes without error.
type StreamedRun[Output any] struct {
	events        EventStream
	result        *RunResult[Output]
	partialOutput func(raw, toolCallID string) (Output, bool, error)
}

// Events streams normalized part lifecycle, final-result, and finish events
// across every model request in the run. The sequence can be ranged once.
func (s *StreamedRun[Output]) Events() EventStream { return s.events }

// Outputs streams typed partial outputs and always ends with the fully
// validated final output. It is an alternative view of Events; consume only
// one view for a run.
func (s *StreamedRun[Output]) Outputs() iter.Seq2[Output, error] {
	return s.outputs(0)
}

// OutputsDebounced groups partial snapshots over interval. A zero interval
// disables grouping. The fully validated final output is always emitted.
func (s *StreamedRun[Output]) OutputsDebounced(interval time.Duration) iter.Seq2[Output, error] {
	if interval < 0 {
		panic("ai: output debounce interval must be non-negative")
	}
	return s.outputs(interval)
}

func (s *StreamedRun[Output]) outputs(interval time.Duration) iter.Seq2[Output, error] {
	return func(yield func(Output, error) bool) {
		parts := map[int]ResponsePart{}
		selectedIndex := -1
		lastStarted := -1
		var pending ResponsePart
		var groupStarted time.Time
		flush := func() bool {
			if pending == nil {
				return true
			}
			part := pending
			pending = nil
			groupStarted = time.Time{}
			return s.yieldPartialOutput(yield, part)
		}
		queue := func(part ResponsePart) bool {
			if interval == 0 {
				return s.yieldPartialOutput(yield, part)
			}
			now := time.Now()
			if pending != nil && now.Sub(groupStarted) >= interval && !flush() {
				return false
			}
			pending = cloneResponsePart(part)
			if groupStarted.IsZero() {
				groupStarted = now
			}
			return true
		}
		for event, err := range s.Events() {
			if err != nil {
				yield(*new(Output), err)
				return
			}
			switch event := event.(type) {
			case PartStartEvent:
				parts[event.Index] = event.Part
				lastStarted = event.Index
			case PartDeltaEvent:
				part, ok := parts[event.Index]
				if !ok {
					continue
				}
				part, err = event.Delta.Apply(part)
				if err != nil {
					yield(*new(Output), err)
					return
				}
				parts[event.Index] = part
				if event.Index == selectedIndex && !queue(part) {
					return
				}
			case FinalResultEvent:
				selectedIndex = lastStarted
				if part, ok := parts[selectedIndex]; ok && !queue(part) {
					return
				}
			}
		}
		if !flush() {
			return
		}
		if s.result != nil {
			yield(s.result.Output, nil)
		}
	}
}

func cloneResponsePart(part ResponsePart) ResponsePart {
	if call, ok := part.(ToolCallPart); ok {
		call.Args = slices.Clone(call.Args)
		return call
	}
	return part
}

func (s *StreamedRun[Output]) yieldPartialOutput(
	yield func(Output, error) bool, part ResponsePart,
) bool {
	var raw, toolCallID string
	switch part := part.(type) {
	case TextPart:
		raw = part.Content
	case ToolCallPart:
		raw, toolCallID = string(part.Args), part.ToolCallID
	default:
		return true
	}
	output, valid, err := s.partialOutput(raw, toolCallID)
	if err != nil {
		yield(*new(Output), err)
		return false
	}
	return !valid || yield(output, nil)
}

// Result returns the final result. It is nil until the event stream has
// been fully consumed without error.
func (s *StreamedRun[Output]) Result() *RunResult[Output] { return s.result }

// RunStream executes the agent loop like Run, but yields events as the
// model produces them. Models that do not implement StreamingModel are
// driven with plain requests; each response is replayed as events.
//
//	stream := agent.RunStream(ctx, "hello", deps)
//	for event, err := range stream.Events() {
//	    if err != nil { ... }
//	    switch event := event.(type) {
//	    case ai.PartStartEvent:
//	        // Handle the first content for event.Part.
//	    case ai.PartDeltaEvent:
//	        // Apply or render event.Delta.
//	    }
//	}
//	result := stream.Result()
func (a *Agent[Deps, Output]) RunStream(
	ctx context.Context, prompt string, deps Deps, opts ...RunOption,
) *StreamedRun[Output] {
	return a.runStreamPrompt(ctx, UserPromptPart{Content: prompt}, deps, opts, true)
}

// RunStreamParts is RunStream with a multimodal prompt.
func (a *Agent[Deps, Output]) RunStreamParts(
	ctx context.Context, contents []UserContent, deps Deps, opts ...RunOption,
) *StreamedRun[Output] {
	return a.runStreamPrompt(ctx, UserPromptPart{Contents: contents}, deps, opts, true)
}

func (a *Agent[Deps, Output]) runStreamPrompt(
	ctx context.Context, prompt UserPromptPart, deps Deps, opts []RunOption, commitFirstOutput bool,
) *StreamedRun[Output] {
	streamedRun := &StreamedRun[Output]{}
	streamedRun.events = func(yield func(StreamEvent, error) bool) {
		cfg := buildRunConfig(opts)
		model := a.model
		if cfg.model != nil {
			model = cfg.model
		}
		ctx, span := startRunSpan(ctx, modelName(model))
		var runErr error
		defer func() {
			if streamedRun.result != nil {
				recordUsage(span, streamedRun.result.usage)
			}
			endSpan(span, runErr)
		}()

		run, err := a.newRun(ctx, prompt, deps, cfg)
		if err != nil {
			runErr = err
			yield(nil, err)
			return
		}
		defer run.cancellation.finish()
		run.recordSelectedModel = func(name string) { recordRunModel(span, name) }
		run.commitStreamedOutput = commitFirstOutput
		streamedRun.partialOutput = func(raw, toolCallID string) (Output, bool, error) {
			return run.validatePartialOutput(run.ctx, raw, toolCallID)
		}

		stopped := false
		core := EventStream(func(yieldCore func(StreamEvent, error) bool) {
			run.emit = func(event StreamEvent) bool {
				if !yieldCore(event, nil) {
					stopped = true
					run.cancellation.stopStream()
					return false
				}
				return true
			}
			result, err := run.wrappedLoop(run.ctx)
			if stopped {
				return
			}
			if err != nil {
				runErr = err
				yieldCore(nil, err)
				return
			}
			streamedRun.result = result
		})
		stream := wrapEventStream(run.ctx, run.info, core, run.capabilities)
		for event, err := range stream {
			if err != nil {
				runErr = err
			}
			if !yield(event, err) {
				return
			}
		}
	}
	return streamedRun
}

func hasEventStreamCapability(capabilities []Capability) bool {
	for _, capability := range capabilities {
		if _, ok := capability.(RunEventStreamWrapper); ok {
			return true
		}
		if _, ok := capability.(StreamEventProcessor); ok {
			return true
		}
	}
	return false
}

func wrapEventStream(
	ctx context.Context, runInfo *RunInfo, stream EventStream, capabilities []Capability,
) EventStream {
	for index := len(capabilities) - 1; index >= 0; index-- {
		capability := capabilities[index]
		if wrapper, ok := capability.(RunEventStreamWrapper); ok {
			stream = wrapper.WrapRunEventStream(ctx, runInfo, stream)
		} else if processor, ok := capability.(StreamEventProcessor); ok {
			stream = processEventStream(ctx, runInfo, stream, processor)
		}
	}
	return stream
}

func processEventStream(
	ctx context.Context, runInfo *RunInfo, stream EventStream, processor StreamEventProcessor,
) EventStream {
	return func(yield func(StreamEvent, error) bool) {
		for event, err := range stream {
			if err != nil {
				yield(nil, err)
				return
			}
			event, err = processor.ProcessStreamEvent(ctx, runInfo, event)
			if err != nil {
				yield(nil, err)
				return
			}
			if event != nil && !yield(event, nil) {
				return
			}
		}
	}
}

// replayAsEvents converts a complete response into the provider deltas a
// streaming model would have produced.
func replayAsEvents(response *ModelResponse) iter.Seq2[ModelStreamEvent, error] {
	return func(yield func(ModelStreamEvent, error) bool) {
		for index, part := range response.Parts {
			partID := strconv.Itoa(index)
			switch part := part.(type) {
			case TextPart:
				if !yield(TextDeltaEvent{
					PartID: partID, Delta: part.Content, ID: part.ID,
					ProviderName: part.ProviderName, ProviderDetails: cloneSchemaMap(part.ProviderDetails),
				}, nil) {
					return
				}
			case ThinkingPart:
				if !yield(ThinkingDeltaEvent{
					PartID: partID, Delta: part.Content, ID: part.ID, SignatureDelta: part.Signature,
					ProviderName: part.ProviderName, ProviderDetails: cloneSchemaMap(part.ProviderDetails),
				}, nil) {
					return
				}
			case ToolCallPart:
				if !yield(ToolCallStartEvent{
					PartID: partID, ToolName: part.ToolName, ToolCallID: part.ToolCallID,
					ToolKind: part.ToolKind, ID: part.ID,
					ProviderName: part.ProviderName, ProviderDetails: cloneSchemaMap(part.ProviderDetails),
				}, nil) {
					return
				}
				if !yield(ToolCallDeltaEvent{PartID: partID, ArgsDelta: string(part.Args)}, nil) {
					return
				}
			}
		}
		state := response.State
		if state == "" {
			state = ModelResponseStateComplete
		}
		yield(FinishEvent{
			Usage: response.Usage, ModelName: response.ModelName, Timestamp: response.Timestamp,
			ProviderName: response.ProviderName, ProviderURL: response.ProviderURL,
			ProviderDetails: cloneSchemaMap(response.ProviderDetails), ProviderResponseID: response.ProviderResponseID,
			FinishReason: response.FinishReason, State: state,
		}, nil)
	}
}
