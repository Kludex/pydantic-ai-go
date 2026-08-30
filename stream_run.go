package ai

import (
	"context"
	"iter"
	"strconv"
)

// StreamedRun is a run in progress. Range over Events to consume it;
// Result is valid once the range completes without error.
type StreamedRun[Output any] struct {
	events EventStream
	result *RunResult[Output]
}

// Events streams normalized part lifecycle, final-result, and finish events
// across every model request in the run. The sequence can be ranged once.
func (s *StreamedRun[Output]) Events() EventStream { return s.events }

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
	return a.runStreamPrompt(ctx, UserPromptPart{Content: prompt}, deps, opts)
}

// RunStreamParts is RunStream with a multimodal prompt.
func (a *Agent[Deps, Output]) RunStreamParts(
	ctx context.Context, contents []UserContent, deps Deps, opts ...RunOption,
) *StreamedRun[Output] {
	return a.runStreamPrompt(ctx, UserPromptPart{Contents: contents}, deps, opts)
}

func (a *Agent[Deps, Output]) runStreamPrompt(
	ctx context.Context, prompt UserPromptPart, deps Deps, opts []RunOption,
) *StreamedRun[Output] {
	streamedRun := &StreamedRun[Output]{}
	streamedRun.events = func(yield func(StreamEvent, error) bool) {
		ctx, span := startRunSpan(ctx, a.model.Name())
		var runErr error
		defer func() {
			if streamedRun.result != nil {
				recordUsage(span, streamedRun.result.usage)
			}
			endSpan(span, runErr)
		}()

		run, err := a.newRun(ctx, prompt, deps, opts)
		if err != nil {
			runErr = err
			yield(nil, err)
			return
		}
		defer run.cancellation.finish()

		stopped := false
		core := EventStream(func(yieldCore func(StreamEvent, error) bool) {
			run.emit = func(event StreamEvent) bool {
				if !yieldCore(event, nil) {
					stopped = true
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
		stream := wrapEventStream(run.ctx, run.info, core, a.capabilities)
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
				if !yield(TextDeltaEvent{PartID: partID, Delta: part.Content}, nil) {
					return
				}
			case ThinkingPart:
				if !yield(ThinkingDeltaEvent{PartID: partID, Delta: part.Content}, nil) {
					return
				}
			case ToolCallPart:
				if !yield(ToolCallStartEvent{
					PartID: partID, ToolName: part.ToolName, ToolCallID: part.ToolCallID,
				}, nil) {
					return
				}
				if !yield(ToolCallDeltaEvent{PartID: partID, ArgsDelta: string(part.Args)}, nil) {
					return
				}
			}
		}
		yield(FinishEvent{Usage: response.Usage, ModelName: response.ModelName}, nil)
	}
}
