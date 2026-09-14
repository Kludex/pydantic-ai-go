package ai

import (
	"context"
	"errors"
	"iter"
	"slices"
	"strconv"
	"sync"
	"time"
)

// StreamedRun is a run in progress. Range over Events to consume it;
// Result is valid once the range completes without error.
type StreamedRun[Output any] struct {
	events        EventStream
	result        *RunResult[Output]
	suspended     *SuspendedRun
	partialOutput func(raw, toolCallID string) (Output, bool, error)
	usageMu       sync.RWMutex
	usage         Usage
}

// SuspendedRun is a resumable snapshot captured when a stream consumer
// detaches from a provider-managed suspended response.
type SuspendedRun struct {
	response ModelResponse
	messages []ModelMessage
	usage    Usage
}

// Response returns the detached suspended response.
func (s *SuspendedRun) Response() ModelResponse { return *cloneModelResponse(&s.response) }

// Messages returns history suitable for Agent.Resume or Agent.ResumeStream.
func (s *SuspendedRun) Messages() []ModelMessage { return cloneModelMessages(s.messages) }

// Usage returns usage accumulated through the detached response.
func (s *SuspendedRun) Usage() Usage { return s.usage.Clone() }

// Usage returns a detached snapshot accumulated so far. During a model stream,
// it includes the provider's latest token snapshot and a best-effort cost.
func (s *StreamedRun[Output]) Usage() Usage {
	s.usageMu.RLock()
	defer s.usageMu.RUnlock()
	return s.usage.Clone()
}

func (s *StreamedRun[Output]) setUsage(usage Usage) {
	s.usageMu.Lock()
	defer s.usageMu.Unlock()
	s.usage = usage.Clone()
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
	return cloneModelResponse(&ModelResponse{Parts: []ResponsePart{part}}).Parts[0]
}

func (s *StreamedRun[Output]) yieldPartialOutput(
	yield func(Output, error) bool, part ResponsePart,
) bool {
	var raw, toolCallID string
	switch part := part.(type) {
	case TextPart:
		raw = part.Content
	case SpeechPart:
		raw = part.Content()
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

// Suspended returns a resumable snapshot after the consumer stops a stream
// whose provider response is suspended. It returns nil for completed,
// failed, explicitly canceled, or non-resumable streams.
func (s *StreamedRun[Output]) Suspended() *SuspendedRun {
	if s.suspended == nil {
		return nil
	}
	return &SuspendedRun{
		response: s.suspended.Response(), messages: s.suspended.Messages(), usage: s.suspended.Usage(),
	}
}

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

// ResumeStream continues a provider response left suspended in history without
// adding a new user prompt. Setup failures are yielded from StreamedRun.Events.
func (a *Agent[Deps, Output]) ResumeStream(
	ctx context.Context, history []ModelMessage, deps Deps, opts ...RunOption,
) *StreamedRun[Output] {
	return a.runStreamPrompt(
		ctx, UserPromptPart{}, deps, suspendedRunOptions(history, opts), true,
	)
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
		capabilities := append(slices.Clone(a.capabilities), cfg.capabilities...)
		ctx, span := startRunSpan(ctx, modelName(model), !hasInstrumentationCapability(capabilities))
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
		closed := false
		closeRun := func() error {
			if closed {
				return nil
			}
			closed = true
			closeErr := run.closeRunResources(context.WithoutCancel(ctx))
			run.cancellation.finish()
			return closeErr
		}
		defer func() { _ = closeRun() }()
		run.recordSelectedModel = func(name string) { recordRunModel(span, name) }
		run.observeUsage = streamedRun.setUsage
		run.publishUsage(nil)
		run.commitStreamedOutput = commitFirstOutput
		streamedRun.partialOutput = func(raw, toolCallID string) (Output, bool, error) {
			return run.validatePartialOutput(run.ctx, raw, toolCallID)
		}

		stopped := false
		core := EventStream(func(yieldCore func(StreamEvent, error) bool) {
			if !run.setEventEmitter(func(event StreamEvent) bool {
				if !yieldCore(event, nil) {
					stopped = true
					run.cancellation.stopStream()
					return false
				}
				return true
			}) {
				if eventErr := run.streamEventError(); eventErr != nil {
					runErr = eventErr
					yieldCore(nil, eventErr)
				}
				return
			}
			result, err := run.wrappedLoop(run.ctx)
			if stopped {
				streamedRun.suspended = run.suspendedSnapshot()
				return
			}
			if err != nil {
				runErr = err
				yieldCore(nil, err)
				return
			}
			streamedRun.result = result
		})
		stream := wrapEventStream(run.ctx, run.info, core, run.capabilities, run.capabilityIDs)
		for event, err := range stream {
			if err != nil {
				runErr = err
			}
			if !yield(event, err) {
				return
			}
		}
		if closeErr := closeRun(); closeErr != nil {
			runErr = errors.Join(runErr, closeErr)
			streamedRun.result = nil
			yield(nil, closeErr)
		}
	}
	return streamedRun
}

func (r *run[Deps, Output]) suspendedSnapshot() *SuspendedRun {
	if r.detachedResponse == nil || r.detachedResponse.State != ModelResponseStateSuspended {
		return nil
	}
	response := cloneModelResponse(r.detachedResponse)
	if response.Timestamp.IsZero() {
		response.Timestamp = time.Now().UTC()
	}
	if response.RunID == "" {
		response.RunID = r.rc.RunID
	}
	if response.ConversationID == "" {
		response.ConversationID = r.rc.ConversationID
	}
	messages := cloneModelMessages(r.messages)
	messages = append(messages, *cloneModelResponse(response))
	usage := r.usage.Clone()
	usage.Add(response.Usage)
	return &SuspendedRun{response: *response, messages: messages, usage: usage}
}

func hasEventStreamCapability(capabilities []Capability) bool {
	for _, capability := range capabilities {
		if _, ok := capability.(RunEventStreamWrapper); ok {
			return true
		}
		if _, ok := capability.(StreamEventProcessor); ok {
			return true
		}
		if _, ok := capability.(EventListener); ok {
			return true
		}
	}
	return false
}

func wrapEventStream(
	ctx context.Context,
	runInfo *RunInfo,
	stream EventStream,
	capabilities []Capability,
	capabilityIDs []string,
) EventStream {
	for index := len(capabilities) - 1; index >= 0; index-- {
		capability := capabilities[index]
		info := *runInfo
		info.capabilityID = capabilityIDs[index]
		if wrapper, ok := capability.(RunEventStreamWrapper); ok {
			stream = wrapper.WrapRunEventStream(ctx, &info, stream)
		} else if processor, ok := capability.(StreamEventProcessor); ok {
			stream = processEventStream(ctx, &info, stream, processor)
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
			case SpeechPart:
				if !yield(SpeechDeltaEvent{PartID: partID, Part: cloneSpeechPart(part)}, nil) {
					return
				}
			case FilePart:
				if !yield(FileEvent{PartID: partID, Part: cloneResponsePart(part).(FilePart)}, nil) {
					return
				}
			case ThinkingPart:
				if !yield(ThinkingDeltaEvent{
					PartID: partID, Delta: part.Content, ID: part.ID, SignatureDelta: part.Signature,
					ProviderName: part.ProviderName, ProviderDetails: cloneSchemaMap(part.ProviderDetails),
				}, nil) {
					return
				}
			case CompactionPart:
				if !yield(CompactionEvent{
					PartID: partID, Content: part.Content, ID: part.ID,
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
			case NativeToolCallPart:
				if !yield(ToolCallStartEvent{
					PartID: partID, ToolName: part.ToolName, ToolCallID: part.ToolCallID,
					ToolKind: part.ToolKind, ID: part.ID,
					ProviderName: part.ProviderName, ProviderDetails: cloneSchemaMap(part.ProviderDetails), Native: true,
				}, nil) {
					return
				}
				if !yield(ToolCallDeltaEvent{PartID: partID, ArgsDelta: string(part.Args)}, nil) {
					return
				}
			case NativeToolReturnPart:
				if !yield(NativeToolReturnEvent{PartID: partID, Part: cloneResponsePart(part).(NativeToolReturnPart)}, nil) {
					return
				}
			}
		}
		state := response.State
		if state == "" {
			state = ModelResponseStateComplete
		}
		yield(FinishEvent{
			Parts: cloneModelResponse(response).Parts,
			Usage: response.Usage, ModelName: response.ModelName, Timestamp: response.Timestamp,
			ProviderName: response.ProviderName, ProviderURL: response.ProviderURL,
			ProviderDetails: cloneSchemaMap(response.ProviderDetails), Metadata: cloneSchemaMap(response.Metadata),
			ProviderResponseID: response.ProviderResponseID, FinishReason: response.FinishReason, State: state,
		}, nil)
	}
}
