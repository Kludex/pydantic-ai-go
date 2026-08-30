package ai

import (
	"context"
	"iter"
)

// StreamedRun is a run in progress. Range over Events to consume it;
// Result is valid once the range completes without error.
type StreamedRun[Output any] struct {
	events iter.Seq2[StreamEvent, error]
	result *RunResult[Output]
}

// Events streams the run's events across every loop iteration: model
// output deltas, tool call starts and argument fragments, and one
// FinishEvent per model request. The sequence can be ranged once.
func (s *StreamedRun[Output]) Events() iter.Seq2[StreamEvent, error] { return s.events }

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
//	    if delta, ok := event.(ai.TextDeltaEvent); ok { fmt.Print(delta.Delta) }
//	}
//	result := stream.Result()
func (a *Agent[Deps, Output]) RunStream(ctx context.Context, prompt string, deps Deps, opts ...RunOption) *StreamedRun[Output] {
	return a.runStreamPrompt(ctx, UserPromptPart{Content: prompt}, deps, opts)
}

// RunStreamParts is RunStream with a multimodal prompt.
func (a *Agent[Deps, Output]) RunStreamParts(ctx context.Context, contents []UserContent, deps Deps, opts ...RunOption) *StreamedRun[Output] {
	return a.runStreamPrompt(ctx, UserPromptPart{Contents: contents}, deps, opts)
}

func (a *Agent[Deps, Output]) runStreamPrompt(ctx context.Context, prompt UserPromptPart, deps Deps, opts []RunOption) *StreamedRun[Output] {
	s := &StreamedRun[Output]{}
	s.events = func(yield func(StreamEvent, error) bool) {
		ctx, span := startRunSpan(ctx, a.model.Name())
		var err error
		defer func() {
			if s.result != nil {
				recordUsage(span, s.result.usage)
			}
			endSpan(span, err)
		}()

		var r *run[Deps, Output]
		r, err = a.newRun(ctx, prompt, deps, opts)
		if err != nil {
			yield(nil, err)
			return
		}
		stopped := false
		r.emit = func(event StreamEvent) bool {
			if !yield(event, nil) {
				stopped = true
				return false
			}
			return true
		}
		var result *RunResult[Output]
		result, err = r.loop(ctx)
		if stopped {
			err = nil // the consumer broke out; not a run failure
			return
		}
		if err != nil {
			yield(nil, err)
			return
		}
		s.result = result
	}
	return s
}

// replayAsEvents converts a complete response into the events a streaming
// model would have produced.
func replayAsEvents(resp *ModelResponse) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		for _, part := range resp.Parts {
			switch p := part.(type) {
			case TextPart:
				if !yield(TextDeltaEvent{Delta: p.Content}, nil) {
					return
				}
			case ThinkingPart:
				if !yield(ThinkingDeltaEvent{Delta: p.Content}, nil) {
					return
				}
			case ToolCallPart:
				if !yield(ToolCallStartEvent{ToolName: p.ToolName, ToolCallID: p.ToolCallID}, nil) {
					return
				}
				if !yield(ToolCallDeltaEvent{ArgsDelta: string(p.Args)}, nil) {
					return
				}
			}
		}
		yield(FinishEvent{Usage: resp.Usage, ModelName: resp.ModelName}, nil)
	}
}
