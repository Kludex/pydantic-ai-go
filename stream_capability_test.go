package ai_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
)

type uppercaseStreamCapability struct {
	runID  string
	events int
}

func (*uppercaseStreamCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (c *uppercaseStreamCapability) ProcessStreamEvent(
	_ context.Context, runInfo *ai.RunInfo, event ai.StreamEvent,
) (ai.StreamEvent, error) {
	c.runID = runInfo.RunID
	c.events++
	switch event := event.(type) {
	case ai.PartStartEvent:
		if text, ok := event.Part.(ai.TextPart); ok {
			text.Content = strings.ToUpper(text.Content)
			event.Part = text
		}
		return event, nil
	case ai.PartDeltaEvent:
		if text, ok := event.Delta.(ai.TextPartDelta); ok {
			text.ContentDelta = strings.ToUpper(text.ContentDelta)
			event.Delta = text
		}
		return event, nil
	case ai.PartEndEvent:
		return nil, nil
	default:
		return event, nil
	}
}

func TestStreamEventProcessorTransformsOnlyConsumerEvents(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.TextDeltaEvent{PartID: "text", Delta: "a"},
			ai.TextDeltaEvent{PartID: "text", Delta: "b"},
			ai.FinishEvent{},
		}
	})
	capability := &uppercaseStreamCapability{}
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(capability))
	stream := agent.RunStream(t.Context(), "go", deps{})
	var text string
	var partEnds int
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		text += streamText(event)
		if _, ok := event.(ai.PartEndEvent); ok {
			partEnds++
		}
	}
	if text != "AB" || partEnds != 0 {
		t.Fatalf("processor did not transform/filter events: text=%q ends=%d", text, partEnds)
	}
	if stream.Result().Output != "ab" || capability.runID == "" || capability.events != 5 {
		t.Fatalf("processor changed run state or missed context: result=%+v capability=%+v", stream.Result(), capability)
	}
}

type recordingStreamCapability struct {
	name  string
	order *[]string
}

func (*recordingStreamCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (c *recordingStreamCapability) ProcessStreamEvent(
	_ context.Context, _ *ai.RunInfo, event ai.StreamEvent,
) (ai.StreamEvent, error) {
	*c.order = append(*c.order, c.name)
	return event, nil
}

func TestStreamEventProcessorMiddlewareOrder(t *testing.T) {
	var order []string
	first := &recordingStreamCapability{name: "first", order: &order}
	second := &recordingStreamCapability{name: "second", order: &order}
	agent := ai.NewAgent[deps, string](newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{ai.TextDeltaEvent{Delta: "ok"}, ai.FinishEvent{}}
	}), ai.WithCapabilities(first, second))
	stream := agent.RunStream(t.Context(), "go", deps{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(order) < 2 || order[0] != "second" || order[1] != "first" {
		t.Fatalf("unexpected stream middleware order: %v", order)
	}
}

type wrapperCapability struct {
	wrapped   bool
	processed bool
	consume   bool
}

func (*wrapperCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (c *wrapperCapability) WrapRunEventStream(
	_ context.Context, _ *ai.RunInfo, stream ai.EventStream,
) ai.EventStream {
	c.wrapped = true
	return func(yield func(ai.StreamEvent, error) bool) {
		if !c.consume {
			return
		}
		for event, err := range stream {
			if !yield(event, err) {
				return
			}
		}
	}
}

func (c *wrapperCapability) ProcessStreamEvent(
	context.Context, *ai.RunInfo, ai.StreamEvent,
) (ai.StreamEvent, error) {
	c.processed = true
	return nil, nil
}

func TestRunEventStreamWrapperTakesPrecedenceAndRunsForRun(t *testing.T) {
	capability := &wrapperCapability{consume: true}
	agent := ai.NewAgent[deps, string](newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{ai.TextDeltaEvent{Delta: "ok"}, ai.FinishEvent{}}
	}), ai.WithCapabilities(capability))
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "ok" || !capability.wrapped || capability.processed {
		t.Fatalf("wrapper precedence failed: result=%+v capability=%+v", result, capability)
	}
}

func TestRunEventStreamWrapperMustDriveRun(t *testing.T) {
	capability := &wrapperCapability{}
	agent := ai.NewAgent[deps, string](newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{ai.TextDeltaEvent{Delta: "unseen"}, ai.FinishEvent{}}
	}), ai.WithCapabilities(capability))
	_, err := agent.Run(t.Context(), "go", deps{})
	var unexpected *ai.UnexpectedModelBehaviorError
	if !errors.As(err, &unexpected) || !strings.Contains(err.Error(), "event stream ended") {
		t.Fatalf("unexpected undriven-stream error: %v", err)
	}
}

type failingStreamCapability struct {
	calls int
}

func (*failingStreamCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (c *failingStreamCapability) ProcessStreamEvent(
	context.Context, *ai.RunInfo, ai.StreamEvent,
) (ai.StreamEvent, error) {
	c.calls++
	return nil, errors.New("event rejected")
}

func TestStreamEventProcessorErrorStopsRun(t *testing.T) {
	newAgent := func(capability *failingStreamCapability) *ai.Agent[deps, string] {
		return ai.NewAgent[deps, string](newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
			return []ai.ModelStreamEvent{ai.TextDeltaEvent{Delta: "unseen"}, ai.FinishEvent{}}
		}), ai.WithCapabilities(capability))
	}
	capability := &failingStreamCapability{}
	stream := newAgent(capability).RunStream(t.Context(), "go", deps{})
	var got error
	for _, err := range stream.Events() {
		if err != nil {
			got = err
		}
	}
	if got == nil || got.Error() != "event rejected" || stream.Result() != nil || capability.calls != 1 {
		t.Fatalf("unexpected processor failure: err=%v result=%+v calls=%d", got, stream.Result(), capability.calls)
	}

	capability = &failingStreamCapability{}
	if _, err := newAgent(capability).Run(t.Context(), "go", deps{}); err == nil || err.Error() != "event rejected" {
		t.Fatalf("Run did not activate the event processor: %v", err)
	}
}

func TestStreamEventProcessorForwardsSourceErrors(t *testing.T) {
	capability := &recordingStreamCapability{name: "processor", order: new([]string)}
	agent := ai.NewAgent[deps, string](
		&streamingErrModel{Model: newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent { return nil })},
		ai.WithCapabilities(capability),
	)
	stream := agent.RunStream(t.Context(), "go", deps{})
	var got error
	for _, err := range stream.Events() {
		if err != nil {
			got = err
		}
	}
	if got == nil || got.Error() != "mid-stream failure" {
		t.Fatalf("source error was not forwarded: %v", got)
	}
}

func TestStreamEventProcessorStopsOnConsumerBreak(t *testing.T) {
	capability := &recordingStreamCapability{name: "processor", order: new([]string)}
	agent := ai.NewAgent[deps, string](newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{ai.TextDeltaEvent{Delta: "a"}, ai.TextDeltaEvent{Delta: "b"}, ai.FinishEvent{}}
	}), ai.WithCapabilities(capability))
	stream := agent.RunStream(t.Context(), "go", deps{})
	for range stream.Events() {
		break
	}
	if stream.Result() != nil {
		t.Fatal("consumer break unexpectedly completed the run")
	}
}
