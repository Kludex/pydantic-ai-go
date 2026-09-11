package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"regexp"
	"slices"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type progressPayload struct {
	Done int
}

func TestToolEmitsTypedCustomEventWithAttribution(t *testing.T) {
	requests := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "call-1", Args: json.RawMessage(`{}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	var listened []*ai.CustomEvent[progressPayload]
	ai.OnEvent(agent, func(
		_ context.Context, _ *ai.RunContext[deps], event *ai.CustomEvent[progressPayload],
	) error {
		listened = append(listened, event)
		return nil
	})
	ai.AddTool(agent, "work", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (string, error) {
		event := ai.NewCustomEvent("progress", progressPayload{Done: 1})
		if err := rc.Emit(event); err != nil {
			return "", err
		}
		return "worked", nil
	})

	stream := agent.RunStream(t.Context(), "go", deps{})
	var emitted []*ai.CustomEvent[progressPayload]
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(*ai.CustomEvent[progressPayload]); ok {
			emitted = append(emitted, event)
		}
	}
	if stream.Result() == nil || stream.Result().Output != "done" || len(emitted) != 1 || len(listened) != 1 {
		t.Fatalf("unexpected custom events: emitted=%+v listened=%+v result=%+v", emitted, listened, stream.Result())
	}
	if emitted[0] != listened[0] || emitted[0].Data.Done != 1 ||
		emitted[0].ToolName != "work" || emitted[0].ToolCallID != "call-1" {
		t.Fatalf("custom event lost type or attribution: %+v", emitted[0])
	}
}

type emittingCapability struct {
	id string
}

func (*emittingCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (capability *emittingCapability) CapabilityID() string { return capability.id }

func (*emittingCapability) BeforeModelRequest(
	_ context.Context, info *ai.RunInfo, request ai.ModelRequestContext,
) (ai.ModelRequestContext, error) {
	return request, info.Emit(ai.NewCapabilityEvent("indexer", "started", progressPayload{Done: 2}))
}

func TestCapabilityEventsUseExplicitAndSyntheticRunIDs(t *testing.T) {
	for _, test := range []struct {
		name string
		id   string
	}{
		{name: "explicit", id: "search"},
		{name: "synthetic"},
	} {
		t.Run(test.name, func(t *testing.T) {
			capability := &emittingCapability{id: test.id}
			agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(capability))
			stream := agent.RunStream(t.Context(), "go", deps{})
			var emitted *ai.CapabilityEvent[progressPayload]
			for event, err := range stream.Events() {
				if err != nil {
					t.Fatal(err)
				}
				if event, ok := event.(*ai.CapabilityEvent[progressPayload]); ok {
					emitted = event
				}
			}
			if emitted == nil || emitted.Kind != "indexer.started" || emitted.Data.Done != 2 {
				t.Fatalf("unexpected capability event: %+v", emitted)
			}
			if test.id != "" && emitted.CapabilityID != test.id {
				t.Fatalf("explicit capability ID was not preserved: %+v", emitted)
			}
			if test.id == "" && !regexp.MustCompile(`^<emittingCapability:[0-9a-f]{6}>$`).MatchString(emitted.CapabilityID) {
				t.Fatalf("unexpected synthetic capability ID %q", emitted.CapabilityID)
			}
		})
	}
}

type listenerCapability struct {
	order      *[]string
	failure    error
	seenID     string
	mutateDone int
}

func (*listenerCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (capability *listenerCapability) OnEvent(
	_ context.Context, info *ai.RunInfo, event ai.StreamEvent,
) error {
	if capability.order != nil {
		*capability.order = append(*capability.order, "capability")
	}
	capability.seenID = info.CapabilityID()
	if typed, ok := event.(*ai.CapabilityEvent[progressPayload]); ok && capability.mutateDone != 0 {
		typed.Data.Done = capability.mutateDone
	}
	return capability.failure
}

func TestAgentEventListenersRunAfterCapabilitiesForRun(t *testing.T) {
	var order []string
	capability := &listenerCapability{order: &order}
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(capability))
	agent.AddEventListener(func(context.Context, *ai.RunContext[deps], ai.StreamEvent) error {
		order = append(order, "agent")
		return nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if len(order) == 0 || len(order)%2 != 0 {
		t.Fatalf("listeners missed events: %v", order)
	}
	for index := 0; index < len(order); index += 2 {
		if !slices.Equal(order[index:index+2], []string{"capability", "agent"}) {
			t.Fatalf("unexpected listener order: %v", order)
		}
	}
}

func TestCapabilityEventListenersCanSettleEventsSynchronously(t *testing.T) {
	listener := &listenerCapability{mutateDone: 9}
	emitter := &emittingCapability{id: "emitter"}
	agent := ai.NewAgent[deps, string](
		fakes.NewTestModel(), ai.WithCapabilities(emitter, listener),
	)
	stream := agent.RunStream(t.Context(), "go", deps{})
	var emitted *ai.CapabilityEvent[progressPayload]
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(*ai.CapabilityEvent[progressPayload]); ok {
			emitted = event
		}
	}
	if emitted == nil || emitted.Data.Done != 9 || listener.seenID == "" {
		t.Fatalf("capability listener did not settle the event: event=%+v listener=%+v", emitted, listener)
	}
}

func TestNestedListenerEmissionPreservesCauseFirstOrder(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	ai.OnEvent(agent, func(
		_ context.Context, rc *ai.RunContext[deps], event *ai.CustomEvent[progressPayload],
	) error {
		if event.Name == "parent" {
			return rc.Emit(ai.NewCustomEvent("child", progressPayload{Done: 2}))
		}
		return nil
	})
	run, err := agent.StartRun(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if err := run.Emit(ai.NewCustomEvent("parent", progressPayload{Done: 1})); err != nil {
		t.Fatal(err)
	}
	var names []string
	for event, err := range run.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(*ai.CustomEvent[progressPayload]); ok {
			names = append(names, event.Name)
		}
	}
	if !slices.Equal(names, []string{"parent", "child"}) {
		t.Fatalf("nested events were reordered: %v", names)
	}
}

func TestAgentRunEmitsBeforeFirstGeneratedEventAndRejectsLateEmission(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	run, err := agent.StartRun(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if err := run.Emit(nil); err == nil {
		t.Fatal("nil event emission succeeded")
	}
	var typedNil *ai.CustomEvent[progressPayload]
	if err := run.Emit(typedNil); err == nil {
		t.Fatal("typed nil event emission succeeded")
	}
	if err := run.Emit(&ai.CustomEvent[progressPayload]{}); err == nil {
		t.Fatal("unnamed custom event emission succeeded")
	}
	if err := run.Emit(&ai.CapabilityEvent[progressPayload]{}); err == nil {
		t.Fatal("unnamed capability event emission succeeded")
	}
	custom := ai.NewCustomEvent("external", progressPayload{Done: 3})
	if err := run.Emit(custom); err != nil {
		t.Fatal(err)
	}
	first, ok, err := run.Next()
	if err != nil || !ok || first != custom {
		t.Fatalf("pre-run event was not first: event=%+v ok=%v err=%v", first, ok, err)
	}
	for range run.Events() {
	}
	if err := run.Emit(ai.NewCustomEvent("late", struct{}{})); err == nil {
		t.Fatal("late event emission succeeded")
	}
}

func TestNilTypedEventListenerIsRejected(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	defer func() {
		if recover() == nil {
			t.Fatal("expected nil-listener panic")
		}
	}()
	ai.OnEvent[deps, string, *ai.CustomEvent[progressPayload]](agent, nil)
}

func TestInvalidEventEmissionIsRejected(t *testing.T) {
	var info ai.RunInfo
	if used, known := info.ContextWindowUsed(); known || used != 0 {
		t.Fatalf("detached run info reported context-window usage: used=%v known=%v", used, known)
	}
	if err := info.Emit(ai.NewCapabilityEvent("cap", "event", struct{}{})); err == nil {
		t.Fatal("detached run info emitted an event")
	}
	var runContext ai.RunContext[deps]
	if err := runContext.Emit(ai.NewCustomEvent("event", struct{}{})); err == nil {
		t.Fatal("detached run context emitted an event")
	}
	for name, fn := range map[string]func(){
		"custom":               func() { ai.NewCustomEvent("", struct{}{}) },
		"capability namespace": func() { ai.NewCapabilityEvent("", "event", struct{}{}) },
		"capability name":      func() { ai.NewCapabilityEvent("cap", "", struct{}{}) },
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			fn()
		})
	}
	defer func() {
		if recover() == nil {
			t.Fatal("expected nil-listener panic")
		}
	}()
	ai.NewAgent[deps, string](fakes.NewTestModel()).AddEventListener(nil)
}

func TestEmittingFrameworkOrCapabilityEventFromApplicationIsRejected(t *testing.T) {
	var runContext *ai.RunContext[deps]
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	agent.AddInstructionsFunc(func(_ context.Context, rc *ai.RunContext[deps]) (string, error) {
		runContext = rc
		if err := rc.Emit(ai.PartStartEvent{}); err == nil {
			return "", errors.New("framework event emission succeeded")
		}
		if err := rc.Emit(ai.NewCapabilityEvent("cap", "event", struct{}{})); err == nil {
			return "", errors.New("capability event emission succeeded")
		}
		return "", nil
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if err := runContext.Emit(ai.NewCustomEvent("late", struct{}{})); err == nil {
		t.Fatal("finished run context emitted an event")
	}
}

func TestCapabilityCannotEmitCustomEvent(t *testing.T) {
	capability := &customEmittingCapability{}
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(capability))
	stream := agent.RunStream(t.Context(), "go", deps{})
	var got error
	for _, err := range stream.Events() {
		if err != nil {
			got = err
		}
	}
	if got == nil || !strings.Contains(got.Error(), "capabilities must emit capability events") {
		t.Fatalf("unexpected capability emission error: %v", got)
	}
}

type customEmittingCapability struct{}

func (*customEmittingCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (*customEmittingCapability) BeforeModelRequest(
	_ context.Context, info *ai.RunInfo, request ai.ModelRequestContext,
) (ai.ModelRequestContext, error) {
	return request, info.Emit(ai.NewCustomEvent("invalid", struct{}{}))
}

func TestCapabilityEventListenerErrorStopsRun(t *testing.T) {
	sentinel := errors.New("capability listener failed")
	capability := &listenerCapability{failure: sentinel}
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(capability))
	if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, sentinel) {
		t.Fatalf("unexpected capability listener error: %v", err)
	}
}

func TestEventListenerErrorStopsRun(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	sentinel := errors.New("listener failed")
	agent.AddEventListener(func(context.Context, *ai.RunContext[deps], ai.StreamEvent) error {
		return sentinel
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, sentinel) {
		t.Fatalf("unexpected listener error: %v", err)
	}
}

func TestEventEmitterObservesConsumerDetachment(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "work", ToolCallID: "work", Args: json.RawMessage(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	emitted := make(chan error, 1)
	ai.AddTool(agent, "work", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (string, error) {
		err := rc.Emit(ai.NewCustomEvent("detach", struct{}{}))
		emitted <- err
		return "done", err
	})
	stream := agent.RunStream(t.Context(), "go", deps{})
	for event := range stream.Events() {
		if _, ok := event.(*ai.CustomEvent[struct{}]); ok {
			break
		}
	}
	if err := <-emitted; err == nil {
		t.Fatal("event emitter did not observe consumer detachment")
	}
}

func TestBufferedRunStreamEventReturnsListenerError(t *testing.T) {
	sentinel := errors.New("listener failed")
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	agent.AddMetadataFunc(func(_ context.Context, rc *ai.RunContext[deps]) (map[string]any, error) {
		return nil, rc.Emit(ai.NewCustomEvent("queued", struct{}{}))
	})
	ai.OnEvent(agent, func(
		context.Context, *ai.RunContext[deps], *ai.CustomEvent[struct{}],
	) error {
		return sentinel
	})
	stream := agent.RunStream(t.Context(), "go", deps{})
	var got error
	for _, err := range stream.Events() {
		if err != nil {
			got = err
		}
	}
	if !errors.Is(got, sentinel) {
		t.Fatalf("unexpected buffered listener error: %v", got)
	}
}

func TestBufferedAgentRunEventReturnsListenerError(t *testing.T) {
	sentinel := errors.New("listener failed")
	capability := &listenerCapability{failure: sentinel}
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(capability))
	run, err := agent.StartRun(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if err := run.Emit(ai.NewCustomEvent("queued", struct{}{})); err != nil {
		t.Fatal(err)
	}
	if _, ok, err := run.Next(); ok || !errors.Is(err, sentinel) {
		t.Fatalf("unexpected buffered listener result: ok=%v err=%v", ok, err)
	}
}

func TestEmittedEventReturnsListenerError(t *testing.T) {
	requests := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "work", Args: json.RawMessage(`{}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "unreachable"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	sentinel := errors.New("listener failed")
	ai.OnEvent(agent, func(
		context.Context, *ai.RunContext[deps], *ai.CustomEvent[progressPayload],
	) error {
		return sentinel
	})
	ai.AddTool(agent, "work", func(
		_ context.Context, rc *ai.RunContext[deps], _ struct{},
	) (string, error) {
		if err := rc.Emit(ai.NewCustomEvent("progress", progressPayload{})); !errors.Is(err, sentinel) {
			return "", errors.New("listener error did not reach emitter")
		}
		if err := rc.Emit(ai.NewCustomEvent("after-error", progressPayload{})); !errors.Is(err, sentinel) {
			return "", errors.New("stored listener error did not reach emitter")
		}
		return "", sentinel
	})
	if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, sentinel) {
		t.Fatalf("unexpected emitted listener error: %v", err)
	}
}
