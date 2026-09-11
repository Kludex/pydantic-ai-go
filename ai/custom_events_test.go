package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"regexp"
	"slices"
	"strings"
	"testing"
	"time"

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
	timeout    time.Duration
}

func (*listenerCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (capability *listenerCapability) EventListenerTimeout() time.Duration { return capability.timeout }

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

type immediateEmittingCapability struct{}

type repeatedImmediateEmittingCapability struct{}

func (*repeatedImmediateEmittingCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (*repeatedImmediateEmittingCapability) CapabilityID() string { return "repeated" }

func (*repeatedImmediateEmittingCapability) BeforeModelRequest(
	_ context.Context, info *ai.RunInfo, request ai.ModelRequestContext,
) (ai.ModelRequestContext, error) {
	first := info.Emit(ai.NewCapabilityEvent("repeated", "first", 1).SetDispatch(ai.EventDispatchImmediate))
	second := info.Emit(ai.NewCapabilityEvent("repeated", "second", 2).SetDispatch(ai.EventDispatchImmediate))
	return request, errors.Join(first, second)
}

func (*immediateEmittingCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (*immediateEmittingCapability) CapabilityID() string { return "immediate" }

func (*immediateEmittingCapability) BeforeModelRequest(
	_ context.Context, info *ai.RunInfo, request ai.ModelRequestContext,
) (ai.ModelRequestContext, error) {
	return request, info.Emit(ai.NewCapabilityEvent("immediate", "decision", 1).
		SetDispatch(ai.EventDispatchImmediate))
}

func TestImmediateCapabilityEventWithoutListeners(t *testing.T) {
	agent := ai.NewAgent[deps, string](
		fakes.NewTestModel(), ai.WithCapabilities(&immediateEmittingCapability{}),
	)
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
}

func TestImmediateCapabilityListenerErrorStopsRun(t *testing.T) {
	sentinel := errors.New("immediate listener failed")
	agent := ai.NewAgent[deps, string](
		fakes.NewTestModel(),
		ai.WithCapabilities(&immediateEmittingCapability{}, &listenerCapability{failure: sentinel}),
	)
	if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, sentinel) {
		t.Fatalf("unexpected immediate listener error: %v", err)
	}
}

func TestRepeatedImmediateCapabilityListenerErrorStopsRun(t *testing.T) {
	sentinel := errors.New("immediate listener failed")
	agent := ai.NewAgent[deps, string](
		fakes.NewTestModel(),
		ai.WithCapabilities(&repeatedImmediateEmittingCapability{}, &listenerCapability{failure: sentinel}),
	)
	if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, sentinel) {
		t.Fatalf("unexpected repeated immediate listener error: %v", err)
	}
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

type contextToolCapability struct {
	observed []bool
}

func (capability *contextToolCapability) Setup(registry *ai.CapabilityRegistry) error {
	registry.AddContextTool(ai.ToolDefinition{Name: "owned", Schema: map[string]any{"type": "object"}}, func(
		_ context.Context, info *ai.RunInfo, _ json.RawMessage,
	) (any, error) {
		if info.CapabilityID() != "owner" || info.ToolName() != "owned" || info.ToolCallID() != "owned-1" {
			return nil, errors.New("capability tool attribution was not available")
		}
		streamEvent := ai.NewCapabilityEvent("owner", "progress", progressPayload{Done: 1})
		if err := info.Emit(streamEvent); err != nil {
			return nil, err
		}
		immediateEvent := ai.NewCapabilityEvent("owner", "decision", progressPayload{Done: 2}).
			SetDispatch(ai.EventDispatchImmediate)
		if err := info.Emit(immediateEvent); err != nil {
			return nil, err
		}
		capability.observed = []bool{streamEvent.Data.Done == 9, immediateEvent.Data.Done == 9}
		return "done", nil
	})
	return nil
}

func (*contextToolCapability) CapabilityID() string { return "owner" }

type innermostEventListener struct {
	order   *[]string
	failure error
}

func (*innermostEventListener) Setup(*ai.CapabilityRegistry) error { return nil }

func (*innermostEventListener) CapabilityOrdering() ai.CapabilityOrdering {
	return ai.CapabilityOrdering{Position: ai.CapabilityInnermost}
}

func (listener *innermostEventListener) OnEvent(context.Context, *ai.RunInfo, ai.StreamEvent) error {
	if listener.order != nil {
		*listener.order = append(*listener.order, "innermost")
	}
	return listener.failure
}

func TestCapabilityOwnedToolAttributionAndDispatchModes(t *testing.T) {
	requests := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "owned", ToolCallID: "owned-1", Args: json.RawMessage(`{}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	owner := &contextToolCapability{}
	listener := &listenerCapability{mutateDone: 9}
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(owner, listener))
	var events []*ai.CapabilityEvent[progressPayload]
	stream := agent.RunStream(t.Context(), "go", deps{})
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(*ai.CapabilityEvent[progressPayload]); ok {
			events = append(events, event)
		}
	}
	if !slices.Equal(owner.observed, []bool{false, true}) {
		t.Fatalf("dispatch mode was not respected: %v", owner.observed)
	}
	if len(events) != 2 {
		t.Fatalf("unexpected capability events: %+v", events)
	}
	for _, event := range events {
		if event.CapabilityID != "owner" || event.ToolName != "owned" || event.ToolCallID != "owned-1" ||
			event.Data.Done != 9 {
			t.Fatalf("event lost attribution or settled data: %+v", event)
		}
	}
}

type nestedImmediateListener struct {
	seen []string
}

func (*nestedImmediateListener) Setup(*ai.CapabilityRegistry) error { return nil }

func (listener *nestedImmediateListener) OnEvent(_ context.Context, info *ai.RunInfo, event ai.StreamEvent) error {
	typed, ok := event.(*ai.CapabilityEvent[progressPayload])
	if !ok {
		return nil
	}
	listener.seen = append(listener.seen, typed.Kind)
	if typed.Kind == "owner.decision" {
		return info.Emit(ai.NewCapabilityEvent("listener", "nested", progressPayload{}).
			SetDispatch(ai.EventDispatchImmediate))
	}
	return nil
}

func (*nestedImmediateListener) CapabilityID() string { return "listener" }

func TestNestedImmediateEventsKeepCauseFirstOrder(t *testing.T) {
	requests := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "owned", ToolCallID: "owned-1", Args: json.RawMessage(`{}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	owner := &contextToolCapability{}
	listener := &nestedImmediateListener{}
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(owner, listener))
	stream := agent.RunStream(t.Context(), "go", deps{})
	var kinds []string
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(*ai.CapabilityEvent[progressPayload]); ok {
			kinds = append(kinds, event.Kind)
		}
	}
	if !slices.Equal(kinds, []string{"owner.progress", "owner.decision", "listener.nested"}) {
		t.Fatalf("nested immediate events were reordered: %v", kinds)
	}
}

func TestAgentListenerRunsBeforeInnermostCapability(t *testing.T) {
	var order []string
	innermost := &innermostEventListener{order: &order}
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(innermost))
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
		if !slices.Equal(order[index:index+2], []string{"agent", "innermost"}) {
			t.Fatalf("unexpected listener order: %v", order)
		}
	}
}

type waitingEventListener struct {
	timeout time.Duration
}

func (*waitingEventListener) Setup(*ai.CapabilityRegistry) error { return nil }

func (listener *waitingEventListener) EventListenerTimeout() time.Duration { return listener.timeout }

func (*waitingEventListener) OnEvent(ctx context.Context, _ *ai.RunInfo, _ ai.StreamEvent) error {
	<-ctx.Done()
	return ctx.Err()
}

func TestInnermostCapabilityListenerErrorStopsRun(t *testing.T) {
	sentinel := errors.New("innermost failed")
	agent := ai.NewAgent[deps, string](
		fakes.NewTestModel(), ai.WithCapabilities(&innermostEventListener{failure: sentinel}),
	)
	if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, sentinel) {
		t.Fatalf("unexpected innermost listener error: %v", err)
	}
}

func TestEventListenerTimeout(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	agent.AddEventListener(func(ctx context.Context, _ *ai.RunContext[deps], _ ai.StreamEvent) error {
		<-ctx.Done()
		return ctx.Err()
	}, ai.WithEventListenerTimeout(time.Millisecond))
	_, err := agent.Run(t.Context(), "go", deps{})
	var timeout *ai.EventListenerTimeoutError
	if !errors.As(err, &timeout) || timeout.Duration != time.Millisecond || !timeout.Timeout() {
		t.Fatalf("unexpected listener timeout: %v", err)
	}

	capabilityAgent := ai.NewAgent[deps, string](
		fakes.NewTestModel(), ai.WithCapabilities(&waitingEventListener{timeout: time.Millisecond}),
	)
	_, err = capabilityAgent.Run(t.Context(), "go", deps{})
	if !errors.As(err, &timeout) {
		t.Fatalf("unexpected capability listener timeout: %v", err)
	}

	timedAgent := ai.NewAgent[deps, string](
		fakes.NewTestModel(), ai.WithCapabilities(&listenerCapability{timeout: time.Second}),
	)
	if _, err := timedAgent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}

	negativeAgent := ai.NewAgent[deps, string](
		fakes.NewTestModel(), ai.WithCapabilities(&listenerCapability{timeout: -time.Second}),
	)
	if _, err := negativeAgent.Run(t.Context(), "go", deps{}); err == nil ||
		!strings.Contains(err.Error(), "must not be negative") {
		t.Fatalf("unexpected negative capability timeout: %v", err)
	}
}

func TestCustomAndCapabilityEventJSONRoundTrip(t *testing.T) {
	custom := ai.NewCustomEvent("progress", progressPayload{Done: 3}).SetUIVisible(false)
	custom.ProjectForUI(func(data progressPayload) any { return map[string]any{"completed": data.Done} })
	projected := custom.Payload().(map[string]any)
	if projected["completed"] != 3 {
		t.Fatalf("unexpected projected payload: %#v", projected)
	}
	encoded, err := json.Marshal(custom)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ai.CustomEvent[progressPayload]
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	payload := decoded.Payload().(map[string]any)
	if _, err := json.Marshal(decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Name != "progress" || decoded.VisibleInUI() || payload["completed"] != float64(3) {
		t.Fatalf("custom event did not round trip: encoded=%s decoded=%+v payload=%#v", encoded, decoded, payload)
	}

	capability := ai.NewCapabilityEvent("index", "decision", progressPayload{Done: 4}).
		SetDispatch(ai.EventDispatchImmediate)
	encoded, err = json.Marshal(capability)
	if err != nil {
		t.Fatal(err)
	}
	var decodedCapability ai.CapabilityEvent[progressPayload]
	if err := json.Unmarshal(encoded, &decodedCapability); err != nil {
		t.Fatal(err)
	}
	if decodedCapability.Kind != "index.decision" || decodedCapability.Data.Done != 4 ||
		decodedCapability.Dispatch != ai.EventDispatchImmediate {
		t.Fatalf("capability event did not round trip: %s %+v", encoded, decodedCapability)
	}

	plain := ai.NewCustomEvent("plain", progressPayload{Done: 1})
	encoded, err = json.Marshal(plain)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"event_kind":"custom"`) {
		t.Fatalf("unexpected plain event JSON: %s", encoded)
	}
	if plain.Payload() != (progressPayload{Done: 1}) {
		t.Fatalf("unexpected default payload: %#v", plain.Payload())
	}
	var streamDispatch ai.CapabilityEvent[progressPayload]
	if err := json.Unmarshal(
		[]byte(`{"kind":"index.stream","data":{"Done":1},"event_kind":"capability"}`), &streamDispatch,
	); err != nil ||
		streamDispatch.Dispatch != ai.EventDispatchStream {
		t.Fatalf("default dispatch was not restored: %+v %v", streamDispatch, err)
	}
	if name, payload, visible, ok := ai.CustomEventUI(plain); !ok || name != "plain" || !visible ||
		payload != (progressPayload{Done: 1}) {
		t.Fatalf("custom event UI projection failed: %q %#v %v %v", name, payload, visible, ok)
	}
	if _, _, _, ok := ai.CustomEventUI(ai.FinishEvent{}); ok {
		t.Fatal("framework event was treated as a custom UI event")
	}
	if _, err := json.Marshal(ai.CapabilityEvent[progressPayload]{Kind: "index.stream"}); err != nil {
		t.Fatal(err)
	}
}

func TestCustomAndCapabilityEventConfigurationErrors(t *testing.T) {
	for name, fn := range map[string]func(){
		"nil projector": func() { ai.NewCustomEvent("x", 1).ProjectForUI(nil) },
		"dispatch":      func() { ai.NewCapabilityEvent("x", "y", 1).SetDispatch("later") },
		"timeout":       func() { ai.WithEventListenerTimeout(0) },
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
	for name, target := range map[string]any{
		"custom malformed":     &ai.CustomEvent[progressPayload]{},
		"capability malformed": &ai.CapabilityEvent[progressPayload]{},
	} {
		t.Run(name, func(t *testing.T) {
			if err := json.Unmarshal([]byte(`{`), target); err == nil {
				t.Fatal("expected JSON error")
			}
		})
	}
	var malformedCustom ai.CustomEvent[progressPayload]
	if err := malformedCustom.UnmarshalJSON([]byte(`{`)); err == nil {
		t.Fatal("custom event accepted malformed JSON")
	}
	var malformedCapability ai.CapabilityEvent[progressPayload]
	if err := malformedCapability.UnmarshalJSON([]byte(`{`)); err == nil {
		t.Fatal("capability event accepted malformed JSON")
	}
	var custom ai.CustomEvent[progressPayload]
	if err := json.Unmarshal([]byte(`{"event_kind":"capability"}`), &custom); err == nil {
		t.Fatal("custom event accepted another event kind")
	}
	var capability ai.CapabilityEvent[progressPayload]
	for _, encoded := range []string{
		`{"event_kind":"custom"}`,
		`{"event_kind":"capability","event_dispatch":"later"}`,
	} {
		if err := json.Unmarshal([]byte(encoded), &capability); err == nil {
			t.Fatalf("capability event accepted %s", encoded)
		}
	}
	invalidDispatch := &ai.CapabilityEvent[int]{Kind: "x.y", Dispatch: "later"}
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	run, err := agent.StartRun(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if err := run.Emit(invalidDispatch); err == nil {
		t.Fatal("run accepted invalid event dispatch")
	}
	if err := run.Close(); err != nil {
		t.Fatal(err)
	}
	nullPayload := ai.NewCustomEvent("null", 1).ProjectForUI(func(int) any { return nil })
	encoded, err := json.Marshal(nullPayload)
	if err != nil {
		t.Fatal(err)
	}
	var decodedNull ai.CustomEvent[int]
	if err := json.Unmarshal(encoded, &decodedNull); err != nil || decodedNull.Payload() != nil {
		t.Fatalf("null UI payload did not round trip: payload=%#v err=%v", decodedNull.Payload(), err)
	}
	if _, err := json.Marshal(ai.NewCustomEvent("bad", make(chan int))); err == nil {
		t.Fatal("custom event marshaled unsupported data")
	}
	badProjection := ai.NewCustomEvent("bad", 1).ProjectForUI(func(int) any { return make(chan int) })
	if _, err := json.Marshal(badProjection); err == nil {
		t.Fatal("custom event marshaled unsupported UI payload")
	}
}

func TestCapabilityToolRegistrationRejectsNilFunctions(t *testing.T) {
	for name, register := range map[string]func(*ai.CapabilityRegistry){
		"plain": func(registry *ai.CapabilityRegistry) {
			registry.AddTool(ai.ToolDefinition{Name: "x"}, nil)
		},
		"context": func(registry *ai.CapabilityRegistry) {
			registry.AddContextTool(ai.ToolDefinition{Name: "x"}, nil)
		},
	} {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			register(&ai.CapabilityRegistry{})
		})
	}
}

func TestNormalizeToolReturnContentPreservesInvalidCandidates(t *testing.T) {
	for _, input := range []map[string]any{
		{"kind": "binary", "media_type": "text/plain", "data": 42},
		{"kind": "binary", "media_type": "text/plain", "data": "eA==", "extra": make(chan int)},
	} {
		normalized := ai.NormalizeToolReturnContent(input)
		if _, ok := normalized.(map[string]any); !ok {
			t.Fatalf("invalid candidate was narrowed: %T", normalized)
		}
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
