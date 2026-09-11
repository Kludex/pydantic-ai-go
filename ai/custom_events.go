package ai

import (
	"context"
	"fmt"
	"reflect"
	"strings"
)

// CustomEvent is an application-defined event with a typed payload.
// Create one with NewCustomEvent and emit it through RunContext or AgentRun.
type CustomEvent[Data any] struct {
	// Name identifies the application event.
	Name string
	// Data is the typed application payload.
	Data Data
	// ToolCallID identifies the tool that emitted the event, when applicable.
	ToolCallID string
	// ToolName identifies the tool that emitted the event, when applicable.
	ToolName string
}

func (*CustomEvent[Data]) streamEventKind() string { return "custom" }

func (event *CustomEvent[Data]) customEventName() string { return event.Name }

func (event *CustomEvent[Data]) stampTool(toolName, toolCallID string) {
	if event.ToolCallID == "" {
		event.ToolCallID = toolCallID
		event.ToolName = toolName
	}
}

// NewCustomEvent creates an application event. Name must not be empty.
func NewCustomEvent[Data any](name string, data Data) *CustomEvent[Data] {
	if strings.TrimSpace(name) == "" {
		panic("ai: custom event name must not be empty")
	}
	return &CustomEvent[Data]{Name: name, Data: data}
}

// CapabilityEvent is a capability-defined event with a typed payload.
// Create one with NewCapabilityEvent and emit it from a capability callback.
type CapabilityEvent[Data any] struct {
	// Kind identifies the event within its capability namespace.
	Kind string
	// Data is the typed capability payload.
	Data Data
	// CapabilityID identifies the capability instance that emitted the event.
	CapabilityID string
	// ToolCallID identifies the capability-owned tool that emitted the event, when applicable.
	ToolCallID string
	// ToolName identifies the capability-owned tool that emitted the event, when applicable.
	ToolName string
}

func (*CapabilityEvent[Data]) streamEventKind() string { return "capability" }

func (event *CapabilityEvent[Data]) capabilityEventKind() string { return event.Kind }

func (event *CapabilityEvent[Data]) stampCapability(capabilityID string) {
	if event.CapabilityID == "" {
		event.CapabilityID = capabilityID
	}
}

func (event *CapabilityEvent[Data]) stampTool(toolName, toolCallID string) {
	if event.ToolCallID == "" {
		event.ToolCallID = toolCallID
		event.ToolName = toolName
	}
}

// NewCapabilityEvent creates a namespaced capability event.
func NewCapabilityEvent[Data any](namespace, name string, data Data) *CapabilityEvent[Data] {
	if strings.TrimSpace(namespace) == "" || strings.TrimSpace(name) == "" ||
		strings.Contains(namespace, "..") || strings.HasPrefix(namespace, ".") || strings.HasSuffix(namespace, ".") {
		panic("ai: capability event namespace and name must not be empty")
	}
	return &CapabilityEvent[Data]{Kind: namespace + "." + name, Data: data}
}

type customStreamEvent interface {
	StreamEvent
	customEventName() string
	stampTool(toolName, toolCallID string)
}

type capabilityStreamEvent interface {
	StreamEvent
	capabilityEventKind() string
	stampCapability(capabilityID string)
	stampTool(toolName, toolCallID string)
}

func validateEmittedEvent(event StreamEvent) error {
	if event == nil {
		return fmt.Errorf("ai: emitted event must not be nil")
	}
	value := reflect.ValueOf(event)
	if value.Kind() == reflect.Pointer && value.IsNil() {
		return fmt.Errorf("ai: emitted event must not be nil")
	}
	switch event := event.(type) {
	case customStreamEvent:
		if strings.TrimSpace(event.customEventName()) == "" {
			return fmt.Errorf("ai: custom event name must not be empty")
		}
	case capabilityStreamEvent:
		if strings.TrimSpace(event.capabilityEventKind()) == "" {
			return fmt.Errorf("ai: capability event kind must not be empty")
		}
	default:
		return fmt.Errorf("ai: only custom and capability events may be emitted, got %T", event)
	}
	return nil
}

// EventListener receives events before consumer-only stream wrappers transform them.
type EventListener interface {
	// OnEvent observes one event. Returning an error stops the run.
	OnEvent(ctx context.Context, info *RunInfo, event StreamEvent) error
}

// EventListenerFunc adapts an application callback into an agent event listener.
type EventListenerFunc[Deps any] func(ctx context.Context, rc *RunContext[Deps], event StreamEvent) error

// AddEventListener registers an application listener for every run event.
func (a *Agent[Deps, Output]) AddEventListener(listener EventListenerFunc[Deps]) {
	if listener == nil {
		panic("ai: event listener must not be nil")
	}
	a.checkNotStarted()
	a.eventListeners = append(a.eventListeners, listener)
}

// OnEvent registers a type-filtered application listener.
func OnEvent[Deps, Output any, Event StreamEvent](
	agent *Agent[Deps, Output], listener func(context.Context, *RunContext[Deps], Event) error,
) {
	if listener == nil {
		panic("ai: event listener must not be nil")
	}
	agent.AddEventListener(func(ctx context.Context, rc *RunContext[Deps], event StreamEvent) error {
		typed, ok := event.(Event)
		if !ok {
			return nil
		}
		return listener(ctx, rc, typed)
	})
}
