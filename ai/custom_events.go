package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"time"
)

// EventDispatch controls when listeners observe a capability event.
type EventDispatch string

const (
	// EventDispatchStream runs listeners when the event reaches its stream position.
	EventDispatchStream EventDispatch = "stream"
	// EventDispatchImmediate runs listeners before Emit returns.
	EventDispatchImmediate EventDispatch = "immediate"
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
	ToolName             string
	uiVisible            *bool
	uiProject            func(Data) any
	restoredUIPayload    any
	hasRestoredUIPayload bool
}

func (*CustomEvent[Data]) streamEventKind() string { return "custom" }

func (event *CustomEvent[Data]) customEventName() string { return event.Name }

func (event *CustomEvent[Data]) stampTool(toolName, toolCallID string) {
	if event.ToolCallID == "" {
		event.ToolCallID = toolCallID
		event.ToolName = toolName
	}
}

func (event *CustomEvent[Data]) cloneForStream() StreamEvent {
	cloned := *event
	return &cloned
}

// VisibleInUI reports whether UI adapters should forward this event.
func (event *CustomEvent[Data]) VisibleInUI() bool {
	return event.uiVisible == nil || *event.uiVisible
}

// Payload returns the frontend payload without attribution envelope fields.
func (event *CustomEvent[Data]) Payload() any {
	if event.uiProject != nil {
		return event.uiProject(event.Data)
	}
	if event.hasRestoredUIPayload {
		return event.restoredUIPayload
	}
	return event.Data
}

// SetUIVisible changes whether UI adapters forward the event and returns the event.
func (event *CustomEvent[Data]) SetUIVisible(visible bool) *CustomEvent[Data] {
	event.uiVisible = &visible
	return event
}

// ProjectForUI sets a typed frontend payload projection and returns the event.
func (event *CustomEvent[Data]) ProjectForUI(project func(Data) any) *CustomEvent[Data] {
	if project == nil {
		panic("ai: custom event UI projector must not be nil")
	}
	event.uiProject = project
	return event
}

// MarshalJSON preserves the typed event envelope across durable JSON boundaries.
func (event CustomEvent[Data]) MarshalJSON() ([]byte, error) {
	var uiPayload json.RawMessage
	if event.uiProject != nil {
		var err error
		uiPayload, err = json.Marshal(event.uiProject(event.Data))
		if err != nil {
			return nil, err
		}
	} else if event.hasRestoredUIPayload {
		var err error
		uiPayload, err = json.Marshal(event.restoredUIPayload)
		if err != nil {
			return nil, err
		}
	}
	return json.Marshal(struct {
		Name       string          `json:"name"`
		Data       Data            `json:"data"`
		ToolCallID string          `json:"tool_call_id,omitempty"`
		ToolName   string          `json:"tool_name,omitempty"`
		EventKind  string          `json:"event_kind"`
		UIVisible  *bool           `json:"ui_visible,omitempty"`
		UIPayload  json.RawMessage `json:"ui_payload,omitempty"`
	}{
		Name: event.Name, Data: event.Data, ToolCallID: event.ToolCallID, ToolName: event.ToolName,
		EventKind: "custom", UIVisible: event.uiVisible, UIPayload: uiPayload,
	})
}

// UnmarshalJSON restores a typed custom event from a durable JSON boundary.
func (event *CustomEvent[Data]) UnmarshalJSON(data []byte) error {
	var wire struct {
		Name       string          `json:"name"`
		Data       Data            `json:"data"`
		ToolCallID string          `json:"tool_call_id"`
		ToolName   string          `json:"tool_name"`
		EventKind  string          `json:"event_kind"`
		UIVisible  *bool           `json:"ui_visible"`
		UIPayload  json.RawMessage `json:"ui_payload"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	if wire.EventKind != "" && wire.EventKind != "custom" {
		return fmt.Errorf("ai: expected custom event, got %q", wire.EventKind)
	}
	*event = CustomEvent[Data]{
		Name: wire.Name, Data: wire.Data, ToolCallID: wire.ToolCallID, ToolName: wire.ToolName,
		uiVisible: wire.UIVisible,
	}
	var fields map[string]json.RawMessage
	_ = json.Unmarshal(data, &fields)
	if payload, ok := fields["ui_payload"]; ok {
		event.hasRestoredUIPayload = true
		_ = json.Unmarshal(payload, &event.restoredUIPayload)
	}
	return nil
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
	// Dispatch controls whether listeners run at stream position or before Emit returns.
	Dispatch EventDispatch
}

func (*CapabilityEvent[Data]) streamEventKind() string { return "capability" }

func (event *CapabilityEvent[Data]) capabilityEventKind() string { return event.Kind }

func (event *CapabilityEvent[Data]) eventDispatch() EventDispatch {
	if event.Dispatch == "" {
		return EventDispatchStream
	}
	return event.Dispatch
}

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

func (event *CapabilityEvent[Data]) cloneForStream() StreamEvent {
	cloned := *event
	return &cloned
}

// MarshalJSON preserves the typed event envelope across durable JSON boundaries.
func (event CapabilityEvent[Data]) MarshalJSON() ([]byte, error) {
	dispatch := event.eventDispatch()
	return json.Marshal(struct {
		Kind         string        `json:"kind"`
		Data         Data          `json:"data"`
		CapabilityID string        `json:"capability_id,omitempty"`
		ToolCallID   string        `json:"tool_call_id,omitempty"`
		ToolName     string        `json:"tool_name,omitempty"`
		EventKind    string        `json:"event_kind"`
		Dispatch     EventDispatch `json:"event_dispatch,omitempty"`
	}{
		Kind: event.Kind, Data: event.Data, CapabilityID: event.CapabilityID,
		ToolCallID: event.ToolCallID, ToolName: event.ToolName, EventKind: "capability", Dispatch: dispatch,
	})
}

// UnmarshalJSON restores a typed capability event from a durable JSON boundary.
func (event *CapabilityEvent[Data]) UnmarshalJSON(data []byte) error {
	var wire struct {
		Kind         string        `json:"kind"`
		Data         Data          `json:"data"`
		CapabilityID string        `json:"capability_id"`
		ToolCallID   string        `json:"tool_call_id"`
		ToolName     string        `json:"tool_name"`
		EventKind    string        `json:"event_kind"`
		Dispatch     EventDispatch `json:"event_dispatch"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	if wire.EventKind != "" && wire.EventKind != "capability" {
		return fmt.Errorf("ai: expected capability event, got %q", wire.EventKind)
	}
	if wire.Dispatch == "" {
		wire.Dispatch = EventDispatchStream
	}
	if wire.Dispatch != EventDispatchStream && wire.Dispatch != EventDispatchImmediate {
		return fmt.Errorf("ai: invalid event dispatch %q", wire.Dispatch)
	}
	*event = CapabilityEvent[Data]{
		Kind: wire.Kind, Data: wire.Data, CapabilityID: wire.CapabilityID,
		ToolCallID: wire.ToolCallID, ToolName: wire.ToolName, Dispatch: wire.Dispatch,
	}
	return nil
}

// NewCapabilityEvent creates a namespaced capability event.
func NewCapabilityEvent[Data any](namespace, name string, data Data) *CapabilityEvent[Data] {
	if strings.TrimSpace(namespace) == "" || strings.TrimSpace(name) == "" ||
		strings.Contains(namespace, "..") || strings.HasPrefix(namespace, ".") || strings.HasSuffix(namespace, ".") {
		panic("ai: capability event namespace and name must not be empty")
	}
	return &CapabilityEvent[Data]{Kind: namespace + "." + name, Data: data, Dispatch: EventDispatchStream}
}

// SetDispatch changes when listeners observe the capability event and returns the event.
func (event *CapabilityEvent[Data]) SetDispatch(dispatch EventDispatch) *CapabilityEvent[Data] {
	if dispatch != EventDispatchStream && dispatch != EventDispatchImmediate {
		panic(fmt.Sprintf("ai: invalid event dispatch %q", dispatch))
	}
	event.Dispatch = dispatch
	return event
}

type customStreamEvent interface {
	StreamEvent
	customEventName() string
	stampTool(toolName, toolCallID string)
	cloneForStream() StreamEvent
}

type capabilityStreamEvent interface {
	StreamEvent
	capabilityEventKind() string
	eventDispatch() EventDispatch
	stampCapability(capabilityID string)
	stampTool(toolName, toolCallID string)
	cloneForStream() StreamEvent
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
		if event.eventDispatch() != EventDispatchStream && event.eventDispatch() != EventDispatchImmediate {
			return fmt.Errorf("ai: invalid event dispatch %q", event.eventDispatch())
		}
	default:
		return fmt.Errorf("ai: only custom and capability events may be emitted, got %T", event)
	}
	return nil
}

// CustomEventUI returns the frontend name, projected payload, and visibility for a custom event.
func CustomEventUI(event StreamEvent) (name string, payload any, visible bool, ok bool) {
	type customUIEvent interface {
		StreamEvent
		customEventName() string
		VisibleInUI() bool
		Payload() any
	}
	custom, ok := event.(customUIEvent)
	if !ok {
		return "", nil, false, false
	}
	return custom.customEventName(), custom.Payload(), custom.VisibleInUI(), true
}

// EventListener receives events before consumer-only stream wrappers transform them.
type EventListener interface {
	// OnEvent observes one event. Returning an error stops the run.
	OnEvent(ctx context.Context, info *RunInfo, event StreamEvent) error
}

// EventListenerTimeoutProvider sets a cooperative timeout for a capability listener.
type EventListenerTimeoutProvider interface {
	// EventListenerTimeout returns the listener deadline. Zero disables it.
	EventListenerTimeout() time.Duration
}

// EventListenerTimeoutError reports a listener that exceeded its deadline.
type EventListenerTimeoutError struct {
	Listener string
	Duration time.Duration
}

// Error implements error.
func (err *EventListenerTimeoutError) Error() string {
	return fmt.Sprintf("ai: event listener %s timed out after %s", err.Listener, err.Duration)
}

// Timeout reports that this error represents a timeout.
func (*EventListenerTimeoutError) Timeout() bool { return true }

// EventListenerFunc adapts an application callback into an agent event listener.
type EventListenerFunc[Deps any] func(ctx context.Context, rc *RunContext[Deps], event StreamEvent) error

type eventListenerConfig struct {
	timeout time.Duration
}

// EventListenerOption configures one application event listener.
type EventListenerOption func(*eventListenerConfig)

// WithEventListenerTimeout applies a cooperative listener deadline.
func WithEventListenerTimeout(timeout time.Duration) EventListenerOption {
	if timeout <= 0 {
		panic("ai: event listener timeout must be positive")
	}
	return func(config *eventListenerConfig) { config.timeout = timeout }
}

type agentEventListener[Deps any] struct {
	listener EventListenerFunc[Deps]
	timeout  time.Duration
}

// AddEventListener registers an application listener for every run event.
func (a *Agent[Deps, Output]) AddEventListener(listener EventListenerFunc[Deps], options ...EventListenerOption) {
	if listener == nil {
		panic("ai: event listener must not be nil")
	}
	var config eventListenerConfig
	for _, option := range options {
		option(&config)
	}
	a.checkNotStarted()
	a.eventListeners = append(a.eventListeners, agentEventListener[Deps]{listener: listener, timeout: config.timeout})
}

// OnEvent registers a type-filtered application listener.
func OnEvent[Deps, Output any, Event StreamEvent](
	agent *Agent[Deps, Output], listener func(context.Context, *RunContext[Deps], Event) error,
	options ...EventListenerOption,
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
	}, options...)
}
