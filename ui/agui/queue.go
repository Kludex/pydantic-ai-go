package agui

import (
	"fmt"
	"sync"
	"time"
)

// EventQueue accepts state and custom events from application dependencies.
type EventQueue struct {
	mu     sync.Mutex
	events []Event
}

// EmitStateSnapshot queues a detached complete application state value.
func (queue *EventQueue) EmitStateSnapshot(snapshot any) error {
	detached, err := detachedJSONValue(snapshot)
	if err != nil {
		return fmt.Errorf("agui: clone state snapshot: %w", err)
	}
	queue.append(Event{Type: EventStateSnapshot, Snapshot: detached})
	return nil
}

// EmitStateDelta queues a detached RFC 6902 JSON Patch array.
func (queue *EventQueue) EmitStateDelta(delta []any) error {
	detached, err := detachedJSONValue(delta)
	if err != nil {
		return fmt.Errorf("agui: clone state delta: %w", err)
	}
	queue.append(Event{Type: EventStateDelta, Delta: detached})
	return nil
}

// EmitCustom queues one detached application-defined event.
func (queue *EventQueue) EmitCustom(name string, value any) error {
	if name == "" {
		return fmt.Errorf("agui: custom event name must not be empty")
	}
	detached, err := detachedJSONValue(value)
	if err != nil {
		return fmt.Errorf("agui: clone custom event: %w", err)
	}
	queue.append(Event{Type: EventCustom, Name: name, Value: detached})
	return nil
}

func (queue *EventQueue) append(event Event) {
	event.Timestamp = time.Now().UnixMilli()
	queue.mu.Lock()
	defer queue.mu.Unlock()
	queue.events = append(queue.events, event)
}

func (queue *EventQueue) drain() []Event {
	queue.mu.Lock()
	defer queue.mu.Unlock()
	events := queue.events
	queue.events = nil
	return events
}
