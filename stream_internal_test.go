package ai

import (
	"iter"
	"testing"
)

type alienStreamEvent struct{}

func (alienStreamEvent) streamEventKind() string { return "alien" }

func TestAccumulateUnknownEvent(t *testing.T) {
	events := iter.Seq2[StreamEvent, error](func(yield func(StreamEvent, error) bool) {
		yield(alienStreamEvent{}, nil)
	})
	if _, err := accumulate(events, nil); err == nil {
		t.Fatal("expected error for unknown event type")
	}
}
