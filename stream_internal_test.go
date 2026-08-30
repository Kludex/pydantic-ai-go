package ai

import (
	"iter"
	"testing"
)

type alienStreamEvent struct{}

func (alienStreamEvent) modelStreamEventKind() string { return "alien" }

func TestAccumulateUnknownEvent(t *testing.T) {
	events := iter.Seq2[ModelStreamEvent, error](func(yield func(ModelStreamEvent, error) bool) {
		yield(alienStreamEvent{}, nil)
	})
	if _, err := accumulate(events, ModelRequestParams{}, nil); err == nil {
		t.Fatal("expected error for unknown event type")
	}
}
