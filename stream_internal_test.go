package ai

import (
	"iter"
	"testing"
)

type alienStreamEvent struct{}

func (alienStreamEvent) modelStreamEventKind() string { return "alien" }

func TestResponseMetadataEventKind(t *testing.T) {
	if kind := (ResponseMetadataEvent{}).modelStreamEventKind(); kind != "response-metadata" {
		t.Fatalf("unexpected response metadata event kind %q", kind)
	}
}

func TestAccumulateUnknownEvent(t *testing.T) {
	events := iter.Seq2[ModelStreamEvent, error](func(yield func(ModelStreamEvent, error) bool) {
		yield(alienStreamEvent{}, nil)
	})
	if _, err := accumulate(events, ModelRequestParams{}, nil, nil); err == nil {
		t.Fatal("expected error for unknown event type")
	}
}
