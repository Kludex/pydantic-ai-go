package ai

import "testing"

type alienEnqueueItem struct{}

func (alienEnqueueItem) enqueueItemKind() string { return "alien" }

func TestBuildEnqueuedMessagesRejectsUnknownItem(t *testing.T) {
	if _, err := buildEnqueuedMessages([]EnqueueItem{alienEnqueueItem{}}); err == nil {
		t.Fatal("expected unknown enqueue item error")
	}
}
