package ai

import (
	"context"
	"fmt"
	"slices"
	"sync"
	"time"
)

// EnqueueItem is content accepted by RunContext.Enqueue. Model messages,
// request parts, and user content implement it.
type EnqueueItem interface {
	enqueueItemKind() string
}

// PendingMessagePriority controls when an enqueued message enters history.
type PendingMessagePriority string

const (
	// PendingMessageASAP delivers before the next model request, or redirects
	// a run that would otherwise finish.
	PendingMessageASAP PendingMessagePriority = "asap"
	// PendingMessageWhenIdle delivers only when a run would otherwise finish.
	PendingMessageWhenIdle PendingMessagePriority = "when_idle"
)

type pendingMessage struct {
	id       string
	priority PendingMessagePriority
	messages []ModelMessage
}

type pendingMessageQueue struct {
	mu      sync.Mutex
	pending []pendingMessage
}

// Enqueue adds items for delivery before the next model request.
func (rc *RunContext[Deps]) Enqueue(items ...EnqueueItem) (string, error) {
	return rc.EnqueueWithPriority(PendingMessageASAP, items...)
}

// EnqueueWhenIdle adds items for delivery when the run would otherwise end.
func (rc *RunContext[Deps]) EnqueueWhenIdle(items ...EnqueueItem) (string, error) {
	return rc.EnqueueWithPriority(PendingMessageWhenIdle, items...)
}

// EnqueueWithPriority adds model messages, request parts, or adjacent user
// content to the current run. The assembled sequence must end in a request.
func (rc *RunContext[Deps]) EnqueueWithPriority(
	priority PendingMessagePriority, items ...EnqueueItem,
) (string, error) {
	if rc.pendingMessages == nil {
		return "", fmt.Errorf("ai: enqueue is only available during an agent run")
	}
	if priority != PendingMessageASAP && priority != PendingMessageWhenIdle {
		return "", fmt.Errorf("ai: invalid pending message priority %q", priority)
	}
	messages, err := buildEnqueuedMessages(items)
	if err != nil {
		return "", err
	}
	if len(messages) == 0 {
		return "", nil
	}
	id := newRunID()
	rc.pendingMessages.add(pendingMessage{id: id, priority: priority, messages: messages})
	return id, nil
}

func buildEnqueuedMessages(items []EnqueueItem) ([]ModelMessage, error) {
	var messages []ModelMessage
	var parts []RequestPart
	var contents []UserContent
	flushContents := func() {
		if len(contents) == 0 {
			return
		}
		parts = append(parts, UserPromptPart{Contents: cloneUserContents(contents)})
		contents = nil
	}
	flushRequest := func() {
		flushContents()
		if len(parts) == 0 {
			return
		}
		messages = append(messages, ModelRequest{Parts: slices.Clone(parts)})
		parts = nil
	}
	for _, item := range items {
		_ = item.enqueueItemKind()
		switch item := item.(type) {
		case ModelMessage:
			flushRequest()
			messages = append(messages, cloneModelMessages([]ModelMessage{item})[0])
		case RequestPart:
			flushContents()
			cloned := cloneModelMessages([]ModelMessage{ModelRequest{Parts: []RequestPart{item}}})[0].(ModelRequest)
			parts = append(parts, cloned.Parts[0])
		case UserContent:
			contents = append(contents, item)
		default:
			return nil, fmt.Errorf("ai: unsupported enqueue item %T", item)
		}
	}
	flushRequest()
	if len(messages) > 0 {
		if _, ok := messages[len(messages)-1].(ModelRequest); !ok {
			return nil, fmt.Errorf("ai: enqueued content must end with a ModelRequest")
		}
	}
	return messages, nil
}

func (q *pendingMessageQueue) add(message pendingMessage) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.pending = append(q.pending, message)
}

func (q *pendingMessageQueue) drain(priority PendingMessagePriority) []pendingMessage {
	q.mu.Lock()
	defer q.mu.Unlock()
	drained := make([]pendingMessage, 0, len(q.pending))
	remaining := make([]pendingMessage, 0, len(q.pending))
	for _, message := range q.pending {
		if message.priority == priority {
			drained = append(drained, message)
		} else {
			remaining = append(remaining, message)
		}
	}
	q.pending = remaining
	return drained
}

func (q *pendingMessageQueue) drainForRedirect() []pendingMessage {
	q.mu.Lock()
	defer q.mu.Unlock()
	drained := make([]pendingMessage, 0, len(q.pending))
	for _, priority := range []PendingMessagePriority{PendingMessageASAP, PendingMessageWhenIdle} {
		for _, message := range q.pending {
			if message.priority == priority {
				drained = append(drained, message)
			}
		}
	}
	q.pending = nil
	return drained
}

func (r *run[Deps, Output]) deliverPendingMessages(priority PendingMessagePriority) error {
	return r.deliverPendingMessageGroups(r.pendingMessages.drain(priority))
}

func (r *run[Deps, Output]) redirectPendingMessages() (bool, error) {
	pending := r.pendingMessages.drainForRedirect()
	if len(pending) == 0 {
		return false, nil
	}
	return true, r.deliverPendingMessageGroups(pending)
}

func (r *run[Deps, Output]) deliverPendingMessageGroups(groups []pendingMessage) error {
	for _, group := range groups {
		messages := stampEnqueuedMessages(group.messages, r.rc.RunID, r.rc.ConversationID)
		r.messages = append(r.messages, messages...)
		if !r.emitStreamEvent(EnqueuedMessagesEvent{
			EnqueueID: group.id, Messages: cloneModelMessages(messages),
		}) {
			return context.Canceled
		}
	}
	return nil
}

func stampEnqueuedMessages(messages []ModelMessage, runID, conversationID string) []ModelMessage {
	messages = cloneModelMessages(messages)
	now := time.Now().UTC()
	for index, message := range messages {
		switch message := message.(type) {
		case ModelRequest:
			message.Parts = stampRequestParts(message.Parts, now)
			if message.Timestamp.IsZero() {
				message.Timestamp = now
			}
			if message.RunID == "" {
				message.RunID = runID
			}
			if message.ConversationID == "" {
				message.ConversationID = conversationID
			}
			if message.State == "" {
				message.State = RequestStateComplete
			}
			messages[index] = message
		case ModelResponse:
			if message.Timestamp.IsZero() {
				message.Timestamp = now
			}
			if message.RunID == "" {
				message.RunID = runID
			}
			if message.ConversationID == "" {
				message.ConversationID = conversationID
			}
			if message.State == "" {
				message.State = ModelResponseStateComplete
			}
			messages[index] = message
		}
	}
	return messages
}
