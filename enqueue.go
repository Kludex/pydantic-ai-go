package ai

import (
	"context"
	"encoding/json"
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

// PendingMessagesMetadataKey identifies deferred-run metadata that preserves
// messages queued for later delivery.
const PendingMessagesMetadataKey = "pydantic_ai_go_pending_messages"

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
	return enqueuePendingMessage(rc.pendingMessages, priority, items)
}

func enqueuePendingMessage(
	queue *pendingMessageQueue, priority PendingMessagePriority, items []EnqueueItem,
) (string, error) {
	if queue == nil {
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
	queue.add(pendingMessage{id: id, priority: priority, messages: messages})
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

func (q *pendingMessageQueue) snapshot() []pendingMessage {
	q.mu.Lock()
	defer q.mu.Unlock()
	messages := make([]pendingMessage, len(q.pending))
	for index, message := range q.pending {
		message.messages = cloneModelMessages(message.messages)
		messages[index] = message
	}
	return messages
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

type persistedPendingMessage struct {
	EnqueueID string                 `json:"enqueue_id"`
	Priority  PendingMessagePriority `json:"priority"`
	Messages  string                 `json:"messages"`
}

func persistPendingMessages(messages []ModelMessage, queue *pendingMessageQueue) error {
	pending := queue.snapshot()
	if len(pending) == 0 {
		return nil
	}
	persisted := make([]persistedPendingMessage, len(pending))
	for index, message := range pending {
		encoded, err := MarshalMessages(message.messages)
		if err != nil {
			return fmt.Errorf("ai: persist pending messages: %w", err)
		}
		persisted[index] = persistedPendingMessage{
			EnqueueID: message.id, Priority: message.priority, Messages: string(encoded),
		}
	}
	encoded, _ := json.Marshal(persisted)
	// A queue is reachable only from a run context created after a model response.
	for index := len(messages) - 1; ; index-- {
		response, ok := messages[index].(ModelResponse)
		if !ok {
			continue
		}
		metadata := make(map[string]any, len(response.Metadata)+1)
		for key, value := range response.Metadata {
			metadata[key] = cloneSchemaValue(value)
		}
		metadata[PendingMessagesMetadataKey] = string(encoded)
		response.Metadata = metadata
		messages[index] = response
		return nil
	}
}

func restorePendingMessages(messages []ModelMessage) ([]pendingMessage, error) {
	for index := len(messages) - 1; index >= 0; index-- {
		response, ok := messages[index].(ModelResponse)
		if !ok || response.Metadata == nil {
			continue
		}
		value, exists := response.Metadata[PendingMessagesMetadataKey]
		if !exists {
			continue
		}
		encoded, ok := value.(string)
		if !ok {
			return nil, fmt.Errorf(
				"ai: restore pending messages: metadata has type %T, expected string", value,
			)
		}
		var persisted []persistedPendingMessage
		if err := json.Unmarshal([]byte(encoded), &persisted); err != nil {
			return nil, fmt.Errorf("ai: restore pending messages: %w", err)
		}
		pending := make([]pendingMessage, len(persisted))
		for pendingIndex, item := range persisted {
			if item.EnqueueID == "" {
				return nil, fmt.Errorf("ai: restore pending messages: enqueue ID must not be empty")
			}
			if item.Priority != PendingMessageASAP && item.Priority != PendingMessageWhenIdle {
				return nil, fmt.Errorf(
					"ai: restore pending messages: invalid priority %q", item.Priority,
				)
			}
			decoded, err := UnmarshalMessages([]byte(item.Messages))
			if err != nil {
				return nil, fmt.Errorf("ai: restore pending messages: %w", err)
			}
			if len(decoded) == 0 {
				return nil, fmt.Errorf("ai: restore pending messages: message group must not be empty")
			}
			if _, ok := decoded[len(decoded)-1].(ModelRequest); !ok {
				return nil, fmt.Errorf("ai: restore pending messages: message group must end with a ModelRequest")
			}
			pending[pendingIndex] = pendingMessage{
				id: item.EnqueueID, priority: item.Priority, messages: decoded,
			}
		}
		response.Metadata = cloneSchemaMap(response.Metadata)
		delete(response.Metadata, PendingMessagesMetadataKey)
		messages[index] = response
		return pending, nil
	}
	return nil, nil
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
