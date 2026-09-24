package realtime

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

var enqueueID atomic.Uint64

// PlayedAudioBytes returns the playback position of the single active audio stream.
// A chunk counts as played after its consumer finishes handling it.
func (session *Session) PlayedAudioBytes() (int, error) {
	session.tapMu.Lock()
	defer session.tapMu.Unlock()
	tap, err := session.singleAudioTapLocked("played audio")
	if err != nil {
		return 0, err
	}
	return tap.playedBytes, nil
}

// WaitForPlayback waits until the single active audio stream has accounted for all emitted audio.
func (session *Session) WaitForPlayback(ctx context.Context) error {
	session.tapMu.Lock()
	tap, err := session.singleAudioTapLocked("wait for playback")
	session.tapMu.Unlock()
	if err != nil {
		return err
	}
	for {
		session.tapMu.Lock()
		complete := tap.ended || len(tap.queue) == 0 &&
			tap.subscribedAtBytes+tap.playedBytes+tap.droppedBytes+tap.pendingDroppedBytes >= session.emittedAudioBytes
		session.tapMu.Unlock()
		if complete {
			return nil
		}
		select {
		case <-ctx.Done():
			return context.Cause(ctx)
		case <-session.playbackProgress:
		}
	}
}

// WaitForReply waits until every requested model reply reaches a turn boundary.
// It returns immediately when no reply is outstanding and also returns when the
// session closes. It does not consume the event stream.
func (session *Session) WaitForReply(ctx context.Context) error {
	if err := session.nextError(); err != nil {
		return err
	}
	session.mu.RLock()
	closed := session.closed
	session.mu.RUnlock()
	if closed {
		return errors.New("realtime: session is closed")
	}
	for {
		session.mu.RLock()
		closed = session.closed
		outstanding := session.responseActive || session.pendingResponses > 0 || session.pendingToolCalls > 0
		session.mu.RUnlock()
		if err := session.nextError(); err != nil {
			return err
		}
		if closed || !outstanding {
			return nil
		}
		select {
		case <-ctx.Done():
			return context.Cause(ctx)
		case <-session.replyProgress:
		}
	}
}

func (session *Session) singleAudioTapLocked(action string) (*audioTap, error) {
	if len(session.audioTaps) != 1 {
		return nil, fmt.Errorf(
			"realtime: %s requires exactly one active audio stream, got %d", action, len(session.audioTaps),
		)
	}
	var tap *audioTap
	for tap = range session.audioTaps {
	}
	return tap, nil
}

func (session *Session) notifyPlaybackLocked() {
	select {
	case session.playbackProgress <- struct{}{}:
	default:
	}
}

// InterruptAtAudio flushes unheard audio and interrupts at a raw PCM playback position.
// It returns false when all emitted audio was already played.
func (session *Session) InterruptAtAudio(ctx context.Context, playedBytes int) (bool, error) {
	if err := session.require(session.profile.SupportsInterruption, "interrupt", "interruption"); err != nil {
		return false, err
	}
	if playedBytes < 0 {
		return false, fmt.Errorf("realtime: played audio bytes must not be negative")
	}

	session.tapMu.Lock()
	if len(session.audioTaps) != 1 {
		count := len(session.audioTaps)
		session.tapMu.Unlock()
		return false, fmt.Errorf("realtime: audio interruption requires exactly one active audio stream, got %d", count)
	}
	var tap *audioTap
	for candidate := range session.audioTaps {
		tap = candidate
	}
	playhead := tap.subscribedAtBytes + playedBytes + tap.droppedBytes
	emitted := session.emittedAudioBytes
	turnStart := session.turnAudioStartBytes
	hasAudio := session.hasAudioPart
	audioPartIndex := session.audioPartIndex
	session.tapMu.Unlock()

	session.mu.RLock()
	responseActive := session.responseActive || session.pendingResponses > 0
	serverCancelling := session.serverCancelling
	activeAssistant := session.activeAssistant
	session.mu.RUnlock()

	if playhead >= emitted {
		waitingForAudio := responseActive && (!hasAudio || activeAssistant != nil && audioPartIndex != activeAssistant.index)
		if !waitingForAudio || serverCancelling {
			return false, nil
		}
		session.tapMu.Lock()
		if activeAssistant != nil {
			session.interruptedAudioPartIndex = activeAssistant.index
			session.hasInterruptedAudioPart = true
		}
		session.tapMu.Unlock()
		if err := session.send(ctx, CancelResponse{}); err != nil {
			return false, err
		}
		session.publish(ResponseInterruptedEvent{})
		return true, nil
	}

	session.tapMu.Lock()
	session.flushAudioTapLocked(tap)
	if hasAudio {
		session.interruptedAudioPartIndex = audioPartIndex
		session.hasInterruptedAudioPart = true
	}
	session.tapMu.Unlock()

	inputs := make([]Input, 0, 2)
	var playedMilliseconds *int
	if session.profile.SupportsOutputTruncation {
		milliseconds := max(0, playhead-turnStart) * 1000 / (session.profile.AudioOutputSampleRate * 2)
		playedMilliseconds = &milliseconds
		inputs = append(inputs, TruncateOutput{AudioEndMilliseconds: milliseconds})
	}
	if !serverCancelling {
		inputs = append(inputs, CancelResponse{})
	}
	if len(inputs) > 0 {
		if err := session.sendInputs(ctx, inputs...); err != nil {
			return false, err
		}
	}
	session.publish(ResponseInterruptedEvent{PlayedMilliseconds: playedMilliseconds})
	return true, nil
}

func (session *Session) flushAudioTaps() {
	session.tapMu.Lock()
	defer session.tapMu.Unlock()
	for tap := range session.audioTaps {
		session.flushAudioTapLocked(tap)
	}
	if session.hasAudioPart {
		session.interruptedAudioPartIndex = session.audioPartIndex
		session.hasInterruptedAudioPart = true
	}
}

func (session *Session) flushAudioTapLocked(tap *audioTap) {
	tap.droppedBytes += tap.pendingDroppedBytes
	tap.pendingDroppedBytes = 0
	for len(tap.queue) > 0 {
		tap.droppedBytes += len(<-tap.queue)
	}
	session.notifyPlaybackLocked()
}

// Enqueue adds text for delivery after an active response completes. It is
// safe to call concurrently and returns an error after the session closes.
func (session *Session) Enqueue(ctx context.Context, content ...any) (string, error) {
	return session.EnqueueWithPriority(ctx, ai.PendingMessageASAP, content...)
}

// EnqueueWhenIdle adds text after active and ASAP work completes. It is safe
// to call concurrently and returns an error after the session closes.
func (session *Session) EnqueueWhenIdle(ctx context.Context, content ...any) (string, error) {
	return session.EnqueueWithPriority(ctx, ai.PendingMessageWhenIdle, content...)
}

// EnqueueWithPriority queues plain text and system prompt parts for a later
// turn. It is safe to call concurrently and returns an error after close.
func (session *Session) EnqueueWithPriority(
	ctx context.Context, priority ai.PendingMessagePriority, content ...any,
) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", context.Cause(ctx)
	}
	if priority != ai.PendingMessageASAP && priority != ai.PendingMessageWhenIdle {
		return "", fmt.Errorf("realtime: invalid pending message priority %q", priority)
	}
	text, err := realtimeEnqueueText(content)
	if err != nil {
		return "", err
	}
	if text == "" {
		return "", nil
	}
	id := fmt.Sprintf("realtime-enqueue-%d", enqueueID.Add(1))
	session.enqueueMu.Lock()
	if session.enqueueClosed {
		session.enqueueMu.Unlock()
		return "", fmt.Errorf("realtime: session is closed")
	}
	session.enqueued = append(session.enqueued, queuedPrompt{id: id, priority: priority, text: text})
	session.enqueueMu.Unlock()
	go session.deliverEnqueued()
	return id, nil
}

func realtimeEnqueueText(content []any) (string, error) {
	parts := make([]string, 0, len(content))
	var appendPart func(any) error
	appendPart = func(item any) error {
		switch item := item.(type) {
		case string:
			if item != "" {
				parts = append(parts, item)
			}
		case ai.TextContent:
			if item.Text != "" {
				parts = append(parts, item.Text)
			}
		case ai.SystemPromptPart:
			if item.Content != "" {
				parts = append(parts, "<system>"+item.Content+"</system>")
			}
		case ai.UserPromptPart:
			if len(item.Contents) == 0 {
				if item.Content != "" {
					parts = append(parts, item.Content)
				}
				return nil
			}
			for _, value := range item.Contents {
				text, ok := value.(ai.TextContent)
				if !ok {
					return fmt.Errorf("realtime: enqueued content supports plain text and system prompt parts only")
				}
				if text.Text != "" {
					parts = append(parts, text.Text)
				}
			}
		case ai.ModelRequest:
			for _, part := range item.Parts {
				if err := appendPart(part); err != nil {
					return err
				}
			}
		default:
			return fmt.Errorf("realtime: enqueued content supports plain text and system prompt parts only, got %T", item)
		}
		return nil
	}
	for _, item := range content {
		if err := appendPart(item); err != nil {
			return "", err
		}
	}
	return strings.Join(parts, "\n\n"), nil
}

func (session *Session) deliverEnqueued() {
	session.deliveryMu.Lock()
	defer session.deliveryMu.Unlock()

	session.mu.RLock()
	active := session.responseActive || session.pendingResponses > 0 || session.pendingToolCalls > 0
	closed := session.closed
	session.mu.RUnlock()
	if active || closed {
		return
	}

	prompts := session.takeEnqueued(ai.PendingMessageASAP)
	if len(prompts) == 0 {
		prompts = session.takeEnqueued(ai.PendingMessageWhenIdle)
	}
	for index, prompt := range prompts {
		last := index == len(prompts)-1
		var input Input = TextContext{Text: prompt.text}
		if last {
			input = TextInput{Text: prompt.text}
			session.reserveResponse()
		}
		if err := session.send(session.ctx, input); err != nil {
			if last {
				session.releaseResponseReservation()
			}
			if ctxErr := context.Cause(session.ctx); ctxErr == nil {
				session.fail(err)
			}
			return
		}
		request := ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: prompt.text}}}
		session.mu.Lock()
		session.history = append(session.history, request)
		session.mu.Unlock()
		session.publish(ai.EnqueuedMessagesEvent{EnqueueID: prompt.id, Messages: []ai.ModelMessage{request}})
	}
}

func (session *Session) closeEnqueue() {
	session.enqueueMu.Lock()
	session.enqueueClosed = true
	session.enqueueMu.Unlock()
}

func (session *Session) takeEnqueued(priority ai.PendingMessagePriority) []queuedPrompt {
	session.enqueueMu.Lock()
	defer session.enqueueMu.Unlock()
	selected := make([]queuedPrompt, 0, len(session.enqueued))
	remaining := make([]queuedPrompt, 0, len(session.enqueued))
	for _, prompt := range session.enqueued {
		if prompt.priority == priority {
			selected = append(selected, prompt)
		} else {
			remaining = append(remaining, prompt)
		}
	}
	session.enqueued = remaining
	return selected
}
