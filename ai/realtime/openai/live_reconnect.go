package openai

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func (connection *LiveConnection) canReconnect() bool {
	return !connection.closed && connection.dial != nil && connection.reconnect != nil && connection.reconnects < connection.reconnect.MaxReconnects
}

func (connection *LiveConnection) redialSession(ctx context.Context) (bool, error) {
	connection.mu.Lock()
	if !connection.canReconnect() {
		connection.mu.Unlock()
		return false, fmt.Errorf("openai GPT-Live: connection dropped; reconnection unavailable or exhausted")
	}
	policy := *connection.reconnect
	history, sessionID, forks := connection.history, connection.sessionID, connection.settings.Store
	connection.mu.Unlock()
	var messages []ai.ModelMessage
	if history != nil {
		messages = history()
	}
	seed, _ := seedLiveItems(messages, true)
	var last error
	for attempt := 0; attempt < policy.MaxAttempts; attempt++ {
		delay := policy.BaseDelay
		for step := 0; step < attempt && delay < policy.MaxDelay; step++ {
			delay = min(policy.MaxDelay, delay*2)
		}
		delay = min(delay, policy.MaxDelay)
		if policy.Jitter && delay > 0 {
			delay = time.Duration(rand.Int64N(int64(delay) + 1))
		}
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return false, context.Cause(ctx)
		case <-timer.C:
		}
		forkFrom := ""
		if forks {
			forkFrom = sessionID
		}
		socket, id, err := connection.dial(ctx, forkFrom, seed)
		forked := forkFrom != "" && err == nil
		if err != nil && forkFrom != "" {
			var protocolError *realtimeHandshakeError
			if errors.As(err, &protocolError) {
				socket, id, err = connection.dial(ctx, "", seed)
			}
		}
		if err != nil {
			last = err
			continue
		}
		connection.mu.Lock()
		if connection.closed {
			connection.mu.Unlock()
			_ = socket.CloseNow()
			return false, fmt.Errorf("openai GPT-Live: connection closed during reconnect")
		}
		_ = connection.socket.CloseNow()
		connection.socket = socket
		connection.sessionID = id
		connection.reconnects++
		connection.responseOpen = false
		connection.inputOpen = false
		connection.hasPause = false
		connection.fragments = [2]string{}
		connection.hasInputEnd = false
		for call := range connection.calls {
			connection.abandoned[call] = true
		}
		clear(connection.calls)
		clear(connection.delegations)
		clear(connection.reasoning)
		connection.seconds = 0
		connection.ended = false
		connection.redial = false
		connection.startReader()
		connection.mu.Unlock()
		return forked || history != nil, nil
	}
	return false, fmt.Errorf("openai GPT-Live: reconnect failed: %w", last)
}

type realtimeHandshakeError struct{ err error }

func (err *realtimeHandshakeError) Error() string { return err.err.Error() }
func (err *realtimeHandshakeError) Unwrap() error { return err.err }
