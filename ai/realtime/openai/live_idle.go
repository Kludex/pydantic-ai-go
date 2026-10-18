package openai

import (
	"context"
	"encoding/base64"
	"time"
)

const idleAudioFrame = 100 * time.Millisecond

func (connection *LiveConnection) streamIdleAudio(ctx context.Context) {
	defer close(connection.idleDone)
	silence := make([]byte, connection.rate/5)
	frame := map[string]any{"type": "session.input_audio.append", "audio": base64.StdEncoding.EncodeToString(silence)}
	timer := time.NewTimer(0)
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
		case <-timer.C:
		}
		connection.mu.Lock()
		if ctx.Err() != nil {
			connection.mu.Unlock()
			return
		}
		if now := time.Now(); !connection.ended && !now.Before(connection.idleFrameDue()) {
			connection.nextIdleFrame = now.Add(idleAudioFrame)
			if connection.write(ctx, frame) == nil {
				connection.audioEnd = connection.queuedAudioStart().Add(idleAudioFrame)
			}
		}
		timer.Reset(min(time.Until(connection.idleFrameDue()), idleAudioFrame))
		connection.mu.Unlock()
	}
}

func (connection *LiveConnection) idleFrameDue() time.Time {
	due := connection.inputAudioEnd.Add(300 * time.Millisecond)
	if due.Before(connection.nextIdleFrame) {
		return connection.nextIdleFrame
	}
	return due
}

func (connection *LiveConnection) queuedAudioStart() time.Time {
	if now := time.Now(); now.After(connection.audioEnd) {
		return now
	}
	return connection.audioEnd
}
