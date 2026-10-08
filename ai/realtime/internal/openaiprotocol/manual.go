package openaiprotocol

import "context"

// DefersAudioCommit reports whether commits wait until a response is requested.
func (connection *Connection) DefersAudioCommit() bool { return connection.config.ManualAudioTurns }

// SetAudioCommitListener installs the callback invoked before a held commit reaches the wire.
func (connection *Connection) SetAudioCommitListener(listener func()) {
	connection.manualMu.Lock()
	connection.commitListener = listener
	connection.manualMu.Unlock()
}

func (connection *Connection) writeManual(ctx context.Context, frame map[string]any) error {
	connection.stateMu.Lock()
	active := connection.responseActive
	speech := connection.speechDetected
	connection.stateMu.Unlock()
	switch frame["type"] {
	case "input_audio_buffer.append":
		connection.audioLatest = true
		if connection.commitHeld || len(connection.heldAudio) > 0 || active {
			connection.heldAudio = append(connection.heldAudio, frame)
			return nil
		}
		connection.audioUncommitted = true
		connection.sentAudio = append(connection.sentAudio, frame)
	case "input_audio_buffer.commit":
		if connection.commitHeld || connection.audioUncommitted || len(connection.heldAudio) > 0 {
			connection.commitHeld = true
			connection.heldCommitted = len(connection.heldAudio)
			connection.audioUncommitted = false
			return nil
		}
	case "input_audio_buffer.clear":
		connection.heldAudio = connection.heldAudio[:connection.heldCommitted]
		if !connection.audioUncommitted {
			return nil
		}
		connection.sentAudio = nil
		connection.audioUncommitted = false
		connection.audioLatest = false
		connection.stateMu.Lock()
		connection.speechDetected = false
		connection.stateMu.Unlock()
	case "response.create":
		if connection.commitHeld {
			if !connection.commitAnnounced {
				connection.commitAnnounced = true
				if connection.commitListener != nil {
					connection.commitListener()
				}
			}
			for connection.heldCommitted > 0 {
				audio := connection.heldAudio[0]
				if err := connection.writeRaw(ctx, audio); err != nil {
					return err
				}
				connection.sentAudio = append(connection.sentAudio, audio)
				connection.heldAudio = connection.heldAudio[1:]
				connection.heldCommitted--
			}
			commit := map[string]any{"type": "input_audio_buffer.commit"}
			if speech {
				commit["event_id"] = frame["event_id"]
			}
			if err := connection.writeRaw(ctx, commit); err != nil {
				return err
			}
			connection.commitHeld = false
			connection.commitAnnounced = false
			connection.sentAudio = nil
			connection.audioLatest = true
			connection.stateMu.Lock()
			connection.speechDetected = false
			connection.stateMu.Unlock()
			if speech {
				return nil
			}
		} else {
			if err := connection.flushHeldAudio(ctx, true); err != nil {
				return err
			}
			if connection.audioLatest && !connection.audioUncommitted {
				if err := connection.writeRaw(ctx, map[string]any{"type": "input_audio_buffer.clear"}); err != nil {
					return err
				}
				connection.audioLatest = false
			}
			if connection.audioUncommitted && connection.commitListener != nil {
				connection.commitListener()
			}
		}
		connection.audioUncommitted = false
		connection.sentAudio = nil
		connection.stateMu.Lock()
		connection.speechDetected = false
		connection.stateMu.Unlock()
	}
	return connection.writeRaw(ctx, frame)
}

func (connection *Connection) flushHeldAudio(ctx context.Context, beforeResponse bool) error {
	for len(connection.heldAudio) > 0 && !connection.commitHeld {
		connection.stateMu.Lock()
		active := connection.responseActive
		connection.stateMu.Unlock()
		if active && !beforeResponse {
			return nil
		}
		audio := connection.heldAudio[0]
		if err := connection.writeRaw(ctx, audio); err != nil {
			return err
		}
		connection.audioUncommitted = true
		connection.audioLatest = true
		connection.sentAudio = append(connection.sentAudio, audio)
		connection.heldAudio = connection.heldAudio[1:]
	}
	return nil
}
