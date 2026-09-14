package realtime_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
)

func TestPlayedAudioAndInterruptionValidation(t *testing.T) {
	profile := fullProfile()
	profile.SupportsInterruption = false
	session, err := realtime.Open(t.Context(), &fakeModel{connection: newFakeConnection(), profile: profile}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.InterruptAtAudio(t.Context(), 0); err == nil {
		t.Fatal("expected unsupported interruption error")
	}
	_ = session.Close(t.Context())

	session, err = realtime.Open(t.Context(), &fakeModel{connection: newFakeConnection(), profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.PlayedAudioBytes(); err == nil {
		t.Fatal("expected missing audio stream error")
	}
	if _, err := session.InterruptAtAudio(t.Context(), 0); err == nil {
		t.Fatal("expected missing audio stream error")
	}
	first := session.StreamAudio(t.Context())
	second := session.StreamAudio(t.Context())
	if _, err := session.PlayedAudioBytes(); err == nil {
		t.Fatal("expected multiple audio stream error")
	}
	if _, err := session.InterruptAtAudio(t.Context(), 0); err == nil {
		t.Fatal("expected ambiguous audio stream error")
	}
	_ = first
	_ = second
	_ = session.Close(t.Context())
}

func TestInterruptAtAudioBranches(t *testing.T) {
	t.Run("negative", func(t *testing.T) {
		session, connection := openSessionWithProfile(t, fullProfile())
		_ = session.StreamAudio(t.Context())
		if _, err := session.InterruptAtAudio(t.Context(), -1); err == nil {
			t.Fatal("expected negative playback error")
		}
		connection.end()
		_ = session.Close(t.Context())
	})

	t.Run("before first audio", func(t *testing.T) {
		session, connection := openSessionWithProfile(t, fullProfile())
		_ = session.StreamAudio(t.Context())
		if err := session.Send(t.Context(), "hello"); err != nil {
			t.Fatal(err)
		}
		<-connection.sent
		connection.events <- realtime.OutputTranscript{Text: "thinking", OutputText: true}
		time.Sleep(time.Millisecond)
		interrupted, err := session.InterruptAtAudio(t.Context(), 0)
		if err != nil || !interrupted {
			t.Fatalf("interrupted=%v err=%v", interrupted, err)
		}
		if _, ok := (<-connection.sent).(realtime.CancelResponse); !ok {
			t.Fatal("expected cancellation")
		}
		connection.end()
		_ = session.Close(t.Context())
	})

	t.Run("all played", func(t *testing.T) {
		session, connection := openSessionWithProfile(t, fullProfile())
		audio := session.StreamAudio(t.Context())
		consumed := make(chan struct{})
		continueStream := make(chan struct{})
		go func() {
			for range audio {
				close(consumed)
				<-continueStream
			}
		}()
		connection.events <- realtime.AudioDelta{Data: []byte{1, 0}, ItemID: "assistant"}
		<-consumed
		close(continueStream)
		deadline := time.Now().Add(time.Second)
		for {
			played, playedErr := session.PlayedAudioBytes()
			if playedErr == nil && played == 2 {
				break
			}
			if time.Now().After(deadline) {
				t.Fatal("audio position did not advance")
			}
			time.Sleep(time.Millisecond)
		}
		interrupted, err := session.InterruptAtAudio(t.Context(), 2)
		if err != nil || interrupted {
			t.Fatalf("interrupted=%v err=%v", interrupted, err)
		}
		connection.end()
		_ = session.Close(t.Context())
	})

	t.Run("without truncation", func(t *testing.T) {
		profile := fullProfile()
		profile.SupportsOutputTruncation = false
		session, connection := openSessionWithProfile(t, profile)
		_ = session.StreamAudio(t.Context())
		connection.events <- realtime.AudioDelta{Data: []byte{1, 0}, ItemID: "assistant"}
		time.Sleep(time.Millisecond)
		interrupted, err := session.InterruptAtAudio(t.Context(), 0)
		if err != nil || !interrupted {
			t.Fatalf("interrupted=%v err=%v", interrupted, err)
		}
		if _, ok := (<-connection.sent).(realtime.CancelResponse); !ok {
			t.Fatal("expected cancellation")
		}
		connection.end()
		_ = session.Close(t.Context())
	})

	t.Run("server cancellation", func(t *testing.T) {
		base := newFakeConnection()
		connection := &interruptingConnection{fakeConnection: base, serverCancels: true}
		session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
		if err != nil {
			t.Fatal(err)
		}
		_ = session.StreamAudio(t.Context())
		connection.events <- realtime.AudioDelta{Data: []byte{1, 0}, ItemID: "assistant"}
		connection.events <- realtime.InputSpeechStarted{}
		for event, eventErr := range session.Events(t.Context()) {
			if eventErr != nil {
				t.Fatal(eventErr)
			}
			if _, ok := event.(realtime.InputSpeechStartEvent); ok {
				break
			}
		}
		interrupted, err := session.InterruptAtAudio(t.Context(), 0)
		if err != nil || !interrupted {
			t.Fatalf("interrupted=%v err=%v", interrupted, err)
		}
		if _, ok := (<-connection.sent).(realtime.TruncateOutput); !ok {
			t.Fatal("expected truncation")
		}
		select {
		case input := <-connection.sent:
			t.Fatalf("unexpected client cancellation: %#v", input)
		default:
		}
		connection.end()
		_ = session.Close(t.Context())
	})

	t.Run("cancel before audio failure", func(t *testing.T) {
		session, connection := openSessionWithProfile(t, fullProfile())
		_ = session.StreamAudio(t.Context())
		if err := session.Send(t.Context(), "hello"); err != nil {
			t.Fatal(err)
		}
		<-connection.sent
		connection.err = errors.New("send failed")
		if _, err := session.InterruptAtAudio(t.Context(), 0); err == nil {
			t.Fatal("expected send error")
		}
		connection.err = nil
		connection.end()
		_ = session.Close(t.Context())
	})

	t.Run("send failure", func(t *testing.T) {
		session, connection := openSessionWithProfile(t, fullProfile())
		_ = session.StreamAudio(t.Context())
		connection.events <- realtime.AudioDelta{Data: []byte{1, 0}, ItemID: "assistant"}
		time.Sleep(time.Millisecond)
		connection.err = errors.New("send failed")
		if _, err := session.InterruptAtAudio(t.Context(), 0); err == nil {
			t.Fatal("expected send error")
		}
		connection.err = nil
		connection.end()
		_ = session.Close(t.Context())
	})
}

func TestInterruptedResponseFlushesAudio(t *testing.T) {
	session, connection := openSessionWithProfile(t, fullProfile(), realtime.WithBargeIn(true))
	audioCtx, cancelAudio := context.WithCancel(t.Context())
	audio := session.StreamAudio(audioCtx)
	connection.events <- realtime.AudioDelta{Data: []byte{1, 0}, ItemID: "assistant"}
	connection.events <- realtime.ResponseDone{Interrupted: true}
	for event, err := range session.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(realtime.TurnCompleteEvent); ok {
			break
		}
	}
	cancelAudio()
	for chunk, err := range audio {
		if len(chunk) != 0 || !errors.Is(err, context.Canceled) {
			t.Fatalf("unexpected flushed audio result: %v, %v", chunk, err)
		}
		break
	}
	connection.end()
	_ = session.Close(t.Context())
}

func TestEnqueueValidationAndContent(t *testing.T) {
	session, connection := openSessionWithProfile(t, fullProfile())
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := session.Enqueue(cancelled, "text"); !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected cancellation error: %v", err)
	}
	if _, err := session.EnqueueWithPriority(t.Context(), "later", "text"); err == nil {
		t.Fatal("expected priority error")
	}
	if id, err := session.Enqueue(t.Context()); err != nil || id != "" {
		t.Fatalf("empty enqueue: id=%q err=%v", id, err)
	}
	for _, content := range []any{
		ai.BinaryContent{Data: []byte("x"), MediaType: "image/png"},
		ai.UserPromptPart{Contents: []ai.UserContent{ai.BinaryContent{Data: []byte("x"), MediaType: "image/png"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.SpeechPart{Speaker: ai.SpeechSpeakerUser}}},
		ai.ModelResponse{},
	} {
		if _, err := session.Enqueue(t.Context(), content); err == nil {
			t.Fatalf("expected content error for %T", content)
		}
	}
	if _, err := session.Enqueue(t.Context(),
		ai.TextContent{Text: "one"},
		ai.UserPromptPart{Content: "two"},
		ai.UserPromptPart{Contents: []ai.UserContent{ai.TextContent{Text: "three"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.SystemPromptPart{Content: "four"}}},
	); err != nil {
		t.Fatal(err)
	}
	select {
	case input := <-connection.sent:
		want := realtime.TextInput{Text: "one\n\ntwo\n\nthree\n\n<system>four</system>"}
		if input != want {
			t.Fatalf("got %#v, want %#v", input, want)
		}
	case <-time.After(time.Second):
		t.Fatal("queued content was not delivered")
	}
	connection.end()
	_ = session.Close(t.Context())
	if _, err := session.Enqueue(t.Context(), "late"); err == nil {
		t.Fatal("expected closed session error")
	}
}

func TestEnqueueSendFailureEndsSession(t *testing.T) {
	connection := newFakeConnection()
	connection.err = errors.New("send failed")
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Enqueue(t.Context(), "queued"); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Second)
	for session.Err() == nil && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if session.Err() == nil {
		t.Fatal("send failure did not end session")
	}
	connection.err = nil
	_ = session.Close(t.Context())
}

func openSessionWithProfile(t *testing.T, profile realtime.Profile, options ...realtime.SessionOption) (*realtime.Session, *fakeConnection) {
	t.Helper()
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: profile}, realtime.ConnectParams{}, options...)
	if err != nil {
		t.Fatal(err)
	}
	return session, connection
}
