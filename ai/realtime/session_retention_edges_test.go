package realtime_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
)

type captureRecoveryConnection struct {
	*fakeConnection
	recovering bool
	available  bool
}

func (connection *captureRecoveryConnection) IsReconnecting() bool { return connection.recovering }
func (connection *captureRecoveryConnection) CanReconnect() bool   { return connection.available }

func TestCaptureDropsAudioDuringReconnect(t *testing.T) {
	connection := &captureRecoveryConnection{fakeConnection: newFakeConnection(), recovering: true, available: true}
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	if err := session.SendAudio(t.Context(), []byte{0, 0}, "audio/pcm"); err != nil {
		t.Fatal(err)
	}
	connection.recovering = false
	connection.err = errors.New("link lost")
	if err := session.SendAudio(t.Context(), []byte{0, 0}, "audio/pcm"); err != nil {
		t.Fatal(err)
	}
	connection.available = false
	if err := session.SendAudio(t.Context(), []byte{0, 0}, "audio/pcm"); err == nil {
		t.Fatal("failure hidden after reconnect exhaustion")
	}
	connection.err = nil
	_ = session.Close(t.Context())
	mode := realtime.AsyncToolCallsAlways
	if profile := realtime.MergeProfile(realtime.DefaultProfile(), realtime.ProfileOverride{AsyncToolCallMode: &mode}); profile.AsyncToolCallMode != mode {
		t.Fatal("async mode override was ignored")
	}
}

func TestAudioBudgetEvictsFinishedAndPendingOutput(t *testing.T) {
	for _, finalized := range []bool{false, true} {
		connection := newFakeConnection()
		session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{},
			realtime.WithAudioRetention(realtime.AudioRetentionAll), realtime.WithAudioRetentionLimit(time.Second))
		if err != nil {
			t.Fatal(err)
		}
		connection.events <- realtime.AudioDelta{Data: make([]byte, 24000)}
		connection.events <- realtime.OutputTranscript{Text: "answer", Final: true}
		for event := range session.Events(t.Context()) {
			if _, ok := event.(ai.PartEndEvent); ok {
				break
			}
		}
		if finalized {
			connection.events <- realtime.ResponseDone{}
			for event := range session.Events(t.Context()) {
				if _, ok := event.(realtime.TurnCompleteEvent); ok {
					break
				}
			}
		}
		if err := session.SendAudio(t.Context(), make([]byte, 48000), "audio/pcm"); err != nil {
			t.Fatal(err)
		}
		connection.events <- realtime.ResponseDone{}
		connection.end()
		for range session.Events(t.Context()) {
		}
		for _, message := range session.NewMessages() {
			if response, ok := message.(ai.ModelResponse); ok {
				for _, part := range response.Parts {
					if speech, ok := part.(ai.SpeechPart); ok && speech.Audio != nil {
						t.Fatal("old output audio survived eviction")
					}
				}
			}
		}
	}
}

func TestAudioBudgetEvictsSegmentsAwaitingTranscripts(t *testing.T) {
	connection := newFakeConnection()
	profile := fullProfile()
	profile.EmitsInputSpeechEvents = true
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: profile}, realtime.ConnectParams{},
		realtime.WithAudioRetention(realtime.AudioRetentionInput), realtime.WithAudioRetentionLimit(time.Second))
	if err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.InputSpeechStarted{ItemID: "first"}
	for event := range session.Events(t.Context()) {
		if _, ok := event.(realtime.InputSpeechStartEvent); ok {
			break
		}
	}
	if err := session.SendAudio(t.Context(), make([]byte, 48000), "audio/pcm"); err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.InputSpeechEnded{ItemID: "first"}
	for event := range session.Events(t.Context()) {
		if _, ok := event.(realtime.InputSpeechEndEvent); ok {
			break
		}
	}
	if err := session.SendAudio(t.Context(), make([]byte, 48000), "audio/pcm"); err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.InputTranscript{ItemID: "first", Text: "spoken", Final: true}
	connection.end()
	for range session.Events(t.Context()) {
	}
	part := session.NewMessages()[0].(ai.ModelRequest).Parts[0].(ai.SpeechPart)
	if part.Audio != nil || part.Transcript == nil {
		t.Fatalf("evicted segment lost transcript: %+v", part)
	}
}

func TestReconnectAbandonsRunningToolBatch(t *testing.T) {
	connection := &turnHistoryConnection{fakeConnection: newFakeConnection(), batch: true}
	started := make(chan struct{})
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{}, realtime.WithToolExecutor(realtime.ToolExecutorFunc(func(ctx context.Context, _ ai.ToolCallPart) (any, error) {
		close(started)
		<-ctx.Done()
		return nil, context.Cause(ctx)
	})))
	if err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.ToolCall{ToolCallID: "c", ToolName: "lookup"}
	<-started
	connection.events <- realtime.SessionReconnected{StateRestored: false}
	for event := range session.Events(t.Context()) {
		if _, ok := event.(realtime.SessionReconnectEvent); ok {
			break
		}
	}
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	if err := session.WaitForReply(ctx); err != nil {
		t.Fatal(err)
	}
	_ = session.Close(t.Context())
	if len(session.NewMessages()) != 2 {
		t.Fatalf("interrupted result was lost: %+v", session.NewMessages())
	}
}
