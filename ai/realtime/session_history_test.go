package realtime_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
)

type turnHistoryConnection struct {
	*fakeConnection
	transcription bool
	batch         bool
	held          bool
	listener      func()
}

func (connection *turnHistoryConnection) InputTranscriptionEnabled() bool {
	return connection.transcription
}
func (connection *turnHistoryConnection) AnswersToolCallsPerResponse() bool { return connection.batch }
func (connection *turnHistoryConnection) DefersAudioCommit() bool           { return connection.held }
func (connection *turnHistoryConnection) SetAudioCommitListener(listener func()) {
	connection.listener = listener
}

func TestSpeechHistoryKeepsSpeakingOrder(t *testing.T) {
	connection := newFakeConnection()
	profile := fullProfile()
	profile.EmitsInputSpeechEvents = true
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: profile}, realtime.ConnectParams{},
		realtime.WithAudioRetention(realtime.AudioRetentionAll), realtime.WithAudioRetentionLimit(time.Second))
	if err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.InputSpeechStarted{ItemID: "first"}
	connection.events <- realtime.InputSpeechEnded{ItemID: "first"}
	connection.events <- realtime.OutputTranscript{Text: "answer", OutputText: true}
	connection.events <- realtime.InputSpeechStarted{ItemID: "second"}
	connection.events <- realtime.InputSpeechEnded{ItemID: "second"}
	connection.events <- realtime.InputTranscript{ItemID: "second", Text: "two", Final: true}
	connection.events <- realtime.InputTranscript{ItemID: "first", Text: "one", Final: true}
	connection.events <- realtime.ResponseDone{}
	connection.end()
	for range session.Events(t.Context()) {
	}
	messages := session.NewMessages()
	if len(messages) != 3 || *messages[0].(ai.ModelRequest).Parts[0].(ai.SpeechPart).Transcript != "one" ||
		messages[1].(ai.ModelResponse).Parts[0].(ai.TextPart).Content != "answer" ||
		*messages[2].(ai.ModelRequest).Parts[0].(ai.SpeechPart).Transcript != "two" {
		t.Fatalf("history: %+v", messages)
	}
}

func TestRetainedAudioBoundsBothSpeakers(t *testing.T) {
	for _, maximum := range []time.Duration{0, time.Second, -1} {
		connection := &turnHistoryConnection{fakeConnection: newFakeConnection()}
		profile := fullProfile()
		profile.EmitsInputSpeechEvents = true
		session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: profile}, realtime.ConnectParams{},
			realtime.WithAudioRetention(realtime.AudioRetentionAll), realtime.WithAudioRetentionLimit(maximum))
		if err != nil {
			t.Fatal(err)
		}
		pcm := make([]byte, 48000)
		if err := session.SendAudio(t.Context(), pcm, "audio/pcm"); err != nil {
			t.Fatal(err)
		}
		if err := session.CommitAudio(t.Context()); err != nil {
			t.Fatal(err)
		}
		if err := session.CommitAudio(t.Context()); err != nil {
			t.Fatal(err)
		}
		connection.events <- realtime.AudioDelta{Data: append(pcm, pcm...), ItemID: "assistant"}
		connection.events <- realtime.OutputTranscript{Text: "spoken", Final: true}
		connection.events <- realtime.ResponseDone{}
		connection.end()
		for range session.Events(t.Context()) {
		}
		messages := session.NewMessages()
		if len(messages) != 2 {
			t.Fatalf("history: %+v", messages)
		}
		input := messages[0].(ai.ModelRequest).Parts[0].(ai.SpeechPart)
		output := messages[1].(ai.ModelResponse).Parts[0].(ai.SpeechPart)
		if maximum == 0 && (input.Audio != nil || output.Audio != nil) {
			t.Fatal("zero budget retained audio")
		}
		if maximum == time.Second && (input.Audio != nil || output.Audio == nil || len(output.Audio.Data) != 48044) {
			t.Fatalf("bounded audio: %+v %+v", input, output)
		}
		if maximum < 0 && (input.Audio == nil || output.Audio == nil) {
			t.Fatal("unbounded audio was dropped")
		}
	}
	if _, err := realtime.Open(t.Context(), &fakeModel{connection: newFakeConnection(), profile: fullProfile()}, realtime.ConnectParams{}, realtime.WithAudioRetentionLimit(-2)); err == nil {
		t.Fatal("accepted negative limit")
	}
}

func TestSpeechWithoutTranscriptionUsesBoundaries(t *testing.T) {
	connection := &turnHistoryConnection{fakeConnection: newFakeConnection()}
	profile := fullProfile()
	profile.EmitsInputSpeechEvents = true
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: profile}, realtime.ConnectParams{}, realtime.WithAudioRetention(realtime.AudioRetentionInput))
	if err != nil {
		t.Fatal(err)
	}
	if err := session.SendAudio(t.Context(), []byte{0, 0}, "audio/pcm"); err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.InputSpeechStarted{ItemID: "speech"}
	connection.events <- realtime.InputSpeechEnded{ItemID: "speech"}
	connection.events <- realtime.InputSpeechEnded{ItemID: "speech"}
	connection.events <- realtime.ResponseDone{}
	connection.end()
	for range session.Events(t.Context()) {
	}
	if len(session.NewMessages()) != 1 {
		t.Fatalf("silence became a turn: %+v", session.NewMessages())
	}
}

func TestHeldAudioCommitFollowsInterveningText(t *testing.T) {
	connection := &turnHistoryConnection{fakeConnection: newFakeConnection(), held: true}
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	if err := session.SendAudio(t.Context(), []byte{0, 0}, "audio/pcm"); err != nil {
		t.Fatal(err)
	}
	if err := session.CommitAudio(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := session.Send(t.Context(), "context", realtime.WithResponse(false)); err != nil {
		t.Fatal(err)
	}
	connection.listener()
	connection.events <- realtime.ResponseDone{}
	connection.end()
	for range session.Events(t.Context()) {
	}
	messages := session.NewMessages()
	if len(messages) != 2 || messages[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content != "context" {
		t.Fatalf("commit ordering: %+v", messages)
	}
}

func TestBargeInFallbackAndPendingTextRollback(t *testing.T) {
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.OutputTranscript{Text: "answer", OutputText: true}
	connection.events <- realtime.InputTranscript{Text: "interruption", Final: true, ItemID: "user"}
	for event := range session.Events(t.Context()) {
		if _, ok := event.(ai.PartEndEvent); ok {
			break
		}
	}
	connection.err = errors.New("send failed")
	if err := session.Send(t.Context(), "refused", realtime.WithResponse(false)); err == nil {
		t.Fatal("accepted failed send")
	}
	connection.err = nil
	deadline := time.Now().Add(6 * time.Second)
	for len(session.NewMessages()) == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if len(session.NewMessages()) != 1 {
		t.Fatal("held turn did not time out")
	}
	connection.end()
	for range session.Events(t.Context()) {
	}
}

func TestReconnectSettlesPartialResponseAndMergedRequests(t *testing.T) {
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{Conversation: &ai.Conversation{DeferredToolRequests: &ai.DeferredToolRequests{}}})
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Send(t.Context(), "one"); err != nil {
		t.Fatal(err)
	}
	if err := session.Send(t.Context(), "two"); err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.ResponseRequestsMerged{Count: 1}
	connection.events <- realtime.OutputTranscript{Text: "partial", OutputText: true}
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
	connection.end()
	for range session.Events(t.Context()) {
	}
	connection = newFakeConnection()
	session, err = realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	connection.err = errors.New("commit failed")
	if err := session.CommitAudio(t.Context()); err == nil {
		t.Fatal("accepted failed commit")
	}
	connection.err = nil
	_ = session.Close(t.Context())
}

func TestToolBatchWaitsForOneAnswer(t *testing.T) {
	for _, early := range []bool{false, true} {
		connection := &turnHistoryConnection{fakeConnection: newFakeConnection(), batch: true}
		release := make(chan struct{})
		started := make(chan struct{}, 2)
		session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{}, realtime.WithToolExecutor(realtime.ToolExecutorFunc(func(ctx context.Context, _ ai.ToolCallPart) (any, error) {
			started <- struct{}{}
			select {
			case <-release:
				return "done", nil
			case <-ctx.Done():
				return nil, context.Cause(ctx)
			}
		})))
		if err != nil {
			t.Fatal(err)
		}
		connection.events <- realtime.ToolCall{ToolCallID: "one", ToolName: "tool"}
		connection.events <- realtime.ToolCall{ToolCallID: "two", ToolName: "tool"}
		<-started
		<-started
		if early {
			close(release)
			<-connection.sent
			<-connection.sent
		}
		connection.events <- realtime.ResponseDone{MoreExpected: true}
		if !early {
			close(release)
			<-connection.sent
			<-connection.sent
		}
		connection.events <- realtime.OutputTranscript{Text: "answer", OutputText: true}
		connection.events <- realtime.ResponseDone{}
		for event := range session.Events(t.Context()) {
			if _, ok := event.(realtime.TurnCompleteEvent); ok {
				break
			}
		}
		ctx, cancel := context.WithTimeout(t.Context(), time.Second)
		if err := session.WaitForReply(ctx); err != nil {
			t.Fatal(err)
		}
		cancel()
		_ = session.Close(t.Context())
	}
}
