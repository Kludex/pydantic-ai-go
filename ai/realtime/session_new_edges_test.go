package realtime_test

import (
	"context"
	"errors"
	"iter"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
)

func TestResponseSendErrorsAndClosedViews(t *testing.T) {
	profile := fullProfile()
	profile.SupportsManualTurnControl = false
	session, connection := openSessionWithProfile(t, profile)
	image := ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"}
	if err := session.Send(t.Context(), image, realtime.WithResponse(true)); err == nil {
		t.Fatal("expected image response capability error")
	}
	connection.end()
	_ = session.Close(t.Context())

	session, connection = openSessionWithProfile(t, fullProfile())
	connection.err = errors.New("send failed")
	if err := session.Send(t.Context(), image, realtime.WithResponse(true)); err == nil {
		t.Fatal("expected image send error")
	}
	if err := session.CreateResponse(t.Context()); err == nil {
		t.Fatal("expected response send error")
	}
	connection.err = nil
	connection.end()
	_ = session.Close(t.Context())

	for range session.StreamAudio(t.Context()) {
		t.Fatal("closed audio stream yielded a value")
	}
	for range session.StreamTranscripts(t.Context()) {
		t.Fatal("closed transcript stream yielded a value")
	}
}

func TestTerminalErrorSurvivesFullEventBuffer(t *testing.T) {
	session, connection := openSessionWithProfile(t, fullProfile())
	for range 128 {
		connection.events <- realtime.InputSpeechEnded{}
	}
	connection.events <- realtime.SessionError{Err: errors.New("terminal")}
	deadline := time.Now().Add(time.Second)
	for !session.Closed() && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if !session.Closed() {
		t.Fatal("session did not close")
	}
	var got error
	for _, err := range session.Events(t.Context()) {
		if err != nil {
			got = err
			break
		}
	}
	if got == nil || !strings.Contains(got.Error(), "terminal") {
		t.Fatalf("unexpected terminal error: %v", got)
	}
	_ = session.Close(t.Context())
}

func TestAutomaticBargeInWithoutInterruptionSupportStandsDown(t *testing.T) {
	connection := newFakeConnection()
	profile := fullProfile()
	profile.SupportsInterruption = false
	session, err := realtime.Open(
		t.Context(), &fakeModel{connection: connection, profile: profile}, realtime.ConnectParams{},
		realtime.WithBargeIn(true),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close(t.Context()) }()

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
	if session.Err() != nil || session.Closed() {
		t.Fatalf("unsupported automatic barge-in terminated the session: %v", session.Err())
	}
	select {
	case input := <-connection.sent:
		t.Fatalf("unsupported automatic barge-in sent %#v", input)
	default:
	}
}

func TestAutomaticBargeInFailureEndsSession(t *testing.T) {
	connection := newFakeConnection()
	session, err := realtime.Open(
		t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{},
		realtime.WithBargeIn(true),
	)
	if err != nil {
		t.Fatal(err)
	}
	_ = session.StreamAudio(t.Context())
	connection.events <- realtime.AudioDelta{Data: []byte{1, 0}, ItemID: "assistant"}
	time.Sleep(time.Millisecond)
	connection.err = errors.New("interrupt failed")
	connection.events <- realtime.InputSpeechStarted{}
	deadline := time.Now().Add(time.Second)
	for session.Err() == nil && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if session.Err() == nil {
		t.Fatal("barge-in failure did not end the session")
	}
	connection.err = nil
	_ = session.Close(t.Context())
}

type blockingToolResultConnection struct {
	*fakeConnection
	started chan struct{}
	release chan struct{}
}

func (connection *blockingToolResultConnection) Send(ctx context.Context, input realtime.Input) error {
	if _, ok := input.(realtime.ToolResult); !ok {
		return connection.fakeConnection.Send(ctx, input)
	}
	close(connection.started)
	<-connection.release
	return nil
}

func (connection *blockingToolResultConnection) Events(ctx context.Context) iter.Seq2[realtime.CodecEvent, error] {
	return connection.fakeConnection.Events(ctx)
}

func TestToolResultCompletingDuringCloseIsDiscarded(t *testing.T) {
	base := newFakeConnection()
	connection := &blockingToolResultConnection{
		fakeConnection: base, started: make(chan struct{}), release: make(chan struct{}),
	}
	var session *realtime.Session
	closed := make(chan error, 1)
	var err error
	session, err = realtime.Open(
		t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{},
		realtime.WithToolExecutor(realtime.ToolExecutorFunc(func(ctx context.Context, call ai.ToolCallPart) (any, error) {
			if call.ToolName == "close" {
				closeErr := session.Close(ctx)
				closed <- closeErr
				return nil, closeErr
			}
			return "done", nil
		})),
	)
	if err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.ToolCall{ToolCallID: "blocked", ToolName: "blocked", Arguments: `{}`}
	<-connection.started
	connection.events <- realtime.ToolCall{ToolCallID: "close", ToolName: "close", Arguments: `{}`}
	select {
	case closeErr := <-closed:
		if closeErr != nil {
			t.Fatal(closeErr)
		}
	case <-time.After(time.Second):
		t.Fatal("tool close did not finish")
	}
	close(connection.release)
}
