package realtime_test

import (
	"context"
	"encoding/binary"
	"errors"
	"iter"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/realtime"
)

type stagedSendConnection struct {
	*fakeConnection
	count  int
	failAt int
}

func (connection *stagedSendConnection) Send(ctx context.Context, input realtime.Input) error {
	connection.count++
	if connection.count == connection.failAt {
		return errors.New("staged write")
	}
	return connection.fakeConnection.Send(ctx, input)
}

func TestRemainingSessionBranches(t *testing.T) {
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()},
		realtime.ConnectParams{}, realtime.WithImageRetention(2, 1))
	if err != nil {
		t.Fatal(err)
	}
	image := ai.BinaryContent{Data: []byte("one"), MediaType: "image/png"}
	if err := session.Send(t.Context(), image); err != nil {
		t.Fatal(err)
	}
	connection.err = errors.New("image write")
	if err := session.Send(t.Context(), image); err == nil {
		t.Fatal("expected image send error")
	}
	connection.err = nil
	connection.end()
	for range session.Events(t.Context()) {
	}

	staged := &stagedSendConnection{fakeConnection: newFakeConnection(), failAt: 2}
	session, err = realtime.Open(t.Context(), &fakeModel{connection: staged, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	played := 1
	if err := session.Interrupt(t.Context(), &played); err == nil {
		t.Fatal("expected truncate send error")
	}
	staged.end()
	for range session.Events(t.Context()) {
	}

	turn := &realtime.TurnDetection{Enabled: true, Sensitivity: "medium"}
	policy := &realtime.ReconnectPolicy{MaxReconnects: 1, MaxDelay: time.Second}
	connection = newFakeConnection()
	session, err = realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()},
		realtime.ConnectParams{Settings: realtime.Settings{TurnDetection: turn, Reconnect: policy}})
	if err != nil {
		t.Fatal(err)
	}
	connection.end()
	for range session.Events(t.Context()) {
	}
}

func TestImageEvictionAcrossHistoryShapes(t *testing.T) {
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()},
		realtime.ConnectParams{}, realtime.WithImageRetention(1, 1))
	if err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.OutputTranscript{Text: "before", Final: true, OutputText: true}
	connection.events <- realtime.ResponseDone{}
	connection.events <- realtime.ToolCall{ToolCallID: "tool", ToolName: "missing"}
	for {
		if _, ok := (<-connection.sent).(realtime.ToolResult); ok {
			break
		}
	}
	if err := session.Send(t.Context(), "text"); err != nil {
		t.Fatal(err)
	}
	for _, content := range []ai.BinaryContent{
		{Data: []byte("one"), MediaType: "image/png"},
		{Data: []byte("two"), MediaType: "image/png"},
	} {
		if err := session.Send(t.Context(), content); err != nil {
			t.Fatal(err)
		}
	}
	connection.end()
	for range session.Events(t.Context()) {
	}
}

func TestToolResultSendFailureAndCloseCancellation(t *testing.T) {
	connection := newFakeConnection()
	connection.err = errors.New("result write")
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()},
		realtime.ConnectParams{}, realtime.WithToolExecutor(realtime.ToolExecutorFunc(func(
			context.Context, ai.ToolCallPart,
		) (any, error) {
			return "done", nil
		})))
	if err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.ToolCall{ToolCallID: "call", ToolName: "tool", Arguments: `{}`}
	foundError := false
	for _, err := range session.Events(t.Context()) {
		if err != nil {
			foundError = true
		}
	}
	if !foundError {
		t.Fatal("expected tool result send failure")
	}

	connection = newFakeConnection()
	started := make(chan struct{})
	session, err = realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()},
		realtime.ConnectParams{}, realtime.WithToolExecutor(realtime.ToolExecutorFunc(func(
			ctx context.Context, _ ai.ToolCallPart,
		) (any, error) {
			close(started)
			<-ctx.Done()
			return nil, ctx.Err()
		})))
	if err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.ToolCall{ToolCallID: "slow", ToolName: "slow", Arguments: `{}`}
	<-started
	_ = session.Close(t.Context())
}

func TestLateToolResultSkipsExistingResultAndUserMessage(t *testing.T) {
	release := make(chan struct{})
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()},
		realtime.ConnectParams{}, realtime.WithToolExecutor(realtime.ToolExecutorFunc(func(
			context.Context, ai.ToolCallPart,
		) (any, error) {
			<-release
			return "done", nil
		})))
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() {
		for range session.Events(t.Context()) {
		}
		close(done)
	}()
	connection.events <- realtime.ToolCall{ToolCallID: "late", ToolName: "late", Arguments: `{}`}
	connection.events <- realtime.ResponseDone{}
	connection.events <- realtime.OutputTranscript{Text: "second", Final: true, OutputText: true}
	connection.events <- realtime.ResponseDone{}
	deadline := time.Now().Add(time.Second)
	for len(session.NewMessages()) < 2 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if err := session.Send(t.Context(), "after"); err != nil {
		t.Fatal(err)
	}
	close(release)
	for {
		if _, ok := (<-connection.sent).(realtime.ToolResult); ok {
			break
		}
	}
	connection.end()
	<-done
}

func TestMalformedShortFormatWAV(t *testing.T) {
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	wav := make([]byte, 44)
	copy(wav[:4], "RIFF")
	binary.LittleEndian.PutUint32(wav[4:8], 20)
	copy(wav[8:12], "WAVE")
	copy(wav[12:16], "fmt ")
	binary.LittleEndian.PutUint32(wav[16:20], 8)
	if err := session.SendAudio(t.Context(), wav, "audio/wav"); err == nil {
		t.Fatal("expected short format chunk error")
	}
	connection.end()
	for range session.Events(t.Context()) {
	}
}

func TestCloseTimesOutWhileToolIgnoresCancellation(t *testing.T) {
	connection := newFakeConnection()
	started := make(chan struct{})
	release := make(chan struct{})
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()},
		realtime.ConnectParams{}, realtime.WithToolExecutor(realtime.ToolExecutorFunc(func(
			context.Context, ai.ToolCallPart,
		) (any, error) {
			close(started)
			<-release
			return "done", nil
		})))
	if err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.ToolCall{ToolCallID: "stubborn", ToolName: "stubborn", Arguments: `{}`}
	<-started
	ctx, cancel := context.WithTimeout(t.Context(), time.Millisecond)
	defer cancel()
	if err := session.Close(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("unexpected close timeout: %v", err)
	}
	close(release)
}

var _ iter.Seq2[any, error]
