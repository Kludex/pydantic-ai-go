package realtime_test

import (
	"context"
	"errors"
	"iter"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/realtime"
)

type unknownCodecEvent struct{ Unknown bool }

func (unknownCodecEvent) RealtimeCodecEventKind() string { return "unknown" }

type historyConnection struct {
	*fakeConnection
	history func() []ai.ModelMessage
}

func (connection *historyConnection) SetMessageHistory(history func() []ai.ModelMessage) {
	connection.history = history
}

type eventErrorConnection struct {
	*fakeConnection
	eventErr error
}

func (connection *eventErrorConnection) Events(context.Context) iter.Seq2[realtime.CodecEvent, error] {
	return func(yield func(realtime.CodecEvent, error) bool) { yield(nil, connection.eventErr) }
}

type noInfoConnection struct{ base *fakeConnection }

func (connection *noInfoConnection) Send(ctx context.Context, input realtime.Input) error {
	return connection.base.Send(ctx, input)
}
func (connection *noInfoConnection) Events(ctx context.Context) iter.Seq2[realtime.CodecEvent, error] {
	return connection.base.Events(ctx)
}
func (connection *noInfoConnection) Close(ctx context.Context) error {
	return connection.base.Close(ctx)
}

type blockingCloseConnection struct{ *fakeConnection }

func (connection *blockingCloseConnection) Close(ctx context.Context) error {
	<-ctx.Done()
	return ctx.Err()
}

func TestSessionHistoryAndIteratorControls(t *testing.T) {
	base := newFakeConnection()
	connection := &historyConnection{fakeConnection: base}
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil || connection.history == nil {
		t.Fatalf("history callback missing: session=%v err=%v", session, err)
	}
	if got := connection.history(); got == nil {
		t.Fatalf("history callback returned nil")
	}
	base.events <- realtime.OutputTranscript{Text: "first", OutputText: true, ItemID: "one"}
	base.events <- realtime.OutputTranscript{Text: "first", Final: true, OutputText: true, ItemID: "one"}
	base.events <- realtime.OutputTranscript{Text: "spoken", ItemID: "two"}
	base.events <- realtime.AudioDelta{Data: []byte{1, 0}, ItemID: "three"}
	base.events <- realtime.ResponseDone{}
	base.events <- realtime.PartStarted{Event: ai.PartStartEvent{
		PartID: "native", Part: ai.NativeToolCallPart{ToolName: "search"},
	}}
	base.events <- realtime.PartEnded{Event: ai.PartEndEvent{
		PartID: "native", Part: ai.NativeToolCallPart{ToolName: "search"},
	}}
	base.events <- realtime.ResponseStarted{ResponseID: "response"}
	base.end()
	count := 0
	for _, err := range session.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		count++
		if count == 1 {
			break
		}
	}
	for range session.Events(t.Context()) {
	}
	if len(session.NewMessages()) == 0 {
		t.Fatal("expected response history")
	}

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	for _, err := range session.Events(ctx) {
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("unexpected iterator cancellation: %v", err)
		}
	}
}

func TestSessionStreamsAndInputErrors(t *testing.T) {
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	inputs := func(yield func(any, error) bool) {
		if !yield("one", nil) {
			return
		}
		yield(ai.BinaryContent{Data: []byte{1, 0}, MediaType: "audio/pcm"}, nil)
	}
	if err := session.SendStream(t.Context(), inputs); err != nil {
		t.Fatal(err)
	}
	inputFailure := errors.New("input")
	if err := session.SendStream(t.Context(), func(yield func(any, error) bool) { yield(nil, inputFailure) }); !errors.Is(err, inputFailure) {
		t.Fatalf("unexpected stream input error: %v", err)
	}
	chunks := func(yield func([]byte, error) bool) {
		if !yield([]byte{1, 0}, nil) {
			return
		}
		yield([]byte{2, 0}, nil)
	}
	if err := session.SendAudioStream(t.Context(), chunks); err != nil {
		t.Fatal(err)
	}
	chunkFailure := errors.New("chunk")
	if err := session.SendAudioStream(t.Context(), func(yield func([]byte, error) bool) { yield(nil, chunkFailure) }); !errors.Is(err, chunkFailure) {
		t.Fatalf("unexpected chunk error: %v", err)
	}
	connection.err = errors.New("write")
	if err := session.SendStream(t.Context(), func(yield func(any, error) bool) { yield("fail", nil) }); err == nil {
		t.Fatal("expected send stream write error")
	}
	if err := session.ClearAudio(t.Context()); err == nil {
		t.Fatal("expected clear write error")
	}
	if err := session.Interrupt(t.Context(), nil); err == nil {
		t.Fatal("expected interrupt write error")
	}
	connection.err = nil
	negative := -1
	if err := session.Interrupt(t.Context(), &negative); err == nil {
		t.Fatal("expected negative playback error")
	}

	audioCtx, cancelAudio := context.WithCancel(t.Context())
	audioErrors := make(chan error, 1)
	go func() {
		for _, err := range session.StreamAudio(audioCtx) {
			audioErrors <- err
			return
		}
	}()
	cancelAudio()
	if err := <-audioErrors; !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected audio tap cancellation: %v", err)
	}
	transcriptCtx, cancelTranscript := context.WithCancel(t.Context())
	transcriptErrors := make(chan error, 1)
	go func() {
		for _, err := range session.StreamTranscripts(transcriptCtx) {
			transcriptErrors <- err
			return
		}
	}()
	cancelTranscript()
	if err := <-transcriptErrors; !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected transcript tap cancellation: %v", err)
	}
	connection.end()
	for range session.Events(t.Context()) {
	}
}

func TestSessionWAVAndRetentionErrors(t *testing.T) {
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()},
		realtime.ConnectParams{}, realtime.WithImageRetention(1, 0))
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Send(t.Context(), ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"}); err != nil {
		t.Fatal(err)
	}
	if len(session.NewMessages()) != 0 {
		t.Fatalf("zero image retention kept history: %+v", session.NewMessages())
	}
	if err := session.SendAudio(t.Context(), nil, "audio/pcm"); err != nil {
		t.Fatal(err)
	}
	if err := session.SendAudio(t.Context(), testWAV([]byte{1, 0}, 16000), "audio/wav"); err == nil {
		t.Fatal("expected WAV rate error")
	}
	truncated := testWAV([]byte{1, 0}, 24000)
	truncated[40] = 20
	if err := session.SendAudio(t.Context(), truncated, "audio/wav"); err == nil {
		t.Fatal("expected truncated WAV error")
	}
	badFormat := testWAV([]byte{1, 0}, 24000)
	badFormat[20] = 3
	if err := session.SendAudio(t.Context(), badFormat, "audio/wav"); err == nil {
		t.Fatal("expected WAV format error")
	}
	connection.end()
	for range session.Events(t.Context()) {
	}
}

func TestSessionTerminalErrors(t *testing.T) {
	root := errors.New("read")
	connection := &eventErrorConnection{fakeConnection: newFakeConnection(), eventErr: root}
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range session.Events(t.Context()) {
		if !errors.Is(err, root) {
			t.Fatalf("unexpected read error: %v", err)
		}
	}
	if !errors.Is(session.Err(), root) {
		t.Fatalf("terminal error was not retained: %v", session.Err())
	}

	for _, event := range []realtime.CodecEvent{
		realtime.SessionError{Err: errors.New("fatal")},
		unknownCodecEvent{Unknown: true},
	} {
		base := newFakeConnection()
		session, err := realtime.Open(t.Context(), &fakeModel{connection: base, profile: fullProfile()}, realtime.ConnectParams{})
		if err != nil {
			t.Fatal(err)
		}
		base.events <- event
		for _, err := range session.Events(t.Context()) {
			if err == nil {
				t.Fatal("expected terminal event error")
			}
		}
	}

	base := newFakeConnection()
	base.err = errors.New("close")
	session, err = realtime.Open(t.Context(), &fakeModel{connection: base, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Close(t.Context()); err == nil {
		t.Fatal("expected close error")
	}
}

func TestSessionNoInfoAndToolOrdering(t *testing.T) {
	base := newFakeConnection()
	connection := &noInfoConnection{base: base}
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	if !session.InputTranscriptionEnabled() || !session.ReconnectRestoresInFlightState() {
		t.Fatal("connection info defaults are wrong")
	}
	base.events <- realtime.ToolCall{ToolCallID: "first", ToolName: "missing"}
	base.events <- realtime.ToolCall{ToolCallID: "second", ToolName: "missing"}
	for results := 0; results < 2; {
		if _, ok := (<-base.sent).(realtime.ToolResult); ok {
			results++
		}
	}
	base.events <- realtime.ResponseDone{}
	base.end()
	for range session.Events(t.Context()) {
	}
	messages := session.NewMessages()
	if len(messages) != 3 {
		t.Fatalf("tool result history is not adjacent: %+v", messages)
	}
	if _, ok := messages[0].(ai.ModelResponse); !ok {
		t.Fatalf("tool response must lead results: %+v", messages)
	}
}
