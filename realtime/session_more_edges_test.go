package realtime_test

import (
	"context"
	"errors"
	"iter"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/realtime"
)

func TestSessionMoreControlAndStreamBranches(t *testing.T) {
	connection := newFakeConnection()
	profile := fullProfile()
	profile.SupportsOutputTruncation = false
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: profile}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	played := 1
	if err := session.Interrupt(t.Context(), &played); err == nil {
		t.Fatal("expected output truncation capability error")
	}
	if err := session.Interrupt(t.Context(), nil); err != nil {
		t.Fatal(err)
	}
	connection.err = errors.New("truncate")
	profile.SupportsOutputTruncation = true
	connection2 := newFakeConnection()
	session2, err := realtime.Open(t.Context(), &fakeModel{connection: connection2, profile: profile}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	connection2.err = errors.New("cancel")
	if err := session2.Interrupt(t.Context(), &played); err == nil {
		t.Fatal("expected cancel error")
	}
	connection.end()
	connection2.end()
	for range session.Events(t.Context()) {
	}
	for range session2.Events(t.Context()) {
	}

	connection = newFakeConnection()
	session, err = realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	connection.err = errors.New("audio write")
	if err := session.SendAudioStream(t.Context(), func(yield func([]byte, error) bool) {
		yield([]byte{1, 0}, nil)
	}); err == nil {
		t.Fatal("expected audio stream send error")
	}
	connection.end()
	for range session.Events(t.Context()) {
	}
}

func TestSessionAnonymousInputAudioAndInterruptedResponse(t *testing.T) {
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()},
		realtime.ConnectParams{}, realtime.WithAudioRetention(realtime.AudioRetentionInput))
	if err != nil {
		t.Fatal(err)
	}
	if err := session.SendAudio(t.Context(), []byte{1, 0}, "audio/pcm"); err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.InputTranscript{Text: "anonymous", Final: true}
	connection.events <- realtime.InputTranscript{Text: "named", Final: true, ItemID: "named"}
	connection.events <- realtime.InputTranscript{Text: "named again", Final: true, ItemID: "named"}
	connection.events <- realtime.OutputTranscript{Text: "partial"}
	connection.events <- realtime.AudioDelta{Data: []byte{2, 0}, ItemID: "filled"}
	connection.events <- realtime.ResponseDone{Interrupted: true}
	connection.end()
	for range session.Events(t.Context()) {
	}
	messages := session.NewMessages()
	foundAudio := false
	foundIncomplete := false
	for _, message := range messages {
		switch message := message.(type) {
		case ai.ModelRequest:
			for _, part := range message.Parts {
				if speech, ok := part.(ai.SpeechPart); ok && speech.Audio != nil {
					foundAudio = true
				}
			}
		case ai.ModelResponse:
			foundIncomplete = foundIncomplete || message.State == ai.ModelResponseStateIncomplete
		}
	}
	if !foundAudio || !foundIncomplete {
		t.Fatalf("audio or interrupted response missing: %+v", messages)
	}
}

func TestSessionLateToolResults(t *testing.T) {
	releases := map[string]chan struct{}{"one": make(chan struct{}), "two": make(chan struct{})}
	started := make(chan string, 2)
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()},
		realtime.ConnectParams{}, realtime.WithToolExecutor(realtime.ToolExecutorFunc(func(
			_ context.Context, call ai.ToolCallPart,
		) (any, error) {
			started <- call.ToolCallID
			<-releases[call.ToolCallID]
			if call.ToolCallID == "two" {
				return func() {}, nil
			}
			return "done", nil
		})))
	if err != nil {
		t.Fatal(err)
	}
	drainDone := make(chan struct{})
	go func() {
		for range session.Events(t.Context()) {
		}
		close(drainDone)
	}()
	connection.events <- realtime.ToolCall{ToolCallID: "one", ToolName: "first", Arguments: `{}`}
	connection.events <- realtime.ToolCall{ToolCallID: "two", ToolName: "second", Arguments: `{}`}
	<-started
	<-started
	connection.events <- realtime.ResponseDone{}
	time.Sleep(time.Millisecond)
	close(releases["one"])
	for {
		if result, ok := (<-connection.sent).(realtime.ToolResult); ok && result.ToolCallID == "one" {
			break
		}
	}
	close(releases["two"])
	for {
		if result, ok := (<-connection.sent).(realtime.ToolResult); ok && result.ToolCallID == "two" {
			if result.Output == "" {
				t.Fatal("fallback rendering was empty")
			}
			break
		}
	}
	connection.end()
	<-drainDone
	messages := session.NewMessages()
	if len(messages) != 3 {
		t.Fatalf("late results were not inserted after response: %+v", messages)
	}
	if _, ok := messages[1].(ai.ModelRequest); !ok {
		t.Fatalf("first result missing: %+v", messages)
	}
}

func TestSessionTapOverflowAndCloseTimeout(t *testing.T) {
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	audioStarted := make(chan struct{})
	audioBlocked := make(chan struct{})
	audioCtx, cancelAudio := context.WithCancel(t.Context())
	go func() {
		close(audioStarted)
		for range session.StreamAudio(audioCtx) {
			<-audioBlocked
		}
	}()
	transcriptStarted := make(chan struct{})
	transcriptBlocked := make(chan struct{})
	transcriptCtx, cancelTranscript := context.WithCancel(t.Context())
	go func() {
		close(transcriptStarted)
		for range session.StreamTranscripts(transcriptCtx) {
			<-transcriptBlocked
		}
	}()
	<-audioStarted
	<-transcriptStarted
	time.Sleep(time.Millisecond)
	drainDone := make(chan struct{})
	go func() {
		for range session.Events(t.Context()) {
		}
		close(drainDone)
	}()
	for range 40 {
		connection.events <- realtime.AudioDelta{Data: []byte{1, 0}, ItemID: "audio"}
	}
	for index := range 520 {
		connection.events <- realtime.InputTranscript{Text: string(rune('a' + index%26)), ItemID: "user"}
	}
	connection.end()
	<-drainDone
	close(audioBlocked)
	close(transcriptBlocked)
	cancelAudio()
	cancelTranscript()

	blocking := &blockingCloseConnection{fakeConnection: newFakeConnection()}
	session, err = realtime.Open(t.Context(), &fakeModel{connection: blocking, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), time.Millisecond)
	defer cancel()
	if err := session.Close(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("unexpected close timeout: %v", err)
	}
}

func TestSessionIteratorCancellationWhileOpen(t *testing.T) {
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	seen := false
	for _, err := range session.Events(ctx) {
		seen = true
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("unexpected iterator error: %v", err)
		}
	}
	if !seen {
		t.Fatal("iterator cancellation was not surfaced")
	}
	connection.end()
}

var _ iter.Seq2[[]byte, error]
