package realtime_test

import (
	"context"
	"slices"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
)

type interruptingConnection struct {
	*fakeConnection
	serverCancels bool
}

func (connection *interruptingConnection) InterruptsResponseOnSpeech() bool {
	return connection.serverCancels
}

func TestSendResponseControl(t *testing.T) {
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close(t.Context()) }()

	if err := session.Send(t.Context(), "context", realtime.WithResponse(false)); err != nil {
		t.Fatal(err)
	}
	image := ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"}
	if err := session.Send(t.Context(), image, realtime.WithResponse(true)); err != nil {
		t.Fatal(err)
	}
	if err := session.Send(t.Context(), ai.BinaryContent{Data: []byte{0, 0}, MediaType: "audio/pcm"}, realtime.WithResponse(true)); err == nil {
		t.Fatal("expected audio response error")
	}

	if input := <-connection.sent; input != (realtime.TextContext{Text: "context"}) {
		t.Fatalf("unexpected text input: %#v", input)
	}
	input := <-connection.sent
	got, ok := input.(realtime.ImageInput)
	if !ok || !got.Respond || !slices.Equal(got.Content.Data, image.Data) {
		t.Fatalf("unexpected image input: %#v", input)
	}
}

func TestMediaViewsSubscribeBeforeIteration(t *testing.T) {
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	audio := session.StreamAudio(t.Context())
	transcripts := session.StreamTranscripts(t.Context())
	connection.events <- realtime.AudioDelta{Data: []byte{1, 0}, ItemID: "assistant"}
	connection.events <- realtime.OutputTranscript{Text: "hello", ItemID: "assistant"}
	time.Sleep(time.Millisecond)

	for chunk, streamErr := range audio {
		if streamErr != nil || !slices.Equal(chunk, []byte{1, 0}) {
			t.Fatalf("unexpected audio: %v, %v", chunk, streamErr)
		}
		break
	}
	for update, streamErr := range transcripts {
		if streamErr != nil || update.Transcript != "hello" {
			t.Fatalf("unexpected transcript: %+v, %v", update, streamErr)
		}
		break
	}
	if err := session.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func TestAutomaticBargeInUsesPlayedAudio(t *testing.T) {
	base := newFakeConnection()
	connection := &interruptingConnection{fakeConnection: base}
	session, err := realtime.Open(
		t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{},
		realtime.WithBargeIn(true),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close(t.Context()) }()

	audio := session.StreamAudio(t.Context())
	playing := make(chan struct{})
	release := make(chan struct{})
	go func() {
		for range audio {
			close(playing)
			<-release
			return
		}
	}()
	if err := session.Send(t.Context(), "speak"); err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.AudioDelta{Data: []byte{1, 0, 2, 0}, ItemID: "assistant"}
	<-playing
	connection.events <- realtime.AudioDelta{Data: []byte{3, 0, 4, 0}, ItemID: "assistant"}
	connection.events <- realtime.InputSpeechStarted{ItemID: "user"}

	var inputs []realtime.Input
	deadline := time.After(time.Second)
	for len(inputs) < 3 {
		select {
		case input := <-connection.sent:
			inputs = append(inputs, input)
		case <-deadline:
			t.Fatalf("timed out waiting for interruption: %#v", inputs)
		}
	}
	if _, ok := inputs[0].(realtime.TextInput); !ok {
		t.Fatalf("unexpected first input: %#v", inputs[0])
	}
	truncate, ok := inputs[1].(realtime.TruncateOutput)
	if !ok || truncate.AudioEndMilliseconds != 0 {
		t.Fatalf("unexpected truncation: %#v", inputs[1])
	}
	if _, ok := inputs[2].(realtime.CancelResponse); !ok {
		t.Fatalf("unexpected cancellation: %#v", inputs[2])
	}
	connection.events <- realtime.AudioDelta{Data: []byte{5, 0}, ItemID: "assistant"}
	time.Sleep(time.Millisecond)
	close(release)
}

func TestEnqueuePriorities(t *testing.T) {
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close(t.Context()) }()

	if _, err := session.Enqueue(t.Context(), "soon", ai.SystemPromptPart{Content: "be brief"}); err != nil {
		t.Fatal(err)
	}
	if _, err := session.EnqueueWhenIdle(t.Context(), "later"); err != nil {
		t.Fatal(err)
	}
	select {
	case input := <-connection.sent:
		if input != (realtime.TextInput{Text: "soon\n\n<system>be brief</system>"}) {
			t.Fatalf("unexpected ASAP input: %#v", input)
		}
	case <-time.After(time.Second):
		t.Fatal("ASAP input was not delivered")
	}
	select {
	case input := <-connection.sent:
		t.Fatalf("idle input arrived before the response boundary: %#v", input)
	case <-time.After(10 * time.Millisecond):
	}
	connection.events <- realtime.ResponseDone{}
	select {
	case input := <-connection.sent:
		if input != (realtime.TextInput{Text: "later"}) {
			t.Fatalf("unexpected idle input: %#v", input)
		}
	case <-time.After(time.Second):
		t.Fatal("idle input was not delivered")
	}
}

func TestEnqueueBatchSolicitsOneResponse(t *testing.T) {
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close(t.Context()) }()
	if err := session.Send(t.Context(), "active"); err != nil {
		t.Fatal(err)
	}
	<-connection.sent
	if _, err := session.Enqueue(t.Context(), "first"); err != nil {
		t.Fatal(err)
	}
	if _, err := session.Enqueue(t.Context(), "second"); err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.ResponseDone{}

	first := <-connection.sent
	second := <-connection.sent
	if first != (realtime.TextContext{Text: "first"}) || second != (realtime.TextInput{Text: "second"}) {
		t.Fatalf("unexpected batch: %#v, %#v", first, second)
	}
}

func TestToolCanCloseSession(t *testing.T) {
	connection := newFakeConnection()
	var session *realtime.Session
	closed := make(chan error, 1)
	var err error
	session, err = realtime.Open(
		t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{},
		realtime.WithToolExecutor(realtime.ToolExecutorFunc(func(ctx context.Context, _ ai.ToolCallPart) (any, error) {
			closeErr := session.Close(ctx)
			closed <- closeErr
			return nil, closeErr
		})),
	)
	if err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.ToolCall{ToolCallID: "hang-up", ToolName: "hang_up", Arguments: `{}`}
	select {
	case closeErr := <-closed:
		if closeErr != nil {
			t.Fatal(closeErr)
		}
	case <-time.After(time.Second):
		t.Fatal("tool deadlocked while closing its session")
	}
}
