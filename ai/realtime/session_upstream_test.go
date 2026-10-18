package realtime_test

import (
	"bytes"
	"context"
	"fmt"
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

func TestResponseDoneCanKeepAnExchangeOpen(t *testing.T) {
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close(t.Context()) }()
	if err := session.Send(t.Context(), "question"); err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.OutputTranscript{Text: "Let me check.", Final: true}
	connection.events <- realtime.ResponseDone{MoreExpected: true}
	waited := make(chan error, 1)
	go func() { waited <- session.WaitForReply(t.Context()) }()
	select {
	case err := <-waited:
		t.Fatalf("wait returned during a stalled exchange: %v", err)
	case <-time.After(10 * time.Millisecond):
	}
	connection.events <- realtime.OutputTranscript{Text: "Done.", Final: true}
	connection.events <- realtime.ResponseDone{}
	select {
	case err := <-waited:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("wait did not return after the exchange")
	}
}

func TestRejectedRealtimeInputRollsBackState(t *testing.T) {
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close(t.Context()) }()
	if err := session.Send(t.Context(), "refused"); err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.InputRejected{InputIndex: 0}
	connection.events <- realtime.InputRejected{InputIndex: 0, Response: true}
	deadline := time.Now().Add(time.Second)
	for len(session.NewMessages()) != 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if len(session.NewMessages()) != 0 {
		t.Fatalf("refused content remained in history: %+v", session.NewMessages())
	}
	image := ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"}
	if err := session.Send(t.Context(), image); err != nil {
		t.Fatal(err)
	}
	connection.events <- realtime.InputRejected{InputIndex: 1}
	deadline = time.Now().Add(time.Second)
	for len(session.NewMessages()) != 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if len(session.NewMessages()) != 0 {
		t.Fatalf("refused image remained in history: %+v", session.NewMessages())
	}
	if err := session.WaitForReply(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func TestAudioViewBuffersFiveMinutes(t *testing.T) {
	connection := newFakeConnection()
	profile := fullProfile()
	profile.AudioOutputSampleRate = 1
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: profile}, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close(t.Context()) }()
	stream := session.StreamAudio(t.Context())
	chunks := make([][]byte, 6)
	for index := range chunks {
		chunks[index] = bytes.Repeat([]byte{byte(index)}, 120)
		connection.events <- realtime.AudioDelta{Data: chunks[index]}
	}
	time.Sleep(10 * time.Millisecond)
	var got [][]byte
	for chunk, streamErr := range stream {
		if streamErr != nil {
			t.Fatal(streamErr)
		}
		got = append(got, chunk)
		if len(got) == 5 {
			break
		}
	}
	if !slices.EqualFunc(got, chunks[1:], slices.Equal) {
		t.Fatalf("unexpected buffered audio: %d chunks", len(got))
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

func TestLateTranscriptKeepsOnePartPerOutputItem(t *testing.T) {
	type step struct {
		item, text string
		final      bool
		outputText bool
	}
	delta := func(item, text string) step { return step{item: item, text: text} }
	done := func(item, text string) step { return step{item: item, text: text, final: true} }
	textual := func(steps ...step) []step {
		for index := range steps {
			steps[index].outputText = true
		}
		return steps
	}
	cases := []struct {
		name    string
		steps   []step
		history []string
		stream  []string
	}{
		{
			name:    "in order",
			steps:   []step{delta("a", "Hello world."), done("a", "Hello world."), delta("b", "Bonjour."), done("b", "Bonjour.")},
			history: []string{"speech Hello world.", "speech Bonjour."},
			stream:  []string{"start 0 ", "delta 0 Hello world.", "end 0 Hello world.", "start 1 ", "delta 1 Bonjour.", "end 1 Bonjour."},
		},
		{
			name:    "late done",
			steps:   []step{delta("a", "Hello"), delta("a", " world."), delta("b", "Bonjour"), delta("b", " tout le monde."), done("a", "Hello world."), done("b", "Bonjour tout le monde.")},
			history: []string{"speech Hello world.", "speech Bonjour tout le monde."},
			stream:  []string{"start 0 ", "delta 0 Hello", "delta 0  world.", "end 0 Hello world.", "start 1 ", "delta 1 Bonjour", "delta 1  tout le monde.", "end 1 Bonjour tout le monde."},
		},
		{
			name:    "late done empty",
			steps:   []step{delta("a", "Hello world."), delta("b", "Bonjour."), done("a", ""), done("b", "Bonjour.")},
			history: []string{"speech Hello world.", "speech Bonjour."},
			stream:  []string{"start 0 ", "delta 0 Hello world.", "end 0 Hello world.", "start 1 ", "delta 1 Bonjour.", "end 1 Bonjour."},
		},
		{
			name:    "late deltas",
			steps:   []step{delta("a", "Hello"), delta("b", "Bonjour."), delta("a", " there"), delta("a", " world."), done("a", "Hello there world."), done("b", "Bonjour.")},
			history: []string{"speech Hello there world.", "speech Bonjour."},
			stream: []string{
				"start 0 ", "delta 0 Hello", "end 0 Hello", "start 1 ", "delta 1 Bonjour.",
				"start 0 Hello", "delta 0  there", "end 0 Hello there", "start 0 Hello there", "delta 0  world.", "end 0 Hello there world.",
				"end 1 Bonjour.",
			},
		},
		{
			name:    "late delta of the other modality",
			steps:   []step{delta("a", "Hello"), delta("b", "Bonjour."), {item: "a", text: " world.", outputText: true}, done("a", "Hello world."), done("b", "Bonjour.")},
			history: []string{"speech Hello", "speech Bonjour.", "text  world.", "speech Hello world."},
			stream: []string{
				"start 0 ", "delta 0 Hello", "end 0 Hello", "start 1 ", "delta 1 Bonjour.", "end 1 Bonjour.",
				"start 2 ", "delta 2  world.", "end 2  world.", "start 3 ", "delta 3 Hello world.", "end 3 Hello world.",
			},
		},
		{
			name:    "late text output",
			steps:   textual(delta("a", "Hello"), delta("b", "Bonjour."), delta("a", " world."), done("a", "Hello world."), done("b", "Bonjour.")),
			history: []string{"text Hello world.", "text Bonjour."},
			stream:  []string{"start 0 ", "delta 0 Hello", "end 0 Hello", "start 1 ", "delta 1 Bonjour.", "start 0 Hello", "delta 0  world.", "end 0 Hello world.", "end 1 Bonjour."},
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			connection := newFakeConnection()
			session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{})
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = session.Close(t.Context()) }()
			for _, step := range test.steps {
				connection.events <- realtime.OutputTranscript{Text: step.text, ItemID: step.item, Final: step.final, OutputText: step.outputText}
			}
			connection.events <- realtime.ResponseDone{}
			var stream []string
			for event, err := range session.Events(t.Context()) {
				if err != nil {
					t.Fatal(err)
				}
				switch event := event.(type) {
				case ai.PartStartEvent:
					stream = append(stream, fmt.Sprintf("start %d %s", event.Index, partText(event.Part)))
				case ai.PartDeltaEvent:
					text := ""
					switch delta := event.Delta.(type) {
					case ai.TextPartDelta:
						text = delta.ContentDelta
					case ai.SpeechPartDelta:
						text = delta.TranscriptDelta
					}
					stream = append(stream, fmt.Sprintf("delta %d %s", event.Index, text))
				case ai.PartEndEvent:
					stream = append(stream, fmt.Sprintf("end %d %s", event.Index, partText(event.Part)))
				}
				if _, ok := event.(realtime.TurnCompleteEvent); ok {
					break
				}
			}
			var history []string
			for _, message := range session.NewMessages() {
				if response, ok := message.(ai.ModelResponse); ok {
					for _, part := range response.Parts {
						kind := "text"
						if _, ok := part.(ai.SpeechPart); ok {
							kind = "speech"
						}
						history = append(history, kind+" "+partText(part))
					}
				}
			}
			if !slices.Equal(history, test.history) {
				t.Errorf("history:\n%q\nwant\n%q", history, test.history)
			}
			if !slices.Equal(stream, test.stream) {
				t.Errorf("stream:\n%q\nwant\n%q", stream, test.stream)
			}
		})
	}
}

func partText(part ai.ResponsePart) string {
	switch part := part.(type) {
	case ai.TextPart:
		return part.Content
	case ai.SpeechPart:
		if part.Transcript != nil {
			return *part.Transcript
		}
	}
	return ""
}

func TestLateTranscriptRestartsSpeechPartWithoutItsAudio(t *testing.T) {
	connection := newFakeConnection()
	session, err := realtime.Open(t.Context(), &fakeModel{connection: connection, profile: fullProfile()}, realtime.ConnectParams{}, realtime.WithAudioRetention(realtime.AudioRetentionOutput))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close(t.Context()) }()
	transcripts := session.StreamTranscripts(t.Context())
	connection.events <- realtime.AudioDelta{Data: make([]byte, 480), ItemID: "a"}
	connection.events <- realtime.OutputTranscript{Text: "Hello", ItemID: "a"}
	connection.events <- realtime.OutputTranscript{Text: "Bonjour.", ItemID: "b"}
	connection.events <- realtime.OutputTranscript{Text: "Hello world.", ItemID: "a", Final: true}
	connection.events <- realtime.OutputTranscript{Text: "Bonjour.", ItemID: "b", Final: true}
	connection.events <- realtime.ResponseDone{}
	var stream []string
	for event, err := range session.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		switch event := event.(type) {
		case ai.PartStartEvent:
			stream = append(stream, fmt.Sprintf("start %s %t", partText(event.Part), event.Part.(ai.SpeechPart).Audio != nil))
		case ai.PartEndEvent:
			stream = append(stream, fmt.Sprintf("end %s %t", partText(event.Part), event.Part.(ai.SpeechPart).Audio != nil))
		}
		if _, ok := event.(realtime.TurnCompleteEvent); ok {
			break
		}
	}
	want := []string{"start  false", "end Hello true", "start  false", "start Hello false", "end Hello world. true", "end Bonjour. false"}
	if !slices.Equal(stream, want) {
		t.Fatalf("stream: %q", stream)
	}
	var recorded []string
	for _, message := range session.NewMessages() {
		if response, ok := message.(ai.ModelResponse); ok {
			for _, part := range response.Parts {
				recorded = append(recorded, fmt.Sprintf("%s %t", partText(part), part.(ai.SpeechPart).Audio != nil))
			}
		}
	}
	if !slices.Equal(recorded, []string{"Hello world. true", "Bonjour. false"}) {
		t.Fatalf("recorded: %q", recorded)
	}
	var updates []string
	for update, err := range transcripts {
		if err != nil {
			t.Fatal(err)
		}
		updates = append(updates, fmt.Sprintf("%d %q %q", update.Index, update.Delta, update.Transcript))
		if len(updates) == 3 {
			break
		}
	}
	if !slices.Equal(updates, []string{`0 "Hello" "Hello"`, `1 "Bonjour." "Bonjour."`, `0 " world." "Hello world."`}) {
		t.Fatalf("transcripts: %q", updates)
	}
}

func TestInterruptAtAudioCutsRepliesAheadOfPlayback(t *testing.T) {
	chunk := make([]byte, 4800)
	speak := func(connection *fakeConnection, itemID string) {
		connection.events <- realtime.AudioDelta{Data: chunk, ItemID: itemID}
		connection.events <- realtime.AudioDelta{Data: chunk, ItemID: itemID}
		connection.events <- realtime.OutputTranscript{Text: "said " + itemID, ItemID: itemID, Final: true}
	}
	until := func(t *testing.T, session *realtime.Session, count int, match func(realtime.Event) bool) {
		t.Helper()
		for event, err := range session.Events(t.Context()) {
			if err != nil {
				t.Fatal(err)
			}
			if match(event) {
				if count--; count == 0 {
					return
				}
			}
		}
	}
	turns := func(event realtime.Event) bool {
		_, ok := event.(realtime.TurnCompleteEvent)
		return ok
	}
	cuts := func(messages []ai.ModelMessage) []string {
		var cuts []string
		for _, message := range messages {
			if response, ok := message.(ai.ModelResponse); ok {
				cut := string(response.State)
				for _, part := range response.Parts {
					if speech := part.(ai.SpeechPart); speech.InterruptedAtMS != nil {
						cut += fmt.Sprintf(" %d", *speech.InterruptedAtMS)
					} else {
						cut += " -"
					}
				}
				cuts = append(cuts, cut)
			}
		}
		return cuts
	}
	interrupt := func(t *testing.T, session *realtime.Session, connection *fakeConnection, played int) []realtime.Input {
		t.Helper()
		if interrupted, err := session.InterruptAtAudio(t.Context(), played); err != nil || !interrupted {
			t.Fatalf("interrupted=%v err=%v", interrupted, err)
		}
		var sent []realtime.Input
		for len(connection.sent) > 0 {
			sent = append(sent, <-connection.sent)
		}
		return sent
	}
	open := func(t *testing.T) (*realtime.Session, *fakeConnection) {
		session, connection := openSessionWithProfile(t, fullProfile())
		t.Cleanup(func() { _ = session.Close(t.Context()) })
		_ = session.StreamAudio(t.Context())
		return session, connection
	}

	t.Run("finished replies", func(t *testing.T) {
		session, connection := open(t)
		speak(connection, "a")
		connection.events <- realtime.ResponseDone{}
		speak(connection, "b")
		connection.events <- realtime.ResponseDone{}
		until(t, session, 2, turns)
		before := session.Messages()

		sent := interrupt(t, session, connection, len(chunk))
		want := []realtime.Input{
			realtime.TruncateOutput{AudioEndMilliseconds: 100, ItemID: "a"},
			realtime.TruncateOutput{AudioEndMilliseconds: 0, ItemID: "b"},
			realtime.CancelResponse{},
		}
		if !slices.Equal(sent, want) {
			t.Fatalf("sent %#v", sent)
		}
		if got := cuts(session.Messages()); !slices.Equal(got, []string{"interrupted 100", "interrupted 0"}) {
			t.Fatalf("history %q", got)
		}
		if got := cuts(before); !slices.Equal(got, []string{"complete -", "complete -"}) {
			t.Fatalf("snapshot %q", got)
		}
	})

	t.Run("reply under way", func(t *testing.T) {
		session, connection := open(t)
		connection.events <- realtime.AudioDelta{Data: chunk, ItemID: "a"}
		connection.events <- realtime.AudioDelta{Data: chunk, ItemID: "a"}
		until(t, session, 2, func(event realtime.Event) bool {
			_, ok := event.(ai.PartDeltaEvent)
			return ok
		})

		sent := interrupt(t, session, connection, len(chunk))
		if !slices.Equal(sent, []realtime.Input{realtime.TruncateOutput{AudioEndMilliseconds: 100, ItemID: "a"}, realtime.CancelResponse{}}) {
			t.Fatalf("sent %#v", sent)
		}
		connection.events <- realtime.ResponseDone{Interrupted: true}
		until(t, session, 1, turns)
		if got := cuts(session.Messages()); !slices.Equal(got, []string{"interrupted 100"}) {
			t.Fatalf("history %q", got)
		}
	})

	t.Run("reply before a reconnect", func(t *testing.T) {
		session, connection := open(t)
		speak(connection, "a")
		connection.events <- realtime.ResponseDone{}
		connection.events <- realtime.SessionReconnected{}
		until(t, session, 1, func(event realtime.Event) bool {
			_, ok := event.(realtime.SessionReconnectEvent)
			return ok
		})

		sent := interrupt(t, session, connection, len(chunk))
		if !slices.Equal(sent, []realtime.Input{realtime.TruncateOutput{AudioEndMilliseconds: 100, ItemID: "a"}, realtime.CancelResponse{}}) {
			t.Fatalf("sent %#v", sent)
		}
		if got := cuts(session.Messages()); !slices.Equal(got, []string{"complete -"}) {
			t.Fatalf("history %q", got)
		}
	})

	t.Run("replies without item ids", func(t *testing.T) {
		session, connection := open(t)
		for range 2 {
			connection.events <- realtime.AudioDelta{Data: chunk}
			connection.events <- realtime.ResponseDone{}
		}
		until(t, session, 2, turns)

		sent := interrupt(t, session, connection, 0)
		if !slices.Equal(sent, []realtime.Input{realtime.TruncateOutput{AudioEndMilliseconds: 0}, realtime.CancelResponse{}}) {
			t.Fatalf("sent %#v", sent)
		}
	})

	t.Run("only the latest items", func(t *testing.T) {
		session, connection := open(t)
		connection.sent = make(chan realtime.Input, 64)
		for item := range 33 {
			connection.events <- realtime.AudioDelta{Data: chunk, ItemID: fmt.Sprint(item)}
			connection.events <- realtime.ResponseDone{}
		}
		until(t, session, 33, turns)

		sent := interrupt(t, session, connection, 0)
		if len(sent) != 33 || sent[0] != (realtime.TruncateOutput{ItemID: "1"}) {
			t.Fatalf("sent %#v", sent)
		}
		history := cuts(session.Messages())
		if history[0] != "complete -" || history[1] != "interrupted 0" || history[32] != "interrupted 0" {
			t.Fatalf("history %q", history)
		}
	})
}
