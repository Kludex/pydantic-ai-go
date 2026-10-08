package openaiprotocol_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	"github.com/Kludex/pydantic-ai-go/ai/realtime/internal/openaiprotocol"
	openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
	xairt "github.com/Kludex/pydantic-ai-go/ai/realtime/xai"
	"github.com/coder/websocket"
)

func TestXAIManualCommitWaitsForResponseRequest(t *testing.T) {
	for _, speech := range []bool{false, true} {
		connection, socket, frames, events := manualWire(t)
		if err := connection.Send(t.Context(), realtime.AudioInput{Data: []byte{1, 0}}); err != nil {
			t.Fatal(err)
		}
		if frame := takeFrame(t, frames); frame["type"] != "input_audio_buffer.append" {
			t.Fatalf("audio: %+v", frame)
		}
		if speech {
			_ = writeFrame(t.Context(), socket, map[string]any{"type": "input_audio_buffer.speech_started", "item_id": "u"})
			select {
			case <-events:
			case <-time.After(time.Second):
				t.Fatal("speech event missing")
			}
		}
		if err := connection.Send(t.Context(), realtime.CommitAudio{}); err != nil {
			t.Fatal(err)
		}
		select {
		case frame := <-frames:
			t.Fatalf("commit replied early: %+v", frame)
		case <-time.After(10 * time.Millisecond):
		}
		if err := connection.Send(t.Context(), realtime.AudioInput{Data: []byte{2, 0}}); err != nil {
			t.Fatal(err)
		}
		if err := connection.Send(t.Context(), realtime.ClearAudio{}); err != nil {
			t.Fatal(err)
		}
		if err := connection.Send(t.Context(), realtime.CreateResponse{}); err != nil {
			t.Fatal(err)
		}
		if frame := takeFrame(t, frames); frame["type"] != "input_audio_buffer.commit" {
			t.Fatalf("commit: %+v", frame)
		}
		if !speech {
			if frame := takeFrame(t, frames); frame["type"] != "response.create" {
				t.Fatalf("silence response: %+v", frame)
			}
		}
		_ = writeFrame(t.Context(), socket, map[string]any{"type": "response.created", "response": map[string]any{"id": "r"}})
		_ = writeFrame(t.Context(), socket, map[string]any{"type": "response.done", "response": map[string]any{"id": "r", "status": "completed"}})
		awaitDone(t, events)
		if err := connection.Send(t.Context(), realtime.CreateResponse{}); err != nil {
			t.Fatal(err)
		}
		if frame := takeFrame(t, frames); frame["type"] != "input_audio_buffer.clear" {
			t.Fatalf("repeat response did not clear empty audio: %+v", frame)
		}
		if frame := takeFrame(t, frames); frame["type"] != "response.create" {
			t.Fatalf("repeat request: %+v", frame)
		}
	}
}

func TestXAIHoldsAudioBehindResponse(t *testing.T) {
	connection, socket, frames, events := manualWire(t)
	if err := connection.Send(t.Context(), realtime.CreateResponse{}); err != nil {
		t.Fatal(err)
	}
	_ = takeFrame(t, frames)
	if err := connection.Send(t.Context(), realtime.AudioInput{Data: []byte{1, 0}}); err != nil {
		t.Fatal(err)
	}
	if err := connection.Send(t.Context(), realtime.AudioInput{Data: []byte{2, 0}}); err != nil {
		t.Fatal(err)
	}
	select {
	case frame := <-frames:
		t.Fatalf("audio cut off response: %+v", frame)
	case <-time.After(10 * time.Millisecond):
	}
	_ = writeFrame(t.Context(), socket, map[string]any{"type": "response.done", "response": map[string]any{"id": "r", "status": "completed"}})
	awaitDone(t, events)
	for range 2 {
		if frame := takeFrame(t, frames); frame["type"] != "input_audio_buffer.append" {
			t.Fatalf("held audio: %+v", frame)
		}
	}
	if err := connection.Send(t.Context(), realtime.ClearAudio{}); err != nil {
		t.Fatal(err)
	}
	if frame := takeFrame(t, frames); frame["type"] != "input_audio_buffer.clear" {
		t.Fatalf("clear: %+v", frame)
	}
	if err := connection.Send(t.Context(), realtime.CommitAudio{}); err != nil {
		t.Fatal(err)
	}
	if frame := takeFrame(t, frames); frame["type"] != "input_audio_buffer.commit" {
		t.Fatalf("empty commit: %+v", frame)
	}
}

func TestXAICoversHeldAudioInExtendedCommit(t *testing.T) {
	connection, socket, frames, events := manualWire(t)
	if err := connection.Send(t.Context(), realtime.AudioInput{Data: []byte{1, 0}}); err != nil {
		t.Fatal(err)
	}
	_ = takeFrame(t, frames)
	if err := connection.Send(t.Context(), realtime.CommitAudio{}); err != nil {
		t.Fatal(err)
	}
	if err := connection.Send(t.Context(), realtime.AudioInput{Data: []byte{2, 0}}); err != nil {
		t.Fatal(err)
	}
	if err := connection.Send(t.Context(), realtime.CommitAudio{}); err != nil {
		t.Fatal(err)
	}
	if err := connection.Send(t.Context(), realtime.TextContext{Text: "context"}); err != nil {
		t.Fatal(err)
	}
	_ = takeFrame(t, frames)
	if err := connection.Send(t.Context(), realtime.CreateResponse{}); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"input_audio_buffer.append", "input_audio_buffer.commit", "response.create"} {
		if frame := takeFrame(t, frames); frame["type"] != kind {
			t.Fatalf("want %s: %+v", kind, frame)
		}
	}
	_ = writeFrame(t.Context(), socket, map[string]any{"type": "response.done", "response": map[string]any{"id": "r", "status": "completed"}})
	awaitDone(t, events)
}

func TestXAIInvalidToolChoice(t *testing.T) {
	if _, err := xairt.NewModel("model", xairt.WithAPIKey("key")).Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{ToolChoice: "invalid"}}); err == nil {
		t.Fatal("accepted invalid tool choice")
	}
}

func TestXAIManualTransportFailures(t *testing.T) {
	for _, extended := range []bool{false, true} {
		connection, _, frames, _ := manualWire(t)
		commits := connection.(realtime.DeferredAudioCommitConnection)
		if !commits.DefersAudioCommit() {
			t.Fatal("manual commit is not deferred")
		}
		if err := connection.Send(t.Context(), realtime.AudioInput{Data: []byte{0, 0}}); err != nil {
			t.Fatal(err)
		}
		_ = takeFrame(t, frames)
		if err := connection.Send(t.Context(), realtime.CommitAudio{}); err != nil {
			t.Fatal(err)
		}
		if extended {
			_ = connection.Send(t.Context(), realtime.AudioInput{Data: []byte{0, 0}})
			_ = connection.Send(t.Context(), realtime.CommitAudio{})
		}
		commits.SetAudioCommitListener(func() { _ = connection.Close(t.Context()) })
		if err := connection.Send(t.Context(), realtime.CreateResponse{}); err == nil {
			t.Fatal("write to closed transport was accepted")
		}
	}
}

func TestManualRepeatedResponsePropagatesClearFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		socket, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = socket.CloseNow() }()
		for range 3 {
			if _, _, err := socket.Read(r.Context()); err != nil {
				return
			}
		}
		_ = writeFrame(r.Context(), socket, map[string]any{"type": "response.done", "response": map[string]any{"status": "completed"}})
		_, _, _ = socket.Read(r.Context())
	}))
	defer server.Close()
	socket, _, err := openaiprotocol.DialSocket(t.Context(), websocketEndpoint(server.URL), nil, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	connection, err := openaiprotocol.New(openaiprotocol.Config{Socket: socket, Mapper: openairt.MapEvent, ManualAudioTurns: true})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = connection.Close(t.Context()) }()
	for _, input := range []realtime.Input{realtime.AudioInput{Data: []byte{0, 0}}, realtime.CommitAudio{}, realtime.CreateResponse{}} {
		if err := connection.Send(t.Context(), input); err != nil {
			t.Fatal(err)
		}
	}
	for event, err := range connection.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(realtime.ResponseDone); ok {
			break
		}
	}
	_ = socket.CloseNow()
	if err := connection.Send(t.Context(), realtime.CreateResponse{}); err == nil {
		t.Fatal("clear failure was hidden")
	}
}

func TestXAIQueuedAudioFailureAndDeferredReply(t *testing.T) {
	for _, closeTransport := range []bool{false, true} {
		connection, socket, frames, _ := manualWire(t, false)
		_ = connection.Send(t.Context(), realtime.CreateResponse{})
		_ = takeFrame(t, frames)
		_ = connection.Send(t.Context(), realtime.AudioInput{Data: []byte{0, 0}})
		if !closeTransport {
			_ = connection.Send(t.Context(), realtime.CreateResponse{})
		}
		_ = writeFrame(t.Context(), socket, map[string]any{"type": "response.done", "response": map[string]any{"status": "completed"}})
		failed := false
		for event, err := range connection.Events(t.Context()) {
			if err != nil {
				failed = true
				break
			}
			if _, ok := event.(realtime.ResponseDone); ok {
				if closeTransport {
					_ = connection.Close(t.Context())
				} else {
					_ = takeFrame(t, frames)
					_ = writeFrame(t.Context(), socket, map[string]any{"type": "response.created", "response": map[string]any{"id": "next"}})
				}
			}
			if _, ok := event.(realtime.ResponseStarted); ok {
				break
			}
		}
		if closeTransport && !failed {
			t.Fatal("held audio failure was hidden")
		}
	}
}

func TestXAIMaxDurationEndsConnectionIterator(t *testing.T) {
	_, socket, _, events := manualWire(t)
	_ = writeFrame(t.Context(), socket, map[string]any{"type": "error", "error": map[string]any{"type": "max_duration", "message": "session cap"}})
	for range events {
	}
}

func manualWire(t *testing.T, pump ...bool) (realtime.Connection, *websocket.Conn, <-chan map[string]any, <-chan realtime.CodecEvent) {
	t.Helper()
	frames := make(chan map[string]any, 32)
	ready := make(chan *websocket.Conn, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		socket, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = socket.CloseNow() }()
		_ = writeFrame(r.Context(), socket, map[string]any{"type": "session.created", "session": map[string]any{}})
		_, _, _ = socket.Read(r.Context())
		_ = writeFrame(r.Context(), socket, map[string]any{"type": "session.updated"})
		ready <- socket
		for {
			_, data, err := socket.Read(r.Context())
			if err != nil {
				return
			}
			var frame map[string]any
			_ = json.Unmarshal(data, &frame)
			frames <- frame
		}
	}))
	model := xairt.NewModel("grok-voice", xairt.WithAPIKey("key"), xairt.WithBaseURL("ws"+strings.TrimPrefix(server.URL, "http")+"/v1"))
	connection, err := model.Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{TurnDetection: &realtime.TurnDetection{Enabled: false}}})
	if err != nil {
		server.Close()
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	events := make(chan realtime.CodecEvent, 32)
	if len(pump) == 0 || pump[0] {
		go func() {
			defer close(events)
			for event, err := range connection.Events(ctx) {
				if err != nil {
					return
				}
				events <- event
			}
		}()
	}
	t.Cleanup(func() { cancel(); _ = connection.Close(context.Background()); server.Close() })
	return connection, <-ready, frames, events
}

func takeFrame(t *testing.T, frames <-chan map[string]any) map[string]any {
	t.Helper()
	select {
	case frame := <-frames:
		return frame
	case <-time.After(time.Second):
		t.Fatal("frame did not arrive")
		return nil
	}
}
func awaitDone(t *testing.T, events <-chan realtime.CodecEvent) {
	t.Helper()
	for {
		select {
		case event := <-events:
			if _, ok := event.(realtime.ResponseDone); ok {
				return
			}
		case <-time.After(time.Second):
			t.Fatal("response did not end")
		}
	}
}
