package openaiprotocol_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	"github.com/Kludex/pydantic-ai-go/ai/realtime/internal/openaiprotocol"
	openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
	"github.com/coder/websocket"
)

func TestManualReconnectReplaysOnlyCommittedAudio(t *testing.T) {
	for _, mode := range []string{"held", "partial", "uncommitted"} {
		t.Run(mode, func(t *testing.T) {
			connection, client, peers, frames := manualReconnectWire(t)
			peer := <-peers
			announced := 0
			connection.SetAudioCommitListener(func() {
				announced++
				if mode == "partial" && announced == 1 {
					_ = client.CloseNow()
				}
			})
			if err := connection.Send(t.Context(), realtime.AudioInput{Data: []byte{1, 0}}); err != nil {
				t.Fatal(err)
			}
			_ = takeFrame(t, frames)
			if mode != "uncommitted" {
				if err := connection.Send(t.Context(), realtime.CommitAudio{}); err != nil {
					t.Fatal(err)
				}
				if err := connection.Send(t.Context(), realtime.AudioInput{Data: []byte{2, 0}}); err != nil {
					t.Fatal(err)
				}
			}
			if mode == "partial" {
				_ = connection.Send(t.Context(), realtime.CommitAudio{})
				if err := connection.Send(t.Context(), realtime.CreateResponse{}); err == nil {
					t.Fatal("accepted failed commit")
				}
			} else {
				_ = peer.CloseNow()
			}
			for event, err := range connection.Events(t.Context()) {
				if err != nil {
					t.Fatal(err)
				}
				if _, ok := event.(realtime.SessionReconnected); ok {
					break
				}
			}
			<-peers
			if err := connection.Send(t.Context(), realtime.CreateResponse{}); err != nil {
				t.Fatal(err)
			}
			if mode == "uncommitted" {
				if frame := takeFrame(t, frames); frame["type"] != "response.create" || announced != 0 {
					t.Fatalf("uncommitted audio survived reconnect: %+v", frame)
				}
				return
			}
			count := 1
			if mode == "partial" {
				count = 2
			}
			for index := range count {
				frame := takeFrame(t, frames)
				if frame["type"] != "input_audio_buffer.append" || frame["audio"] != base64.StdEncoding.EncodeToString([]byte{byte(index + 1), 0}) {
					t.Fatalf("replayed audio: %+v", frame)
				}
			}
			for _, kind := range []string{"input_audio_buffer.commit", "response.create"} {
				if frame := takeFrame(t, frames); frame["type"] != kind {
					t.Fatalf("want %s: %+v", kind, frame)
				}
			}
			if announced != 1 {
				t.Fatalf("commit was announced %d times", announced)
			}
		})
	}
}

func TestManualDeferredReplySendsHeldAudioFirst(t *testing.T) {
	connection, peer, frames, _ := manualWire(t, false)
	commits := 0
	connection.(realtime.DeferredAudioCommitConnection).SetAudioCommitListener(func() { commits++ })
	_ = connection.Send(t.Context(), realtime.CreateResponse{})
	_ = takeFrame(t, frames)
	_ = connection.Send(t.Context(), realtime.AudioInput{Data: []byte{1, 0}})
	_ = connection.Send(t.Context(), realtime.CreateResponse{})
	_ = writeFrame(t.Context(), peer, map[string]any{"type": "response.done", "response": map[string]any{"status": "completed"}})
	for event, err := range connection.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(realtime.ResponseDone); ok {
			for _, kind := range []string{"input_audio_buffer.append", "response.create"} {
				if frame := takeFrame(t, frames); frame["type"] != kind {
					t.Fatalf("want %s before deferred reply: %+v", kind, frame)
				}
			}
			if commits != 1 {
				t.Fatal("implicit audio commit did not notify history")
			}
			if err := connection.Send(t.Context(), realtime.AudioInput{Data: []byte{2, 0}}); err != nil {
				t.Fatal(err)
			}
			_ = writeFrame(t.Context(), peer, map[string]any{"type": "response.created", "response": map[string]any{"id": "next"}})
		}
		if _, ok := event.(realtime.ResponseStarted); ok {
			break
		}
	}
}

func TestManualDeferredReplyPropagatesHeldAudioFailure(t *testing.T) {
	var client *websocket.Conn
	connection, socket, peers, frames := manualReconnectWire(t, func(data []byte) ([]realtime.CodecEvent, error) {
		_ = client.CloseNow()
		return openairt.MapEvent(data)
	})
	client = socket
	peer := <-peers
	_ = connection.Send(t.Context(), realtime.CreateResponse{})
	_ = takeFrame(t, frames)
	_ = connection.Send(t.Context(), realtime.AudioInput{Data: []byte{1, 0}})
	_ = connection.Send(t.Context(), realtime.CreateResponse{})
	_ = writeFrame(t.Context(), peer, map[string]any{"type": "response.done", "response": map[string]any{"status": "completed"}})
	failed := false
	for _, err := range connection.Events(t.Context()) {
		failed = err != nil
		break
	}
	if !failed {
		t.Fatal("held audio failure was hidden")
	}
}

func manualReconnectWire(t *testing.T, mappers ...openaiprotocol.Mapper) (*openaiprotocol.Connection, *websocket.Conn, <-chan *websocket.Conn, <-chan map[string]any) {
	t.Helper()
	peers := make(chan *websocket.Conn, 4)
	frames := make(chan map[string]any, 32)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		socket, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = socket.CloseNow() }()
		peers <- socket
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
	dial := func(ctx context.Context, _ []ai.ModelMessage) (*websocket.Conn, string, error) {
		socket, _, err := openaiprotocol.DialSocket(ctx, websocketEndpoint(server.URL), nil, server.Client())
		return socket, "model", err
	}
	client, _, err := dial(t.Context(), nil)
	if err != nil {
		server.Close()
		t.Fatal(err)
	}
	mapper := openairt.MapEvent
	if len(mappers) > 0 {
		mapper = mappers[0]
	}
	connection, err := openaiprotocol.New(openaiprotocol.Config{
		Socket: client, Mapper: mapper, ManualAudioTurns: true, Dial: dial,
		Reconnect: &realtime.ReconnectPolicy{MaxAttempts: 1, MaxReconnects: 1, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = connection.Close(context.Background()); server.Close() })
	return connection, client, peers, frames
}
