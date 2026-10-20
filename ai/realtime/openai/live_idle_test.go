package openai_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
	"github.com/coder/websocket"
)

func idleAudio() realtime.ConnectParams {
	return realtime.ConnectParams{Settings: realtime.Settings{Provider: map[string]any{"openai_live_idle_audio": true}}}
}

func nextAudio(t *testing.T, sent <-chan map[string]any) []byte {
	t.Helper()
	for {
		select {
		case frame := <-sent:
			if frame["type"] == "session.input_audio.append" {
				data, _ := base64.StdEncoding.DecodeString(frame["audio"].(string))
				return data
			}
		case <-time.After(3 * time.Second):
			t.Fatal("no audio received")
			return nil
		}
	}
}

func silent(data []byte) bool { return bytes.Equal(data, make([]byte, 4800)) }

func TestLiveIdleAudioIsOffByDefault(t *testing.T) {
	_, peer := livePair(t, realtime.ConnectParams{})
	select {
	case frame := <-peer.sent:
		t.Fatalf("unexpected frame: %v", frame)
	case <-time.After(500 * time.Millisecond):
	}
}

func TestLiveIdleAudioStreamsSilenceInRealTime(t *testing.T) {
	started := time.Now()
	connection, peer := livePair(t, idleAudio())
	if data := nextAudio(t, peer.sent); !silent(data) || time.Since(started) < 300*time.Millisecond {
		t.Fatalf("first frame: %d bytes after %s", len(data), time.Since(started))
	}
	frames, deadline := 0, time.After(time.Second)
counting:
	for {
		select {
		case <-peer.sent:
			frames++
		case <-deadline:
			break counting
		}
	}
	if frames < 5 || frames > 12 {
		t.Fatalf("frames per second: %d", frames)
	}

	sentAt := time.Now()
	speech := bytes.Repeat([]byte{1, 0}, 24000)
	if err := connection.Send(t.Context(), realtime.AudioInput{Data: speech}); err != nil {
		t.Fatal(err)
	}
	for silent(nextAudio(t, peer.sent)) {
	}
	if data := nextAudio(t, peer.sent); !silent(data) || time.Since(sentAt) < 1300*time.Millisecond {
		t.Fatalf("silence resumed after %s", time.Since(sentAt))
	}
}

func TestLiveIdleAudioStopsBeforeSessionClose(t *testing.T) {
	connection, peer := livePair(t, idleAudio())
	nextAudio(t, peer.sent)
	for _, err := range connection.EndSession(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
	}
	var last map[string]any
	for drained := false; !drained; {
		select {
		case frame := <-peer.sent:
			last = frame
		case <-time.After(300 * time.Millisecond):
			drained = true
		}
	}
	if last["type"] != "session.close" {
		t.Fatalf("last frame: %v", last)
	}
}

func TestLiveIdleAudioRejectsFailedSendsAndSettings(t *testing.T) {
	connection, peer := livePair(t, idleAudio())
	_ = peer.socket.CloseNow()
	deadline := time.Now().Add(5 * time.Second)
	for connection.Send(t.Context(), realtime.AudioInput{Data: make([]byte, 48000)}) == nil {
		if time.Now().After(deadline) {
			t.Fatal("send on dropped socket kept succeeding")
		}
		time.Sleep(10 * time.Millisecond)
	}
	for event, err := range connection.Events(t.Context()) {
		if failure, ok := event.(realtime.SessionError); err != nil || !ok || failure.Recoverable {
			t.Fatalf("drop not reported: %#v %v", event, err)
		}
	}
	if err := connection.Send(t.Context(), realtime.AudioInput{Data: make([]byte, 2)}); err == nil {
		t.Fatal("send after reported drop")
	}

	model := openairt.NewLiveModel("gpt-live-1", openairt.WithAPIKey("secret"), openairt.WithLiveSettings(openairt.LiveSettings{IdleAudio: true}))
	if _, err := model.ConnectWebRTC(t.Context(), realtime.WebRTCSession{Provider: "openai", ID: "id"}, realtime.ConnectParams{}); err == nil {
		t.Fatal("sideband idle audio")
	}
	if _, err := model.Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{Provider: map[string]any{"openai_live_idle_audio": "yes"}}}); err == nil {
		t.Fatal("non-boolean idle audio")
	}
}

func TestLiveIdleAudioContinuesAfterReconnect(t *testing.T) {
	var attempts atomic.Int32
	replacement := make(chan map[string]any, 64)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		socket, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = socket.CloseNow() }()
		socket.SetReadLimit(1 << 20)
		if _, _, err := socket.Read(r.Context()); err != nil {
			return
		}
		_ = writeFrame(r.Context(), socket, map[string]any{"type": "session.started", "session": map[string]any{"id": "s", "model": "gpt-live-1"}})
		dropping := attempts.Add(1) == 1
		for {
			_, data, err := socket.Read(r.Context())
			if err != nil {
				return
			}
			var frame map[string]any
			_ = json.Unmarshal(data, &frame)
			if !dropping {
				replacement <- frame
			} else if audio, _ := base64.StdEncoding.DecodeString(frame["audio"].(string)); !silent(audio) {
				return
			}
		}
	}))
	defer server.Close()
	model := openairt.NewLiveModel("gpt-live-1", openairt.WithAPIKey("key"), openairt.WithBaseURL(server.URL+"/v1"),
		openairt.WithLiveSettings(openairt.LiveSettings{IdleAudio: true}))
	connection, err := model.Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{Reconnect: &realtime.ReconnectPolicy{BaseDelay: time.Millisecond}}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = connection.Close(context.Background()) }()
	if err := connection.Send(t.Context(), realtime.AudioInput{Data: make([]byte, 10*48000)}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	go func() {
		for range connection.Events(ctx) {
		}
	}()
	if data := nextAudio(t, replacement); !silent(data) {
		t.Fatalf("replacement audio: %d bytes", len(data))
	}
}
