package openaiprotocol_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
	"github.com/coder/websocket"
)

func TestIdleTimeoutDoesNotBecomeUserSpeech(t *testing.T) {
	connection, socket, _, events := protocolWire(t)
	for _, frame := range []map[string]any{
		{"type": "input_audio_buffer.timeout_triggered", "item_id": "idle"},
		{"type": "conversation.item.input_audio_transcription.delta", "item_id": "idle", "delta": ""},
		{"type": "conversation.item.input_audio_transcription.completed", "item_id": "idle", "transcript": ""},
		{"type": "input_audio_buffer.timeout_triggered", "item_id": "failed"},
		{"type": "conversation.item.input_audio_transcription.failed", "item_id": "failed", "error": map[string]any{"message": "silence"}},
		{"type": "response.done", "response": map[string]any{"id": "followup", "status": "completed"}},
	} {
		_ = writeFrame(t.Context(), socket, frame)
	}
	select {
	case event := <-events:
		if _, ok := event.(realtime.ResponseDone); !ok {
			t.Fatalf("idle transcription escaped: %+v", event)
		}
	case <-time.After(time.Second):
		t.Fatal("missing follow-up")
	}
	_ = connection.Close(t.Context())
}

func TestReplayKeepsTranscriptlessUserAndDropsMedia(t *testing.T) {
	var dials atomic.Int32
	drop, redial, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
	frames := make(chan map[string]any, 16)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call := dials.Add(1)
		if call == 2 {
			close(redial)
			<-release
		}
		socket, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = socket.CloseNow() }()
		_ = writeFrame(r.Context(), socket, map[string]any{"type": "session.created", "session": map[string]any{}})
		_, _, _ = socket.Read(r.Context())
		_ = writeFrame(r.Context(), socket, map[string]any{"type": "session.updated"})
		if call == 1 {
			<-drop
			return
		}
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
	defer server.Close()
	connection, err := openairt.NewModel("gpt-realtime", openairt.WithAPIKey("key"), openairt.WithBaseURL(wsBase(server.URL))).Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{
		Reconnect: &realtime.ReconnectPolicy{MaxAttempts: 1, MaxReconnects: 1, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond},
	}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = connection.Close(t.Context()) }()
	text := "transcript"
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.SpeechPart{Speaker: ai.SpeechSpeakerUser}, ai.SpeechPart{Speaker: ai.SpeechSpeakerUser, Transcript: &text},
			ai.UserPromptPart{Contents: []ai.UserContent{ai.BinaryContent{MediaType: "image/png"}, ai.TextContent{Text: "caption"}}}, ai.ToolReturnPart{ToolCallID: "c", Content: "result"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.SpeechPart{Speaker: ai.SpeechSpeakerAssistant, Transcript: &text, Audio: &ai.BinaryContent{MediaType: "audio/wav"}}, ai.TextPart{Content: "answer"}}},
	}
	connection.(realtime.HistoryAwareConnection).SetMessageHistory(func() []ai.ModelMessage { return (ai.ModelRequestContext{Messages: history}).Clone().Messages })
	recovery := connection.(realtime.ReconnectingConnection)
	if !recovery.CanReconnect() {
		t.Fatal("missing reconnect budget")
	}
	done := make(chan struct{})
	go func() {
		for event := range connection.Events(t.Context()) {
			if _, ok := event.(realtime.SessionReconnected); ok {
				break
			}
		}
		close(done)
	}()
	close(drop)
	<-redial
	if !recovery.IsReconnecting() {
		t.Fatal("missing reconnect state")
	}
	if err := connection.Send(t.Context(), realtime.AudioInput{Data: []byte{0, 0}}); err != nil {
		t.Fatal(err)
	}
	close(release)
	<-done
	if recovery.CanReconnect() {
		t.Fatal("reconnect budget not exhausted")
	}
	marker := protocolFrame(t, frames)["item"].(map[string]any)["content"].([]any)[0].(map[string]any)["text"]
	if marker != "[The user spoke; no transcript is available.]" {
		t.Fatalf("marker: %v", marker)
	}
	for range 5 {
		_ = protocolFrame(t, frames)
	}
	if history[1].(ai.ModelResponse).Parts[0].(ai.SpeechPart).Audio == nil {
		t.Fatal("replay mutated caller history")
	}
}

func TestDropInterruptedBoundaryAllowsConsumerToStop(t *testing.T) {
	server := handshakeServer(t, func(_ map[string]any, socket *websocket.Conn, ctx context.Context) {
		_ = writeFrame(ctx, socket, map[string]any{"type": "session.updated"})
		_, _, _ = socket.Read(ctx)
	})
	defer server.Close()
	connection, err := openairt.NewModel("gpt-realtime", openairt.WithAPIKey("key"), openairt.WithBaseURL(wsBase(server.URL))).Connect(t.Context(), realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	_ = connection.Send(t.Context(), realtime.CreateResponse{})
	for event := range connection.Events(t.Context()) {
		if done, ok := event.(realtime.ResponseDone); !ok || !done.Interrupted {
			t.Fatalf("drop: %+v", event)
		}
		break
	}
	_ = connection.Close(t.Context())
}

func TestMergedBoundaryAllowsConsumerToStop(t *testing.T) {
	server := handshakeServer(t, func(_ map[string]any, socket *websocket.Conn, ctx context.Context) {
		_ = writeFrame(ctx, socket, map[string]any{"type": "session.updated"})
		for range 4 {
			_, _, _ = socket.Read(ctx)
		}
		_ = writeFrame(ctx, socket, map[string]any{"type": "response.created", "response": map[string]any{"id": "one"}})
		_ = writeFrame(ctx, socket, map[string]any{"type": "response.done", "response": map[string]any{"id": "one", "status": "completed"}})
		_, _, _ = socket.Read(ctx)
		_ = writeFrame(ctx, socket, map[string]any{"type": "response.created", "response": map[string]any{"id": "two"}})
		_, _, _ = socket.Read(ctx)
	})
	defer server.Close()
	connection, err := openairt.NewModel("gpt-realtime", openairt.WithAPIKey("key"), openairt.WithBaseURL(wsBase(server.URL))).Connect(t.Context(), realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{"one", "two", "three"} {
		_ = connection.Send(t.Context(), realtime.TextInput{Text: text})
	}
	for event := range connection.Events(t.Context()) {
		if _, ok := event.(realtime.ResponseRequestsMerged); ok {
			break
		}
	}
	_ = connection.Close(t.Context())
}
