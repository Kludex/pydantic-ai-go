package openaiprotocol_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
	"github.com/coder/websocket"
)

func TestParallelToolResultsAskForOneAnswer(t *testing.T) {
	for _, early := range []bool{false, true} {
		connection, socket, frames, events := protocolWire(t)
		_ = writeFrame(t.Context(), socket, map[string]any{"type": "response.created", "response": map[string]any{"id": "tools"}})
		for _, id := range []string{"a", "b"} {
			_ = writeFrame(t.Context(), socket, map[string]any{"type": "response.function_call_arguments.done", "response_id": "tools", "call_id": id, "name": "lookup", "arguments": "{}"})
		}
		for range 3 {
			<-events
		}
		if !early {
			_ = writeFrame(t.Context(), socket, map[string]any{"type": "response.done", "response": map[string]any{"id": "tools", "status": "completed"}})
			<-events
		}
		for _, id := range []string{"a", "b"} {
			if err := connection.Send(t.Context(), realtime.ToolResult{ToolCallID: id, Output: "result"}); err != nil {
				t.Fatal(err)
			}
			if frame := protocolFrame(t, frames); frame["type"] != "conversation.item.create" {
				t.Fatalf("result: %+v", frame)
			}
		}
		if early {
			select {
			case frame := <-frames:
				t.Fatalf("early tool answer: %+v", frame)
			case <-time.After(10 * time.Millisecond):
			}
			_ = writeFrame(t.Context(), socket, map[string]any{"type": "response.done", "response": map[string]any{"id": "tools", "status": "completed"}})
			<-events
		}
		if frame := protocolFrame(t, frames); frame["type"] != "response.create" {
			t.Fatalf("answer: %+v", frame)
		}
		select {
		case frame := <-frames:
			t.Fatalf("second tool answer: %+v", frame)
		case <-time.After(10 * time.Millisecond):
		}
		if batched, ok := connection.(realtime.ToolBatchConnection); !ok || !batched.AnswersToolCallsPerResponse() {
			t.Fatal("missing batch accounting")
		}
	}
}

func TestDeferredRequestsReportMergedReservations(t *testing.T) {
	connection, socket, frames, events := protocolWire(t)
	for _, text := range []string{"one", "two", "three"} {
		if err := connection.Send(t.Context(), realtime.TextInput{Text: text}); err != nil {
			t.Fatal(err)
		}
	}
	for range 4 {
		_ = protocolFrame(t, frames)
	}
	_ = writeFrame(t.Context(), socket, map[string]any{"type": "response.created", "response": map[string]any{"id": "first"}})
	<-events
	_ = writeFrame(t.Context(), socket, map[string]any{"type": "response.done", "response": map[string]any{"id": "first", "status": "completed"}})
	<-events
	_ = protocolFrame(t, frames)
	_ = writeFrame(t.Context(), socket, map[string]any{"type": "response.created", "response": map[string]any{"id": "second"}})
	select {
	case event := <-events:
		if merged, ok := event.(realtime.ResponseRequestsMerged); !ok || merged.Count != 1 {
			t.Fatalf("merged: %+v", event)
		}
	case <-time.After(time.Second):
		t.Fatal("missing merged requests")
	}
}

func TestToolBatchSendFailureReachesEvents(t *testing.T) {
	frames := make(chan map[string]any, 8)
	server := handshakeServer(t, func(_ map[string]any, socket *websocket.Conn, ctx context.Context) {
		_ = writeFrame(ctx, socket, map[string]any{"type": "session.updated"})
		_ = writeFrame(ctx, socket, map[string]any{"type": "response.function_call_arguments.done", "response_id": "tools", "call_id": "c", "name": "lookup", "arguments": "{}"})
		_, data, _ := socket.Read(ctx)
		var frame map[string]any
		_ = json.Unmarshal(data, &frame)
		frames <- frame
		_ = writeFrame(ctx, socket, map[string]any{"type": "response.done", "response": map[string]any{"id": "tools", "status": "completed", "usage": map[string]any{"input_tokens": 1}}})
		_, _, _ = socket.Read(ctx)
	})
	defer server.Close()
	connection, err := openairt.NewModel("gpt-realtime", openairt.WithAPIKey("key"), openairt.WithBaseURL(wsBase(server.URL))).Connect(t.Context(), realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = connection.Close(t.Context()) }()
	failed := false
	for event, eventErr := range connection.Events(t.Context()) {
		if eventErr != nil {
			failed = true
			break
		}
		switch event.(type) {
		case realtime.ToolCall:
			if err := connection.Send(t.Context(), realtime.ToolResult{ToolCallID: "c", Output: "result"}); err != nil {
				t.Fatal(err)
			}
			_ = protocolFrame(t, frames)
		case realtime.SessionUsage:
			_ = connection.Close(t.Context())
		}
	}
	if !failed {
		t.Fatal("tool batch response write failure was lost")
	}
}

func TestSeedlessSpeechRemainsInReplay(t *testing.T) {
	items := openairt.SeedItems([]ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.SpeechPart{Speaker: ai.SpeechSpeakerUser}}}}, realtime.DefaultProfile())
	if len(items) != 0 {
		t.Fatal("content-less seed should be skipped")
	}
}

func protocolWire(t *testing.T) (realtime.Connection, *websocket.Conn, <-chan map[string]any, <-chan realtime.CodecEvent) {
	t.Helper()
	frames := make(chan map[string]any, 32)
	ready := make(chan *websocket.Conn, 1)
	server := handshakeServer(t, func(_ map[string]any, socket *websocket.Conn, ctx context.Context) {
		_ = writeFrame(ctx, socket, map[string]any{"type": "session.updated"})
		ready <- socket
		for {
			_, data, err := socket.Read(ctx)
			if err != nil {
				return
			}
			var frame map[string]any
			_ = json.Unmarshal(data, &frame)
			frames <- frame
		}
	})
	connection, err := openairt.NewModel("gpt-realtime", openairt.WithAPIKey("key"), openairt.WithBaseURL(wsBase(server.URL))).Connect(t.Context(), realtime.ConnectParams{})
	if err != nil {
		server.Close()
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	events := make(chan realtime.CodecEvent, 32)
	go func() {
		defer close(events)
		for event, err := range connection.Events(ctx) {
			if err != nil {
				return
			}
			events <- event
		}
	}()
	t.Cleanup(func() { cancel(); _ = connection.Close(context.Background()); server.Close() })
	return connection, <-ready, frames, events
}
func handshakeServer(t *testing.T, afterUpdate func(map[string]any, *websocket.Conn, context.Context)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		socket, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = socket.CloseNow() }()
		_ = writeFrame(r.Context(), socket, map[string]any{"type": "session.created", "session": map[string]any{"model": "served"}})
		_, data, err := socket.Read(r.Context())
		if err != nil {
			return
		}
		var frame map[string]any
		_ = json.Unmarshal(data, &frame)
		afterUpdate(frame, socket, r.Context())
	}))
}

func wsBase(serverURL string) string { return "ws" + strings.TrimPrefix(serverURL, "http") + "/v1" }

func writeFrame(ctx context.Context, socket *websocket.Conn, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return socket.Write(ctx, websocket.MessageText, data)
}

func protocolFrame(t *testing.T, frames <-chan map[string]any) map[string]any {
	t.Helper()
	select {
	case frame := <-frames:
		return frame
	case <-time.After(time.Second):
		t.Fatal("frame did not arrive")
		return nil
	}
}
