package openai_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
	"github.com/coder/websocket"
)

type writeBoundary struct {
	io.ReadWriteCloser
	limit   *atomic.Int32
	writes  *atomic.Int32
	discard *atomic.Bool
}

func (body writeBoundary) Write(data []byte) (int, error) {
	if body.discard.Load() {
		return len(data), nil
	}
	if maximum := body.limit.Load(); maximum > 0 && body.writes.Add(1) >= maximum {
		return 0, errors.New("injected write failure")
	}
	return body.ReadWriteCloser.Write(data)
}

func failingLivePair(t *testing.T) (*openairt.LiveConnection, *livePeer, *atomic.Int32, *atomic.Int32, *atomic.Bool) {
	t.Helper()
	limit, writes, discard := &atomic.Int32{}, &atomic.Int32{}, &atomic.Bool{}
	client := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		response, err := http.DefaultTransport.RoundTrip(request)
		if err == nil && response.StatusCode == 101 {
			response.Body = writeBoundary{ReadWriteCloser: response.Body.(io.ReadWriteCloser), limit: limit, writes: writes, discard: discard}
		}
		return response, err
	})}
	connection, peer := livePair(t, realtime.ConnectParams{}, openairt.WithHTTPClient(client))
	return connection, peer, limit, writes, discard
}

func TestLiveTransportSendFailures(t *testing.T) {
	for _, test := range []struct {
		input realtime.Input
		limit int32
	}{
		{realtime.ImageInput{Content: ai.BinaryContent{Data: []byte("png"), MediaType: "image/png"}, Respond: true}, 1},
		{realtime.ToolResult{ToolCallID: "c", Content: []ai.UserContent{ai.TextContent{Text: "extra"}}}, 1},
		{realtime.ToolResult{ToolCallID: "c", Content: []ai.UserContent{ai.TextContent{Text: "extra"}}}, 2},
	} {
		connection, _, limit, _, _ := failingLivePair(t)
		limit.Store(test.limit)
		if err := connection.Send(t.Context(), test.input); err == nil {
			t.Fatalf("write succeeded: %T", test.input)
		}
	}
	connection, peer, limit, _, _ := failingLivePair(t)
	peer.frame(t, delegation("d"))
	peer.frame(t, delegated("d", liveCall("c")))
	for event, err := range connection.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if call, ok := event.(realtime.ToolCall); ok {
			if err := connection.Send(t.Context(), realtime.ToolResult{ToolCallID: call.ToolCallID}); err != nil {
				t.Fatal(err)
			}
			_ = peer.input(t)
			break
		}
	}
	limit.Store(1)
	peer.frame(t, delegated("d", backendTerminal("response.completed", nil)))
	failed := false
	for _, err := range connection.Events(t.Context()) {
		if err != nil {
			failed = true
			break
		}
	}
	if !failed {
		t.Fatal("continuation write failure lost")
	}
}

func TestLiveEndSessionTimeoutAndTransportClose(t *testing.T) {
	failed, _, limit, _, _ := failingLivePair(t)
	limit.Store(1)
	for _, err := range failed.EndSession(t.Context()) {
		if err == nil {
			t.Fatal("write failure lost")
		}
	}
	connection, _, _, _, discard := failingLivePair(t)
	discard.Store(true)
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Millisecond)
	defer cancel()
	for range connection.EndSession(ctx) {
		t.Fatal("unexpected ending report")
	}
	connection, peer, _, _, discard := failingLivePair(t)
	discard.Store(true)
	go func() { time.Sleep(time.Millisecond); _ = peer.socket.CloseNow() }()
	for range connection.EndSession(t.Context()) {
		t.Fatal("unexpected report from dropped provider")
	}
}

func TestLiveClosedConnectionDoesNotDeliverBufferedFrames(t *testing.T) {
	connection, peer := livePair(t, realtime.ConnectParams{})
	peer.frame(t, map[string]any{"type": "future"})
	time.Sleep(time.Millisecond)
	_ = connection.Close(t.Context())
	for attempt := 0; attempt < 100; attempt++ {
		for range connection.Events(t.Context()) {
			t.Fatal("event after close")
		}
	}
}

func TestLiveGracefulSocketCloseSettlesSpeech(t *testing.T) {
	for _, stopEarly := range []bool{false, true} {
		connection, peer := livePair(t, realtime.ConnectParams{}, openairt.WithLiveSettings(openairt.LiveSettings{TurnSilence: time.Hour}))
		peer.frame(t, map[string]any{"type": "session.output_transcript.delta", "delta": "Finished"})
		go func() { _ = peer.socket.Close(websocket.StatusNormalClosure, "") }()
		done := false
		for event, err := range connection.Events(t.Context()) {
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := event.(realtime.ResponseDone); ok {
				done = true
				if stopEarly {
					break
				}
			}
		}
		if !done {
			t.Fatal("graceful close left speech open")
		}
	}
}
