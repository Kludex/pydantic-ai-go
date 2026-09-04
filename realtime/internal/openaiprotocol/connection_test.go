package openaiprotocol_test

import (
	"context"
	"errors"
	"iter"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/realtime"
	"github.com/Kludex/pydantic-ai-go/realtime/internal/openaiprotocol"
	"github.com/coder/websocket"
)

type unsupportedInput struct{}

func (unsupportedInput) RealtimeInputKind() string { return "unsupported" }

func TestConnectionSendAndEvents(t *testing.T) {
	frames := make(chan struct {
		kind websocket.MessageType
		data string
	}, 16)
	var receivedMu sync.Mutex
	var received [][]byte
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		socket, err := websocket.Accept(writer, request, nil)
		if err != nil {
			t.Error(err)
			return
		}
		defer func() { _ = socket.CloseNow() }()
		readDone := make(chan struct{})
		go func() {
			defer close(readDone)
			for {
				_, data, err := socket.Read(request.Context())
				if err != nil {
					return
				}
				receivedMu.Lock()
				received = append(received, slices.Clone(data))
				receivedMu.Unlock()
			}
		}()
		for frame := range frames {
			if err := socket.Write(request.Context(), frame.kind, []byte(frame.data)); err != nil {
				break
			}
		}
		<-readDone
	}))
	defer server.Close()

	socket, _, err := openaiprotocol.DialSocket(t.Context(), websocketEndpoint(server.URL), http.Header{}, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	mapper := func(data []byte) ([]realtime.CodecEvent, error) {
		switch string(data) {
		case "created":
			return []realtime.CodecEvent{realtime.ResponseStarted{ResponseID: "response"}}, nil
		case "audio":
			return []realtime.CodecEvent{realtime.AudioDelta{Data: []byte{1, 0}, ItemID: "item"}}, nil
		case "done":
			return []realtime.CodecEvent{realtime.ResponseDone{}}, nil
		case "duplicate":
			return []realtime.CodecEvent{
				realtime.ToolCall{ToolCallID: "same", ToolName: "tool"},
				realtime.ToolCall{ToolCallID: "same", ToolName: "tool"},
			}, nil
		case "bad":
			return nil, errors.New("bad frame")
		default:
			return nil, nil
		}
	}
	connection, err := openaiprotocol.New(openaiprotocol.Config{
		Provider: "test", Model: "model", Socket: socket, ServerModel: "served", Mapper: mapper,
		InputTranscriptionEnabled: true, RestoresInFlightState: true, SupportsImages: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if connection.ModelName() != "served" || !connection.InputTranscriptionEnabled() ||
		!connection.ReconnectRestoresInFlightState() {
		t.Fatal("connection info mismatch")
	}
	connection.SetMessageHistory(func() []ai.ModelMessage { return []ai.ModelMessage{ai.ModelRequest{}} })

	for _, input := range []realtime.Input{
		realtime.AudioInput{Data: []byte{1, 0}},
		realtime.TextInput{Text: "hello"},
		realtime.ImageInput{Content: ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"}},
		realtime.ToolResult{ToolCallID: "call", Output: "result", Content: []ai.UserContent{
			ai.TextContent{Text: "detail"}, ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"},
			ai.CachePoint{}, nil,
		}},
		realtime.ToolResult{ToolCallID: "plain", Output: "result"},
		realtime.CommitAudio{}, realtime.ClearAudio{}, realtime.CreateResponse{}, realtime.CancelResponse{},
	} {
		if err := connection.Send(t.Context(), input); err != nil {
			t.Fatalf("send %T: %v", input, err)
		}
	}
	if err := connection.Send(t.Context(), realtime.AudioInput{Data: []byte{1}}); err == nil {
		t.Fatal("expected odd audio error")
	}
	if err := connection.Send(t.Context(), unsupportedInput{}); err == nil {
		t.Fatal("expected unsupported input error")
	}
	for _, content := range [][]ai.UserContent{
		{ai.AudioURL{URL: "https://example.com/audio.mp3"}},
		{ai.BinaryContent{Data: []byte("audio"), MediaType: "audio/wav"}},
	} {
		if err := connection.Send(t.Context(), realtime.ToolResult{Content: content}); err == nil {
			t.Fatalf("expected tool content error for %+v", content)
		}
	}

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	eventDone := make(chan []realtime.CodecEvent, 1)
	go func() {
		var events []realtime.CodecEvent
		for event, err := range connection.Events(ctx) {
			if err == nil {
				events = append(events, event)
			}
			if len(events) == 5 {
				break
			}
		}
		eventDone <- events
	}()
	frames <- struct {
		kind websocket.MessageType
		data string
	}{websocket.MessageBinary, "ignored"}
	for _, frame := range []string{"created", "audio", "bad", "duplicate", "done"} {
		frames <- struct {
			kind websocket.MessageType
			data string
		}{websocket.MessageText, frame}
	}
	events := <-eventDone
	if len(events) != 5 {
		t.Fatalf("unexpected mapped events: %+v", events)
	}
	if err := connection.Send(t.Context(), realtime.TruncateOutput{AudioEndMilliseconds: 100}); err != nil {
		t.Fatal(err)
	}
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	for _, input := range []realtime.Input{
		realtime.TextInput{Text: "cancelled"}, realtime.ToolResult{ToolCallID: "cancelled"}, realtime.CreateResponse{},
	} {
		if err := connection.Send(cancelled, input); err == nil {
			t.Fatalf("expected cancelled send error for %T", input)
		}
	}
	if err := connection.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := connection.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	for _, input := range []realtime.Input{
		realtime.CommitAudio{}, realtime.TextInput{Text: "closed"}, realtime.CreateResponse{},
		realtime.ToolResult{ToolCallID: "closed", Output: "closed"},
	} {
		if err := connection.Send(t.Context(), input); err == nil {
			t.Fatalf("expected closed write error for %T", input)
		}
	}
	failedWrite, err := openaiprotocol.New(openaiprotocol.Config{
		Provider: "test", Model: "model", Socket: socket, Mapper: mapper,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := failedWrite.Send(t.Context(), realtime.CreateResponse{}); err == nil {
		t.Fatal("expected response creation write error")
	}
	close(frames)
	receivedMu.Lock()
	if len(received) < 8 {
		t.Fatalf("too few client frames: %d", len(received))
	}
	receivedMu.Unlock()

	if _, err := openaiprotocol.New(openaiprotocol.Config{}); err == nil {
		t.Fatal("expected invalid connection error")
	}
	headers := http.Header{"X-Test": []string{"one"}}
	cloned := openaiprotocol.CloneHeader(headers)
	cloned["X-Test"][0] = "two"
	if headers.Get("X-Test") != "one" {
		t.Fatal("headers were not detached")
	}
}

func TestConnectionWithoutImagesAndInactiveControls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		socket, err := websocket.Accept(writer, request, nil)
		if err != nil {
			return
		}
		defer func() { _ = socket.CloseNow() }()
		<-request.Context().Done()
	}))
	defer server.Close()
	socket, _, err := openaiprotocol.DialSocket(t.Context(), websocketEndpoint(server.URL), nil, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	connection, err := openaiprotocol.New(openaiprotocol.Config{Socket: socket, Mapper: func([]byte) ([]realtime.CodecEvent, error) {
		return nil, nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	if err := connection.Send(t.Context(), realtime.ImageInput{}); err == nil {
		t.Fatal("expected image error")
	}
	if err := connection.Send(t.Context(), realtime.CancelResponse{}); err != nil {
		t.Fatal(err)
	}
	if err := connection.Send(t.Context(), realtime.TruncateOutput{AudioEndMilliseconds: 1}); err != nil {
		t.Fatal(err)
	}
	_ = connection.Close(t.Context())
}

func TestConnectionReconnect(t *testing.T) {
	var connections atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		socket, err := websocket.Accept(writer, request, nil)
		if err != nil {
			return
		}
		if connections.Add(1) == 1 {
			_ = socket.Close(websocket.StatusInternalError, "drop")
			return
		}
		_ = socket.Write(request.Context(), websocket.MessageText, []byte("done"))
		<-request.Context().Done()
	}))
	defer server.Close()
	dial := func(ctx context.Context, _ []ai.ModelMessage) (*websocket.Conn, string, error) {
		socket, _, err := openaiprotocol.DialSocket(ctx, websocketEndpoint(server.URL), nil, server.Client())
		return socket, "reconnected", err
	}
	socket, _, err := openaiprotocol.DialSocket(t.Context(), websocketEndpoint(server.URL), nil, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	connection, err := openaiprotocol.New(openaiprotocol.Config{
		Socket: socket, Dial: dial, Mapper: func(data []byte) ([]realtime.CodecEvent, error) {
			if string(data) == "done" {
				return []realtime.CodecEvent{realtime.ResponseDone{}}, nil
			}
			return nil, nil
		},
		Reconnect: &realtime.ReconnectPolicy{
			MaxAttempts: 2, MaxReconnects: 2, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond, Jitter: true,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var events []realtime.CodecEvent
	for event, err := range connection.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, event)
		break
	}
	if len(events) != 1 || connection.ModelName() != "reconnected" {
		t.Fatalf("reconnect events mismatch: %+v model=%q", events, connection.ModelName())
	}
	_ = connection.Close(t.Context())
}

func TestConnectionCancelledRead(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		socket, err := websocket.Accept(writer, request, nil)
		if err == nil {
			defer func() { _ = socket.CloseNow() }()
			<-request.Context().Done()
		}
	}))
	defer server.Close()
	socket, _, err := openaiprotocol.DialSocket(t.Context(), websocketEndpoint(server.URL), nil, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	connection, _ := openaiprotocol.New(openaiprotocol.Config{Socket: socket, Mapper: func([]byte) ([]realtime.CodecEvent, error) {
		return nil, nil
	}})
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	for range connection.Events(ctx) {
		t.Fatal("cancelled read emitted an event")
	}
	_ = connection.Close(t.Context())
}

func TestConnectionReconnectContinues(t *testing.T) {
	var count atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		socket, err := websocket.Accept(writer, request, nil)
		if err != nil {
			return
		}
		if count.Add(1) == 1 {
			_ = socket.Close(websocket.StatusInternalError, "drop")
			return
		}
		defer func() { _ = socket.CloseNow() }()
		_ = socket.Write(request.Context(), websocket.MessageText, []byte("done"))
		<-request.Context().Done()
	}))
	defer server.Close()
	dial := func(ctx context.Context, _ []ai.ModelMessage) (*websocket.Conn, string, error) {
		socket, _, err := openaiprotocol.DialSocket(ctx, websocketEndpoint(server.URL), nil, server.Client())
		return socket, "model", err
	}
	socket, _, err := openaiprotocol.DialSocket(t.Context(), websocketEndpoint(server.URL), nil, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	connection, _ := openaiprotocol.New(openaiprotocol.Config{
		Socket: socket, Dial: dial,
		Mapper: func([]byte) ([]realtime.CodecEvent, error) {
			return []realtime.CodecEvent{realtime.ResponseDone{}}, nil
		},
		Reconnect: &realtime.ReconnectPolicy{MaxAttempts: 1, MaxReconnects: 1},
	})
	countEvents := 0
	for event, err := range connection.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		countEvents++
		if _, ok := event.(realtime.ResponseDone); ok {
			break
		}
	}
	if countEvents != 2 {
		t.Fatalf("unexpected reconnect event count: %d", countEvents)
	}
	_ = connection.Close(t.Context())
}

func TestConnectionNoReconnect(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		socket, err := websocket.Accept(writer, request, nil)
		if err == nil {
			_ = socket.Close(websocket.StatusInternalError, "drop")
		}
	}))
	defer server.Close()
	socket, _, err := openaiprotocol.DialSocket(t.Context(), websocketEndpoint(server.URL), nil, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	connection, _ := openaiprotocol.New(openaiprotocol.Config{Socket: socket, Mapper: func([]byte) ([]realtime.CodecEvent, error) {
		return nil, nil
	}})
	for event, err := range connection.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if failure, ok := event.(realtime.SessionError); !ok || failure.Recoverable {
			t.Fatalf("unexpected close event: %+v", event)
		}
	}
	_ = connection.Close(t.Context())
}

func TestConnectionReconnectFailuresAndConsumerStops(t *testing.T) {
	newDropped := func(t *testing.T) (*websocket.Conn, func()) {
		t.Helper()
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			socket, err := websocket.Accept(writer, request, nil)
			if err == nil {
				_ = socket.Close(websocket.StatusInternalError, "drop")
			}
		}))
		socket, _, err := openaiprotocol.DialSocket(t.Context(), websocketEndpoint(server.URL), nil, server.Client())
		if err != nil {
			server.Close()
			t.Fatal(err)
		}
		return socket, server.Close
	}

	for name, config := range map[string]openaiprotocol.Config{
		"limit": {
			Reconnect: &realtime.ReconnectPolicy{MaxAttempts: 1, MaxReconnects: 0},
			Dial: func(context.Context, []ai.ModelMessage) (*websocket.Conn, string, error) {
				return nil, "", errors.New("unused")
			},
		},
		"failed": {
			Reconnect: &realtime.ReconnectPolicy{
				MaxAttempts: 2, MaxReconnects: 1, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond,
			},
			Dial: func(context.Context, []ai.ModelMessage) (*websocket.Conn, string, error) {
				return nil, "", errors.New("dial failed")
			},
		},
		"nil socket": {
			Reconnect: &realtime.ReconnectPolicy{MaxAttempts: 1, MaxReconnects: 1},
			Dial: func(context.Context, []ai.ModelMessage) (*websocket.Conn, string, error) {
				return nil, "", nil
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			socket, cleanup := newDropped(t)
			defer cleanup()
			config.Socket = socket
			config.Mapper = func([]byte) ([]realtime.CodecEvent, error) { return nil, nil }
			connection, err := openaiprotocol.New(config)
			if err != nil {
				t.Fatal(err)
			}
			connection.SetMessageHistory(func() []ai.ModelMessage { return []ai.ModelMessage{ai.ModelRequest{}} })
			found := false
			for _, err := range connection.Events(t.Context()) {
				found = err != nil
			}
			if !found {
				t.Fatal("expected reconnect failure")
			}
			_ = connection.Close(t.Context())
		})
	}

	socket, cleanup := newDropped(t)
	defer cleanup()
	connection, _ := openaiprotocol.New(openaiprotocol.Config{
		Socket: socket, Mapper: func([]byte) ([]realtime.CodecEvent, error) { return nil, nil },
		Reconnect: &realtime.ReconnectPolicy{
			MaxAttempts: 2, MaxReconnects: 1, BaseDelay: time.Second, MaxDelay: time.Millisecond,
		},
		Dial: func(context.Context, []ai.ModelMessage) (*websocket.Conn, string, error) {
			return nil, "", errors.New("dial")
		},
	})
	ctx, cancel := context.WithTimeout(t.Context(), time.Millisecond)
	defer cancel()
	for range connection.Events(ctx) {
	}
	_ = connection.Close(t.Context())

	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		socket, err := websocket.Accept(writer, request, nil)
		if err != nil {
			return
		}
		defer func() { _ = socket.CloseNow() }()
		_ = socket.Write(request.Context(), websocket.MessageText, []byte("bad"))
		_ = socket.Write(request.Context(), websocket.MessageText, []byte("event"))
		<-request.Context().Done()
	}))
	defer server.Close()
	for _, stopOn := range []string{"bad", "event"} {
		socket, _, err := openaiprotocol.DialSocket(t.Context(), websocketEndpoint(server.URL), nil, server.Client())
		if err != nil {
			t.Fatal(err)
		}
		connection, _ := openaiprotocol.New(openaiprotocol.Config{
			Socket: socket, Mapper: func(data []byte) ([]realtime.CodecEvent, error) {
				if string(data) == "bad" {
					if stopOn == "bad" {
						return nil, errors.New("bad")
					}
					return nil, nil
				}
				return []realtime.CodecEvent{realtime.ResponseDone{}}, nil
			},
		})
		for range connection.Events(t.Context()) {
			break
		}
		_ = connection.Close(t.Context())
	}
}

func websocketEndpoint(serverURL string) string {
	return "ws" + strings.TrimPrefix(serverURL, "http")
}

var _ iter.Seq2[realtime.CodecEvent, error]
