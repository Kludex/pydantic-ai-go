package xai_test

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/realtime"
	xairt "github.com/Kludex/pydantic-ai-go/realtime/xai"
	"github.com/coder/websocket"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

func TestXAISettingsVariants(t *testing.T) {
	for _, settings := range []realtime.Settings{
		{
			OutputModality:          realtime.OutputModalityAudio,
			InputTranscriptionModel: ptr(""), Thinking: ai.ThinkingLevelDisabled,
			TurnDetection: &realtime.TurnDetection{Enabled: false},
		},
		{
			OutputModality:          realtime.OutputModalityAudio,
			InputTranscriptionModel: ptr("auto"), Thinking: ai.ThinkingLevelHigh,
			TurnDetection: &realtime.TurnDetection{
				Enabled: true, Sensitivity: "low", PrefixPadding: time.Millisecond, SilenceDuration: 2 * time.Millisecond,
			},
		},
	} {
		server := xaiHandshakeServer(t, func(socket *websocket.Conn, ctx context.Context) {
			_ = writeFrame(ctx, socket, map[string]any{"type": "session.updated", "session": map[string]any{}})
		})
		model := xairt.NewModel("grok-voice-latest",
			xairt.WithAPIKey("key"), xairt.WithBaseURL(server.URL+"/v1"), xairt.WithHTTPClient(server.Client()),
		)
		connection, err := model.Connect(t.Context(), realtime.ConnectParams{Settings: settings})
		if err != nil {
			t.Fatal(err)
		}
		_ = connection.Close(t.Context())
		server.Close()
	}
}

func TestXAIHandshakeAndEndpointErrors(t *testing.T) {
	for _, baseURL := range []string{"://bad", "ftp://example.com/v1"} {
		if _, err := xairt.NewModel("model", xairt.WithAPIKey("key"), xairt.WithBaseURL(baseURL)).Connect(
			t.Context(), realtime.ConnectParams{},
		); err == nil {
			t.Fatalf("expected URL error for %q", baseURL)
		}
	}
	for name, handler := range map[string]http.HandlerFunc{
		"status": func(writer http.ResponseWriter, _ *http.Request) { http.Error(writer, "bad", http.StatusUnauthorized) },
		"closed": func(writer http.ResponseWriter, request *http.Request) {
			socket, _ := websocket.Accept(writer, request, nil)
			_ = socket.Close(websocket.StatusInternalError, "closed")
		},
		"malformed": func(writer http.ResponseWriter, request *http.Request) {
			socket, _ := websocket.Accept(writer, request, nil)
			defer func() { _ = socket.CloseNow() }()
			_ = socket.Write(request.Context(), websocket.MessageText, []byte(`{`))
		},
		"binary": func(writer http.ResponseWriter, request *http.Request) {
			socket, _ := websocket.Accept(writer, request, nil)
			defer func() { _ = socket.CloseNow() }()
			_ = socket.Write(request.Context(), websocket.MessageBinary, []byte("ignored"))
			_ = writeFrame(request.Context(), socket, map[string]any{
				"type": "error", "error": map[string]any{"message": "bad"},
			})
		},
	} {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(handler)
			defer server.Close()
			model := xairt.NewModel("model", xairt.WithAPIKey("key"), xairt.WithBaseURL(server.URL+"/v1"),
				xairt.WithHTTPClient(server.Client()))
			if _, err := model.Connect(t.Context(), realtime.ConnectParams{}); err == nil {
				t.Fatal("expected handshake error")
			}
		})
	}
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("network")
	})}
	if _, err := xairt.NewModel("model", xairt.WithAPIKey("key"), xairt.WithHTTPClient(client)).Connect(
		t.Context(), realtime.ConnectParams{},
	); err == nil {
		t.Fatal("expected transport error")
	}
}

func TestXAIConfigureErrorsAndOverrides(t *testing.T) {
	for _, behavior := range []func(*websocket.Conn, context.Context){
		func(socket *websocket.Conn, ctx context.Context) {
			_ = writeFrame(ctx, socket, map[string]any{
				"type": "session.created", "event_id": "e", "session": map[string]any{},
			})
			_ = socket.Close(websocket.StatusInternalError, "before update")
		},
		func(socket *websocket.Conn, ctx context.Context) {
			_ = writeFrame(ctx, socket, map[string]any{
				"type": "session.created", "event_id": "e", "session": map[string]any{},
			})
			_, _, _ = socket.Read(ctx)
			_ = socket.Close(websocket.StatusInternalError, "before updated")
		},
		func(socket *websocket.Conn, ctx context.Context) {
			_ = writeFrame(ctx, socket, map[string]any{
				"type": "session.created", "event_id": "e", "session": map[string]any{},
			})
			_, _, _ = socket.Read(ctx)
			_ = writeFrame(ctx, socket, map[string]any{"type": "error", "error": map[string]any{"message": "bad"}})
		},
	} {
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			socket, err := websocket.Accept(writer, request, nil)
			if err == nil {
				behavior(socket, request.Context())
			}
		}))
		model := xairt.NewModel("model", xairt.WithAPIKey("key"), xairt.WithBaseURL(server.URL+"/v1"),
			xairt.WithHTTPClient(server.Client()))
		if _, err := model.Connect(t.Context(), realtime.ConnectParams{}); err == nil {
			t.Fatal("expected configure error")
		}
		server.Close()
	}

	yes := true
	server := xaiHandshakeServer(t, func(socket *websocket.Conn, ctx context.Context) {
		_ = writeFrame(ctx, socket, map[string]any{"type": "session.updated", "session": map[string]any{}})
	})
	defer server.Close()
	model := xairt.NewModel("grok-voice-latest",
		xairt.WithAPIKey("key"), xairt.WithBaseURL(server.URL+"/v1"), xairt.WithHTTPClient(server.Client()),
		xairt.WithSettings(xairt.Settings{TurnDetection: map[string]any{"type": "server_vad"}}),
		xairt.WithProfile(realtime.ProfileOverride{SupportsThinking: &yes}),
	)
	connection, err := model.Connect(t.Context(), realtime.ConnectParams{
		Settings: realtime.Settings{
			OutputModality: realtime.OutputModalityAudio,
			Provider: map[string]any{
				"xai_voice": "eve", "xai_turn_detection": map[string]any{"type": "semantic_vad"},
			},
		},
		Request: ai.ModelRequestParams{Tools: []ai.ToolDefinition{{Name: "tool", Schema: map[string]any{"type": "object"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	_ = connection.Close(t.Context())
}

func TestXAISessionUpdateSerializationError(t *testing.T) {
	server := xaiHandshakeServer(t, func(*websocket.Conn, context.Context) {})
	defer server.Close()
	model := xairt.NewModel("model", xairt.WithAPIKey("key"), xairt.WithBaseURL(server.URL+"/v1"),
		xairt.WithHTTPClient(server.Client()))
	if _, err := model.Connect(t.Context(), realtime.ConnectParams{Request: ai.ModelRequestParams{
		Tools: []ai.ToolDefinition{{Name: "bad", Schema: map[string]any{"bad": func() {}}}},
	}}); err == nil {
		t.Fatal("expected session update serialization error")
	}
}

func TestXAIReconnectUsesConversationID(t *testing.T) {
	var calls atomic.Int32
	seenResume := make(chan bool, 1)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		call := calls.Add(1)
		if call == 2 {
			seenResume <- request.URL.Query().Get("conversation_id") == "conversation"
		}
		socket, err := websocket.Accept(writer, request, nil)
		if err != nil {
			return
		}
		_ = writeFrame(request.Context(), socket, map[string]any{
			"type": "session.created", "event_id": "event", "session": map[string]any{"model": "served"},
		})
		_, _, _ = socket.Read(request.Context())
		_ = writeFrame(request.Context(), socket, map[string]any{
			"type": "conversation.created", "conversation": map[string]any{"id": "conversation"},
		})
		_ = writeFrame(request.Context(), socket, map[string]any{"type": "session.updated", "session": map[string]any{}})
		if call == 1 {
			_ = socket.Close(websocket.StatusInternalError, "drop")
			return
		}
		defer func() { _ = socket.CloseNow() }()
		_ = writeFrame(request.Context(), socket, map[string]any{
			"type": "response.done", "response": map[string]any{"status": "completed"},
		})
		<-request.Context().Done()
	}))
	defer server.Close()
	model := xairt.NewModel("model", xairt.WithAPIKey("key"), xairt.WithBaseURL(server.URL+"/v1"),
		xairt.WithHTTPClient(server.Client()))
	connection, err := model.Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{
		Reconnect: &realtime.ReconnectPolicy{MaxAttempts: 1, MaxReconnects: 1, MaxDelay: time.Second},
	}})
	if err != nil {
		t.Fatal(err)
	}
	for event, err := range connection.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(realtime.ResponseDone); ok {
			break
		}
	}
	if !<-seenResume {
		t.Fatal("reconnect omitted conversation ID")
	}
	_ = connection.Close(t.Context())
}

func TestXAIMapOpenAIError(t *testing.T) {
	if _, err := xairt.MapEvent([]byte(`{"type":"response.audio.delta","delta":"!"}`)); err == nil {
		t.Fatal("expected delegated mapping error")
	}
}

func xaiHandshakeServer(
	t *testing.T, afterUpdate func(socket *websocket.Conn, ctx context.Context),
) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		socket, err := websocket.Accept(writer, request, nil)
		if err != nil {
			return
		}
		defer func() { _ = socket.CloseNow() }()
		_ = writeFrame(request.Context(), socket, map[string]any{
			"type": "session.created", "event_id": "event", "session": map[string]any{"model": "served"},
		})
		_, _, err = socket.Read(request.Context())
		if err != nil {
			return
		}
		afterUpdate(socket, request.Context())
		<-request.Context().Done()
	}))
}

func ptr[T any](value T) *T { return &value }
