package openai_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
	"github.com/coder/websocket"
)

func TestOpenAIManualTurnAndSerializationErrors(t *testing.T) {
	for _, disabled := range []bool{true, false} {
		server := handshakeServer(t, func(_ map[string]any, socket *websocket.Conn, ctx context.Context) {
			_ = writeFrame(ctx, socket, map[string]any{"type": "session.updated", "session": map[string]any{}})
		})
		model := openairt.NewModel("model", openairt.WithAPIKey("key"), openairt.WithBaseURL(server.URL+"/v1"),
			openairt.WithHTTPClient(server.Client()))
		connection, err := model.Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{
			OutputModality: realtime.OutputModalityAudio,
			TurnDetection:  &realtime.TurnDetection{Enabled: disabled},
		}})
		if err != nil {
			t.Fatal(err)
		}
		_ = connection.Close(t.Context())
		server.Close()
	}

	invalid := realtime.Settings{
		OutputModality: realtime.OutputModalityAudio,
		Provider:       map[string]any{"openai_truncation": func() {}},
	}
	server := handshakeServer(t, func(map[string]any, *websocket.Conn, context.Context) {})
	defer server.Close()
	model := openairt.NewModel("model", openairt.WithAPIKey("key"), openairt.WithBaseURL(server.URL+"/v1"),
		openairt.WithHTTPClient(server.Client()))
	if _, err := model.Connect(t.Context(), realtime.ConnectParams{Settings: invalid}); err == nil {
		t.Fatal("expected session update serialization error")
	}
	if _, err := model.CreateClientSecret(t.Context(), "", nil, invalid, 0); err == nil {
		t.Fatal("expected client secret serialization error")
	}
	if _, err := model.AnswerWebRTCOffer(t.Context(), "offer", "", nil, invalid); err == nil {
		t.Fatal("expected WebRTC serialization error")
	}
}

func TestOpenAIConfigureSocketWriteFailures(t *testing.T) {
	behaviors := []func(*websocket.Conn, context.Context){
		func(socket *websocket.Conn, ctx context.Context) {
			_ = writeFrame(ctx, socket, map[string]any{"type": "session.created", "session": map[string]any{}})
			_ = socket.Close(websocket.StatusInternalError, "before update")
		},
		func(socket *websocket.Conn, ctx context.Context) {
			_ = writeFrame(ctx, socket, map[string]any{"type": "session.created", "session": map[string]any{}})
			_, _, _ = socket.Read(ctx)
			_ = socket.Close(websocket.StatusInternalError, "before updated")
		},
	}
	for index, behavior := range behaviors {
		t.Run(string(rune('a'+index)), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				socket, err := websocket.Accept(writer, request, nil)
				if err == nil {
					behavior(socket, request.Context())
				}
			}))
			defer server.Close()
			model := openairt.NewModel("model", openairt.WithAPIKey("key"), openairt.WithBaseURL(server.URL+"/v1"),
				openairt.WithHTTPClient(server.Client()))
			params := realtime.ConnectParams{Settings: realtime.Settings{OutputModality: realtime.OutputModalityAudio}}
			if _, err := model.Connect(t.Context(), params); err == nil {
				t.Fatal("expected configure failure")
			}
		})
	}
}

func TestOpenAISidebandErrors(t *testing.T) {
	session := providerSession{provider: "openai", id: "call"}
	for _, baseURL := range []string{"://bad", "ftp://example.com/v1"} {
		model := openairt.NewModel("model", openairt.WithAPIKey("key"), openairt.WithBaseURL(baseURL))
		if _, err := model.ConnectWebRTC(t.Context(), session, realtime.ConnectParams{}); err == nil {
			t.Fatalf("expected sideband URL error for %q", baseURL)
		}
	}
	for name, handler := range map[string]http.HandlerFunc{
		"status": func(writer http.ResponseWriter, _ *http.Request) { http.Error(writer, "bad", http.StatusUnauthorized) },
		"closed": func(writer http.ResponseWriter, request *http.Request) {
			socket, _ := websocket.Accept(writer, request, nil)
			_ = socket.Close(websocket.StatusInternalError, "closed")
		},
	} {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(handler)
			defer server.Close()
			model := openairt.NewModel("model", openairt.WithAPIKey("key"), openairt.WithBaseURL(server.URL+"/v1"),
				openairt.WithHTTPClient(server.Client()))
			if _, err := model.ConnectWebRTC(t.Context(), session, realtime.ConnectParams{}); err == nil {
				t.Fatal("expected sideband handshake error")
			}
		})
	}
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, context.DeadlineExceeded
	})}
	model := openairt.NewModel("model", openairt.WithAPIKey("key"), openairt.WithHTTPClient(client))
	if _, err := model.ConnectWebRTC(t.Context(), session, realtime.ConnectParams{}); err == nil {
		t.Fatal("expected sideband transport error")
	}
}

func TestOpenAIEndpointAndLocationErrors(t *testing.T) {
	for _, baseURL := range []string{"://bad", "ftp://example.com/v1"} {
		model := openairt.NewModel("model", openairt.WithAPIKey("key"), openairt.WithBaseURL(baseURL))
		if _, err := model.CreateClientSecret(t.Context(), "", nil, realtime.Settings{}, 0); err == nil {
			t.Fatalf("expected HTTP endpoint error for %q", baseURL)
		}
		if _, err := model.AnswerWebRTCOffer(t.Context(), "offer", "", nil, realtime.Settings{}); err == nil {
			t.Fatalf("expected WebRTC endpoint error for %q", baseURL)
		}
	}
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Location", "%")
		_, _ = writer.Write([]byte("answer"))
	}))
	defer server.Close()
	model := openairt.NewModel("model", openairt.WithAPIKey("key"), openairt.WithBaseURL(server.URL+"/v1"),
		openairt.WithHTTPClient(server.Client()))
	if _, err := model.AnswerWebRTCOffer(t.Context(), "offer", "", nil, realtime.Settings{}); err == nil {
		t.Fatal("expected malformed location error")
	}
}

func TestOpenAIBinaryHandshakeFrame(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		socket, err := websocket.Accept(writer, request, nil)
		if err != nil {
			return
		}
		defer func() { _ = socket.CloseNow() }()
		_ = socket.Write(request.Context(), websocket.MessageBinary, []byte("ignored"))
		_ = writeFrame(request.Context(), socket, map[string]any{"type": "session.created", "session": map[string]any{}})
		_, _, _ = socket.Read(request.Context())
		_ = writeFrame(request.Context(), socket, map[string]any{"type": "session.updated", "session": map[string]any{}})
		<-request.Context().Done()
	}))
	defer server.Close()
	model := openairt.NewModel("model", openairt.WithAPIKey("key"), openairt.WithBaseURL(server.URL+"/v1"),
		openairt.WithHTTPClient(server.Client()))
	connection, err := model.Connect(t.Context(), realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	_ = connection.Close(t.Context())
}

var _ = time.Second
