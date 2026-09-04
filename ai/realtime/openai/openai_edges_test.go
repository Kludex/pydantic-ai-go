package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
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

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

type providerSession struct{ provider, id string }

func (session providerSession) ProviderName() string { return session.provider }
func (session providerSession) SessionID() string    { return session.id }

type errorReader struct{}

func (errorReader) Read([]byte) (int, error) { return 0, errors.New("read failed") }
func (errorReader) Close() error             { return nil }

func TestOpenAIConnectValidationAndPortableSettings(t *testing.T) {
	if _, err := openairt.NewModel("", openairt.WithAPIKey("key")).Connect(t.Context(), realtime.ConnectParams{}); err == nil {
		t.Fatal("expected model name error")
	}
	if _, err := openairt.NewModel("model", openairt.WithAPIKey("")).Connect(t.Context(), realtime.ConnectParams{}); err == nil {
		t.Fatal("expected API key error")
	}
	for _, baseURL := range []string{"://bad", "ftp://example.com/v1"} {
		if _, err := openairt.NewModel("model", openairt.WithAPIKey("key"), openairt.WithBaseURL(baseURL)).Connect(
			t.Context(), realtime.ConnectParams{},
		); err == nil {
			t.Fatalf("expected base URL error for %q", baseURL)
		}
	}

	updated := make(chan map[string]any, 1)
	server := handshakeServer(t, func(frame map[string]any, socket *websocket.Conn, ctx context.Context) {
		updated <- frame
		_ = writeFrame(ctx, socket, map[string]any{"type": "session.updated", "session": map[string]any{}})
	})
	defer server.Close()
	transcriptionOff := ""
	no := false
	model := openairt.NewModel("gpt-realtime-2.1",
		openairt.WithAPIKey("key"), openairt.WithBaseURL(wsBase(server.URL)), openairt.WithHTTPClient(server.Client()),
	)
	connection, err := model.Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{
		OutputModality: realtime.OutputModalityAudio, InputTranscriptionModel: &transcriptionOff,
		TurnDetection: &realtime.TurnDetection{
			Enabled: true, Sensitivity: "high", PrefixPadding: time.Millisecond, SilenceDuration: 2 * time.Millisecond,
		},
		ParallelToolCalls: &no, ToolChoice: realtime.ToolChoiceRequired,
	}})
	if err != nil {
		t.Fatal(err)
	}
	frame := <-updated
	if frame["type"] != "session.update" {
		t.Fatalf("unexpected update: %+v", frame)
	}
	_ = connection.Close(t.Context())

	server2 := handshakeServer(t, func(_ map[string]any, socket *websocket.Conn, ctx context.Context) {
		_ = writeFrame(ctx, socket, map[string]any{"type": "session.updated", "session": map[string]any{}})
	})
	defer server2.Close()
	model = openairt.NewModel("gpt-realtime", openairt.WithAPIKey("key"), openairt.WithBaseURL(wsBase(server2.URL)),
		openairt.WithHTTPClient(server2.Client()), openairt.WithSettings(openairt.Settings{OutputSpeed: 2}))
	if _, err := model.Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{OutputModality: realtime.OutputModalityAudio}}); err == nil {
		t.Fatal("expected speed validation error")
	}
}

func TestOpenAIHandshakeErrors(t *testing.T) {
	for name, handler := range map[string]http.HandlerFunc{
		"status": func(writer http.ResponseWriter, _ *http.Request) {
			http.Error(writer, "denied", http.StatusUnauthorized)
		},
		"malformed": func(writer http.ResponseWriter, request *http.Request) {
			socket, _ := websocket.Accept(writer, request, nil)
			defer func() { _ = socket.CloseNow() }()
			_ = socket.Write(request.Context(), websocket.MessageText, []byte(`{`))
		},
		"provider": func(writer http.ResponseWriter, request *http.Request) {
			socket, _ := websocket.Accept(writer, request, nil)
			defer func() { _ = socket.CloseNow() }()
			_ = writeFrame(request.Context(), socket, map[string]any{
				"type": "error", "error": map[string]any{"message": "bad session"},
			})
		},
		"closed": func(writer http.ResponseWriter, request *http.Request) {
			socket, _ := websocket.Accept(writer, request, nil)
			_ = socket.Close(websocket.StatusInternalError, "closed")
		},
	} {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(handler)
			defer server.Close()
			model := openairt.NewModel("model", openairt.WithAPIKey("key"), openairt.WithBaseURL(wsBase(server.URL)),
				openairt.WithHTTPClient(server.Client()))
			if _, err := model.Connect(t.Context(), realtime.ConnectParams{}); err == nil {
				t.Fatal("expected handshake error")
			}
		})
	}

	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("network")
	})}
	if _, err := openairt.NewModel("model", openairt.WithAPIKey("key"), openairt.WithHTTPClient(client)).Connect(
		t.Context(), realtime.ConnectParams{},
	); err == nil {
		t.Fatal("expected network handshake error")
	}
}

func TestOpenAIClientSecretErrors(t *testing.T) {
	if _, err := openairt.NewModel("model", openairt.WithAPIKey("")).CreateClientSecret(
		t.Context(), "", nil, realtime.Settings{}, 0,
	); err == nil {
		t.Fatal("expected missing key error")
	}
	badSetting := realtime.Settings{Provider: map[string]any{"openai_output_speed": 2.0}}
	if _, err := openairt.NewModel("model", openairt.WithAPIKey("key")).CreateClientSecret(
		t.Context(), "", nil, badSetting, 0,
	); err == nil {
		t.Fatal("expected settings error")
	}
	for name, handler := range map[string]http.HandlerFunc{
		"status":     func(writer http.ResponseWriter, _ *http.Request) { http.Error(writer, "bad", http.StatusBadRequest) },
		"json":       func(writer http.ResponseWriter, _ *http.Request) { _, _ = writer.Write([]byte(`{`)) },
		"incomplete": func(writer http.ResponseWriter, _ *http.Request) { _, _ = writer.Write([]byte(`{}`)) },
	} {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(handler)
			defer server.Close()
			model := openairt.NewModel("model", openairt.WithAPIKey("key"), openairt.WithBaseURL(server.URL+"/v1"),
				openairt.WithHTTPClient(server.Client()))
			if _, err := model.CreateClientSecret(t.Context(), "", nil, realtime.Settings{}, 0); err == nil {
				t.Fatal("expected client secret error")
			}
		})
	}
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("network")
	})}
	model := openairt.NewModel("model", openairt.WithAPIKey("key"), openairt.WithHTTPClient(client))
	if _, err := model.CreateClientSecret(t.Context(), "", nil, realtime.Settings{}, 0); err == nil {
		t.Fatal("expected transport error")
	}
}

func TestOpenAIWebRTCErrorsAndSideband(t *testing.T) {
	model := openairt.NewModel("model", openairt.WithAPIKey("key"))
	if _, err := model.AnswerWebRTCOffer(t.Context(), "", "", nil, realtime.Settings{}); err == nil {
		t.Fatal("expected empty offer error")
	}
	if _, err := model.AnswerWebRTCOffer(t.Context(), "offer", "", nil, realtime.Settings{
		Provider: map[string]any{"openai_output_speed": 2.0},
	}); err == nil {
		t.Fatal("expected WebRTC settings error")
	}
	for name, handler := range map[string]http.HandlerFunc{
		"status":   func(writer http.ResponseWriter, _ *http.Request) { http.Error(writer, "bad", http.StatusBadRequest) },
		"location": func(writer http.ResponseWriter, _ *http.Request) { _, _ = writer.Write([]byte("answer")) },
		"query location": func(writer http.ResponseWriter, _ *http.Request) {
			writer.Header().Set("Location", "/v1/realtime/calls?call_id=query-call")
			_, _ = writer.Write([]byte("answer"))
		},
	} {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(handler)
			defer server.Close()
			model := openairt.NewModel("model", openairt.WithAPIKey("key"), openairt.WithBaseURL(server.URL+"/v1"),
				openairt.WithHTTPClient(server.Client()))
			answer, err := model.AnswerWebRTCOffer(t.Context(), "offer", "", nil, realtime.Settings{})
			if name == "query location" {
				if err != nil || answer.Session.ID != "query-call" {
					t.Fatalf("unexpected query call ID: %+v %v", answer, err)
				}
			} else if err == nil {
				t.Fatal("expected negotiation error")
			}
		})
	}
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("network")
	})}
	if _, err := openairt.NewModel("model", openairt.WithAPIKey("key"), openairt.WithHTTPClient(client)).AnswerWebRTCOffer(
		t.Context(), "offer", "", nil, realtime.Settings{},
	); err == nil {
		t.Fatal("expected negotiation transport error")
	}
	readClient := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK, Status: "200 OK", Header: http.Header{"Location": []string{"/v1/realtime/calls/id"}},
			Body: errorReader{},
		}, nil
	})}
	if _, err := openairt.NewModel("model", openairt.WithAPIKey("key"), openairt.WithHTTPClient(readClient)).AnswerWebRTCOffer(
		t.Context(), "offer", "", nil, realtime.Settings{},
	); err == nil {
		t.Fatal("expected answer read error")
	}

	if _, err := model.ConnectWebRTC(t.Context(), providerSession{}, realtime.ConnectParams{}); err == nil {
		t.Fatal("expected invalid sideband session error")
	}
	sideband := handshakeServer(t, func(_ map[string]any, socket *websocket.Conn, ctx context.Context) {
		_ = writeFrame(ctx, socket, map[string]any{"type": "session.updated", "session": map[string]any{}})
	})
	defer sideband.Close()
	sidebandModel := openairt.NewModel("model", openairt.WithAPIKey("key"), openairt.WithBaseURL(wsBase(sideband.URL)),
		openairt.WithHTTPClient(sideband.Client()))
	connection, err := sidebandModel.ConnectWebRTC(t.Context(), providerSession{provider: "openai", id: "call"},
		realtime.ConnectParams{Settings: realtime.Settings{OutputModality: realtime.OutputModalityAudio}})
	if err != nil {
		t.Fatal(err)
	}
	_ = connection.Close(t.Context())
}

func TestOpenAISeedItems(t *testing.T) {
	profile := openairt.NewModel("model").Profile()
	transcript := "speech"
	items := openairt.SeedItems([]ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.UserPromptPart{}, ai.UserPromptPart{Content: "text"},
			ai.UserPromptPart{Contents: []ai.UserContent{
				ai.TextContent{Text: "rich"}, ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"},
			}},
			ai.SpeechPart{Speaker: ai.SpeechSpeakerUser, Transcript: &transcript},
			ai.SpeechPart{Speaker: ai.SpeechSpeakerUser, Audio: &ai.BinaryContent{Data: []byte{1}, MediaType: "audio/pcm"}},
			ai.ToolReturnPart{ToolCallID: "return", Content: func() {}},
			ai.RetryPromptPart{ToolCallID: "retry", Content: "retry"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.TextPart{Content: "answer"}, ai.SpeechPart{Speaker: ai.SpeechSpeakerAssistant, Transcript: &transcript},
			ai.ToolCallPart{ToolName: "tool", ToolCallID: "call"},
		}},
	}, profile)
	if len(items) < 8 {
		t.Fatalf("seed items missing: %+v", items)
	}
}

func handshakeServer(
	t *testing.T, afterUpdate func(frame map[string]any, socket *websocket.Conn, ctx context.Context),
) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		socket, err := websocket.Accept(writer, request, nil)
		if err != nil {
			t.Error(err)
			return
		}
		defer func() { _ = socket.CloseNow() }()
		_ = writeFrame(request.Context(), socket, map[string]any{
			"type": "session.created", "session": map[string]any{"model": "served"},
		})
		_, data, err := socket.Read(request.Context())
		if err != nil {
			return
		}
		var frame map[string]any
		_ = json.Unmarshal(data, &frame)
		afterUpdate(frame, socket, request.Context())
		<-request.Context().Done()
	}))
}

func wsBase(serverURL string) string { return "ws" + strings.TrimPrefix(serverURL, "http") + "/v1" }

var _ io.ReadCloser = errorReader{}
