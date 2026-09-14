package azure_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	azurert "github.com/Kludex/pydantic-ai-go/ai/realtime/azure"
	openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
	"github.com/coder/websocket"
)

type providerSession struct{ provider, id string }

func (session providerSession) ProviderName() string { return session.provider }
func (session providerSession) SessionID() string    { return session.id }

func TestDatazoneModelsUseVoiceLive(t *testing.T) {
	for _, name := range []string{"gpt-realtime-datazone", "gpt-realtime-1.5-datazone"} {
		model, err := azurert.NewModel(name, azurert.Config{
			Endpoint: "https://example.openai.azure.com", APIKey: "key",
		})
		if err != nil {
			t.Fatal(err)
		}
		if model.Profile().SupportsWebRTC {
			t.Fatalf("%s unexpectedly supports WebRTC", name)
		}
	}
}

func TestAzureGARealtimeSession(t *testing.T) {
	var mutex sync.Mutex
	var update map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/openai/v1/realtime" || request.URL.Query().Get("model") != "gpt-realtime-2" ||
			request.Header.Get("api-key") != "key" {
			t.Errorf("unexpected Azure handshake: %s %s", request.URL, request.Header)
		}
		socket, err := websocket.Accept(writer, request, nil)
		if err != nil {
			t.Error(err)
			return
		}
		defer func() { _ = socket.CloseNow() }()
		_ = writeFrame(request.Context(), socket, map[string]any{
			"type": "session.created", "session": map[string]any{"model": "served-azure"},
		})
		_, data, err := socket.Read(request.Context())
		if err != nil {
			return
		}
		mutex.Lock()
		_ = json.Unmarshal(data, &update)
		mutex.Unlock()
		_ = writeFrame(request.Context(), socket, map[string]any{"type": "session.updated", "session": map[string]any{}})
		_ = writeFrame(request.Context(), socket, map[string]any{
			"type": "response.text.delta", "delta": "hello", "item_id": "item",
		})
		_ = writeFrame(request.Context(), socket, map[string]any{
			"type": "response.text.done", "text": "hello", "item_id": "item",
		})
		_ = writeFrame(request.Context(), socket, map[string]any{
			"type": "response.done", "response": map[string]any{"id": "response", "status": "completed"},
		})
		<-request.Context().Done()
	}))
	defer server.Close()
	yes := true
	transcription := "auto"
	model, err := azurert.NewModel("gpt-realtime-2", azurert.Config{
		Endpoint: server.URL + "/ignored/path", APIKey: "key", HTTPClient: server.Client(),
		Headers: http.Header{"X-Test": []string{"value"}},
	}, azurert.WithSettings(azurert.Settings{OpenAI: structOpenAISettings()}),
		azurert.WithProfile(realtime.ProfileOverride{SupportsThinking: &yes}))
	if err != nil {
		t.Fatal(err)
	}
	session, err := realtime.Open(t.Context(), model, realtime.ConnectParams{
		Messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "seed"}}}},
		Settings: realtime.Settings{
			OutputModality: realtime.OutputModalityText, MaxTokens: 100,
			ParallelToolCalls: &yes, ToolChoice: realtime.ToolChoiceAuto,
			InputTranscriptionModel: &transcription, Thinking: ai.ThinkingLevelHigh,
			TurnDetection: &realtime.TurnDetection{Enabled: true, Sensitivity: "high"},
			Provider:      map[string]any{"openai_voice": "coral"},
		},
		Request: ai.ModelRequestParams{
			Instructions: "Be concise.", Tools: []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if model.Name() != "gpt-realtime-2" || model.ProviderName() != "azure" || !model.Profile().SupportsThinking {
		t.Fatalf("unexpected Azure model: %+v", model.Profile())
	}
	for event, err := range session.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(realtime.TurnCompleteEvent); ok {
			break
		}
	}
	mutex.Lock()
	if update["type"] != "session.update" {
		t.Fatalf("session config missing: %+v", update)
	}
	mutex.Unlock()
	_ = session.Close(t.Context())
}

func TestAzureVoiceLiveSession(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/voice-live/realtime" || request.URL.Query().Get("model") != "gpt-5" ||
			request.URL.Query().Get("api-version") != "preview" || request.Header.Get("api-key") != "voice-key" {
			t.Errorf("unexpected Voice Live handshake: %s %s", request.URL, request.Header)
		}
		socket, err := websocket.Accept(writer, request, nil)
		if err != nil {
			return
		}
		defer func() { _ = socket.CloseNow() }()
		_ = writeFrame(request.Context(), socket, map[string]any{
			"type": "session.created", "session": map[string]any{"model": "served-voice"},
		})
		_, _, _ = socket.Read(request.Context())
		_ = writeFrame(request.Context(), socket, map[string]any{"type": "session.updated", "session": map[string]any{}})
		_ = writeFrame(request.Context(), socket, map[string]any{"type": "response.text.delta", "delta": "voice"})
		_ = writeFrame(request.Context(), socket, map[string]any{"type": "response.text.done", "text": "voice"})
		_ = writeFrame(request.Context(), socket, map[string]any{
			"type": "response.done", "response": map[string]any{"status": "completed"},
		})
		<-request.Context().Done()
	}))
	defer server.Close()
	model, err := azurert.NewModel("gpt-5", azurert.Config{
		Endpoint: server.URL, APIKey: "key", VoiceLiveEndpoint: server.URL,
		VoiceLiveAPIKey: "voice-key", VoiceLiveAPIVersion: "preview", HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if model.Profile().SupportsWebRTC {
		t.Fatal("Voice Live-only model advertised WebRTC")
	}
	session, err := realtime.Open(t.Context(), model, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	for event, err := range session.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(realtime.TurnCompleteEvent); ok {
			break
		}
	}
	_ = session.Close(t.Context())
}

func TestAzureWebRTC(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/openai/v1/realtime/client_secrets":
			writer.Header().Set("Content-Type", "application/json")
			_, _ = writer.Write([]byte(`{"value":"ephemeral","expires_at":2000000000,"session":{"id":"session"}}`))
		case "/openai/v1/realtime/calls":
			if request.Header.Get("Authorization") != "Bearer ephemeral" {
				t.Errorf("unexpected signaling auth: %s", request.Header)
			}
			writer.Header().Set("Location", "/openai/v1/realtime/calls/rtc_azure")
			_, _ = writer.Write([]byte("answer"))
		default:
			http.NotFound(writer, request)
		}
	}))
	defer server.Close()
	model, err := azurert.NewModel("gpt-realtime", azurert.Config{
		Endpoint: server.URL, APIKey: "key", HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	secret, err := model.CreateClientSecret(t.Context(), "instructions", nil, realtime.Settings{}, time.Minute)
	if err != nil || secret.Value != "ephemeral" {
		t.Fatalf("unexpected client secret: %+v %v", secret, err)
	}
	answer, err := model.AnswerWebRTCOffer(t.Context(), "offer", "instructions", nil, realtime.Settings{})
	if err != nil || answer.Session.ID != "rtc_azure" || answer.SDP != "answer" {
		t.Fatalf("unexpected answer: %+v %v", answer, err)
	}
}

func TestAzureEntraAuthentication(t *testing.T) {
	server := handshakeServer(t, func(request *http.Request) {
		if request.Header.Get("Authorization") != "Bearer token" {
			t.Errorf("missing Entra token: %s", request.Header)
		}
	})
	defer server.Close()
	calls := 0
	model, err := azurert.NewModel("gpt-realtime", azurert.Config{
		Endpoint: server.URL, HTTPClient: server.Client(), TokenProvider: func(context.Context) (string, error) {
			calls++
			return "token", nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	connection, err := model.Connect(t.Context(), realtime.ConnectParams{})
	if err != nil || calls != 1 {
		t.Fatalf("unexpected Entra connection: %v calls=%d", err, calls)
	}
	_ = connection.Close(t.Context())
}

func structOpenAISettings() openairt.Settings {
	return openairt.Settings{
		Voice: "alloy", InputNoiseReduction: "near_field", OutputSpeed: 1,
		TurnDetection: map[string]any{"type": "server_vad"}, Truncation: "auto",
	}
}

func handshakeServer(t *testing.T, inspect func(*http.Request)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		inspect(request)
		socket, err := websocket.Accept(writer, request, nil)
		if err != nil {
			return
		}
		defer func() { _ = socket.CloseNow() }()
		_ = writeFrame(request.Context(), socket, map[string]any{
			"type": "session.created", "session": map[string]any{"model": "served"},
		})
		_, _, _ = socket.Read(request.Context())
		_ = writeFrame(request.Context(), socket, map[string]any{"type": "session.updated", "session": map[string]any{}})
		<-request.Context().Done()
	}))
}

func writeFrame(ctx context.Context, socket *websocket.Conn, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return socket.Write(ctx, websocket.MessageText, data)
}
