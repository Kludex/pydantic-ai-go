package azure_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	azurert "github.com/Kludex/pydantic-ai-go/ai/realtime/azure"
	openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
	"github.com/coder/websocket"
)

func TestVoiceLiveSettingsAndWarnings(t *testing.T) {
	for _, test := range []struct {
		name        string
		thinking    ai.ThinkingLevel
		voice       any
		vad         map[string]any
		temperature float64
		invalid     bool
	}{
		{"gpt-5", ai.ThinkingLevelEnabled, "en-US-AvaNeural", map[string]any{"type": "semantic_vad", "eagerness": "high"}, 1, false},
		{"gpt-5", ai.ThinkingLevelDisabled, map[string]any{"type": "azure-custom", "name": "custom"}, nil, 0, false},
		{"gpt-realtime-2", ai.ThinkingLevelHigh, "ava", nil, 2, false},
		{"gpt-realtime-2", "", nil, nil, 3, true},
	} {
		t.Run(test.name+string(test.thinking), func(t *testing.T) {
			updates := make(chan map[string]any, 1)
			warnings := make(chan string, 4)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				socket, err := websocket.Accept(w, r, nil)
				if err != nil {
					return
				}
				defer func() { _ = socket.CloseNow() }()
				_ = writeFrame(r.Context(), socket, map[string]any{"type": "session.created", "session": map[string]any{}})
				_, data, err := socket.Read(r.Context())
				if err != nil {
					return
				}
				var update map[string]any
				_ = json.Unmarshal(data, &update)
				updates <- update["session"].(map[string]any)
				_ = writeFrame(r.Context(), socket, map[string]any{"type": "session.updated"})
				_ = socket.Write(r.Context(), websocket.MessageText, []byte(`{`))
				_ = writeFrame(r.Context(), socket, map[string]any{"type": "warning", "warning": map[string]any{"message": "voice fallback", "code": "fallback", "param": "voice"}})
				_ = writeFrame(r.Context(), socket, map[string]any{"type": "response.done", "response": map[string]any{"id": "r", "status": "completed"}})
				_, _, _ = socket.Read(r.Context())
			}))
			defer server.Close()
			model, err := azurert.NewModel(test.name, azurert.Config{Endpoint: server.URL, APIKey: "key", VoiceLiveEndpoint: server.URL, VoiceLiveAPIKey: "key"},
				azurert.WithSettings(azurert.Settings{VoiceLive: true, OpenAI: openairt.Settings{TurnDetection: test.vad}, WarningHandler: func(message string) { warnings <- message }}))
			if err != nil {
				t.Fatal(err)
			}
			connection, err := model.Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{Thinking: test.thinking, Provider: map[string]any{
				"azure_voice_live_voice": test.voice, "azure_voice_live_temperature": test.temperature,
			}}})
			if test.invalid {
				if err == nil {
					t.Fatal("accepted invalid temperature")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = connection.Close(t.Context()) }()
			config := <-updates
			if test.thinking == ai.ThinkingLevelEnabled {
				if _, ok := config["temperature"]; ok {
					t.Fatal("reasoning temperature was sent")
				}
				if config["turn_detection"].(map[string]any)["type"] != "azure_semantic_vad" {
					t.Fatalf("VAD: %+v", config)
				}
			} else if config["temperature"] != test.temperature {
				t.Fatalf("temperature: %+v", config)
			}
			for event, eventErr := range connection.Events(t.Context()) {
				if eventErr != nil {
					t.Fatal(eventErr)
				}
				if _, done := event.(realtime.ResponseDone); done {
					break
				}
			}
			found := false
			for len(warnings) > 0 {
				found = strings.Contains(<-warnings, "voice fallback") || found
			}
			if !found {
				t.Fatal("provider warning was dropped")
			}
		})
	}
}

func TestAzureRouteAndToolChoiceValidation(t *testing.T) {
	model, _ := azurert.NewModel("gpt-4o-realtime", azurert.Config{Endpoint: "https://example.com", APIKey: "key"})
	live := realtime.Settings{Provider: map[string]any{"azure_voice_live": true}}
	if _, err := model.Connect(t.Context(), realtime.ConnectParams{Settings: live}); err == nil {
		t.Fatal("accepted GA-only Voice Live route")
	}
	if _, err := model.CreateClientSecret(t.Context(), "", nil, live, 0); err == nil {
		t.Fatal("accepted GA-only Voice Live secret")
	}
	if _, err := model.ConnectWebRTC(t.Context(), providerSession{provider: "azure", id: "call"}, realtime.ConnectParams{Settings: live}); err == nil {
		t.Fatal("accepted GA-only Voice Live sideband")
	}
	for _, choice := range []realtime.ToolChoice{realtime.ToolChoiceRequired, "invalid"} {
		settings := realtime.Settings{ToolChoice: choice}
		if _, err := model.ConnectWebRTC(t.Context(), providerSession{provider: "azure", id: "call"}, realtime.ConnectParams{Settings: settings}); err == nil {
			t.Fatal("accepted tool choice")
		}
		if _, err := model.CreateClientSecret(t.Context(), "", nil, settings, 0); err == nil {
			t.Fatal("accepted secret tool choice")
		}
	}
	server := handshakeServer(t, func(*http.Request) {})
	defer server.Close()
	model, _ = azurert.NewModel("gpt-5", azurert.Config{Endpoint: server.URL, APIKey: "key", VoiceLiveEndpoint: server.URL, VoiceLiveAPIKey: "key"},
		azurert.WithSettings(azurert.Settings{VoiceLiveTemperature: ptr(1.0)}))
	connection, err := model.Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{TurnDetection: &realtime.TurnDetection{Enabled: false}}})
	if err != nil {
		t.Fatal(err)
	}
	_ = connection.Close(t.Context())
}
