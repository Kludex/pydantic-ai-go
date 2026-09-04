package google_test

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/realtime"
	googlert "github.com/Kludex/pydantic-ai-go/realtime/google"
	"google.golang.org/genai"
)

type unsupportedInput struct{}

func (unsupportedInput) RealtimeInputKind() string { return "unsupported" }

func TestGoogleOptionsAndOfficialConnector(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")
	client, err := genai.NewClient(t.Context(), &genai.ClientConfig{
		APIKey: "key", Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{BaseURL: "http://127.0.0.1:1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	yes := true
	model := googlert.NewModel("model", googlert.WithClient(client), googlert.WithAPIKey("key"),
		googlert.WithHTTPClient(http.DefaultClient), googlert.WithBaseURL("http://127.0.0.1:1"),
		googlert.WithProfile(realtime.ProfileOverride{SupportsThinking: &yes}))
	if _, err := model.Connect(t.Context(), realtime.ConnectParams{}); err == nil {
		t.Fatal("expected official SDK connection error")
	}

	vertex := googlert.NewModel("model", googlert.WithVertex("project", "location"), googlert.WithBaseURL("http://127.0.0.1:1"))
	if vertex.ProviderName() != "google-vertex" {
		t.Fatalf("unexpected Vertex provider: %s", vertex.ProviderName())
	}
	if _, err := vertex.Connect(t.Context(), realtime.ConnectParams{}); err == nil {
		t.Fatal("expected Vertex connection error")
	}

	if _, err := googlert.NewModel("model", googlert.WithClient(nil), googlert.WithBaseURL("://bad")).Connect(
		t.Context(), realtime.ConnectParams{},
	); err == nil {
		t.Fatal("expected client construction error")
	}
}

func TestGoogleSettingsBranches(t *testing.T) {
	for _, test := range []struct {
		settings googlert.Settings
		common   realtime.Settings
	}{
		{
			settings: googlert.Settings{ConfigOverrides: func(config *genai.LiveConnectConfig) {
				config.MaxOutputTokens = 77
			}},
			common: realtime.Settings{
				InputTranscriptionModel: ptr(""), Thinking: ai.ThinkingLevelDisabled,
				TurnDetection: &realtime.TurnDetection{
					Enabled: true, Sensitivity: "low", SilenceDuration: time.Millisecond,
				},
				Reconnect: &realtime.ReconnectPolicy{MaxAttempts: 1, MaxReconnects: 1, MaxDelay: time.Second},
				Provider: map[string]any{
					"google_voice": "Kore", "google_language_code": "fr-FR", "google_async_tool_calls": true,
				},
			},
		},
	} {
		live := newFakeSession()
		connector := &fakeConnector{session: live}
		model := googlert.NewModel("gemini-2.5-flash-native-audio-latest",
			googlert.WithConnector(connector), googlert.WithSettings(test.settings))
		connection, err := model.Connect(t.Context(), realtime.ConnectParams{Settings: test.common})
		if err != nil {
			t.Fatal(err)
		}
		if connector.config.MaxOutputTokens != 77 || connector.config.InputAudioTranscription != nil ||
			connector.config.SessionResumption == nil || connector.config.ThinkingConfig == nil {
			t.Fatalf("settings were not mapped: %+v", connector.config)
		}
		_ = connection.Close(t.Context())
	}
}

func TestGoogleSeedErrors(t *testing.T) {
	transcript := "speech"
	tests := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{
			ai.BinaryContent{Data: []byte("pdf"), MediaType: "application/pdf"},
		}}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.SpeechPart{Speaker: ai.SpeechSpeakerUser}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.SpeechPart{Speaker: ai.SpeechSpeakerAssistant}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.FilePart{Content: ai.BinaryContent{Data: []byte("x"), MediaType: "image/png"}}}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{Content: func() {}}, ai.RetryPromptPart{Content: "retry"},
			ai.SpeechPart{Speaker: ai.SpeechSpeakerUser, Transcript: &transcript},
		}},
	}
	for index, message := range tests {
		t.Run(string(rune('a'+index)), func(t *testing.T) {
			live := newFakeSession()
			connector := &fakeConnector{session: live}
			model := googlert.NewModel("model", googlert.WithConnector(connector))
			_, err := model.Connect(t.Context(), realtime.ConnectParams{Messages: []ai.ModelMessage{message}})
			if index < 4 && err == nil {
				t.Fatal("expected seed error")
			}
			if index == 4 && err != nil {
				t.Fatal(err)
			}
		})
	}

	live := newFakeSession()
	live.err = errors.New("seed failed")
	model := googlert.NewModel("model", googlert.WithConnector(&fakeConnector{session: live}))
	if _, err := model.Connect(t.Context(), realtime.ConnectParams{Messages: []ai.ModelMessage{ai.ModelRequest{
		Parts: []ai.RequestPart{ai.UserPromptPart{Content: "seed"}},
	}}}); err == nil {
		t.Fatal("expected seed transport error")
	}
}

func TestGoogleConnectionEventsAndSendErrors(t *testing.T) {
	live := newFakeSession()
	connector := &fakeConnector{session: live}
	model := googlert.NewModel("model", googlert.WithConnector(connector))
	connection, err := model.Connect(t.Context(), realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	if connection.(interface{ InputTranscriptionEnabled() bool }).InputTranscriptionEnabled() != true ||
		connection.(interface{ ReconnectRestoresInFlightState() bool }).ReconnectRestoresInFlightState() != true {
		t.Fatal("connection info mismatch")
	}
	if err := connection.Send(t.Context(), unsupportedInput{}); err == nil {
		t.Fatal("expected unsupported send error")
	}
	live.receive <- nil
	live.receive <- &genai.LiveServerMessage{GoAway: &genai.LiveServerGoAway{}}
	for event, err := range connection.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(realtime.SessionError); ok {
			break
		}
	}
	_ = connection.Close(t.Context())

	live = newFakeSession()
	connection, err = googlert.NewModel("model", googlert.WithConnector(&fakeConnector{session: live})).Connect(
		t.Context(), realtime.ConnectParams{},
	)
	if err != nil {
		t.Fatal(err)
	}
	live.once.Do(func() {
		close(live.closed)
		close(live.receive)
	})
	found := false
	for _, err := range connection.Events(t.Context()) {
		found = err != nil
	}
	if !found {
		t.Fatal("expected receive error")
	}
	_ = connection.Close(t.Context())
}

func TestGoogleSendFailures(t *testing.T) {
	for _, input := range []realtime.Input{
		realtime.AudioInput{Data: []byte{1, 0}}, realtime.ImageInput{}, realtime.TextInput{Text: "text"},
		realtime.ToolResult{ToolCallID: "call", Output: "result"},
	} {
		live := newFakeSession()
		live.err = errors.New("send")
		connection, err := googlert.NewModel("model", googlert.WithConnector(&fakeConnector{session: live})).Connect(
			t.Context(), realtime.ConnectParams{},
		)
		if err != nil {
			t.Fatal(err)
		}
		if err := connection.Send(t.Context(), input); err == nil {
			t.Fatalf("expected send error for %T", input)
		}
	}
}

func TestGoogleEventConsumerStops(t *testing.T) {
	live := newFakeSession()
	connection, err := googlert.NewModel("model", googlert.WithConnector(&fakeConnector{session: live})).Connect(
		t.Context(), realtime.ConnectParams{},
	)
	if err != nil {
		t.Fatal(err)
	}
	live.receive <- &genai.LiveServerMessage{UsageMetadata: &genai.UsageMetadata{PromptTokenCount: 1}}
	for range connection.Events(t.Context()) {
		break
	}
	_ = connection.Close(t.Context())
}

func TestGoogleClientCreationSuccessThenConnectFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		http.Error(writer, "not websocket", http.StatusBadRequest)
	}))
	defer server.Close()
	model := googlert.NewModel("model", googlert.WithAPIKey("key"), googlert.WithBaseURL(server.URL),
		googlert.WithHTTPClient(server.Client()))
	if _, err := model.Connect(t.Context(), realtime.ConnectParams{}); err == nil {
		t.Fatal("expected live connection failure")
	}
}

func ptr[T any](value T) *T { return &value }
