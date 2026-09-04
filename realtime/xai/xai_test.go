package xai_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/realtime"
	xairt "github.com/Kludex/pydantic-ai-go/realtime/xai"
	"github.com/coder/websocket"
)

func TestXAIRealtimeSession(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Query().Get("model") != "grok-voice-latest" || request.Header.Get("Authorization") != "Bearer secret" {
			t.Errorf("unexpected handshake: %s %s", request.URL, request.Header)
		}
		socket, err := websocket.Accept(writer, request, nil)
		if err != nil {
			t.Error(err)
			return
		}
		defer func() { _ = socket.CloseNow() }()
		_ = writeFrame(request.Context(), socket, map[string]any{
			"type": "session.created", "event_id": "event", "session": map[string]any{"model": "served-grok"},
		})
		if _, _, err := socket.Read(request.Context()); err != nil {
			t.Error(err)
			return
		}
		_ = writeFrame(request.Context(), socket, map[string]any{
			"type": "conversation.created", "conversation": map[string]any{"id": "conversation"},
		})
		_ = writeFrame(request.Context(), socket, map[string]any{"type": "session.updated", "session": map[string]any{}})
		_ = writeFrame(request.Context(), socket, map[string]any{
			"type": "conversation.item.input_audio_transcription.updated", "transcript": "hello", "item_id": "user",
		})
		_ = writeFrame(request.Context(), socket, map[string]any{
			"type": "conversation.item.input_audio_transcription.completed", "transcript": "hello", "item_id": "user",
		})
		_ = writeFrame(request.Context(), socket, map[string]any{
			"type": "response.done", "response": map[string]any{
				"id": "response", "status": "completed", "output": []any{},
				"usage": map[string]any{
					"input_tokens": 2, "output_tokens": 1, "billable_audio_seconds": 3,
					"input_token_details":  map[string]any{"grok_tokens": 4},
					"output_token_details": map[string]any{"grok_tokens": 5},
				},
			},
		})
		<-request.Context().Done()
	}))
	defer server.Close()

	yes := true
	transcription := "auto"
	model := xairt.NewModel("grok-voice-latest",
		xairt.WithAPIKey("secret"), xairt.WithBaseURL(server.URL+"/v1"),
		xairt.WithHTTPClient(server.Client()), xairt.WithHeaders(http.Header{"X-Test": []string{"value"}}),
		xairt.WithSettings(xairt.Settings{Voice: "eve", TurnDetection: map[string]any{"type": "server_vad"}}),
		xairt.WithProfile(realtime.ProfileOverride{SupportsThinking: &yes}),
	)
	session, err := realtime.Open(t.Context(), model, realtime.ConnectParams{
		Messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "seed"}}}},
		Settings: realtime.Settings{
			MaxTokens: 100, ParallelToolCalls: &yes, ToolChoice: realtime.ToolChoiceAuto,
			InputTranscriptionModel: &transcription, Thinking: ai.ThinkingLevelHigh,
			Reconnect: &realtime.ReconnectPolicy{MaxAttempts: 1, MaxReconnects: 1, MaxDelay: time.Second},
			Provider: map[string]any{
				"xai_voice": "ara", "xai_turn_detection": map[string]any{"type": "semantic_vad"},
			},
		},
		Request: ai.ModelRequestParams{Tools: []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if session.Profile().SupportsTextOutput || !session.Profile().SupportsThinking ||
		!session.ReconnectRestoresInFlightState() {
		t.Fatalf("unexpected xAI profile")
	}
	for event, err := range session.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(realtime.TurnCompleteEvent); ok {
			break
		}
	}
	if session.Usage().Details["billable_audio_seconds"] != 3 || session.Usage().Details["input_grok_tokens"] != 4 {
		t.Fatalf("xAI usage details missing: %+v", session.Usage())
	}
	_ = session.Close(t.Context())
}

func TestXAIMapEvent(t *testing.T) {
	for _, frame := range []string{
		`{"type":"conversation.item.input_audio_transcription.updated","transcript":"one","item_id":"i"}`,
		`{"type":"conversation.item.input_audio_transcription.completed","transcript":"one","item_id":"i"}`,
		`{"type":"response.done","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1,"input_token_details":{"grok_tokens":2},"output_token_details":{"grok_tokens":3},"billable_audio_seconds":4}}}`,
	} {
		events, err := xairt.MapEvent([]byte(frame))
		if err != nil || len(events) == 0 {
			t.Fatalf("map %s: events=%+v err=%v", frame, events, err)
		}
	}
	if _, err := xairt.MapEvent([]byte(`{`)); err == nil {
		t.Fatal("expected malformed event error")
	}
}

func TestXAIValidation(t *testing.T) {
	if _, err := xairt.NewModel("", xairt.WithAPIKey("key")).Connect(t.Context(), realtime.ConnectParams{}); err == nil {
		t.Fatal("expected model name error")
	}
	if _, err := xairt.NewModel("model", xairt.WithAPIKey("")).Connect(t.Context(), realtime.ConnectParams{}); err == nil {
		t.Fatal("expected API key error")
	}
	if _, err := xairt.NewModel("model", xairt.WithAPIKey("key")).Connect(t.Context(), realtime.ConnectParams{
		Settings: realtime.Settings{OutputModality: realtime.OutputModalityText},
	}); err == nil {
		t.Fatal("expected text output error")
	}
	optional := ai.WebSearchTool{Optional: true}
	if optional.Kind() == "" {
		t.Fatal("unexpected native tool")
	}
}

func writeFrame(ctx context.Context, socket *websocket.Conn, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return socket.Write(ctx, websocket.MessageText, data)
}
