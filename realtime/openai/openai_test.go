package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/realtime"
	openairt "github.com/Kludex/pydantic-ai-go/realtime/openai"
	"github.com/coder/websocket"
)

func TestOpenAIRealtimeSession(t *testing.T) {
	var mutex sync.Mutex
	var received []map[string]any
	upgraded := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Authorization") != "Bearer secret" || request.URL.Query().Get("model") != "gpt-realtime-2.1" {
			t.Errorf("unexpected handshake: %s %s", request.Header, request.URL)
		}
		socket, err := websocket.Accept(writer, request, nil)
		if err != nil {
			t.Error(err)
			return
		}
		defer func() { _ = socket.CloseNow() }()
		_ = writeFrame(request.Context(), socket, map[string]any{
			"type": "session.created", "session": map[string]any{"model": "gpt-realtime-2026"},
		})
		_, update, err := socket.Read(request.Context())
		if err != nil {
			t.Error(err)
			return
		}
		var updateFrame map[string]any
		_ = json.Unmarshal(update, &updateFrame)
		mutex.Lock()
		received = append(received, updateFrame)
		mutex.Unlock()
		_ = writeFrame(request.Context(), socket, map[string]any{"type": "session.updated", "session": map[string]any{}})
		close(upgraded)

		go func() {
			for {
				_, data, readErr := socket.Read(request.Context())
				if readErr != nil {
					return
				}
				var frame map[string]any
				_ = json.Unmarshal(data, &frame)
				mutex.Lock()
				received = append(received, frame)
				mutex.Unlock()
			}
		}()
		for _, frame := range []map[string]any{
			{"type": "input_audio_buffer.speech_started", "item_id": "user"},
			{"type": "conversation.item.input_audio_transcription.delta", "delta": "hello", "item_id": "user"},
			{"type": "conversation.item.input_audio_transcription.completed", "transcript": "hello", "item_id": "user"},
			{"type": "response.created", "response": map[string]any{"id": "response"}},
			{"type": "response.output_audio.delta", "delta": "AQACAA==", "item_id": "assistant"},
			{"type": "response.output_audio_transcript.delta", "delta": "hi", "item_id": "assistant"},
			{"type": "response.output_audio_transcript.done", "transcript": "hi", "item_id": "assistant"},
			{"type": "response.done", "response": map[string]any{
				"id": "response", "status": "completed", "output": []any{},
				"usage": map[string]any{"input_tokens": 3, "output_tokens": 2},
			}},
		} {
			if err := writeFrame(request.Context(), socket, frame); err != nil {
				return
			}
		}
		<-request.Context().Done()
	}))
	defer server.Close()

	baseURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/v1"
	yes := true
	transcription := "auto"
	transcript := "spoken"
	model := openairt.NewModel("gpt-realtime-2.1",
		openairt.WithAPIKey("secret"), openairt.WithBaseURL(baseURL), openairt.WithHTTPClient(server.Client()),
		openairt.WithHeaders(http.Header{"X-Test": []string{"value"}}),
		openairt.WithSettings(openairt.Settings{
			Voice: "alloy", InputNoiseReduction: "near_field", OutputSpeed: 1.0,
			TurnDetection: map[string]any{"type": "server_vad"}, Truncation: "auto",
		}),
		openairt.WithProfile(realtime.ProfileOverride{SupportsAsyncToolCalls: &yes}),
	)
	session, err := realtime.Open(t.Context(), model, realtime.ConnectParams{
		Messages: []ai.ModelMessage{
			ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Content: "history"},
				ai.UserPromptPart{Contents: []ai.UserContent{
					ai.TextContent{Text: "rich"}, ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"},
				}},
				ai.SpeechPart{Speaker: ai.SpeechSpeakerUser, Transcript: &transcript},
				ai.SpeechPart{Speaker: ai.SpeechSpeakerUser, Audio: &ai.BinaryContent{Data: []byte{1, 0}, MediaType: "audio/pcm"}},
				ai.ToolReturnPart{ToolName: "lookup", ToolCallID: "old", Content: map[string]any{"ok": true}},
				ai.RetryPromptPart{ToolName: "lookup", ToolCallID: "retry", Content: "again"},
			}},
			ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.TextPart{Content: "answer"},
				ai.SpeechPart{Speaker: ai.SpeechSpeakerAssistant, Transcript: &transcript},
				ai.ToolCallPart{ToolName: "lookup", ToolCallID: "old", Args: json.RawMessage(`{}`)},
			}},
		},
		Settings: realtime.Settings{
			MaxTokens: 100, ParallelToolCalls: &yes, ToolChoice: realtime.ToolChoiceAuto,
			InputTranscriptionModel: &transcription, Thinking: ai.ThinkingLevelHigh,
			Provider: map[string]any{
				"openai_voice": "coral", "openai_input_noise_reduction": "far_field",
				"openai_output_speed": 1.25, "openai_turn_detection": map[string]any{"type": "semantic_vad"},
				"openai_truncation": map[string]any{"type": "retention_ratio"},
			},
		},
		Request: ai.ModelRequestParams{
			Instructions: "Be concise.", Tools: []ai.ToolDefinition{{Name: "lookup", Schema: map[string]any{"type": "object"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	<-upgraded
	if session.Profile().SupportsAsyncToolCalls != true || session.InputTranscriptionEnabled() != true ||
		session.ReconnectRestoresInFlightState() {
		t.Fatalf("unexpected OpenAI profile or connection info")
	}
	if err := session.Send(t.Context(), "question"); err != nil {
		t.Fatal(err)
	}
	if err := session.SendAudio(t.Context(), []byte{1, 0}, "audio/pcm"); err != nil {
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
	messages := session.Messages()
	found := false
	for _, message := range messages {
		if response, ok := message.(ai.ModelResponse); ok {
			found = response.ModelName == "gpt-realtime-2026" && response.Usage.InputTokens == 3
		}
	}
	if !found {
		t.Fatalf("provider response was not retained: %+v", messages)
	}
	deadline := time.Now().Add(time.Second)
	for {
		mutex.Lock()
		frameCount := len(received)
		mutex.Unlock()
		if frameCount >= 3 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for client frames: got %d", frameCount)
		}
		time.Sleep(time.Millisecond)
	}
	if err := session.Close(t.Context()); err != nil && !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	mutex.Lock()
	defer mutex.Unlock()
	if received[0]["type"] != "session.update" {
		t.Fatalf("unexpected client frames: %+v", received)
	}
}

func TestOpenAIMapEvent(t *testing.T) {
	frames := []string{
		`{"type":"response.created","response":{"id":"r"}}`,
		`{"type":"response.audio.delta","delta":"AQI=","item_id":"a"}`,
		`{"type":"response.audio_transcript.delta","delta":"x"}`,
		`{"type":"response.audio_transcript.done","transcript":"x"}`,
		`{"type":"response.text.delta","delta":"x"}`,
		`{"type":"response.text.done","text":"x"}`,
		`{"type":"conversation.item.input_audio_transcription.failed","item_id":"i","error":{"message":"bad"}}`,
		`{"type":"input_audio_buffer.speech_stopped","item_id":"i"}`,
		`{"type":"output_audio_buffer.started"}`,
		`{"type":"output_audio_buffer.cleared"}`,
		`{"type":"response.function_call_arguments.done","call_id":"c","name":"f","arguments":"{}"}`,
		`{"type":"conversation.created","conversation":{"id":"conversation"}}`,
		`{"type":"conversation.item.created","item":{"id":"item","call_id":"call"}}`,
		`{"type":"error","error":{"message":"failure"}}`,
		`{"type":"error","error":{}}`,
		`{"type":"rate_limits.updated"}`,
		`{"type":"unknown"}`,
		`{"type":"response.done","response":{"id":"r","status":"cancelled","status_details":{"reason":"max_output_tokens"},"output":[{"type":"function_call","call_id":"c","name":"f","arguments":"{}"}],"usage":{"input_tokens":10,"output_tokens":5,"input_token_details":{"audio_tokens":2,"cached_tokens":1,"cached_tokens_details":{"audio_tokens":1},"text_tokens":8,"image_tokens":1},"output_token_details":{"audio_tokens":3,"reasoning_tokens":2,"text_tokens":2}}}}`,
		`{"type":"response.done","response":{"status":"incomplete","status_details":{"reason":"max_tokens"}}}`,
		`{"type":"response.done","response":{"status":"incomplete","status_details":{"reason":"content_filter"}}}`,
		`{"type":"response.done","response":{"status":"incomplete","status_details":{"reason":"tool_calls"}}}`,
	}
	count := 0
	for _, frame := range frames {
		events, err := openairt.MapEvent([]byte(frame))
		if err != nil {
			t.Fatalf("map %s: %v", frame, err)
		}
		count += len(events)
	}
	if count < 18 {
		t.Fatalf("too few mapped events: %d", count)
	}
	for _, frame := range []string{`{`, `{"type":"response.audio.delta","delta":"!"}`} {
		if _, err := openairt.MapEvent([]byte(frame)); err == nil {
			t.Fatalf("expected malformed frame error for %s", frame)
		}
	}
}

func TestOpenAIWebRTC(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/v1/realtime/client_secrets":
			writer.Header().Set("Content-Type", "application/json")
			_, _ = writer.Write([]byte(`{"value":"ephemeral","expires_at":2000000000,"session":{"id":"s"}}`))
		case "/v1/realtime/calls":
			writer.Header().Set("Location", "/v1/realtime/calls/rtc_123")
			_, _ = writer.Write([]byte("answer"))
		default:
			http.NotFound(writer, request)
		}
	}))
	defer server.Close()
	model := openairt.NewModel("gpt-realtime",
		openairt.WithAPIKey("secret"), openairt.WithBaseURL(server.URL+"/v1"), openairt.WithHTTPClient(server.Client()),
	)
	secret, err := model.CreateClientSecret(t.Context(), "instructions", nil, realtime.Settings{}, time.Minute)
	if err != nil || secret.Value != "ephemeral" || secret.ExpiresAt.IsZero() {
		t.Fatalf("unexpected secret: %+v %v", secret, err)
	}
	answer, err := model.AnswerWebRTCOffer(t.Context(), "offer", "instructions", nil, realtime.Settings{})
	if err != nil || answer.SDP != "answer" || answer.Session.ID != "rtc_123" {
		t.Fatalf("unexpected WebRTC answer: %+v %v", answer, err)
	}
}

func writeFrame(ctx context.Context, socket *websocket.Conn, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return socket.Write(ctx, websocket.MessageText, data)
}
