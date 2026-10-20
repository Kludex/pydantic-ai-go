package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
	"github.com/coder/websocket"
)

func TestThinkingProfileUsesModelFamilyBoundary(t *testing.T) {
	for _, test := range []struct {
		name string
		want bool
	}{
		{name: "gpt-realtime-2", want: true},
		{name: "gpt-realtime-2.1", want: true},
		{name: "gpt-realtime-2-mini", want: true},
		{name: "gpt-realtime-2025-08-28"},
		{name: "gpt-realtime-mini-2025-10-06"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := openairt.NewModel(test.name).Profile().SupportsThinking; got != test.want {
				t.Fatalf("SupportsThinking = %v, want %v", got, test.want)
			}
		})
	}
}

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
			InputTranscriptionModel: &transcription, Thinking: ai.ThinkingLevelDisabled,
			Provider: map[string]any{
				"openai_voice": "coral", "openai_input_noise_reduction": "far_field",
				"openai_output_speed": 1.25, "openai_turn_detection": map[string]any{"type": "semantic_vad"},
				"openai_truncation":       map[string]any{"type": "retention_ratio"},
				"openai_prune_tool_items": true, "openai_keep_tool_items": []string{"lookup"},
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
	sessionConfig := received[0]["session"].(map[string]any)
	if received[0]["type"] != "session.update" || sessionConfig["reasoning"].(map[string]any)["effort"] != "none" {
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

func TestOpenAIRejectedAndFailedEvents(t *testing.T) {
	events, err := openairt.MapEvent([]byte(`{"type":"error","error":{"event_id":"pydantic_ai.response.2-3","message":"refused"}}`))
	if err != nil || len(events) != 3 {
		t.Fatalf("unexpected refusal: %+v err=%v", events, err)
	}
	for index, expected := range []int{2, 3} {
		rejected, ok := events[index].(realtime.InputRejected)
		if !ok || rejected.InputIndex != expected || !rejected.Response {
			t.Fatalf("unexpected rejected input: %#v", events[index])
		}
	}
	contentEvents, contentErr := openairt.MapEvent([]byte(`{"type":"error","error":{"event_id":"pydantic_ai.content.4","message":"refused"}}`))
	contentRejected, contentOK := contentEvents[0].(realtime.InputRejected)
	if contentErr != nil || !contentOK || contentRejected.Response || contentRejected.InputIndex != 4 {
		t.Fatalf("unexpected content refusal: %+v err=%v", contentEvents, contentErr)
	}
	for _, eventID := range []string{
		"foreign", "pydantic_ai.content", "pydantic_ai.other.1", "pydantic_ai.content.bad", "pydantic_ai.content.-1",
	} {
		events, err = openairt.MapEvent([]byte(`{"type":"error","error":{"event_id":"` + eventID + `","message":"refused"}}`))
		if err != nil || len(events) != 1 {
			t.Fatalf("foreign event ID was accepted: %+v err=%v", events, err)
		}
	}
	events, err = openairt.MapEvent([]byte(`{"type":"response.done","response":{"id":"failed","status":"failed","status_details":{"error":{"type":"invalid_request_error","code":"image","message":"unsafe"}},"output":[]}}`))
	if err != nil || len(events) != 2 {
		t.Fatalf("unexpected failed response: %+v err=%v", events, err)
	}
	failure, ok := events[0].(realtime.SessionError)
	done, doneOK := events[1].(realtime.ResponseDone)
	if !ok || !failure.Recoverable || !strings.Contains(failure.Err.Error(), "unsafe") || !doneOK ||
		done.ProviderDetails["error"] == nil {
		t.Fatalf("failed response was not surfaced: %+v", events)
	}
	for _, frame := range []string{
		`{"type":"response.done","response":{"status":"failed","output":[]}}`,
		`{"type":"response.done","response":{"status":"failed","status_details":{"error":{"type":"server_error","code":"oops"}},"output":[]}}`,
	} {
		events, err = openairt.MapEvent([]byte(frame))
		if err != nil || len(events) != 2 {
			t.Fatalf("failed response fallback missing: %+v err=%v", events, err)
		}
	}
}

func TestOpenAIRealtimeImageUsage(t *testing.T) {
	events, err := openairt.MapEvent([]byte(`{"type":"response.done","response":{"status":"completed","usage":{"input_tokens":10,"output_tokens":0,"input_token_details":{"image_tokens":8,"cached_tokens_details":{"image_tokens":3}}}}}`))
	if err != nil {
		t.Fatal(err)
	}
	usage := events[0].(realtime.SessionUsage).Usage
	if usage.InputImageTokens != 8 || usage.CacheImageReadTokens != 3 {
		t.Fatalf("unexpected image usage: %+v", usage)
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

func TestOpenAIPrunesAnsweredToolRounds(t *testing.T) {
	type peerSocket struct {
		socket *websocket.Conn
		frames chan map[string]any
	}
	peers := make(chan peerSocket, 2)
	server := handshakeServer(t, func(_ map[string]any, socket *websocket.Conn, ctx context.Context) {
		_ = writeFrame(ctx, socket, map[string]any{"type": "session.updated", "session": map[string]any{}})
		frames := make(chan map[string]any, 64)
		peers <- peerSocket{socket: socket, frames: frames}
		for {
			_, data, err := socket.Read(ctx)
			if err != nil {
				return
			}
			var frame map[string]any
			_ = json.Unmarshal(data, &frame)
			frames <- frame
		}
	})
	defer server.Close()
	model := openairt.NewModel("gpt-realtime", openairt.WithAPIKey("secret"), openairt.WithBaseURL(wsBase(server.URL)),
		openairt.WithHTTPClient(server.Client()),
		openairt.WithSettings(openairt.Settings{PruneToolItems: true, KeepToolItems: []string{"profile"}}))
	connection, err := model.Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{
		Reconnect: &realtime.ReconnectPolicy{MaxAttempts: 1, MaxReconnects: 1},
	}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = connection.Close(t.Context()) }()
	peer := <-peers
	var historyMu sync.Mutex
	var history []ai.ModelMessage
	connection.(interface {
		SetMessageHistory(func() []ai.ModelMessage)
	}).SetMessageHistory(func() []ai.ModelMessage {
		historyMu.Lock()
		defer historyMu.Unlock()
		return history
	})
	events := make(chan realtime.CodecEvent, 64)
	go func() {
		for event, err := range connection.Events(t.Context()) {
			if err != nil {
				return
			}
			events <- event
		}
	}()
	send := func(frames ...map[string]any) {
		for _, frame := range frames {
			if err := writeFrame(t.Context(), peer.socket, frame); err != nil {
				t.Fatal(err)
			}
		}
	}
	added := func(item map[string]any) map[string]any {
		return map[string]any{"type": "conversation.item.added", "item": item}
	}
	call := func(id, callID, name string) map[string]any {
		return map[string]any{"type": "function_call", "id": id, "call_id": callID, "name": name, "arguments": "{}"}
	}
	created := func(id string) map[string]any {
		return map[string]any{"type": "response.created", "response": map[string]any{"id": id}}
	}
	done := func(id, status string, output ...any) map[string]any {
		return map[string]any{"type": "response.done", "response": map[string]any{"id": id, "status": status, "output": output}}
	}
	next := func(kind string) map[string]any {
		for {
			select {
			case frame := <-peer.frames:
				if frame["type"] == kind {
					return frame
				}
			case <-time.After(time.Second):
				t.Fatalf("no %s frame", kind)
			}
		}
	}
	waitFor := func(count int) {
		for count > 0 {
			select {
			case event := <-events:
				if _, ok := event.(realtime.ResponseDone); ok {
					count--
				}
			case <-time.After(time.Second):
				t.Fatal("no response done")
			}
		}
	}
	send(
		created("r1"), added(call("fc1", "call_1", "lookup")), added(call("fc2", "call_2", "profile")),
		added(map[string]any{"type": "function_call", "id": "fc3"}),
		done("r1", "completed", call("fc1", "call_1", "lookup"), call("fc2", "call_2", "profile")),
	)
	waitFor(1)
	image := ai.BinaryContent{Data: []byte("png"), MediaType: "image/png"}
	if err := connection.Send(t.Context(), realtime.ToolResult{ToolCallID: "call_1", Output: "found", Content: []ai.UserContent{image}}); err != nil {
		t.Fatal(err)
	}
	next("conversation.item.create")
	content := next("conversation.item.create")["item"].(map[string]any)
	if err := connection.Send(t.Context(), realtime.ToolResult{ToolCallID: "call_2", Output: "kept", Content: []ai.UserContent{image}}); err != nil {
		t.Fatal(err)
	}
	next("conversation.item.create")
	if kept := next("conversation.item.create")["item"].(map[string]any); kept["id"] != nil {
		t.Fatalf("a kept tool's content was named for pruning: %v", kept)
	}
	next("response.create")
	send(
		added(map[string]any{"type": "function_call_output", "id": "out1", "call_id": "call_1"}),
		added(map[string]any{"type": "function_call_output", "id": "out2", "call_id": "call_2"}),
		added(map[string]any{"type": "message", "id": content["id"], "role": "user"}),
		created("r2"), done("r2", "completed", map[string]any{"type": "message"}),
	)
	waitFor(1)
	var deleted []string
	for range 3 {
		deleted = append(deleted, next("conversation.item.delete")["item_id"].(string))
	}
	if want := []string{"fc1", "out1", "pydantic_ai_content_0"}; !slices.Equal(deleted, want) || content["id"] != want[2] {
		t.Fatalf("deleted %v, want %v (content %v)", deleted, want, content["id"])
	}

	message := map[string]any{"type": "message"}
	send(
		created("r3"), added(call("fc4", "call_4", "lookup")), done("r3", "completed", call("fc4", "call_4", "lookup")),
		created("r4"), added(map[string]any{"type": "function_call_output", "id": "out4", "call_id": "call_4"}),
		done("r4", "completed", message),
		done("unknown", "completed", message),
		created("r5"), done("r5", "cancelled", message),
		created("r6"), done("r6", "completed"),
		map[string]any{"type": "response.done"}, added(nil),
	)
	waitFor(5)
	select {
	case frame := <-peer.frames:
		t.Fatalf("an output added after its answer started was pruned: %v", frame)
	case <-time.After(50 * time.Millisecond):
	}

	historyMu.Lock()
	history = []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "hello"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "lookup", ToolCallID: "call_1", Args: []byte(`{}`)},
			ai.ToolCallPart{ToolName: "profile", ToolCallID: "call_2", Args: []byte(`{}`)},
		}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{ToolName: "lookup", ToolCallID: "call_1", Content: "found"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.RetryPromptPart{ToolName: "lookup", ToolCallID: "call_1", Content: "again"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{ToolName: "profile", ToolCallID: "call_2", Content: "kept"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "lookup", ToolCallID: "call_1"}}},
	}
	historyMu.Unlock()
	_ = peer.socket.CloseNow()
	peer = <-peers
	var calls []string
	for {
		frame := next("conversation.item.create")
		item := frame["item"].(map[string]any)
		if callID, ok := item["call_id"].(string); ok {
			calls = append(calls, callID)
		}
		if item["type"] == "function_call_output" {
			break
		}
	}
	if want := []string{"call_2", "call_2"}; !slices.Equal(calls, want) {
		t.Fatalf("replayed calls %v, want %v", calls, want)
	}
}
