package openai_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	"github.com/Kludex/pydantic-ai-go/ai/realtime/infer"
	openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
	"github.com/coder/websocket"
)

type livePeer struct {
	socket *websocket.Conn
	sent   chan map[string]any
	start  map[string]any
	model  *openairt.LiveModel
}

func livePair(t *testing.T, params realtime.ConnectParams, options ...openairt.Option) (*openairt.LiveConnection, *livePeer) {
	t.Helper()
	ready := make(chan *livePeer, 1)
	sent := make(chan map[string]any, 256)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/live/sessions" || r.Header.Get("Authorization") != "Bearer secret" {
			t.Errorf("handshake: %s %s", r.URL, r.Header)
		}
		socket, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = socket.CloseNow() }()
		socket.SetReadLimit(4 << 20)
		_, data, err := socket.Read(r.Context())
		if err != nil {
			return
		}
		var frame map[string]any
		_ = json.Unmarshal(data, &frame)
		session := frame["session"].(map[string]any)
		started := map[string]any{"id": "live-1", "model": "gpt-live-1", "delegation": session["delegation"]}
		_ = writeFrame(r.Context(), socket, map[string]any{"type": "session.started", "session": started})
		ready <- &livePeer{socket: socket, sent: sent, start: session}
		for {
			_, data, err := socket.Read(r.Context())
			if err != nil {
				return
			}
			var input map[string]any
			_ = json.Unmarshal(data, &input)
			sent <- input
			if input["type"] == "session.close" {
				_ = writeFrame(r.Context(), socket, map[string]any{"type": "session.closed", "reason": "close_requested", "usage": map[string]any{"seconds": 9.0}})
			}
		}
	}))
	t.Cleanup(server.Close)
	defaults := []openairt.Option{openairt.WithAPIKey("secret"), openairt.WithBaseURL(server.URL + "/v1"), openairt.WithLiveSettings(openairt.LiveSettings{TurnSilence: 10 * time.Millisecond})}
	model := openairt.NewLiveModel("gpt-live-1+gpt-5", append(defaults, options...)...)
	connection, err := model.Connect(t.Context(), params)
	if err != nil {
		t.Fatal(err)
	}
	live := connection.(*openairt.LiveConnection)
	t.Cleanup(func() { _ = live.Close(context.Background()) })
	peer := <-ready
	peer.model = model
	return live, peer
}

func (peer *livePeer) frame(t *testing.T, value any) {
	t.Helper()
	if err := writeFrame(t.Context(), peer.socket, value); err != nil {
		t.Fatal(err)
	}
}
func (peer *livePeer) input(t *testing.T) map[string]any {
	t.Helper()
	select {
	case frame := <-peer.sent:
		return frame
	case <-time.After(time.Second):
		t.Fatal("no input received")
		return nil
	}
}
func (peer *livePeer) end(t *testing.T, reason string, seconds float64) {
	t.Helper()
	peer.frame(t, map[string]any{"type": "session.closed", "reason": reason, "usage": map[string]any{"seconds": seconds}})
	go func() { _ = peer.socket.Close(websocket.StatusNormalClosure, "") }()
}
func collectLive(t *testing.T, connection *openairt.LiveConnection) []realtime.CodecEvent {
	t.Helper()
	var events []realtime.CodecEvent
	for event, err := range connection.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, event)
	}
	return events
}
func delegated(id string, event map[string]any) map[string]any {
	return map[string]any{"type": "response.event", "delegation_id": id, "event": event}
}
func delegation(id string) map[string]any {
	return map[string]any{"type": "session.delegation.created", "delegation": map[string]any{"id": id, "target": "responses"}}
}
func liveCall(id string) map[string]any {
	return map[string]any{"type": "response.output_item.done", "item": map[string]any{"type": "function_call", "call_id": id, "name": "lookup", "arguments": ""}}
}
func backendTerminal(kind string, usage any) map[string]any {
	return map[string]any{"type": kind, "response": map[string]any{"id": "resp-1", "model": "gpt-5", "usage": usage}}
}

func TestLiveInferenceAndProfile(t *testing.T) {
	model, err := infer.Model("openai:gpt-live-1+gpt-5")
	if err != nil {
		t.Fatal(err)
	}
	live, ok := model.(*openairt.LiveModel)
	if !ok || live.Name() != "gpt-live-1" || live.ProviderName() != "openai" || live.ProviderURL() == "" {
		t.Fatalf("model: %#v", model)
	}
	profile := live.Profile()
	if !profile.SynthesizesTurnBoundary || !profile.SupportsThinking || !profile.ImageInputRequiresResponse || profile.SupportsTextOutput || profile.SupportsManualTurnControl {
		t.Fatalf("profile: %+v", profile)
	}
	profile.SupportedNativeTools["web_search"] = false
	if !live.Profile().SupportedNativeTools["web_search"] {
		t.Fatal("aliased profile")
	}
	if _, err := live.CreateClientSecret(t.Context(), "", nil, realtime.Settings{}, 0); err == nil {
		t.Fatal("client secret accepted")
	}
}

func TestLiveConfigAndHistory(t *testing.T) {
	yes, no := true, false
	text := "speech"
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.SystemPromptPart{Content: "system"}, ai.UserPromptPart{Contents: []ai.UserContent{ai.TextContent{Text: "one"}, ai.TextContent{Text: "two"}}}, ai.SpeechPart{Transcript: &text}, ai.ToolReturnPart{ToolName: "lookup", Content: map[string]any{"ok": true}}, ai.RetryPromptPart{ToolName: "lookup", Content: "try again"}, ai.RetryPromptPart{Content: "retry"}, ai.UserPromptPart{Content: "  "}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "answer"}, ai.ThinkingPart{Content: "thought", Signature: "not replayed"}, ai.SpeechPart{Transcript: &text}, ai.ToolCallPart{ToolName: "lookup", Args: json.RawMessage(`{}`)}}},
	}
	connection, peer := livePair(t, realtime.ConnectParams{Messages: messages, Request: ai.ModelRequestParams{Instructions: "work", Tools: []ai.ToolDefinition{{Name: "lookup", Description: "find", Schema: map[string]any{"type": "object"}, Strict: &yes}}, NativeTools: []ai.NativeTool{ai.WebSearchTool{UserLocation: &ai.WebSearchUserLocation{City: "Berlin", Country: "DE", Region: "BE", Timezone: "Europe/Berlin"}, AllowedDomains: []string{"example.org"}, BlockedDomains: []string{"example.com"}, ExternalWebAccess: &no}, ai.CodeExecutionTool{Optional: true}}}, Settings: realtime.Settings{Thinking: ai.ThinkingLevelHigh, ParallelToolCalls: &no, ToolChoice: realtime.ToolChoiceAuto, Provider: map[string]any{"openai_voice": "marin", "openai_live_instructions": "talk", "openai_live_store": true, "openai_live_turn_silence_ms": 15, "openai_live_delegation": map[string]any{"model": "gpt-6-sol", "instructions": "extra", "max_output_tokens": 100, "verbosity": "low", "service_tier": "priority"}}}})
	if connection.ModelName() != "gpt-live-1" || !connection.InputTranscriptionEnabled() || connection.ReconnectRestoresInFlightState() {
		t.Fatal("connection info")
	}
	responses := peer.start["delegation"].(map[string]any)["responses"].(map[string]any)
	if responses["model"] != "gpt-6-sol" || responses["instructions"] != "work\n\nsystem\n\nextra" || responses["parallel_tool_calls"] != false || responses["reasoning"].(map[string]any)["effort"] != "high" || len(responses["tools"].([]any)) != 2 {
		t.Fatalf("config: %+v", peer.start)
	}
	if len(peer.start["input"].([]any)) != 9 {
		t.Fatalf("history: %+v", peer.start["input"])
	}
	peer.end(t, "remote_hangup", 0)
	_ = collectLive(t, connection)
}

func TestLiveInputWireProtocol(t *testing.T) {
	connection, peer := livePair(t, realtime.ConnectParams{})
	inputs := []realtime.Input{realtime.AudioInput{Data: []byte{1, 0}}, realtime.TextInput{Text: "hello"}, realtime.TextContext{Text: "silent"}, realtime.ImageInput{Content: ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"}, Respond: true}, realtime.CreateResponse{}, realtime.ToolResult{ToolCallID: "unknown", Output: "result", Content: []ai.UserContent{ai.TextContent{Text: "extra"}, ai.BinaryContent{Data: []byte("png"), MediaType: "image/png"}, ai.BinaryContent{Data: []byte("pdf"), MediaType: "application/pdf"}, ai.UploadedFile{FileID: "file-1", ProviderName: "openai"}, ai.UploadedFile{FileID: "file-2", ProviderName: "openai", MediaType: "image/png"}, ai.CachePoint{}, ai.ImageURL{URL: "https://example.com/i.png", ForceDownload: ai.FileDownloadNever}, ai.DocumentURL{URL: "https://example.com/f.pdf", ForceDownload: ai.FileDownloadNever}}}}
	for _, input := range inputs {
		if err := connection.Send(t.Context(), input); err != nil {
			t.Fatal(err)
		}
	}
	want := []string{"session.input_audio.append", "session.commentary.append", "session.thinking.append", "response.item.create", "response.create", "response.create", "response.item.create", "response.item.create", "response.create"}
	for _, kind := range want {
		frame := peer.input(t)
		if frame["type"] != kind {
			t.Fatalf("input: %+v, want %s", frame, kind)
		}
	}
	for _, input := range []realtime.Input{realtime.AudioInput{Data: []byte{1}}, realtime.CommitAudio{}, realtime.ClearAudio{}, realtime.CancelResponse{}, realtime.TruncateOutput{}, realtime.ImageInput{}, realtime.TextInput{Text: strings.Repeat(" a", 501)}} {
		if err := connection.Send(t.Context(), input); err == nil {
			t.Fatalf("accepted %T", input)
		}
	}
	if err := connection.Send(t.Context(), realtime.TextInput{Text: strings.Repeat("<|endoftext|>", 500)}); err != nil {
		t.Fatal(err)
	}
	_ = peer.input(t)
	if err := connection.Send(t.Context(), realtime.TextContext{Text: strings.Repeat("<|endofprompt|>", 501)}); err == nil {
		t.Fatal("special token limit")
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := connection.Send(ctx, realtime.CreateResponse{}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := connection.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := connection.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := connection.Send(t.Context(), realtime.CreateResponse{}); err == nil {
		t.Fatal("send after close")
	}
	for range connection.Events(t.Context()) {
		t.Fatal("event after close")
	}
	for range connection.EndSession(t.Context()) {
		t.Fatal("usage after close")
	}
}

func TestLiveTurnTimelineAndUsage(t *testing.T) {
	connection, peer := livePair(t, realtime.ConnectParams{})
	peer.frame(t, map[string]any{"type": "session.output_audio.delta", "delta": base64.StdEncoding.EncodeToString([]byte{20, 0})})
	peer.frame(t, map[string]any{"type": "session.input_transcript.delta", "delta": "Hello.", "start_ms": 0, "end_ms": 100})
	peer.frame(t, map[string]any{"type": "session.output_transcript.delta", "delta": "Yes."})
	peer.frame(t, map[string]any{"type": "session.input_transcript.delta", "delta": ".", "start_ms": 100, "end_ms": 110})
	peer.frame(t, map[string]any{"type": "session.output_transcript.delta", "delta": "It's me."})
	peer.frame(t, map[string]any{"type": "session.output_audio.delta", "delta": base64.StdEncoding.EncodeToString([]byte{100, 0})})
	peer.frame(t, map[string]any{"type": "session.output_audio.delta", "delta": base64.StdEncoding.EncodeToString(make([]byte, 24002))})
	peer.frame(t, map[string]any{"type": "session.output_audio.delta", "delta": "AAA="})
	peer.frame(t, map[string]any{"type": "session.usage.updated", "usage": map[string]any{"seconds": 5}, "context_window": map[string]any{"usage_ratio": 0.5}})
	peer.frame(t, map[string]any{"type": "session.usage.updated", "usage": map[string]any{"seconds": 5}, "context_window": map[string]any{"usage_ratio": 0.2}})
	peer.frame(t, map[string]any{"type": "session.usage.updated", "usage": map[string]any{"seconds": 4}})
	var events []realtime.CodecEvent
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	for event, err := range connection.Events(ctx) {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, event)
		if _, ok := event.(realtime.ResponseDone); ok {
			break
		}
	}
	var text string
	audio := 0
	seconds := 0.0
	ratios := []float64{}
	for _, event := range events {
		switch event := event.(type) {
		case realtime.OutputTranscript:
			text += event.Text
		case realtime.AudioDelta:
			audio += len(event.Data)
		case realtime.SessionUsage:
			seconds += event.Usage.AudioSeconds
			if event.ContextWindowUsed != nil {
				ratios = append(ratios, *event.ContextWindowUsed)
			}
		}
	}
	if text != "Yes. It's me." || audio != 24002 || seconds != 5 || !reflect.DeepEqual(ratios, []float64{0.5, 0.2}) {
		t.Fatalf("events: %+v / text %s audio %d usage %g", events, text, audio, seconds)
	}
	for report, err := range connection.EndSession(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if report.Usage.AudioSeconds != 4 {
			t.Fatalf("final usage: %+v", report)
		}
	}
	if peer.input(t)["type"] != "session.close" {
		t.Fatal("did not end session")
	}
	for range connection.EndSession(t.Context()) {
		t.Fatal("duplicated final usage")
	}
}

func TestLiveDelegatedToolsWaitForTerminal(t *testing.T) {
	connection, peer := livePair(t, realtime.ConnectParams{})
	peer.frame(t, delegation("d"))
	peer.frame(t, delegated("d", liveCall("c1")))
	peer.frame(t, delegated("d", liveCall("c1")))
	peer.frame(t, delegated("d", liveCall("c2")))
	peer.frame(t, delegated("d", backendTerminal("response.completed", map[string]any{"input_tokens": 100, "output_tokens": 10, "input_tokens_details": map[string]any{"cached_tokens": 20, "cache_write_tokens": 2}, "output_tokens_details": map[string]any{"reasoning_tokens": 3}})))
	peer.frame(t, delegated("d", backendTerminal("response.completed", nil)))
	var calls int
	requests := 0
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	for event, err := range connection.Events(ctx) {
		if err != nil {
			t.Fatal(err)
		}
		switch event := event.(type) {
		case realtime.ToolCall:
			calls++
			if !event.ResponseUsageFollows {
				t.Fatal("no usage pending")
			}
			if err := connection.Send(t.Context(), realtime.ToolResult{ToolCallID: event.ToolCallID, Output: "ok"}); err != nil {
				t.Fatal(err)
			}
			if peer.input(t)["type"] != "response.item.create" {
				t.Fatal("result wire")
			}
			select {
			case frame := <-peer.sent:
				t.Fatalf("premature continuation: %+v", frame)
			default:
			}
		case realtime.SessionUsage:
			requests += event.Usage.Requests
			if event.Usage.InputTokens > 0 && (event.Usage.CacheReadTokens != 20 || event.Usage.CacheWriteTokens != 2 || event.Usage.ReasoningTokens != 3 || event.ProviderDetails["delegated_model"] != "gpt-5") {
				t.Fatalf("usage: %+v", event)
			}
		case realtime.ResponseDone:
			goto done
		}
	}
done:
	if calls != 2 || requests != 2 || peer.input(t)["type"] != "response.create" {
		t.Fatalf("calls=%d requests=%d", calls, requests)
	}
}

func TestLiveBackendFailureAndNativeSearch(t *testing.T) {
	connection, peer := livePair(t, realtime.ConnectParams{})
	peer.frame(t, delegation("d"))
	peer.frame(t, delegated("d", map[string]any{"type": "response.output_item.done", "item": map[string]any{"type": "reasoning", "id": "rs", "encrypted_content": "opaque", "summary": []any{map[string]any{"text": "thinking"}, map[string]any{"text": "more"}}}}))
	peer.frame(t, delegated("d", map[string]any{"type": "response.output_item.done", "item": map[string]any{"type": "web_search_call", "id": "ws", "status": "completed", "action": map[string]any{"type": "search", "query": "weather"}}}))
	peer.frame(t, delegated("d", liveCall("failed")))
	peer.frame(t, delegated("d", backendTerminal("response.failed", nil)))
	peer.frame(t, delegated("none", map[string]any{"type": "error", "message": "backend error", "code": "oops"}))
	peer.frame(t, delegated("none", backendTerminal("response.incomplete", nil)))
	peer.frame(t, map[string]any{"type": "session.delegation.created", "delegation": map[string]any{"id": "client", "target": "client"}})
	peer.frame(t, map[string]any{"type": "session.delegation.created", "delegation": map[string]any{"id": "d2", "target": "responses"}})
	peer.frame(t, map[string]any{"type": "error", "error": map[string]any{"code": nil, "message": "Responses handoff incomplete."}})
	peer.end(t, "remote_hangup", 6)
	events := collectLive(t, connection)
	parts := 0
	failures := 0
	for _, event := range events {
		switch event := event.(type) {
		case realtime.PartStarted:
			parts++
		case realtime.SessionError:
			failures++
			if event.Err.Error() == "" {
				t.Fatal("empty error")
			}
		}
	}
	if parts != 4 || failures != 5 {
		t.Fatalf("parts %d failures %d: %+v", parts, failures, events)
	}
	// Ended sessions reject sending even when the call was abandoned.
	if err := connection.Send(t.Context(), realtime.ToolResult{ToolCallID: "failed"}); err == nil {
		t.Fatal("send after ended")
	}
}

func TestLiveMalformedEventsAreRecoverable(t *testing.T) {
	frames := []string{`null`, `[]`, `{`, `{"type":"session.output_audio.delta"}`, `{"type":"session.output_audio.delta","delta":"%%%"}`, `{"type":"session.output_audio.delta","delta":"AA=="}`, `{"type":"session.input_transcript.delta","delta":"hello"}`, `{"type":"session.output_transcript.delta","delta":2}`, `{"type":"session.delegation.created"}`, `{"type":"response.event","event":[]}`, `{"type":"session.usage.updated","usage":{}}`, `{"type":"session.usage.updated","usage":{"seconds":0},"context_window":{}}`, `{"type":"session.closed"}`, `{"type":"error","error":{}}`, `{"type":"response.event","event":{"type":"response.completed"}}`, `{"type":"response.event","event":{"type":"response.output_item.done"}}`, `{"type":"response.event","event":{"type":"response.output_item.done","item":{"type":"function_call"}}}`, `{"type":"response.event","event":{"type":"error"}}`, `{"type":"response.event","event":{"type":"response.output_item.done","item":{"type":"web_search_call"}}}`}
	connection, peer := livePair(t, realtime.ConnectParams{})
	for _, frame := range frames {
		if err := peer.socket.Write(t.Context(), websocket.MessageText, []byte(frame)); err != nil {
			t.Fatal(err)
		}
	}
	for _, frame := range []string{`{}`, `{"type":"future"}`, `{"type":"response.event","event":{"type":"future"}}`, `{"type":"response.event","event":{"type":"response.output_item.done","item":{"type":"message"}}}`} {
		_ = peer.socket.Write(t.Context(), websocket.MessageText, []byte(frame))
	}
	_ = peer.socket.Write(t.Context(), websocket.MessageBinary, []byte("ignore"))
	peer.end(t, "remote_hangup", 0)
	events := collectLive(t, connection)
	if len(events) != len(frames) {
		t.Fatalf("events=%d want=%d: %+v", len(events), len(frames), events)
	}
	for _, event := range events {
		if failure, ok := event.(realtime.SessionError); !ok || !failure.Recoverable {
			t.Fatalf("not recoverable: %+v", event)
		}
	}
}
