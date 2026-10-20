package elevenlabs_test

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	"github.com/Kludex/pydantic-ai-go/ai/realtime/elevenlabs"
	"github.com/coder/websocket"
)

// fakeAgent serves the REST preflight and one conversation socket per dial.
type fakeAgent struct {
	t         *testing.T
	agent     map[string]any
	workspace []map[string]any
	dependent map[string][]string

	mu        sync.Mutex
	requests  []string
	bodies    map[string][]map[string]any
	initiated chan map[string]any
	sockets   chan *websocket.Conn
	metadata  map[string]any
	server    *httptest.Server
	rejectWS  int
	// failing answers these "METHOD /path" requests with 500; "decode" ones with invalid JSON.
	failing   map[string]bool
	malform   map[string]bool
	handshake func(ctx context.Context, socket *websocket.Conn)
}

func newFakeAgent(t *testing.T, agent map[string]any) *fakeAgent {
	t.Helper()
	fake := &fakeAgent{
		t: t, agent: agent, dependent: map[string][]string{}, bodies: map[string][]map[string]any{},
		failing: map[string]bool{}, malform: map[string]bool{},
		initiated: make(chan map[string]any, 4), sockets: make(chan *websocket.Conn, 4),
		metadata: map[string]any{
			"conversation_id": "conv_1", "agent_output_audio_format": "pcm_16000", "user_input_audio_format": "pcm_16000",
		},
	}
	fake.server = httptest.NewServer(http.HandlerFunc(fake.serve))
	t.Cleanup(fake.server.Close)
	return fake
}

func (fake *fakeAgent) serve(writer http.ResponseWriter, request *http.Request) {
	fake.mu.Lock()
	fake.requests = append(fake.requests, request.Method+" "+request.URL.Path)
	if request.Body != nil && request.Method != http.MethodGet {
		var body map[string]any
		_ = json.NewDecoder(request.Body).Decode(&body)
		fake.bodies[request.Method+" "+request.URL.Path] = append(fake.bodies[request.Method+" "+request.URL.Path], body)
	}
	fake.mu.Unlock()
	if request.URL.Path != "/ws" && request.Header.Get("xi-api-key") != "key" {
		writer.WriteHeader(http.StatusUnauthorized)
		return
	}
	if fake.failing[request.Method+" "+request.URL.Path] {
		writer.WriteHeader(http.StatusInternalServerError)
		return
	}
	if fake.malform[request.Method+" "+request.URL.Path] {
		_, _ = writer.Write([]byte("{"))
		return
	}
	reply := func(value any) { _ = json.NewEncoder(writer).Encode(value) }
	switch {
	case request.URL.Path == "/v1/convai/agents/agent_1" && request.Method == http.MethodGet:
		reply(fake.agent)
	case request.URL.Path == "/v1/convai/agents/agent_1" && request.Method == http.MethodPatch:
		reply(map[string]any{})
	case request.URL.Path == "/v1/convai/conversation/get-signed-url":
		if request.URL.Query().Get("agent_id") != "agent_1" {
			writer.WriteHeader(http.StatusBadRequest)
			return
		}
		reply(map[string]any{"signed_url": "ws" + strings.TrimPrefix(fake.server.URL, "http") + "/ws"})
	case request.URL.Path == "/v1/convai/tools" && request.Method == http.MethodGet:
		cursor := request.URL.Query().Get("cursor")
		if cursor == "" && len(fake.workspace) > 1 {
			reply(map[string]any{"tools": fake.workspace[:1], "has_more": true, "next_cursor": "page2"})
			return
		}
		start := 0
		if cursor != "" {
			start = 1
		}
		reply(map[string]any{"tools": fake.workspace[min(start, len(fake.workspace)):], "has_more": cursor == "page2", "next_cursor": "page2"})
	case request.URL.Path == "/v1/convai/tools" && request.Method == http.MethodPost:
		reply(map[string]any{"id": "created_" + fake.bodies["POST /v1/convai/tools"][len(fake.bodies["POST /v1/convai/tools"])-1]["tool_config"].(map[string]any)["name"].(string)})
	case strings.HasSuffix(request.URL.Path, "/dependent-agents"):
		id := strings.TrimSuffix(strings.TrimPrefix(request.URL.Path, "/v1/convai/tools/"), "/dependent-agents")
		var agents []map[string]any
		for _, agent := range fake.dependent[id] {
			agents = append(agents, map[string]any{"id": agent})
		}
		reply(map[string]any{"agents": agents})
	case strings.HasPrefix(request.URL.Path, "/v1/convai/tools/") && request.Method == http.MethodPatch:
		reply(map[string]any{})
	case strings.HasPrefix(request.URL.Path, "/v1/convai/tools/"):
		id := strings.TrimPrefix(request.URL.Path, "/v1/convai/tools/")
		for _, tool := range fake.workspace {
			if tool["id"] == id {
				reply(tool)
				return
			}
		}
		writer.WriteHeader(http.StatusNotFound)
	case request.URL.Path == "/ws":
		if fake.rejectWS != 0 {
			writer.WriteHeader(fake.rejectWS)
			_, _ = writer.Write([]byte("expired"))
			return
		}
		socket, err := websocket.Accept(writer, request, nil)
		if err != nil {
			return
		}
		defer func() { _ = socket.CloseNow() }()
		_, data, err := socket.Read(request.Context())
		if err != nil {
			return
		}
		var initiation map[string]any
		_ = json.Unmarshal(data, &initiation)
		fake.initiated <- initiation
		if fake.handshake != nil {
			fake.handshake(request.Context(), socket)
			return
		}
		send(request.Context(), socket, map[string]any{"type": "ping", "ping_event": map[string]any{"event_id": 1}})
		_, _, _ = socket.Read(request.Context())
		send(request.Context(), socket, map[string]any{"type": "conversation_initiation_metadata", "conversation_initiation_metadata_event": fake.metadata})
		fake.sockets <- socket
		<-request.Context().Done()
	default:
		fake.t.Errorf("unexpected request %s %s", request.Method, request.URL)
		writer.WriteHeader(http.StatusNotFound)
	}
}

func send(ctx context.Context, socket *websocket.Conn, value any) {
	data, _ := json.Marshal(value)
	_ = socket.Write(ctx, websocket.MessageText, data)
}

func (fake *fakeAgent) model(options ...elevenlabs.Option) *elevenlabs.Model {
	return elevenlabs.NewModel("agent_1", append([]elevenlabs.Option{
		elevenlabs.WithAPIKey("key"), elevenlabs.WithBaseURL(fake.server.URL + "/"),
		elevenlabs.WithHTTPClient(fake.server.Client()),
		elevenlabs.WithSettings(elevenlabs.Settings{WarningHandler: func(string) {}}),
	}, options...)...)
}

func permissions(paths ...string) map[string]any {
	root := map[string]any{}
	for _, path := range paths {
		node := root
		keys := strings.Split(path, ".")
		for _, key := range keys[:len(keys)-1] {
			child, ok := node[key].(map[string]any)
			if !ok {
				child = map[string]any{}
				node[key] = child
			}
			node = child
		}
		node[keys[len(keys)-1]] = true
	}
	return root
}

func agentConfig(tools []map[string]any, toolIDs []string, allowed ...string) map[string]any {
	return map[string]any{
		"agent_id": "agent_1",
		"conversation_config": map[string]any{
			"agent":        map[string]any{"prompt": map[string]any{"tools": tools, "tool_ids": toolIDs}},
			"conversation": map[string]any{"client_events": []string{"audio", "context_usage"}},
		},
		"platform_settings": map[string]any{"overrides": map[string]any{
			"conversation_config_override": permissions(allowed...), "custom_llm_extra_body": true,
		}},
	}
}

var lookupTool = ai.ToolDefinition{Name: "lookup", Description: "Look up a city.", Schema: map[string]any{
	"type": "object", "additionalProperties": false, "required": []any{"city"},
	"properties": map[string]any{"city": map[string]any{"type": "string", "description": "The city."}},
}}

var lookupRemote = map[string]any{
	"type": "client", "name": "lookup", "description": "Look up a city.", "expects_response": true,
	"parameters": map[string]any{
		"type": "object", "required": []any{"city"}, "description": "",
		"properties": map[string]any{"city": map[string]any{
			"type": "string", "description": "The city.", "enum": nil, "is_omitted": false,
		}},
	},
}

func connect(t *testing.T, model *elevenlabs.Model, params realtime.ConnectParams) (*elevenlabs.Connection, chan realtime.CodecEvent) {
	t.Helper()
	connection, err := model.Connect(t.Context(), params)
	if err != nil {
		t.Fatal(err)
	}
	live := connection.(*elevenlabs.Connection)
	events := make(chan realtime.CodecEvent, 64)
	go func() {
		defer close(events)
		for event, err := range live.Events(t.Context()) {
			if err != nil {
				return
			}
			events <- event
		}
	}()
	t.Cleanup(func() { _ = live.Close(context.Background()) })
	return live, events
}

func next(t *testing.T, events chan realtime.CodecEvent) realtime.CodecEvent {
	t.Helper()
	select {
	case event, ok := <-events:
		if !ok {
			t.Fatal("events ended")
		}
		return event
	case <-time.After(3 * time.Second):
		t.Fatal("no event")
		return nil
	}
}

func TestConnectSendsOverridesAndStreamsAReply(t *testing.T) {
	fake := newFakeAgent(t, agentConfig([]map[string]any{lookupRemote}, []string{"tool_lookup"},
		"agent.prompt.prompt", "agent.prompt.llm", "agent.first_message", "agent.language", "tts.voice_id",
		"tts.model_id", "tts.stability", "tts.speed", "tts.similarity_boost"))
	stability, speed, similarity := 0.4, 1.05, 0.7
	model := fake.model(elevenlabs.WithSettings(elevenlabs.Settings{
		VoiceID: "voice", Language: "de", FirstMessage: "Hallo!", LLM: "gpt-5.2",
		TTS:              elevenlabs.TTSSettings{ModelID: "eleven_v3_conversational", Stability: &stability, Speed: &speed, SimilarityBoost: &similarity},
		DynamicVariables: map[string]any{"name": "Ada"}, CustomLLMExtraBody: map[string]any{"x": 1}, UserID: "user",
		ConfigOverride: map[string]any{"agent": map[string]any{"prompt": map[string]any{"temperature": 0.2}}},
		WarningHandler: func(string) {},
	}))
	if model.Name() != "agent_1" || model.ProviderName() != "elevenlabs" || !strings.HasPrefix(model.ProviderURL(), "http") ||
		model.Profile().AudioInputSampleRate != 16000 || model.Profile().SupportsInterruption {
		t.Fatalf("unexpected identity %q %q %+v", model.Name(), model.ProviderName(), model.Profile())
	}
	connection, events := connect(t, model, realtime.ConnectParams{
		Request: ai.ModelRequestParams{Instructions: "Be brief.", Tools: []ai.ToolDefinition{lookupTool}},
	})
	initiation := <-fake.initiated
	overrides := initiation["conversation_config_override"].(map[string]any)
	agent := overrides["agent"].(map[string]any)
	prompt := agent["prompt"].(map[string]any)
	tts := overrides["tts"].(map[string]any)
	if initiation["type"] != "conversation_initiation_client_data" || prompt["prompt"] != "Be brief." ||
		prompt["llm"] != "gpt-5.2" || prompt["temperature"] != 0.2 || agent["first_message"] != "Hallo!" ||
		agent["language"] != "de" || tts["voice_id"] != "voice" || tts["model_id"] != "eleven_v3_conversational" ||
		tts["speed"] != 1.05 || initiation["user_id"] != "user" || initiation["custom_llm_extra_body"] == nil ||
		initiation["dynamic_variables"].(map[string]any)["name"] != "Ada" {
		t.Fatalf("unexpected initiation %v", initiation)
	}
	if connection.ConversationID() != "conv_1" || connection.ModelName() != "" || !connection.InputTranscriptionEnabled() ||
		connection.ReconnectRestoresInFlightState() {
		t.Fatal("unexpected connection info")
	}
	socket := <-fake.sockets
	ctx := t.Context()
	send(ctx, socket, map[string]any{"type": "user_transcript", "user_transcription_event": map[string]any{"user_transcript": "Hi"}})
	send(ctx, socket, map[string]any{"type": "audio", "audio_event": map[string]any{"audio_base_64": "AAAA", "event_id": 7}})
	send(ctx, socket, map[string]any{"type": "agent_response", "agent_response_event": map[string]any{
		"agent_response": "[laughs] Hello [1] there [sighs]", "response_id": "resp_1", "event_id": 7,
	}})
	send(ctx, socket, map[string]any{"type": "audio", "audio_event": map[string]any{"audio_base_64": "AAAA", "event_id": 7}})
	send(ctx, socket, map[string]any{"type": "context_usage", "context_usage_event": map[string]any{"context_tokens": 500, "context_limit_tokens": 1000}})
	if transcript := next(t, events).(realtime.InputTranscript); transcript.Text != "Hi" || !transcript.Final {
		t.Fatalf("input transcript %+v", transcript)
	}
	next(t, events)
	if transcript := next(t, events).(realtime.OutputTranscript); transcript.Text != "Hello [1] there" || !transcript.Final {
		t.Fatalf("output transcript %+v", transcript)
	}
	next(t, events)
	done := next(t, events).(realtime.ResponseDone)
	if done.Interrupted || done.ProviderResponseID != "resp_1" || done.ProviderDetails["conversation_id"] != "conv_1" ||
		strings.Join(done.ProviderDetails["audio_tags"].([]string), ",") != "laughs,sighs" {
		t.Fatalf("response done %+v", done)
	}
	usage := next(t, events).(realtime.SessionUsage)
	if usage.Usage.InputTokens != 500 || *usage.ContextWindowUsed != 0.5 || connection.ContextLimitTokens() != 1000 {
		t.Fatalf("usage %+v", usage)
	}

	if err := connection.Send(ctx, realtime.AudioInput{Data: []byte{1, 0}}); err != nil {
		t.Fatal(err)
	}
	if err := connection.Send(ctx, realtime.TextInput{Text: "hello"}); err != nil {
		t.Fatal(err)
	}
	if err := connection.Send(ctx, realtime.TextContext{Text: "context"}); err != nil {
		t.Fatal(err)
	}
	var sent []map[string]any
	for range 3 {
		_, data, _ := socket.Read(ctx)
		var frame map[string]any
		_ = json.Unmarshal(data, &frame)
		sent = append(sent, frame)
	}
	if sent[0]["user_audio_chunk"] != "AQA=" || sent[1]["type"] != "user_message" || sent[2]["type"] != "contextual_update" {
		t.Fatalf("sent %v", sent)
	}
	for _, input := range []realtime.Input{realtime.AudioInput{Data: []byte{1}}, realtime.CommitAudio{}} {
		if err := connection.Send(ctx, input); err == nil {
			t.Fatalf("%T accepted", input)
		}
	}
}

func TestToolCallsBatchAndResultsGoBack(t *testing.T) {
	fake := newFakeAgent(t, agentConfig([]map[string]any{lookupRemote}, []string{"tool_lookup"}))
	connection, events := connect(t, fake.model(), realtime.ConnectParams{
		Request: ai.ModelRequestParams{Tools: []ai.ToolDefinition{lookupTool}},
	})
	socket := <-fake.sockets
	ctx := t.Context()
	send(ctx, socket, map[string]any{"type": "client_tool_call", "client_tool_call": map[string]any{
		"tool_name": "lookup", "tool_call_id": "call_1", "parameters": map[string]any{"city": "Paris"},
	}})
	send(ctx, socket, map[string]any{"type": "client_tool_call", "client_tool_call": map[string]any{
		"tool_name": "lookup", "tool_call_id": "call_2", "parameters": `{"city":"Rome"}`, "expects_response": false,
	}})
	first := next(t, events).(realtime.ToolCall)
	second := next(t, events).(realtime.ToolCall)
	if first.Arguments != `{"city":"Paris"}` || second.Arguments != `{"city":"Rome"}` {
		t.Fatalf("calls %+v %+v", first, second)
	}
	done := next(t, events).(realtime.ResponseDone)
	if !done.MoreExpected || done.FinishReason != ai.FinishReasonToolCall {
		t.Fatalf("calls did not end their response: %+v", done)
	}
	if err := connection.Send(ctx, realtime.ToolResult{ToolCallID: "call_2", Output: "ignored"}); err != nil {
		t.Fatal(err)
	}
	if err := connection.Send(ctx, realtime.ToolResult{ToolCallID: "call_1", Output: "Sunny",
		Content: []ai.UserContent{ai.TextContent{Text: "Warm"}, ai.CachePoint{}}}); err != nil {
		t.Fatal(err)
	}
	_, data, _ := socket.Read(ctx)
	var result map[string]any
	_ = json.Unmarshal(data, &result)
	if result["type"] != "client_tool_result" || result["tool_call_id"] != "call_1" || result["result"] != "Sunny\n\nWarm" {
		t.Fatalf("result %v", result)
	}
	if err := connection.Send(ctx, realtime.ToolResult{ToolCallID: "call_3",
		Content: []ai.UserContent{ai.BinaryContent{Data: []byte("x"), MediaType: "image/png"}}}); err == nil {
		t.Fatal("media tool result accepted")
	}
	send(ctx, socket, map[string]any{"type": "client_tool_call", "client_tool_call": map[string]any{
		"tool_name": "lookup", "tool_call_id": "call_4",
	}})
	if call := next(t, events).(realtime.ToolCall); call.Arguments != "{}" {
		t.Fatalf("call without parameters %+v", call)
	}
	send(ctx, socket, map[string]any{"type": "vad_score"})
	if done := next(t, events).(realtime.ResponseDone); !done.MoreExpected {
		t.Fatalf("a following frame did not end the calls: %+v", done)
	}
}

func TestTextOutputStreamsAndInterruptionsEndReplies(t *testing.T) {
	fake := newFakeAgent(t, agentConfig(nil, nil, "conversation.text_only"))
	_, events := connect(t, fake.model(), realtime.ConnectParams{
		Settings: realtime.Settings{OutputModality: realtime.OutputModalityText},
	})
	if initiation := <-fake.initiated; initiation["conversation_config_override"].(map[string]any)["conversation"].(map[string]any)["text_only"] != true {
		t.Fatalf("text-only override missing: %v", initiation)
	}
	socket := <-fake.sockets
	ctx := t.Context()
	for _, part := range []map[string]any{
		{"type": "start", "text": ""}, {"type": "delta", "text": "Hi "}, {"type": "delta", "text": "[lau"},
		{"type": "delta", "text": "ghs] there"}, {"type": "stop", "text": ""},
	} {
		send(ctx, socket, map[string]any{"type": "agent_chat_response_part", "text_response_part": part})
	}
	send(ctx, socket, map[string]any{"type": "agent_response", "agent_response_event": map[string]any{"agent_response": "Hi [laughs] there"}})
	var text string
	for {
		event := next(t, events)
		if transcript, ok := event.(realtime.OutputTranscript); ok {
			if !transcript.OutputText {
				t.Fatal("text output was not marked")
			}
			if transcript.Final {
				if text != "Hithere" && text != "Hi there" || transcript.Text != "Hi there" {
					t.Fatalf("streamed %q, final %q", text, transcript.Text)
				}
				continue
			}
			text += transcript.Text
		}
		if done, ok := event.(realtime.ResponseDone); ok {
			if done.Interrupted {
				t.Fatal("unexpected interruption")
			}
			break
		}
	}

	send(ctx, socket, map[string]any{"type": "agent_chat_response_part", "text_response_part": map[string]any{"type": "delta", "text": "Long"}})
	send(ctx, socket, map[string]any{"type": "interruption"})
	send(ctx, socket, map[string]any{"type": "agent_response", "agent_response_event": map[string]any{"agent_response": "Lo"}})
	next(t, events)
	next(t, events)
	if done := next(t, events).(realtime.ResponseDone); !done.Interrupted {
		t.Fatalf("interrupted reply %+v", done)
	}
	send(ctx, socket, map[string]any{"type": "agent_response_correction", "agent_response_correction_event": map[string]any{"corrected_agent_response": "Lo"}})
	send(ctx, socket, map[string]any{"type": "agent_response_complete"})
	send(ctx, socket, map[string]any{"type": "client_error", "error_event": map[string]any{"error_name": "bad"}})
	if failure := next(t, events).(realtime.SessionError); !failure.Recoverable || !strings.Contains(failure.Err.Error(), "bad") {
		t.Fatalf("client error %+v", failure)
	}
	send(ctx, socket, map[string]any{"type": "client_error"})
	if failure := next(t, events).(realtime.SessionError); !strings.Contains(failure.Err.Error(), "unknown error") {
		t.Fatalf("client error %+v", failure)
	}
}

func TestAudioRepliesEndWithTheirLastAudio(t *testing.T) {
	fake := newFakeAgent(t, agentConfig(nil, nil))
	_, events := connect(t, fake.model(elevenlabs.WithSettings(elevenlabs.Settings{KeepAudioTags: true, WarningHandler: func(string) {}})), realtime.ConnectParams{})
	socket := <-fake.sockets
	ctx := t.Context()
	audio := func(eventID int) {
		send(ctx, socket, map[string]any{"type": "audio", "audio_event": map[string]any{"audio_base_64": "AAAA", "event_id": eventID}})
	}
	reply := func(text string, eventID int) {
		send(ctx, socket, map[string]any{"type": "agent_response", "agent_response_event": map[string]any{"agent_response": text, "event_id": eventID}})
	}
	audio(1)
	reply("[laughs] One", 1)
	audio(2)
	next(t, events)
	if transcript := next(t, events).(realtime.OutputTranscript); transcript.Text != "[laughs] One" {
		t.Fatalf("tags were stripped: %q", transcript.Text)
	}
	if done := next(t, events).(realtime.ResponseDone); done.ProviderDetails["audio_tags"] != nil {
		t.Fatalf("audio of the next turn did not end the reply: %+v", done)
	}
	next(t, events)
	reply("Two", 2)
	next(t, events)
	send(ctx, socket, map[string]any{"type": "interruption"})
	send(ctx, socket, map[string]any{"type": "agent_response_correction", "agent_response_correction_event": map[string]any{"corrected_agent_response": "Tw"}})
	if done := next(t, events).(realtime.ResponseDone); !done.Interrupted || done.ProviderDetails["corrected_agent_response"] != "Tw" {
		t.Fatalf("correction did not end the reply: %+v", done)
	}
	audio(3)
	next(t, events)
	reply("Three", 3)
	next(t, events)
	// Without a frame that marks it, the reply ends after a second without audio.
	if done := next(t, events).(realtime.ResponseDone); done.Interrupted {
		t.Fatalf("grace end %+v", done)
	}
	send(ctx, socket, map[string]any{"type": "agent_response_correction", "agent_response_correction_event": map[string]any{"corrected_agent_response": "x"}})
	send(ctx, socket, map[string]any{"type": "interruption"})
	send(ctx, socket, map[string]any{"type": "queue_status", "queue_status_event": map[string]any{"status": "waiting"}})
	audio(4)
	send(ctx, socket, map[string]any{"type": "queue_status", "queue_status_event": map[string]any{"status": "admitted"}})
	audio(5)
	if delta := next(t, events).(realtime.AudioDelta); len(delta.Data) != 3 {
		t.Fatalf("hold audio was streamed: %+v", delta)
	}
	send(ctx, socket, map[string]any{"type": "audio", "audio_event": map[string]any{"audio_base_64": "%%%"}})
	if failure := next(t, events).(realtime.SessionError); !failure.Recoverable {
		t.Fatalf("malformed audio %+v", failure)
	}
	_ = socket.Write(ctx, websocket.MessageText, []byte("not json"))
	if failure := next(t, events).(realtime.SessionError); !failure.Recoverable {
		t.Fatalf("malformed frame %+v", failure)
	}
	_ = socket.Write(ctx, websocket.MessageBinary, []byte("ignored"))
	send(ctx, socket, map[string]any{"type": "ping", "ping_event": map[string]any{"event_id": 2}})
	if _, data, _ := socket.Read(ctx); !strings.Contains(string(data), `"pong"`) {
		t.Fatalf("ping not answered: %s", data)
	}
	for _, frame := range []string{"queue_status", "audio", "client_tool_call", "mcp_tool_call", "context_usage",
		"agent_response", "agent_response_correction", "user_transcript"} {
		send(ctx, socket, map[string]any{"type": frame})
		if failure := next(t, events).(realtime.SessionError); !failure.Recoverable {
			t.Fatalf("%s without payload %+v", frame, failure)
		}
	}
	_ = socket.Close(4300, "queue timed out")
	for {
		event := next(t, events)
		if done, ok := event.(realtime.ResponseDone); ok && !done.Interrupted {
			t.Fatalf("lost response not interrupted %+v", done)
		}
		if failure, ok := event.(realtime.SessionError); ok {
			if failure.Recoverable || !strings.Contains(failure.Err.Error(), "concurrency queue") {
				t.Fatalf("queue timeout %+v", failure)
			}
			break
		}
	}
}

func TestMCPCallsAreNativeParts(t *testing.T) {
	fake := newFakeAgent(t, agentConfig(nil, nil))
	var warnings []string
	_, events := connect(t, fake.model(elevenlabs.WithSettings(elevenlabs.Settings{
		WarningHandler: func(message string) { warnings = append(warnings, message) },
	})), realtime.ConnectParams{})
	socket := <-fake.sockets
	ctx := t.Context()
	mcp := func(id, state string, extra map[string]any) {
		call := map[string]any{"service_id": "docs", "tool_call_id": id, "tool_name": "search", "state": state}
		for key, value := range extra {
			call[key] = value
		}
		send(ctx, socket, map[string]any{"type": "mcp_tool_call", "mcp_tool_call": call})
	}
	mcp("m1", "loading", map[string]any{"parameters": map[string]any{"q": "go"}})
	mcp("m1", "success", map[string]any{"result": []any{"found"}})
	mcp("m2", "awaiting_approval", nil)
	mcp("m3", "awaiting_approval", nil)
	mcp("m3", "failure", map[string]any{"error_message": "denied upstream"})
	mcp("m4", "failure", map[string]any{"error_message": "broken"})
	var parts []ai.ResponsePart
	for len(parts) < 7 {
		if ended, ok := next(t, events).(realtime.PartEnded); ok {
			parts = append(parts, ended.Event.Part)
		}
	}
	call := parts[0].(ai.NativeToolCallPart)
	if call.ToolName != "mcp_server:docs" || !strings.Contains(string(call.Args), `"tool_args":{"q":"go"}`) ||
		parts[1].(ai.NativeToolReturnPart).Outcome != ai.ToolReturnOutcomeSuccess ||
		parts[4].(ai.NativeToolReturnPart).Outcome != ai.ToolReturnOutcomeDenied ||
		parts[6].(ai.NativeToolReturnPart).Outcome != ai.ToolReturnOutcomeFailed {
		t.Fatalf("parts %+v", parts)
	}
	for range 2 {
		_, data, _ := socket.Read(ctx)
		if !strings.Contains(string(data), `"is_approved":false`) {
			t.Fatalf("approval not denied: %s", data)
		}
	}
	if len(warnings) != 1 {
		t.Fatalf("warnings %v", warnings)
	}
	send(ctx, socket, map[string]any{"type": "agent_response_complete"})
	var denied ai.NativeToolReturnPart
	for {
		event := next(t, events)
		if ended, ok := event.(realtime.PartEnded); ok {
			denied = ended.Event.Part.(ai.NativeToolReturnPart)
		}
		if _, ok := event.(realtime.ResponseDone); ok {
			break
		}
	}
	if denied.ToolCallID != "m2" || denied.Outcome != ai.ToolReturnOutcomeDenied {
		t.Fatalf("pending denial %+v", denied)
	}
	mcp("m5", "awaiting_approval", nil)
	_, _, _ = socket.Read(ctx)
	_ = socket.CloseNow()
	var lost []realtime.CodecEvent
	for event := range events {
		lost = append(lost, event)
	}
	if failure, ok := lost[len(lost)-1].(realtime.SessionError); !ok || failure.Recoverable {
		t.Fatalf("drop %+v", lost)
	}
}

func TestToolSyncModes(t *testing.T) {
	differing := map[string]any{"type": "client", "name": "lookup", "description": "Old.", "parameters": map[string]any{"type": "object"}, "response_timeout_secs": 40}
	extra := map[string]any{"type": "client", "name": "stale", "description": "", "parameters": map[string]any{"type": "object"}}
	fake := newFakeAgent(t, agentConfig([]map[string]any{differing, extra}, []string{"tool_lookup", "tool_stale", "tool_webhook", "tool_unknown"}))
	fake.workspace = []map[string]any{
		{"id": "tool_lookup", "tool_config": differing},
		{"id": "tool_stale", "tool_config": extra},
		{"id": "tool_webhook", "tool_config": map[string]any{"type": "webhook", "name": "hook"}},
		{"id": "tool_orphan", "tool_config": lookupRemote},
	}
	params := realtime.ConnectParams{Request: ai.ModelRequestParams{Tools: []ai.ToolDefinition{lookupTool}}}
	_, err := fake.model().Connect(t.Context(), params)
	if err == nil || !strings.Contains(err.Error(), `tool "lookup" differs in description`) ||
		!strings.Contains(err.Error(), `agent client tool "stale" is not defined`) {
		t.Fatalf("error mode: %v", err)
	}
	sync := func(settings elevenlabs.Settings) (realtime.Connection, error) {
		settings.WarningHandler = func(string) {}
		return fake.model(elevenlabs.WithSettings(settings)).Connect(t.Context(), params)
	}
	connection, err := sync(elevenlabs.Settings{ToolSync: elevenlabs.ToolSyncSync})
	if err != nil {
		t.Fatal(err)
	}
	_ = connection.Close(t.Context())
	<-fake.initiated
	patched := fake.bodies["PATCH /v1/convai/tools/tool_lookup"]
	repointed := fake.bodies["PATCH /v1/convai/agents/agent_1"]
	if len(patched) != 1 || patched[0]["tool_config"].(map[string]any)["response_timeout_secs"] != 40.0 || len(repointed) != 1 {
		t.Fatalf("sync wrote %v %v", patched, repointed)
	}
	ids := repointed[0]["conversation_config"].(map[string]any)["agent"].(map[string]any)["prompt"].(map[string]any)["tool_ids"]
	if encoded, _ := json.Marshal(ids); string(encoded) != `["tool_webhook","tool_unknown","tool_lookup"]` {
		t.Fatalf("re-pointed %s", encoded)
	}

	fake.dependent["tool_lookup"] = []string{"agent_other"}
	if connection, err = sync(elevenlabs.Settings{ToolSync: elevenlabs.ToolSyncSync}); err != nil {
		t.Fatal(err)
	}
	_ = connection.Close(t.Context())
	<-fake.initiated
	if created := fake.bodies["POST /v1/convai/tools"]; len(created) != 1 {
		t.Fatalf("a shared tool was not copied: %v", created)
	}

	fake.agent = agentConfig(nil, []string{"tool_webhook"})
	if connection, err = sync(elevenlabs.Settings{ToolSync: elevenlabs.ToolSyncSync}); err != nil {
		t.Fatal(err)
	}
	_ = connection.Close(t.Context())
	<-fake.initiated
	if created := fake.bodies["POST /v1/convai/tools"]; len(created) != 1 {
		t.Fatalf("an orphaned matching tool was not adopted: %v", created)
	}
	fake.dependent["tool_orphan"] = []string{"agent_other"}
	if connection, err = sync(elevenlabs.Settings{ToolSync: elevenlabs.ToolSyncSync}); err != nil {
		t.Fatal(err)
	}
	_ = connection.Close(t.Context())
	<-fake.initiated
	if created := fake.bodies["POST /v1/convai/tools"]; len(created) != 2 {
		t.Fatalf("a used orphan was adopted: %v", created)
	}

	undescribed := ai.ToolDefinition{Name: "bad", Schema: map[string]any{"type": "object", "properties": map[string]any{
		"x": map[string]any{"type": "string"}, "y": map[string]any{"anyOf": []any{map[string]any{"type": "string"}, map[string]any{"type": "integer"}}, "description": "Y"},
		"z": map[string]any{"type": "array", "description": "Z", "items": map[string]any{"type": "string"}},
	}}}
	writes := func() int {
		return len(fake.bodies["POST /v1/convai/tools"]) + len(fake.bodies["PATCH /v1/convai/agents/agent_1"])
	}
	before := writes()
	_, err = fake.model(elevenlabs.WithSettings(elevenlabs.Settings{ToolSync: elevenlabs.ToolSyncSync, WarningHandler: func(string) {}})).Connect(t.Context(),
		realtime.ConnectParams{Request: ai.ModelRequestParams{Tools: []ai.ToolDefinition{undescribed}}})
	if err == nil || !strings.Contains(err.Error(), `"x" has none`) || !strings.Contains(err.Error(), `parameter "y"`) ||
		!strings.Contains(err.Error(), `"z[]" has none`) || writes() != before {
		t.Fatalf("unsyncable tool: %v (%d writes)", err, writes()-before)
	}
	if connection, err = sync(elevenlabs.Settings{ToolSync: elevenlabs.ToolSyncOff}); err != nil {
		t.Fatal(err)
	}
	_ = connection.Close(t.Context())
	<-fake.initiated
	if _, err = sync(elevenlabs.Settings{ToolSync: "bogus"}); err == nil {
		t.Fatal("unknown sync mode accepted")
	}
}

func TestToolChoiceRestrictsAttachedTools(t *testing.T) {
	fake := newFakeAgent(t, agentConfig([]map[string]any{lookupRemote}, []string{"tool_lookup", "tool_webhook"}, "agent.prompt.tool_ids"))
	fake.workspace = []map[string]any{
		{"id": "tool_lookup", "tool_config": lookupRemote},
		{"id": "tool_webhook", "tool_config": map[string]any{"type": "webhook", "name": "hook"}},
	}
	params := realtime.ConnectParams{
		Request:  ai.ModelRequestParams{Tools: []ai.ToolDefinition{lookupTool}},
		Settings: realtime.Settings{ToolChoice: realtime.ToolChoiceNone},
	}
	connection, err := fake.model().Connect(t.Context(), params)
	if err != nil {
		t.Fatal(err)
	}
	_ = connection.Close(t.Context())
	initiation := <-fake.initiated
	ids := initiation["conversation_config_override"].(map[string]any)["agent"].(map[string]any)["prompt"].(map[string]any)["tool_ids"]
	if encoded, _ := json.Marshal(ids); string(encoded) != `["tool_webhook"]` {
		t.Fatalf("restricted %s", encoded)
	}
	params.Settings.ToolChoice = realtime.ToolChoiceRequired
	if _, err := fake.model().Connect(t.Context(), params); err == nil {
		t.Fatal("forced tool choice accepted")
	}
	params.Settings.ToolChoice = realtime.ToolChoiceNone
	connection, err = fake.model(elevenlabs.WithSettings(elevenlabs.Settings{ToolSync: elevenlabs.ToolSyncSync, WarningHandler: func(string) {}})).Connect(t.Context(), params)
	if err != nil {
		t.Fatal(err)
	}
	_ = connection.Close(t.Context())
	<-fake.initiated
}

func TestConnectRefusals(t *testing.T) {
	fake := newFakeAgent(t, agentConfig(nil, nil))
	empty := ""
	for name, test := range map[string]struct {
		model    *elevenlabs.Model
		params   realtime.ConnectParams
		contains string
	}{
		"instructions":   {fake.model(), realtime.ConnectParams{Request: ai.ModelRequestParams{Instructions: "x"}}, "agent.prompt.prompt"},
		"text":           {fake.model(), realtime.ConnectParams{Settings: realtime.Settings{OutputModality: realtime.OutputModalityText}}, "conversation.text_only"},
		"reconnect":      {fake.model(), realtime.ConnectParams{Settings: realtime.Settings{Reconnect: &realtime.ReconnectPolicy{}}}, "reconnect"},
		"turn off":       {fake.model(), realtime.ConnectParams{Settings: realtime.Settings{TurnDetection: &realtime.TurnDetection{}}}, "always on"},
		"turn tuned":     {fake.model(), realtime.ConnectParams{Settings: realtime.Settings{TurnDetection: &realtime.TurnDetection{Enabled: true, Sensitivity: "low"}}}, "cannot be configured"},
		"transcription":  {fake.model(), realtime.ConnectParams{Settings: realtime.Settings{InputTranscriptionModel: &empty}}, "cannot be disabled"},
		"seeding":        {fake.model(), realtime.ConnectParams{Messages: []ai.ModelMessage{ai.ModelRequest{}}}, "seed"},
		"agent":          {elevenlabs.NewModel("", elevenlabs.WithAPIKey("key")), realtime.ConnectParams{}, "agent ID"},
		"key":            {elevenlabs.NewModel("agent_1", elevenlabs.WithAPIKey("")), realtime.ConnectParams{}, "ELEVENLABS_API_KEY"},
		"bad key":        {elevenlabs.NewModel("agent_1", elevenlabs.WithAPIKey("wrong"), elevenlabs.WithBaseURL(fake.server.URL)), realtime.ConnectParams{}, "status 401"},
		"unreachable":    {elevenlabs.NewModel("agent_1", elevenlabs.WithAPIKey("key"), elevenlabs.WithBaseURL("http://127.0.0.1:1")), realtime.ConnectParams{}, "could not reach"},
		"custom llm":     {fake.model(elevenlabs.WithSettings(elevenlabs.Settings{CustomLLMExtraBody: map[string]any{}, LLM: "x"})), realtime.ConnectParams{}, "agent.prompt.llm"},
		"invalid prefix": {fake.model(), realtime.ConnectParams{Settings: realtime.Settings{Provider: map[string]any{"elevenlabs_agents_tool_sync": "bogus"}}}, "unknown tool sync"},
	} {
		if _, err := test.model.Connect(t.Context(), test.params); err == nil || !strings.Contains(err.Error(), test.contains) {
			t.Fatalf("%s: %v", name, err)
		}
	}
	noExtra := agentConfig(nil, nil)
	noExtra["platform_settings"].(map[string]any)["overrides"].(map[string]any)["custom_llm_extra_body"] = false
	fake.agent = noExtra
	if _, err := fake.model(elevenlabs.WithSettings(elevenlabs.Settings{CustomLLMExtraBody: map[string]any{}})).Connect(t.Context(), realtime.ConnectParams{}); err == nil ||
		!strings.Contains(err.Error(), "custom_llm_extra_body") {
		t.Fatalf("custom llm extra body: %v", err)
	}

	fake.agent = agentConfig(nil, nil)
	fake.metadata["user_input_audio_format"] = "ulaw_8000"
	if _, err := fake.model().Connect(t.Context(), realtime.ConnectParams{}); err == nil || !strings.Contains(err.Error(), "ulaw_8000") {
		t.Fatalf("audio format: %v", err)
	}
	<-fake.initiated
	fake.metadata["user_input_audio_format"] = "pcm_24000"
	fake.metadata["agent_output_audio_format"] = "pcm_24000"
	connection, err := fake.model(elevenlabs.WithProfile(realtime.ProfileOverride{AudioInputSampleRate: 24000, AudioOutputSampleRate: 24000})).Connect(t.Context(), realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	_ = connection.Close(t.Context())
	<-fake.initiated
	fake.rejectWS = http.StatusForbidden
	var apiErr *elevenlabs.APIError
	if _, err := fake.model().Connect(t.Context(), realtime.ConnectParams{}); !errors.As(err, &apiErr) || apiErr.StatusCode != 403 || !apiErr.IsModelAPIError() {
		t.Fatalf("rejected upgrade: %v", err)
	}
}

func TestPreflightResolvesAttachedToolsAndWarnsWithoutContextUsage(t *testing.T) {
	config := agentConfig(nil, []string{"tool_lookup", "tool_gone"})
	config["conversation_config"].(map[string]any)["conversation"] = map[string]any{"client_events": []string{"audio"}}
	fake := newFakeAgent(t, config)
	fake.workspace = []map[string]any{{"id": "tool_lookup", "tool_config": lookupRemote}}
	var warnings []string
	model := fake.model(elevenlabs.WithSettings(elevenlabs.Settings{WarningHandler: func(message string) { warnings = append(warnings, message) }}))
	for range 2 {
		connection, err := model.Connect(t.Context(), realtime.ConnectParams{Request: ai.ModelRequestParams{Tools: []ai.ToolDefinition{lookupTool}}})
		if err != nil {
			t.Fatal(err)
		}
		_ = connection.Close(t.Context())
		<-fake.initiated
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0], "context_usage") {
		t.Fatalf("warnings %v", warnings)
	}
}

func TestAudioTagsAreStrippedAcrossStreamedPieces(t *testing.T) {
	fake := newFakeAgent(t, agentConfig(nil, nil, "conversation.text_only"))
	_, events := connect(t, fake.model(), realtime.ConnectParams{
		Settings: realtime.Settings{OutputModality: realtime.OutputModalityText},
	})
	socket := <-fake.sockets
	ctx := t.Context()
	whole := "[strong French accent] Hi [laughs] there, see [1], [x], [Your Name] and [text](https://example.com) [sighs]"
	for _, piece := range []string{"[strong French", " accent] Hi ", "[lau", "ghs] there, see [", "1], [x], [Your Name] and [text]", "(https://example.com) [sig", "hs]"} {
		send(ctx, socket, map[string]any{"type": "agent_chat_response_part", "text_response_part": map[string]any{"type": "delta", "text": piece}})
	}
	send(ctx, socket, map[string]any{"type": "agent_response", "agent_response_event": map[string]any{"agent_response": whole}})
	var streamed string
	for {
		switch event := next(t, events).(type) {
		case realtime.OutputTranscript:
			if !event.Final {
				streamed += event.Text
				continue
			}
			want := "Hi there, see [1], [x], [Your Name] and [text](https://example.com)"
			if event.Text != want || streamed != want {
				t.Fatalf("final %q, streamed %q", event.Text, streamed)
			}
		case realtime.ResponseDone:
			if tags := strings.Join(event.ProviderDetails["audio_tags"].([]string), ","); tags != "strong French accent,laughs,sighs" {
				t.Fatalf("tags %q", tags)
			}
			return
		}
	}
}

func TestToolMismatchesCompareTheStoredDialect(t *testing.T) {
	remote := func(change func(parameters map[string]any)) map[string]any {
		parameters := map[string]any{"type": "object", "required": []any{"a"}, "properties": map[string]any{
			"a": map[string]any{"type": "string", "description": "A", "enum": []any{"x"}},
			"l": map[string]any{"type": "array", "description": "L", "items": map[string]any{"type": "string", "description": "I"}},
			"n": map[string]any{"type": "string", "description": "N"},
		}}
		if change != nil {
			change(parameters)
		}
		return map[string]any{"type": "client", "name": "t", "description": "T", "parameters": parameters}
	}
	local := ai.ToolDefinition{Name: "t", Description: "T", Schema: map[string]any{
		"type": "object", "required": []string{"a"}, "properties": map[string]any{
			"a": map[string]any{"type": "string", "description": "A", "enum": []string{"x"}},
			"l": map[string]any{"type": "array", "description": "L", "items": map[string]any{"type": "string"}},
			"n": map[string]any{"anyOf": []any{map[string]any{"type": "string"}, map[string]any{"type": "null"}}},
		},
	}}
	property := func(parameters map[string]any, name string) map[string]any {
		return parameters["properties"].(map[string]any)[name].(map[string]any)
	}
	for name, test := range map[string]struct {
		change func(map[string]any)
		match  bool
	}{
		"same":        {nil, true},
		"type":        {func(p map[string]any) { property(p, "a")["type"] = "integer" }, false},
		"enum":        {func(p map[string]any) { property(p, "a")["enum"] = []any{"y"} }, false},
		"enum absent": {func(p map[string]any) { property(p, "a")["enum"] = nil }, false},
		"description": {func(p map[string]any) { property(p, "a")["description"] = "B" }, false},
		"required":    {func(p map[string]any) { p["required"] = []any{} }, false},
		"missing":     {func(p map[string]any) { delete(p["properties"].(map[string]any), "a") }, false},
		"renamed": {func(p map[string]any) {
			p["properties"].(map[string]any)["b"] = property(p, "a")
			delete(p["properties"].(map[string]any), "a")
		}, false},
		"items":        {func(p map[string]any) { delete(property(p, "l"), "items") }, false},
		"remote items": {func(p map[string]any) { property(p, "n")["items"] = map[string]any{"type": "string"} }, false},
		"no response":  {func(map[string]any) {}, false},
	} {
		config := remote(test.change)
		if name == "no response" {
			config["expects_response"] = false
		}
		fake := newFakeAgent(t, agentConfig([]map[string]any{config}, nil))
		connection, err := fake.model().Connect(t.Context(), realtime.ConnectParams{Request: ai.ModelRequestParams{Tools: []ai.ToolDefinition{local}}})
		if (err == nil) != test.match {
			t.Fatalf("%s: %v", name, err)
		}
		if connection != nil {
			_ = connection.Close(t.Context())
		}
	}
}

func TestRESTFailuresSurface(t *testing.T) {
	stale := map[string]any{"type": "client", "name": "lookup", "description": "Old.", "parameters": map[string]any{"type": "object"}}
	for _, step := range []string{
		"GET /v1/convai/agents/agent_1", "GET /v1/convai/tools/tool_lookup", "GET /v1/convai/tools", "PATCH /v1/convai/agents/agent_1",
		"GET /v1/convai/tools/tool_lookup/dependent-agents", "PATCH /v1/convai/tools/tool_lookup",
		"PATCH /v1/convai/agents/agent_1", "GET /v1/convai/conversation/get-signed-url",
	} {
		for _, mode := range []string{"fail", "malform"} {
			fake := newFakeAgent(t, agentConfig(nil, []string{"tool_lookup"}))
			fake.workspace = []map[string]any{{"id": "tool_lookup", "tool_config": stale}}
			if mode == "fail" {
				fake.failing[step] = true
			} else {
				fake.malform[step] = true
			}
			model := fake.model(elevenlabs.WithSettings(elevenlabs.Settings{ToolSync: elevenlabs.ToolSyncSync, WarningHandler: func(string) {}}))
			_, err := model.Connect(t.Context(), realtime.ConnectParams{Request: ai.ModelRequestParams{Tools: []ai.ToolDefinition{lookupTool}}})
			if err == nil && !strings.HasPrefix(step, "PATCH") {
				t.Fatalf("%s %s: connected", mode, step)
			}
		}
	}
	for _, step := range []string{"GET /v1/convai/tools/tool_lookup/dependent-agents", "POST /v1/convai/tools", "PATCH /v1/convai/agents/agent_1"} {
		fake := newFakeAgent(t, agentConfig([]map[string]any{stale}, []string{"tool_lookup"}))
		fake.workspace = []map[string]any{{"id": "tool_lookup", "tool_config": stale}, {"id": "tool_orphan", "tool_config": lookupRemote}}
		fake.dependent["tool_lookup"] = []string{"agent_other"}
		fake.failing[step] = true
		model := fake.model(elevenlabs.WithSettings(elevenlabs.Settings{ToolSync: elevenlabs.ToolSyncSync, WarningHandler: func(string) {}}))
		if _, err := model.Connect(t.Context(), realtime.ConnectParams{Request: ai.ModelRequestParams{Tools: []ai.ToolDefinition{lookupTool}}}); err == nil {
			t.Fatalf("%s: connected", step)
		}
		fake.agent = agentConfig(nil, nil)
		switch step {
		case "POST /v1/convai/tools":
			fake.dependent["tool_orphan"] = []string{"agent_other"}
		case "GET /v1/convai/tools/tool_lookup/dependent-agents":
			fake.failing["GET /v1/convai/tools/tool_orphan/dependent-agents"] = true
		}
		if _, err := model.Connect(t.Context(), realtime.ConnectParams{Request: ai.ModelRequestParams{Tools: []ai.ToolDefinition{lookupTool}}}); err == nil {
			t.Fatalf("%s while adopting: connected", step)
		}
	}
	fake := newFakeAgent(t, agentConfig([]map[string]any{lookupRemote}, []string{"tool_lookup"}, "agent.prompt.tool_ids"))
	fake.failing["GET /v1/convai/tools"] = true
	if _, err := fake.model().Connect(t.Context(), realtime.ConnectParams{
		Request:  ai.ModelRequestParams{Tools: []ai.ToolDefinition{lookupTool}},
		Settings: realtime.Settings{ToolChoice: realtime.ToolChoiceNone},
	}); err == nil {
		t.Fatal("restricting without a tool listing connected")
	}
	fake = newFakeAgent(t, agentConfig(nil, nil))
	fake.malform["GET /v1/convai/conversation/get-signed-url"] = false
	empty := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if strings.HasSuffix(request.URL.Path, "get-signed-url") {
			_, _ = writer.Write([]byte(`{}`))
			return
		}
		fake.serve(writer, request)
	}))
	defer empty.Close()
	if _, err := elevenlabs.NewModel("agent_1", elevenlabs.WithAPIKey("key"), elevenlabs.WithBaseURL(empty.URL),
		elevenlabs.WithHTTPClient(empty.Client()), elevenlabs.WithSettings(elevenlabs.Settings{WarningHandler: func(string) {}})).Connect(t.Context(), realtime.ConnectParams{}); err == nil ||
		!strings.Contains(err.Error(), "no URL") {
		t.Fatalf("empty signed URL: %v", err)
	}
	invalid := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if strings.HasSuffix(request.URL.Path, "get-signed-url") {
			_, _ = writer.Write([]byte(`{"signed_url":"ws://127.0.0.1:1/ws"}`))
			return
		}
		fake.serve(writer, request)
	}))
	defer invalid.Close()
	if _, err := elevenlabs.NewModel("agent_1", elevenlabs.WithAPIKey("key"), elevenlabs.WithBaseURL(invalid.URL),
		elevenlabs.WithHTTPClient(invalid.Client()), elevenlabs.WithSettings(elevenlabs.Settings{WarningHandler: func(string) {}})).Connect(t.Context(), realtime.ConnectParams{}); err == nil ||
		!strings.Contains(err.Error(), "dial conversation") {
		t.Fatalf("unreachable socket: %v", err)
	}
	if _, err := elevenlabs.NewModel("agent_1", elevenlabs.WithAPIKey("key"), elevenlabs.WithBaseURL("://")).Connect(t.Context(), realtime.ConnectParams{}); err == nil {
		t.Fatal("invalid base URL connected")
	}
}

func TestHandshakeFailures(t *testing.T) {
	for name, handshake := range map[string]func(context.Context, *websocket.Conn){
		"close": func(context.Context, *websocket.Conn) {},
		"binary": func(ctx context.Context, socket *websocket.Conn) {
			_ = socket.Write(ctx, websocket.MessageBinary, []byte("x"))
		},
		"malformed": func(ctx context.Context, socket *websocket.Conn) {
			_ = socket.Write(ctx, websocket.MessageText, []byte("x"))
		},
		"error": func(ctx context.Context, socket *websocket.Conn) {
			send(ctx, socket, map[string]any{"type": "client_error"})
		},
		"pong": func(ctx context.Context, socket *websocket.Conn) {
			send(ctx, socket, map[string]any{"type": "ping", "ping_event": map[string]any{}})
			_ = socket.CloseNow()
		},
	} {
		fake := newFakeAgent(t, agentConfig(nil, nil))
		fake.handshake = handshake
		if _, err := fake.model().Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{HandshakeTimeout: time.Second}}); err == nil ||
			!strings.Contains(err.Error(), "handshake") {
			t.Fatalf("%s: %v", name, err)
		}
	}
}

func TestDefaultWarningsGoToSlog(t *testing.T) {
	var logged strings.Builder
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logged, nil)))
	defer slog.SetDefault(previous)
	config := agentConfig(nil, nil)
	config["conversation_config"].(map[string]any)["conversation"] = map[string]any{}
	fake := newFakeAgent(t, config)
	connection, err := elevenlabs.NewModel("agent_1", elevenlabs.WithAPIKey("key"), elevenlabs.WithBaseURL(fake.server.URL),
		elevenlabs.WithHTTPClient(fake.server.Client())).Connect(t.Context(), realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	_ = connection.Close(t.Context())
	<-fake.initiated
	if !strings.Contains(logged.String(), "context_usage") {
		t.Fatalf("warning not logged: %q", logged.String())
	}
}

func TestEventStreamEdges(t *testing.T) {
	fake := newFakeAgent(t, agentConfig(nil, nil))
	delete(fake.metadata, "conversation_id")
	connection, err := fake.model().Connect(t.Context(), realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	socket := <-fake.sockets
	ctx := t.Context()
	// A consumer that stops takes no further events.
	send(ctx, socket, map[string]any{"type": "audio", "audio_event": map[string]any{"audio_base_64": "AAAA", "event_id": 1}})
	send(ctx, socket, map[string]any{"type": "agent_response", "agent_response_event": map[string]any{"agent_response": "Hi", "event_id": 1}})
	send(ctx, socket, map[string]any{"type": "user_transcript", "user_transcription_event": map[string]any{"user_transcript": "Next"}})
	var seen []realtime.CodecEvent
	for event := range connection.Events(ctx) {
		seen = append(seen, event)
		if _, ok := event.(realtime.ResponseDone); ok {
			break
		}
	}
	if done := seen[len(seen)-1].(realtime.ResponseDone); done.ProviderDetails != nil {
		t.Fatalf("details without a conversation %+v", done)
	}
	send(ctx, socket, map[string]any{"type": "audio", "audio_event": map[string]any{"audio_base_64": "AAAA", "event_id": 2}})
	send(ctx, socket, map[string]any{"type": "agent_response", "agent_response_event": map[string]any{"agent_response": "Hi", "event_id": 2}})
	for event := range connection.Events(ctx) {
		if _, ok := event.(realtime.OutputTranscript); ok {
			break
		}
	}
	// The grace end is the one event its timer emits; a consumer stopping there ends the stream.
	for range connection.Events(ctx) {
		break
	}
	send(ctx, socket, map[string]any{"type": "client_tool_call", "client_tool_call": map[string]any{"tool_name": "t", "tool_call_id": "c"}})
	for range connection.Events(ctx) {
		break
	}
	_ = socket.CloseNow()
	var lost []realtime.CodecEvent
	for event := range connection.Events(ctx) {
		lost = append(lost, event)
	}
	if done, ok := lost[0].(realtime.ResponseDone); !ok || !done.MoreExpected {
		t.Fatalf("calls open at a drop did not end their response: %+v", lost)
	}
	_ = connection.Close(ctx)
	for range connection.Events(ctx) {
		t.Fatal("events after close")
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	for range connection.Events(canceled) {
		t.Fatal("events after cancel")
	}
}

func TestSyncRefusalsCostNoWrites(t *testing.T) {
	fake := newFakeAgent(t, agentConfig(nil, nil))
	sync := fake.model(elevenlabs.WithSettings(elevenlabs.Settings{ToolSync: elevenlabs.ToolSyncSync, WarningHandler: func(string) {}}))
	if _, err := sync.Connect(t.Context(), realtime.ConnectParams{Request: ai.ModelRequestParams{
		Instructions: "x", Tools: []ai.ToolDefinition{lookupTool},
	}}); err == nil || !strings.Contains(err.Error(), "agent.prompt.prompt") || len(fake.bodies) != 0 {
		t.Fatalf("refused override synced first: %v %v", err, fake.bodies)
	}
	for name, schema := range map[string]map[string]any{
		"union":     {"anyOf": []any{map[string]any{"type": "string"}, map[string]any{"type": "integer"}}, "description": "U"},
		"two nulls": {"anyOf": []any{map[string]any{"type": "null"}, map[string]any{"type": "null"}}, "description": "N"},
		"one":       {"anyOf": []any{map[string]any{"type": "string"}}, "description": "O"},
	} {
		tool := ai.ToolDefinition{Name: "t", Schema: map[string]any{"type": "object", "properties": map[string]any{"p": schema}}}
		if _, err := sync.Connect(t.Context(), realtime.ConnectParams{Request: ai.ModelRequestParams{Tools: []ai.ToolDefinition{tool}}}); err == nil ||
			!strings.Contains(err.Error(), "cannot express") {
			t.Fatalf("%s: %v", name, err)
		}
	}
	nullable := ai.ToolDefinition{Name: "lookup", Description: "Look up a city.", Schema: map[string]any{
		"type": "object", "required": []any{"city"}, "properties": map[string]any{"city": map[string]any{
			"description": "The city.", "anyOf": []any{map[string]any{"type": "null"}, map[string]any{"type": "string"}},
		}},
	}}
	fake.agent = agentConfig([]map[string]any{lookupRemote}, nil)
	connection, err := fake.model().Connect(t.Context(), realtime.ConnectParams{Request: ai.ModelRequestParams{Tools: []ai.ToolDefinition{nullable}}})
	if err != nil {
		t.Fatalf("a nullable parameter does not match its stored type: %v", err)
	}
	_ = connection.Close(t.Context())
	<-fake.initiated
}
