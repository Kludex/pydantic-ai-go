package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

func TestLiveRejectUnsupportedSettings(t *testing.T) {
	empty := ""
	for _, common := range []realtime.Settings{{TurnDetection: &realtime.TurnDetection{}}, {MaxTokens: 1}, {InputTranscriptionModel: &empty}, {OutputModality: realtime.OutputModalityText}, {ToolChoice: realtime.ToolChoiceRequired}, {ToolChoice: "bogus"}, {Reconnect: &realtime.ReconnectPolicy{MaxAttempts: -1}}} {
		if _, err := openairt.NewLiveModel("gpt-live-1").Connect(t.Context(), realtime.ConnectParams{Settings: common}); err == nil {
			t.Fatalf("accepted %+v", common)
		}
	}
	for _, key := range []string{"openai_voice", "openai_live_instructions", "openai_live_delegation", "openai_live_data_channel", "openai_live_store", "openai_live_turn_silence_ms", "openai_turn_detection", "openai_truncation", "openai_input_noise_reduction", "openai_output_speed"} {
		if _, err := openairt.NewLiveModel("gpt-live-1").Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{Provider: map[string]any{key: []any{}}}}); err == nil {
			t.Fatalf("accepted %s", key)
		}
	}
	for _, options := range [][]openairt.Option{{openairt.WithAPIKey("")}, {openairt.WithAPIKey("key"), openairt.WithBaseURL("://")}, {openairt.WithAPIKey("key"), openairt.WithLiveSettings(openairt.LiveSettings{TurnSilence: -1})}, {openairt.WithAPIKey("key"), openairt.WithProfile(realtime.ProfileOverride{AudioInputSampleRate: 16000})}} {
		if _, err := openairt.NewLiveModel("gpt-live-1", options...).Connect(t.Context(), realtime.ConnectParams{}); err == nil {
			t.Fatal("invalid model accepted")
		}
	}
	if _, err := openairt.NewLiveModel("", openairt.WithAPIKey("key")).Connect(t.Context(), realtime.ConnectParams{}); err == nil {
		t.Fatal("empty model")
	}
	if _, err := openairt.NewLiveModel("gpt-live-1").Connect(t.Context(), realtime.ConnectParams{Request: ai.ModelRequestParams{NativeTools: []ai.NativeTool{ai.CodeExecutionTool{}}}}); err == nil {
		t.Fatal("unsupported native tool")
	}
	audio := ai.BinaryContent{Data: []byte{1, 0}, MediaType: "audio/pcm"}
	for _, message := range []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{audio}}}}, ai.ModelRequest{Parts: []ai.RequestPart{ai.SpeechPart{Audio: &audio}}}, ai.ModelResponse{Parts: []ai.ResponsePart{ai.SpeechPart{Audio: &audio}}}} {
		if _, err := openairt.NewLiveModel("gpt-live-1").Connect(t.Context(), realtime.ConnectParams{Messages: []ai.ModelMessage{message}}); err == nil {
			t.Fatal("media seeded")
		}
	}
}

func TestLiveConfigurationPrecedence(t *testing.T) {
	for _, test := range []struct {
		backend    string
		thinking   ai.ThinkingLevel
		delegation map[string]any
		expected   string
	}{
		{"gpt-5", ai.ThinkingLevelDisabled, nil, "none"}, {"gpt-6-sol", ai.ThinkingLevelMinimal, nil, "low"}, {"o3", ai.ThinkingLevelDisabled, nil, ""}, {"gpt-4.1", ai.ThinkingLevelHigh, nil, ""}, {"gpt-5", ai.ThinkingLevelEnabled, nil, "medium"}, {"gpt-5", ai.ThinkingLevelLow, map[string]any{"reasoning_effort": "high", "parallel_tool_calls": true}, "high"},
	} {
		t.Run(test.backend+string(test.thinking), func(t *testing.T) {
			delegation := test.delegation
			if delegation == nil {
				delegation = map[string]any{}
			}
			delegation["model"] = test.backend
			connection, peer := livePair(t, realtime.ConnectParams{Settings: realtime.Settings{Thinking: test.thinking, ToolChoice: realtime.ToolChoiceNone, Provider: map[string]any{"openai_live_delegation": delegation, "openai_live_turn_silence_ms": 1.5, "openai_live_data_channel": map[string]any{}, "openai_live_store": false, "unrelated": "ignored"}}, Request: ai.ModelRequestParams{Tools: []ai.ToolDefinition{{Name: "ignored"}}, NativeTools: []ai.NativeTool{ai.WebSearchTool{SearchContextSize: ai.WebSearchContextLow}}}}, openairt.WithSettings(openairt.Settings{Voice: "cedar"}), openairt.WithLiveSettings(openairt.LiveSettings{Delegation: map[string]any{"model": "auto"}}))
			responses := peer.start["delegation"].(map[string]any)["responses"].(map[string]any)
			got := ""
			if reasoning, ok := responses["reasoning"].(map[string]any); ok {
				got = reasoning["effort"].(string)
			}
			if got != test.expected {
				t.Fatalf("effort=%s expected=%s", got, test.expected)
			}
			_ = connection.Close(t.Context())
		})
	}
	connection, peer := livePair(t, realtime.ConnectParams{Settings: realtime.Settings{Provider: map[string]any{"openai_live_delegation": map[string]any{"model": "auto"}}}})
	if peer.start["delegation"].(map[string]any)["responses"].(map[string]any)["model"] != "gpt-6-sol" {
		t.Fatal("auto backend")
	}
	_ = connection.Close(t.Context())
}

func TestLiveHandshakeFailures(t *testing.T) {
	for _, test := range []string{"status", "error", "malformed", "json", "timeout", "closed"} {
		t.Run(test, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if test == "status" {
					w.WriteHeader(401)
					return
				}
				socket, err := websocket.Accept(w, r, nil)
				if err != nil {
					return
				}
				defer func() { _ = socket.CloseNow() }()
				_, _, err = socket.Read(r.Context())
				if err != nil {
					return
				}
				switch test {
				case "error":
					_ = writeFrame(r.Context(), socket, map[string]any{"type": "error", "error": map[string]any{"message": "refused"}})
				case "malformed":
					_ = writeFrame(r.Context(), socket, map[string]any{"type": "session.started", "session": map[string]any{}})
				case "json":
					_ = socket.Write(r.Context(), websocket.MessageText, []byte("invalid"))
				case "timeout":
					_, _, _ = socket.Read(r.Context())
				}
			}))
			defer server.Close()
			_, err := openairt.NewLiveModel("gpt-live-1", openairt.WithAPIKey("key"), openairt.WithBaseURL(server.URL)).Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{HandshakeTimeout: 10 * time.Millisecond}})
			if err == nil || err.Error() == "" {
				t.Fatal("handshake accepted")
			}
			if test == "error" || test == "malformed" {
				if errors.Unwrap(err) == nil {
					t.Fatal("not wrapped")
				}
			}
		})
	}
	_, err := openairt.NewLiveModel("gpt-live-1", openairt.WithAPIKey("key"), openairt.WithLiveSettings(openairt.LiveSettings{Delegation: map[string]any{"max_output_tokens": func() {}}})).Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{HandshakeTimeout: time.Millisecond}})
	if err == nil {
		t.Fatal("invalid JSON config")
	}
}

func TestLiveWebRTCAndHangUp(t *testing.T) {
	offers := make(chan map[string]any, 1)
	attached := make(chan *websocket.Conn, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/attach"):
			if r.URL.EscapedPath() != "/v1/live/sessions/session%2Fid/attach" {
				t.Errorf("attach path: %s", r.URL)
			}
			socket, err := websocket.Accept(w, r, nil)
			if err != nil {
				return
			}
			defer func() { _ = socket.CloseNow() }()
			_ = writeFrame(r.Context(), socket, map[string]any{"type": "session.started", "session": map[string]any{"id": "session/id", "model": "gpt-live-1", "delegation": map[string]any{"type": "responses", "responses": map[string]any{"model": "gpt-5"}}}})
			attached <- socket
			for {
				_, _, err = socket.Read(r.Context())
				if err != nil {
					return
				}
				t.Error("sideband sent configuration or session.close")
			}
		case strings.HasSuffix(r.URL.Path, "/hangup"):
			w.WriteHeader(204)
		default:
			var offer map[string]any
			_ = json.NewDecoder(r.Body).Decode(&offer)
			offers <- offer
			_, _ = io.WriteString(w, `{"session":{"id":"session/id"},"transport":{"sdp":"answer"}}`)
		}
	}))
	defer server.Close()
	model := openairt.NewLiveModel("gpt-live-1", openairt.WithAPIKey("key"), openairt.WithBaseURL(server.URL+"/v1"), openairt.WithLiveSettings(openairt.LiveSettings{TurnSilence: time.Millisecond, Voice: "marin"}), openairt.WithProfile(realtime.ProfileOverride{AudioInputSampleRate: 123}))
	answer, err := model.AnswerWebRTCOffer(t.Context(), "offer", "work", []ai.ToolDefinition{{Name: "lookup"}}, realtime.Settings{})
	if err != nil {
		t.Fatal(err)
	}
	if answer.SDP != "answer" || answer.Session.ID != "session/id" {
		t.Fatal(answer)
	}
	offer := <-offers
	config := offer["session"].(map[string]any)
	if _, ok := config["audio"].(map[string]any)["format"]; ok {
		t.Fatal("format on WebRTC")
	}
	dataChannel := config["client"].(map[string]any)["data_channel"].(map[string]any)
	if len(dataChannel["allowed_client_events"].([]any)) != 0 || len(dataChannel["allowed_server_events"].([]any)) != 0 {
		t.Fatal("open data channel")
	}
	connection, err := model.ConnectWebRTC(t.Context(), answer.Session, realtime.ConnectParams{})
	if err != nil {
		t.Fatal(err)
	}
	live := connection.(*openairt.LiveConnection)
	socket := <-attached
	_ = writeFrame(t.Context(), socket, map[string]any{"type": "session.output_audio.delta", "delta": "ZAA="})
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	for event, err := range live.Events(ctx) {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(realtime.AudioDelta); ok {
			t.Fatal("sideband forwarded audio")
		}
		if _, ok := event.(realtime.ResponseDone); ok {
			break
		}
	}
	for range live.EndSession(t.Context()) {
		t.Fatal("sideband ended call")
	}
	if err := live.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := model.HangUp(t.Context(), answer.Session); err != nil {
		t.Fatal(err)
	}
	ga := openairt.NewModel("gpt-realtime", openairt.WithAPIKey("key"), openairt.WithBaseURL(server.URL+"/v1"))
	if err := ga.HangUp(t.Context(), answer.Session); err != nil {
		t.Fatal(err)
	}
}

func TestLiveWebRTCValidation(t *testing.T) {
	if _, err := openairt.NewModel("gpt-realtime", openairt.WithAPIKey("key")).CreateClientSecret(t.Context(), "", nil, realtime.Settings{ToolChoice: realtime.ToolChoiceRequired}, 0); err == nil {
		t.Fatal("required persistent tool choice accepted")
	}
	model := openairt.NewLiveModel("gpt-live-1", openairt.WithAPIKey("key"))
	if _, err := model.AnswerWebRTCOffer(t.Context(), "", "", nil, realtime.Settings{}); err == nil {
		t.Fatal("empty offer")
	}
	for _, common := range []realtime.Settings{{MaxTokens: 1}, {Provider: map[string]any{"openai_live_store": "bad"}}} {
		if _, err := model.AnswerWebRTCOffer(t.Context(), "offer", "", nil, common); err == nil {
			t.Fatal("invalid settings")
		}
	}
	for _, session := range []realtime.ProviderSession{nil, realtime.WebRTCSession{Provider: "google", ID: "id"}, realtime.WebRTCSession{Provider: "openai"}} {
		if _, err := model.ConnectWebRTC(t.Context(), session, realtime.ConnectParams{}); err == nil {
			t.Fatal("invalid session")
		}
		if err := model.HangUp(t.Context(), session); err == nil {
			t.Fatal("invalid hangup")
		}
	}
	for _, params := range []realtime.ConnectParams{{Messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "seed"}}}}}, {Messages: []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: []ai.UserContent{ai.BinaryContent{MediaType: "image/png"}}}}}}}, {Settings: realtime.Settings{MaxTokens: 1}}, {Settings: realtime.Settings{Provider: map[string]any{"openai_live_store": "bad"}}}} {
		if _, err := model.ConnectWebRTC(t.Context(), realtime.WebRTCSession{Provider: "openai", ID: "id"}, params); err == nil {
			t.Fatal("invalid sideband")
		}
	}
	for _, body := range []string{`{}`, `not json`} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, body) }))
		bad := openairt.NewLiveModel("gpt-live-1", openairt.WithBaseURL(server.URL))
		if _, err := bad.AnswerWebRTCOffer(t.Context(), "offer", "", nil, realtime.Settings{}); err == nil {
			t.Fatal("invalid answer")
		}
		if _, err := bad.ConnectWebRTC(t.Context(), realtime.WebRTCSession{Provider: "openai", ID: "id"}, realtime.ConnectParams{}); err == nil {
			t.Fatal("invalid sideband handshake")
		}
		server.Close()
	}
}

func TestOpenAIHangUpErrors(t *testing.T) {
	session := realtime.WebRTCSession{Provider: "openai", ID: "gone/id"}
	for _, test := range []struct {
		status int
		body   string
		live   bool
		ok     bool
	}{{400, `{"error":{"code":"session_id_not_found"}}`, true, true}, {400, `{"error":{"code":"call_id_not_found"}}`, false, true}, {404, `{"error":{"code":"session_id_not_found"}}`, true, false}, {400, `{"error":{"code":"different"}}`, true, false}} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !strings.Contains(r.URL.EscapedPath(), "gone%2Fid/hangup") {
				t.Errorf("path %s", r.URL)
			}
			w.WriteHeader(test.status)
			_, _ = io.WriteString(w, test.body)
		}))
		var err error
		if test.live {
			err = openairt.NewLiveModel("gpt-live-1", openairt.WithBaseURL(server.URL)).HangUp(t.Context(), session)
		} else {
			err = openairt.NewModel("gpt-realtime", openairt.WithBaseURL(server.URL)).HangUp(t.Context(), session)
		}
		if (err == nil) != test.ok {
			t.Fatalf("hangup: %v", err)
		}
		server.Close()
	}
	for _, url := range []string{"://", "http://127.0.0.1:1"} {
		if err := openairt.NewLiveModel("gpt-live-1", openairt.WithBaseURL(url)).HangUp(t.Context(), session); err == nil {
			t.Fatal("bad endpoint")
		}
	}
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 400, Body: readFailure{}}, nil
	})}
	if err := openairt.NewModel("gpt-realtime", openairt.WithHTTPClient(client)).HangUp(t.Context(), session); err == nil {
		t.Fatal("read failure")
	}
}

type readFailure struct{}

func (readFailure) Read([]byte) (int, error) { return 0, fmt.Errorf("read failed") }
func (readFailure) Close() error             { return nil }
