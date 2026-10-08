package openai_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
	"github.com/coder/websocket"
)

func TestLiveReconnectForkAndReplay(t *testing.T) {
	for _, mode := range []string{"replay", "fork", "fallback", "retry", "expired", "nohistory", "cancel", "exhausted", "defaults", "close-during-dial", "cancel-backoff"} {
		t.Run(mode, func(t *testing.T) {
			var attempts atomic.Int32
			configs := make(chan map[string]any, 10)
			retryStarted := make(chan struct{}, 1)
			release := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				attempt := attempts.Add(1)
				if mode == "close-during-dial" && attempt == 2 {
					retryStarted <- struct{}{}
					<-release
				}
				if mode == "retry" && attempt == 2 || mode == "exhausted" && attempt > 1 {
					w.WriteHeader(500)
					return
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
				configs <- frame["session"].(map[string]any)
				if mode == "fallback" && strings.HasSuffix(r.URL.Path, "/fork") {
					_ = writeFrame(r.Context(), socket, map[string]any{"type": "error", "error": map[string]any{"message": "cannot fork"}})
					return
				}
				if strings.HasSuffix(r.URL.Path, "/fork") && r.URL.EscapedPath() != "/v1/live/sessions/s%2F1/fork" {
					t.Errorf("fork escaped path: %s", r.URL)
				}
				_ = writeFrame(r.Context(), socket, map[string]any{"type": "session.started", "session": map[string]any{"id": "s/1", "model": "gpt-live-1"}})
				if attempt == 1 {
					_ = writeFrame(r.Context(), socket, delegation("d"))
					_ = writeFrame(r.Context(), socket, delegated("d", liveCall("old")))
					if mode == "expired" {
						_ = writeFrame(r.Context(), socket, map[string]any{"type": "session.closed", "reason": "expired", "usage": map[string]any{"seconds": 1}})
						_ = socket.Close(websocket.StatusNormalClosure, "")
					}
					return
				}
				if mode == "close-during-dial" {
					_, _, _ = socket.Read(r.Context())
					return
				}
				_ = writeFrame(r.Context(), socket, map[string]any{"type": "session.usage.updated", "usage": map[string]any{"seconds": 2}})
				_ = writeFrame(r.Context(), socket, map[string]any{"type": "session.closed", "reason": "remote_hangup", "usage": map[string]any{"seconds": 2}})
				_ = socket.Close(websocket.StatusNormalClosure, "")
			}))
			defer server.Close()
			stored := mode == "fork" || mode == "fallback"
			policy := &realtime.ReconnectPolicy{MaxAttempts: 3, MaxReconnects: 1, BaseDelay: time.Millisecond, MaxDelay: 2 * time.Millisecond, Jitter: true}
			if mode == "defaults" {
				policy = &realtime.ReconnectPolicy{}
			}
			if mode == "cancel-backoff" {
				policy.BaseDelay = 100 * time.Millisecond
				policy.MaxDelay = 100 * time.Millisecond
				policy.Jitter = false
			}
			model := openairt.NewLiveModel("gpt-live-1", openairt.WithAPIKey("key"), openairt.WithBaseURL(server.URL+"/v1"), openairt.WithLiveSettings(openairt.LiveSettings{Store: stored, TurnSilence: time.Millisecond}))
			connection, err := model.Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{Reconnect: policy}})
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = connection.Close(t.Context()) }()
			live := connection.(*openairt.LiveConnection)
			ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
			defer cancel()
			if mode != "nohistory" {
				live.SetMessageHistory(func() []ai.ModelMessage {
					if mode == "cancel-backoff" {
						time.AfterFunc(20*time.Millisecond, cancel)
						return nil
					}
					messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: strings.Repeat(" a", 9000)}}}}
					for index := 0; index < 150; index++ {
						messages = append(messages, ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "recent"}, ai.UserPromptPart{Contents: []ai.UserContent{ai.BinaryContent{MediaType: "image/png"}, ai.TextContent{Text: "text"}}}, ai.SpeechPart{Audio: &ai.BinaryContent{MediaType: "audio/pcm"}}}})
					}
					return messages
				})
			}
			if mode == "close-during-dial" {
				go func() { <-retryStarted; _ = live.Close(context.Background()); close(release) }()
			}
			reconnects := 0
			seconds := 0.0
			failed := false
			for event, err := range live.Events(ctx) {
				if err != nil {
					failed = true
					break
				}
				switch event := event.(type) {
				case realtime.ToolCall:
					if mode == "cancel" {
						cancel()
					}
				case realtime.SessionReconnected:
					reconnects++
					if event.StateRestored != (mode != "nohistory") {
						t.Fatalf("restored: %+v", event)
					}
					if err := live.Send(t.Context(), realtime.ToolResult{ToolCallID: "old"}); err != nil {
						t.Fatal(err)
					}
					if mode == "nohistory" {
						break
					}
				case realtime.SessionUsage:
					seconds += event.Usage.AudioSeconds
				}
			}
			if mode == "cancel" {
				if reconnects != 0 {
					t.Fatal("reconnected after cancel")
				}
				return
			}
			if mode == "close-during-dial" || mode == "exhausted" || mode == "cancel-backoff" {
				if !failed {
					t.Fatal("reconnect failure missing")
				}
				return
			}
			if reconnects != 1 || seconds < 2 {
				t.Fatalf("reconnects=%d usage=%g", reconnects, seconds)
			}
			<-configs
			replacement := <-configs
			if mode == "fork" || mode == "fallback" {
				if len(replacement) != 0 {
					t.Fatalf("fork config: %+v", replacement)
				}
				if mode == "fallback" {
					replacement = <-configs
				}
			}
			if mode == "replay" || mode == "fallback" || mode == "retry" || mode == "expired" || mode == "defaults" {
				input := replacement["input"].([]any)
				if len(input) != 128 {
					t.Fatalf("replay len=%d", len(input))
				}
			}
		})
	}
}

func TestLiveReplayTokenBudgetAndDroppedWithoutPolicy(t *testing.T) {
	var attempt atomic.Int32
	received := make(chan map[string]any, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
		received <- frame["session"].(map[string]any)
		_ = writeFrame(r.Context(), socket, map[string]any{"type": "session.started", "session": map[string]any{"model": "gpt-live-1", "id": "s"}})
		if attempt.Add(1) == 1 {
			return
		}
		_ = socket.Close(websocket.StatusNormalClosure, "")
	}))
	defer server.Close()
	connection, err := openairt.NewLiveModel("gpt-live-1", openairt.WithAPIKey("key"), openairt.WithBaseURL(server.URL)).Connect(t.Context(), realtime.ConnectParams{Settings: realtime.Settings{Reconnect: &realtime.ReconnectPolicy{MaxAttempts: 1, MaxReconnects: 1, BaseDelay: time.Millisecond, MaxDelay: time.Millisecond}}})
	if err != nil {
		t.Fatal(err)
	}
	live := connection.(*openairt.LiveConnection)
	defer func() { _ = live.Close(t.Context()) }()
	live.SetMessageHistory(func() []ai.ModelMessage {
		var messages []ai.ModelMessage
		for index := 0; index < 30; index++ {
			messages = append(messages, ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: strings.Repeat(" a", 500)}}})
		}
		messages = append(messages, ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: strings.Repeat(" a", 9000)}}})
		return messages
	})
	for event, err := range live.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(realtime.SessionReconnected); ok {
			break
		}
	}
	<-received
	replay := <-received
	if len(replay["input"].([]any)) != 16 {
		t.Fatalf("token-budget replay: %+v", replay)
	}
	connection, peer := livePair(t, realtime.ConnectParams{})
	_ = peer.socket.CloseNow()
	for _, err := range connection.Events(t.Context()) {
		if err == nil {
			t.Fatal("drop error absent")
		}
		break
	}
}
