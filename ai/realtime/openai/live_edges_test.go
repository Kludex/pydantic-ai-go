package openai_test

import (
	"context"
	"encoding/json"
	"errors"
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

func TestLiveToolMediaValidationIsAtomic(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "octet"):
			w.Header().Set("Content-Type", "application/octet-stream")
		case strings.Contains(r.URL.Path, "png"):
			w.Header().Set("Content-Type", "image/png")
		case strings.Contains(r.URL.Path, "audio"):
			w.Header().Set("Content-Type", "audio/wav")
		case strings.Contains(r.URL.Path, "unknown"):
			w.Header().Set("Content-Type", "application/unknown")
		default:
			w.Header().Set("Content-Type", "application/pdf")
		}
		_, _ = io.WriteString(w, "file")
	}))
	defer server.Close()
	connection, peer := livePair(t, realtime.ConnectParams{})
	good := [][]ai.UserContent{{ai.ImageURL{URL: server.URL + "/png", ForceDownload: ai.FileDownloadAllowLocal, VendorMetadata: map[string]any{"detail": "low"}}}, {ai.ImageURL{URL: server.URL + "/octet.png", ForceDownload: ai.FileDownloadAllowLocal}}, {ai.DocumentURL{URL: server.URL + "/pdf", ForceDownload: ai.FileDownloadAllowLocal}}, {ai.DocumentURL{URL: server.URL + "/octet.pdf", ForceDownload: ai.FileDownloadAllowLocal}}, {ai.UploadedFile{FileID: "f", ProviderName: "openai", MediaType: "image/png", VendorMetadata: map[string]any{"detail": "high"}}}}
	for _, content := range good {
		if err := connection.Send(t.Context(), realtime.ToolResult{ToolCallID: "c", Content: content}); err != nil {
			t.Fatal(err)
		}
		for index := 0; index < 3; index++ {
			_ = peer.input(t)
		}
	}
	bad := [][]ai.UserContent{{nil}, {ai.AudioURL{}}, {ai.VideoURL{}}, {ai.BinaryContent{MediaType: "audio/wav"}}, {ai.BinaryContent{MediaType: "application/unknown"}}, {ai.UploadedFile{ProviderName: "google", FileID: "f"}}, {ai.UploadedFile{ProviderName: "openai", FileID: "f", MediaType: "audio/wav"}}, {ai.ImageURL{ForceDownload: "bad"}}, {ai.DocumentURL{ForceDownload: "bad"}}, {ai.ImageURL{URL: "bad", ForceDownload: ai.FileDownloadSafe}}, {ai.DocumentURL{URL: "bad", ForceDownload: ai.FileDownloadSafe}}, {ai.DocumentURL{URL: server.URL + "/audio.wav", ForceDownload: ai.FileDownloadNever}}, {ai.ImageURL{URL: server.URL + "/pdf", ForceDownload: ai.FileDownloadAllowLocal}}, {ai.ImageURL{URL: server.URL + "/octet.unknown", ForceDownload: ai.FileDownloadAllowLocal}}, {ai.DocumentURL{URL: server.URL + "/octet.unknown", ForceDownload: ai.FileDownloadAllowLocal}}, {ai.DocumentURL{URL: server.URL + "/unknown", ForceDownload: ai.FileDownloadAllowLocal}}, {ai.DocumentURL{URL: server.URL + "/audio", MediaType: "application/pdf", ForceDownload: ai.FileDownloadAllowLocal}}}
	for _, content := range bad {
		if err := connection.Send(t.Context(), realtime.ToolResult{ToolCallID: "c", Output: "must not send", Content: content}); err == nil {
			t.Fatalf("accepted: %+v", content)
		}
		select {
		case frame := <-peer.sent:
			t.Fatalf("partially sent: %+v", frame)
		default:
		}
	}
}

func TestLiveErrorCloseAndAbandonedTools(t *testing.T) {
	for _, mode := range []string{"abandoned", "incomplete", "backend-unavailable", "content", "drift", "input-only", "negative-audio", "unsigned-reasoning"} {
		t.Run(mode, func(t *testing.T) {
			connection, peer := livePair(t, realtime.ConnectParams{})
			switch mode {
			case "abandoned", "incomplete":
				peer.frame(t, delegation("d"))
				peer.frame(t, delegated("d", liveCall("c")))
				frame := backendTerminal("response.failed", nil)
				if mode == "incomplete" {
					frame = backendTerminal("response.incomplete", nil)
					frame["response"].(map[string]any)["error"] = map[string]any{"message": "bad", "code": "oops"}
				}
				peer.frame(t, delegated("d", frame))
			case "backend-unavailable":
				peer.frame(t, map[string]any{"type": "error", "error": map[string]any{"code": nil, "message": "Model `gpt-5` does not exist"}})
			case "content":
				peer.frame(t, map[string]any{"type": "session.output_transcript.delta", "delta": "unfinished"})
				peer.frame(t, map[string]any{"type": "session.closed", "reason": "content", "usage": map[string]any{"seconds": 1}})
			case "drift":
				peer.frame(t, map[string]any{"type": "session.closed", "reason": "future", "usage": "drift"})
			case "input-only":
				peer.frame(t, map[string]any{"type": "session.input_transcript.delta", "delta": "Hi", "start_ms": 0, "end_ms": 1})
			case "negative-audio":
				peer.frame(t, map[string]any{"type": "session.output_audio.delta", "delta": "nP8="})
				peer.frame(t, map[string]any{"type": "session.input_transcript.delta", "delta": "User speaking", "start_ms": 1, "end_ms": 2})
				peer.frame(t, map[string]any{"type": "session.output_audio.delta", "delta": "AAA="})
			case "unsigned-reasoning":
				peer.frame(t, delegated("", map[string]any{"type": "response.output_item.done", "item": map[string]any{"type": "reasoning", "id": "r", "encrypted_content": "opaque"}}))
				peer.frame(t, delegated("", map[string]any{"type": "response.output_item.done", "item": map[string]any{"type": "web_search_call", "id": "search", "status": "completed"}}))
				peer.end(t, "close_requested", 1)
			}
			ctx, cancel := context.WithTimeout(t.Context(), time.Second)
			defer cancel()
			for event, err := range connection.Events(ctx) {
				if err != nil {
					t.Fatal(err)
				}
				if failure, ok := event.(realtime.SessionError); ok {
					if mode == "abandoned" || mode == "incomplete" {
						if err := connection.Send(t.Context(), realtime.ToolResult{ToolCallID: "c"}); err != nil {
							t.Fatal(err)
						}
						select {
						case frame := <-peer.sent:
							t.Fatalf("sent abandoned: %+v", frame)
						default:
						}
						break
					}
					if mode == "backend-unavailable" {
						var liveError *openairt.LiveError
						if !errors.As(failure.Err, &liveError) || liveError.Code != "live_backend_model_unavailable" {
							t.Fatal(failure)
						}
					}
				}
				if mode == "input-only" {
					if transcript, ok := event.(realtime.InputTranscript); ok && transcript.Final {
						break
					}
				}
				if mode == "negative-audio" {
					if _, ok := event.(realtime.ResponseDone); ok {
						break
					}
				}
			}
		})
	}
}

func TestLiveToolContinuationAfterTerminal(t *testing.T) {
	connection, peer := livePair(t, realtime.ConnectParams{})
	peer.frame(t, delegation("d"))
	peer.frame(t, delegated("d", liveCall("c1")))
	peer.frame(t, delegated("d", liveCall("c2")))
	peer.frame(t, delegated("d", backendTerminal("response.completed", nil)))
	for event, err := range connection.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(realtime.SessionUsage); ok {
			break
		}
	}
	if err := connection.Send(t.Context(), realtime.ToolResult{ToolCallID: "c1"}); err != nil {
		t.Fatal(err)
	}
	_ = peer.input(t)
	if err := connection.Send(t.Context(), realtime.ToolResult{ToolCallID: "c2"}); err != nil {
		t.Fatal(err)
	}
	_ = peer.input(t)
	if peer.input(t)["type"] != "response.create" {
		t.Fatal("continuation missing")
	}
}

func TestLiveEndSessionCancellationAndLateBackendUsage(t *testing.T) {
	connection, peer := livePair(t, realtime.ConnectParams{})
	peer.frame(t, delegated("", backendTerminal("response.completed", map[string]any{"input_tokens": 7, "output_tokens": 3})))
	_ = peer.socket.Write(t.Context(), websocket.MessageBinary, []byte("ignore"))
	reports := 0
	for report, err := range connection.EndSession(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if report.ResponseScoped {
			t.Fatal("scoped closing usage")
		}
		reports++
		if reports == 1 {
			if report.Usage.InputTokens != 7 {
				t.Fatal(report)
			}
		}
	}
	if reports != 2 {
		t.Fatal(reports)
	}
	connection, _ = livePair(t, realtime.ConnectParams{})
	for range connection.EndSession(t.Context()) {
		break
	}
	connection, peer = livePair(t, realtime.ConnectParams{})
	_ = peer.socket.CloseNow()
	for _, err := range connection.EndSession(t.Context()) {
		if err == nil {
			t.Fatal("write should fail")
		}
	}
	connection, _ = livePair(t, realtime.ConnectParams{})
	// Cancellation preserves a pending read for EndSession rather than cancelling the websocket.
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	for range connection.Events(ctx) {
		t.Fatal("event on cancelled iterator")
	}
	for range connection.EndSession(ctx) {
		t.Fatal("usage on cancelled ending")
	}
}

type boundLiveModel struct {
	model      *openairt.LiveModel
	connection realtime.Connection
}

func (model boundLiveModel) Name() string              { return model.model.Name() }
func (model boundLiveModel) ProviderName() string      { return model.model.ProviderName() }
func (model boundLiveModel) ProviderURL() string       { return model.model.ProviderURL() }
func (model boundLiveModel) Profile() realtime.Profile { return model.model.Profile() }
func (model boundLiveModel) Connect(context.Context, realtime.ConnectParams) (realtime.Connection, error) {
	return model.connection, nil
}

func TestLivePublicSessionToolsAndFinalUsage(t *testing.T) {
	connection, peer := livePair(t, realtime.ConnectParams{})
	session, err := realtime.Open(t.Context(), boundLiveModel{peer.model, connection}, realtime.ConnectParams{}, realtime.WithToolExecutor(realtime.ToolExecutorFunc(func(context.Context, ai.ToolCallPart) (any, error) { return "result", nil })))
	if err != nil {
		t.Fatal(err)
	}
	image := ai.BinaryContent{Data: []byte("png"), MediaType: "image/png"}
	if err := session.Send(t.Context(), image); err == nil {
		t.Fatal("passive image accepted")
	}
	if err := session.Send(t.Context(), image, realtime.WithResponse(true)); err != nil {
		t.Fatal(err)
	}
	_ = peer.input(t)
	_ = peer.input(t)
	peer.frame(t, map[string]any{"type": "session.input_transcript.delta", "delta": "Hello", "start_ms": 0, "end_ms": 10})
	peer.frame(t, delegation("d"))
	peer.frame(t, delegated("d", liveCall("c")))
	_ = peer.input(t)
	peer.frame(t, delegated("d", backendTerminal("response.completed", map[string]any{"input_tokens": 10, "output_tokens": 2})))
	_ = peer.input(t)
	peer.frame(t, delegated("d", backendTerminal("response.completed", map[string]any{"input_tokens": 20, "output_tokens": 3})))
	peer.frame(t, map[string]any{"type": "session.output_transcript.delta", "delta": "Done"})
	peer.frame(t, map[string]any{"type": "session.usage.updated", "usage": map[string]any{"seconds": 5}, "context_window": map[string]any{"usage_ratio": 0.4}})
	for event, err := range session.Events(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(realtime.TurnCompleteEvent); ok {
			break
		}
	}
	if used, ok := session.ContextWindowUsed(); !ok || used != 0.4 {
		t.Fatalf("context %g %v", used, ok)
	}
	if err := session.WaitForReply(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := session.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if session.Usage().AudioSeconds != 9 || session.Usage().Requests != 2 || session.Usage().InputTokens != 30 {
		t.Fatalf("usage: %+v", session.Usage())
	}
	messages := session.NewMessages()
	var responses []ai.ModelResponse
	for _, message := range messages {
		if response, ok := message.(ai.ModelResponse); ok {
			responses = append(responses, response)
		}
	}
	if len(responses) != 2 || responses[0].Usage.InputTokens != 10 || responses[1].Usage.InputTokens != 20 || responses[0].ProviderDetails["delegated_model"] != "gpt-5" {
		encoded, _ := json.Marshal(messages)
		t.Fatalf("history: %s", encoded)
	}
	if err := session.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
}
