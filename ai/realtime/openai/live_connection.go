package openai

import (
	"context"
	"encoding/base64"
	"fmt"
	"iter"
	"strings"
	"sync"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	"github.com/coder/websocket"
)

type liveDial func(context.Context, string, []map[string]any) (*websocket.Conn, string, error)

type liveDelegation struct {
	pending      map[string]bool
	inFlight     bool
	continuation bool
}

type liveFrame struct {
	kind websocket.MessageType
	data []byte
	err  error
}

// LiveConnection translates the GPT-Live timeline into realtime session events.
// Send and Close are safe alongside one Events consumer. EndSession requires that consumer to have stopped.
type LiveConnection struct {
	mu           sync.Mutex
	socket       *websocket.Conn
	frames       chan liveFrame
	stop         chan struct{}
	closed       bool
	ended        bool
	redial       bool
	model        string
	backend      string
	sessionID    string
	baseURL      string
	rate         int
	sideband     bool
	settings     LiveSettings
	reconnect    *realtime.ReconnectPolicy
	reconnects   int
	dial         liveDial
	history      func() []ai.ModelMessage
	responseOpen bool
	inputOpen    bool
	lastVoice    time.Time
	pauseBytes   int
	hasPause     bool
	lastInputEnd float64
	hasInputEnd  bool
	fragments    [2]string
	delegations  map[string]*liveDelegation
	calls        map[string]string
	abandoned    map[string]bool
	seenCalls    map[string]bool
	reasoning    map[string][]ai.ResponsePart
	nativeIndex  int
	seconds      float64
}

func newLiveConnection(socket *websocket.Conn, started map[string]any, common realtime.Settings, settings LiveSettings, rate int, baseURL string, sideband bool, dial liveDial) *LiveConnection {
	connection := &LiveConnection{socket: socket, model: stringValue(started["model"]),
		backend: stringValue(object(object(started["delegation"])["responses"])["model"]), sessionID: stringValue(started["id"]),
		baseURL: baseURL, rate: rate, sideband: sideband, settings: settings, reconnect: common.Reconnect, dial: dial,
		stop: make(chan struct{}), delegations: map[string]*liveDelegation{}, calls: map[string]string{},
		abandoned: map[string]bool{}, seenCalls: map[string]bool{}, reasoning: map[string][]ai.ResponsePart{}}
	connection.startReader()
	return connection
}

func (connection *LiveConnection) startReader() {
	connection.frames = make(chan liveFrame, 1)
	frames, socket := connection.frames, connection.socket
	go func() {
		for {
			kind, data, err := socket.Read(context.Background())
			select {
			case frames <- liveFrame{kind: kind, data: data, err: err}:
			case <-connection.stop:
				return
			}
			if err != nil {
				return
			}
		}
	}()
}

// ModelName returns the model negotiated with the server.
func (connection *LiveConnection) ModelName() string {
	connection.mu.Lock()
	defer connection.mu.Unlock()
	return connection.model
}

// InputTranscriptionEnabled reports GPT-Live's mandatory transcription.
func (*LiveConnection) InputTranscriptionEnabled() bool { return true }

// ReconnectRestoresInFlightState reports that interrupted work cannot be resumed.
func (*LiveConnection) ReconnectRestoresInFlightState() bool { return false }

// SetMessageHistory installs the portable history reader used after a drop.
func (connection *LiveConnection) SetMessageHistory(history func() []ai.ModelMessage) {
	connection.mu.Lock()
	defer connection.mu.Unlock()
	connection.history = history
}

// Send appends audio or context, or answers a delegated Responses function call.
func (connection *LiveConnection) Send(ctx context.Context, input realtime.Input) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	var content []any
	var err error
	if result, ok := input.(realtime.ToolResult); ok {
		content, err = liveToolContent(ctx, result.Content)
		if err != nil {
			return err
		}
	}
	connection.mu.Lock()
	defer connection.mu.Unlock()
	if connection.closed || connection.ended {
		return fmt.Errorf("openai GPT-Live: connection is closed")
	}
	switch input := input.(type) {
	case realtime.AudioInput:
		if len(input.Data)%2 != 0 {
			return fmt.Errorf("openai GPT-Live: PCM16 audio length must be even")
		}
		return connection.write(ctx, map[string]any{"type": "session.input_audio.append", "audio": base64.StdEncoding.EncodeToString(input.Data)})
	case realtime.TextInput:
		return connection.appendText(ctx, "session.commentary.append", input.Text)
	case realtime.TextContext:
		return connection.appendText(ctx, "session.thinking.append", input.Text)
	case realtime.ImageInput:
		if !input.Respond || len(input.Content.Data) == 0 || !strings.HasPrefix(input.Content.MediaType, "image/") {
			return fmt.Errorf("openai GPT-Live: an image requires Respond=true")
		}
		if err := connection.write(ctx, map[string]any{"type": "response.item.create", "item": map[string]any{"type": "message", "role": "user", "content": []any{map[string]any{"type": "input_image", "image_url": liveDataURL(input.Content)}}}}); err != nil {
			return err
		}
		return connection.write(ctx, map[string]any{"type": "response.create"})
	case realtime.CreateResponse:
		return connection.write(ctx, map[string]any{"type": "response.create"})
	case realtime.ToolResult:
		if connection.abandoned[input.ToolCallID] {
			delete(connection.abandoned, input.ToolCallID)
			return nil
		}
		delegation := connection.delegations[connection.calls[input.ToolCallID]]
		if err := connection.write(ctx, map[string]any{"type": "response.item.create", "item": map[string]any{"type": "function_call_output", "call_id": input.ToolCallID, "output": input.Output}}); err != nil {
			return err
		}
		if len(content) > 0 {
			if err := connection.write(ctx, map[string]any{"type": "response.item.create", "item": map[string]any{"type": "message", "role": "user", "content": content}}); err != nil {
				return err
			}
		}
		delete(connection.calls, input.ToolCallID)
		if delegation == nil {
			return connection.write(ctx, map[string]any{"type": "response.create"})
		}
		delete(delegation.pending, input.ToolCallID)
		if len(delegation.pending) > 0 {
			return nil
		}
		if delegation.inFlight {
			delegation.continuation = true
			return nil
		}
		return connection.continueDelegation(ctx, delegation)
	default:
		return fmt.Errorf("openai GPT-Live: manual turn control, cancellation, and truncation are unavailable (%T)", input)
	}
}

func (connection *LiveConnection) appendText(ctx context.Context, kind, text string) error {
	if len(text) > 500 && liveTokenCount(text) > 500 {
		return fmt.Errorf("openai GPT-Live: context accepts at most 500 tokens per send")
	}
	return connection.write(ctx, map[string]any{"type": kind, "delegation_id": nil, "content": text})
}

func (connection *LiveConnection) write(ctx context.Context, value any) error {
	return writeJSON(ctx, connection.socket, value)
}

func (connection *LiveConnection) continueDelegation(ctx context.Context, delegation *liveDelegation) error {
	delegation.inFlight = true
	delegation.continuation = false
	return connection.write(ctx, map[string]any{"type": "response.create"})
}

// Events yields normalized events. Silence closes turns only when no delegated work remains.
func (connection *LiveConnection) Events(ctx context.Context) iter.Seq2[realtime.CodecEvent, error] {
	return func(yield func(realtime.CodecEvent, error) bool) {
		for {
			connection.mu.Lock()
			frames := connection.frames
			delay := time.Hour
			if len(connection.delegations) == 0 && (connection.responseOpen || connection.inputOpen) {
				delay = max(0, time.Until(connection.lastVoice.Add(connection.settings.TurnSilence)))
			}
			connection.mu.Unlock()
			timer := time.NewTimer(delay)
			var events []realtime.CodecEvent
			select {
			case <-ctx.Done():
				timer.Stop()
				return
			case <-connection.stop:
				timer.Stop()
				return
			case frame := <-frames:
				timer.Stop()
				connection.mu.Lock()
				closed := connection.closed
				connection.mu.Unlock()
				if closed {
					return
				}
				if frame.err != nil {
					connection.mu.Lock()
					normal := websocket.CloseStatus(frame.err) == websocket.StatusNormalClosure || websocket.CloseStatus(frame.err) == websocket.StatusGoingAway
					redial := connection.redial
					if normal && !redial {
						connection.ended = true
						events = connection.settleTurns(false)
					}
					connection.mu.Unlock()
					if normal && !redial {
						for _, event := range events {
							if !yield(event, nil) {
								return
							}
						}
						return
					}
					restored, err := connection.redialSession(ctx)
					if err != nil {
						yield(nil, err)
						return
					}
					if !yield(realtime.SessionReconnected{StateRestored: restored}, nil) {
						return
					}
					continue
				}
				if frame.kind != websocket.MessageText {
					continue
				}
				connection.mu.Lock()
				var err error
				events, err = connection.mapLiveFrame(frame.data)
				if err != nil {
					events = append(events, realtime.SessionError{Err: err, Recoverable: true})
				}
				connection.mu.Unlock()
			case <-timer.C:
			}
			connection.mu.Lock()
			if len(connection.delegations) == 0 && (connection.inputOpen || connection.responseOpen) && time.Since(connection.lastVoice) >= connection.settings.TurnSilence {
				events = append(events, connection.settleTurns(false)...)
			}
			connection.mu.Unlock()
			for _, event := range events {
				if !yield(event, nil) {
					return
				}
				if failed, ok := event.(realtime.SessionError); ok && !failed.Recoverable {
					return
				}
			}
			connection.mu.Lock()
			var err error
			for _, delegation := range connection.delegations {
				if delegation.continuation && !delegation.inFlight {
					err = connection.continueDelegation(ctx, delegation)
					if err != nil {
						break
					}
				}
			}
			connection.mu.Unlock()
			if err != nil {
				yield(nil, err)
				return
			}
		}
	}
}

// EndSession ends an owned media session and yields its final usage. Sidebands leave the call running.
func (connection *LiveConnection) EndSession(ctx context.Context) iter.Seq2[realtime.SessionUsage, error] {
	return func(yield func(realtime.SessionUsage, error) bool) {
		if ctx.Err() != nil {
			return
		}
		connection.mu.Lock()
		if connection.closed || connection.sideband {
			connection.mu.Unlock()
			return
		}
		ended := connection.ended
		var err error
		if !ended {
			err = connection.write(ctx, map[string]any{"type": "session.close"})
		}
		connection.mu.Unlock()
		if err != nil {
			yield(realtime.SessionUsage{}, err)
			return
		}
		for !ended {
			select {
			case <-ctx.Done():
				return
			case frame := <-connection.frames:
				if frame.err != nil {
					return
				}
				if frame.kind != websocket.MessageText {
					continue
				}
				connection.mu.Lock()
				events, _ := connection.mapLiveFrame(frame.data)
				ended = connection.ended
				connection.mu.Unlock()
				for _, event := range events {
					if report, ok := event.(realtime.SessionUsage); ok {
						report.ResponseScoped = false
						if !yield(report, nil) {
							return
						}
					}
				}
			}
		}
	}
}

// Close releases this websocket without ending a browser's media session.
func (connection *LiveConnection) Close(context.Context) error {
	connection.mu.Lock()
	defer connection.mu.Unlock()
	if connection.closed {
		return nil
	}
	connection.closed = true
	close(connection.stop)
	return connection.socket.CloseNow()
}

func liveDataURL(content ai.BinaryContent) string {
	return "data:" + content.MediaType + ";base64," + base64.StdEncoding.EncodeToString(content.Data)
}

var _ realtime.ConnectionInfo = (*LiveConnection)(nil)
var _ realtime.HistoryAwareConnection = (*LiveConnection)(nil)
