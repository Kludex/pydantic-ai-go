package openaiprotocol

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"math/rand/v2"
	"net/http"
	"slices"
	"sync"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/realtime"
	"github.com/coder/websocket"
)

// Dial opens and fully configures one websocket session.
type Dial func(ctx context.Context, history []ai.ModelMessage) (*websocket.Conn, string, error)

// Mapper translates one provider frame.
type Mapper func(data []byte) ([]realtime.CodecEvent, error)

// Config defines one OpenAI-protocol connection.
type Config struct {
	Provider                  string
	Model                     string
	Socket                    *websocket.Conn
	ServerModel               string
	Dial                      Dial
	Mapper                    Mapper
	Reconnect                 *realtime.ReconnectPolicy
	InputTranscriptionEnabled bool
	RestoresInFlightState     bool
	SupportsImages            bool
	OutputSampleRate          int
}

// Connection implements the common OpenAI realtime websocket protocol.
type Connection struct {
	config Config

	mu      sync.RWMutex
	socket  *websocket.Conn
	model   string
	history func() []ai.ModelMessage
	closed  bool

	stateMu             sync.Mutex
	responseActive      bool
	activeResponseID    string
	currentItemID       string
	currentContentIndex int
	generatedAudioBytes int
	reconnects          int
	seenToolCalls       map[string]struct{}
}

// New returns a configured protocol connection.
func New(config Config) (*Connection, error) {
	if config.Socket == nil || config.Mapper == nil {
		return nil, fmt.Errorf("realtime: OpenAI protocol connection requires a socket and mapper")
	}
	if config.OutputSampleRate == 0 {
		config.OutputSampleRate = realtime.DefaultAudioSampleRate
	}
	return &Connection{
		config: config, socket: config.Socket, model: config.ServerModel, seenToolCalls: map[string]struct{}{},
	}, nil
}

// ModelName returns the model reported by the server handshake.
func (connection *Connection) ModelName() string {
	connection.mu.RLock()
	defer connection.mu.RUnlock()
	return connection.model
}

// InputTranscriptionEnabled reports whether transcript events are configured.
func (connection *Connection) InputTranscriptionEnabled() bool {
	return connection.config.InputTranscriptionEnabled
}

// ReconnectRestoresInFlightState reports provider reconnect behavior.
func (connection *Connection) ReconnectRestoresInFlightState() bool {
	return connection.config.RestoresInFlightState
}

// SetMessageHistory installs the live history callback used by reconnect replay.
func (connection *Connection) SetMessageHistory(history func() []ai.ModelMessage) {
	connection.mu.Lock()
	defer connection.mu.Unlock()
	connection.history = history
}

// Send writes one normalized input frame.
func (connection *Connection) Send(ctx context.Context, input realtime.Input) error {
	switch input := input.(type) {
	case realtime.AudioInput:
		if len(input.Data)%2 != 0 {
			return fmt.Errorf("realtime: PCM16 audio length must be even")
		}
		return connection.writeJSON(ctx, map[string]any{
			"type": "input_audio_buffer.append", "audio": base64.StdEncoding.EncodeToString(input.Data),
		})
	case realtime.TextInput:
		if err := connection.writeJSON(ctx, map[string]any{
			"type": "conversation.item.create",
			"item": map[string]any{
				"type": "message", "role": "user",
				"content": []any{map[string]any{"type": "input_text", "text": input.Text}},
			},
		}); err != nil {
			return err
		}
		return connection.requestResponse(ctx)
	case realtime.ImageInput:
		if !connection.config.SupportsImages {
			return fmt.Errorf("realtime: %s does not support image input", connection.config.Provider)
		}
		return connection.writeJSON(ctx, map[string]any{
			"type": "conversation.item.create",
			"item": map[string]any{
				"type": "message", "role": "user",
				"content": []any{map[string]any{
					"type": "input_image", "image_url": dataURL(input.Content),
				}},
			},
		})
	case realtime.ToolResult:
		parts, err := userContent(input.Content, connection.config.SupportsImages)
		if err != nil {
			return err
		}
		if err := connection.writeJSON(ctx, map[string]any{
			"type": "conversation.item.create",
			"item": map[string]any{
				"type": "function_call_output", "call_id": input.ToolCallID, "output": input.Output,
			},
		}); err != nil {
			return err
		}
		if len(parts) > 0 {
			err = connection.writeJSON(ctx, map[string]any{
				"type": "conversation.item.create",
				"item": map[string]any{"type": "message", "role": "user", "content": parts},
			})
			if err == nil {
				err = connection.requestResponse(ctx)
			}
			return err
		}
		return connection.requestResponse(ctx)
	case realtime.CommitAudio:
		return connection.writeJSON(ctx, map[string]any{"type": "input_audio_buffer.commit"})
	case realtime.ClearAudio:
		return connection.writeJSON(ctx, map[string]any{"type": "input_audio_buffer.clear"})
	case realtime.CreateResponse:
		return connection.requestResponse(ctx)
	case realtime.CancelResponse:
		connection.stateMu.Lock()
		active := connection.responseActive
		connection.stateMu.Unlock()
		if !active {
			return nil
		}
		return connection.writeJSON(ctx, map[string]any{"type": "response.cancel"})
	case realtime.TruncateOutput:
		connection.stateMu.Lock()
		itemID := connection.currentItemID
		contentIndex := connection.currentContentIndex
		maximum := connection.generatedAudioBytes * 1000 / (connection.config.OutputSampleRate * 2)
		connection.stateMu.Unlock()
		if itemID == "" {
			return nil
		}
		end := min(input.AudioEndMilliseconds, maximum)
		return connection.writeJSON(ctx, map[string]any{
			"type": "conversation.item.truncate", "item_id": itemID,
			"content_index": contentIndex, "audio_end_ms": end,
		})
	default:
		return fmt.Errorf("realtime: %s does not support %T input", connection.config.Provider, input)
	}
}

func (connection *Connection) requestResponse(ctx context.Context) error {
	connection.stateMu.Lock()
	if connection.responseActive {
		connection.stateMu.Unlock()
		return nil
	}
	connection.responseActive = true
	connection.stateMu.Unlock()
	if err := connection.writeJSON(ctx, map[string]any{"type": "response.create"}); err != nil {
		connection.stateMu.Lock()
		connection.responseActive = false
		connection.stateMu.Unlock()
		return err
	}
	return nil
}

func (connection *Connection) writeJSON(ctx context.Context, value any) error {
	data, _ := json.Marshal(value)
	connection.mu.RLock()
	socket := connection.socket
	closed := connection.closed
	connection.mu.RUnlock()
	if closed {
		return fmt.Errorf("realtime: connection is closed")
	}
	return socket.Write(ctx, websocket.MessageText, data)
}

// Events reads, maps, and reconnects the websocket stream.
func (connection *Connection) Events(ctx context.Context) iter.Seq2[realtime.CodecEvent, error] {
	return func(yield func(realtime.CodecEvent, error) bool) {
		for {
			connection.mu.RLock()
			socket := connection.socket
			connection.mu.RUnlock()
			messageType, data, err := socket.Read(ctx)
			if err != nil {
				if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || ctx.Err() != nil {
					return
				}
				reconnected, reconnectErr := connection.reconnect(ctx)
				if reconnectErr != nil {
					yield(nil, reconnectErr)
					return
				}
				if reconnected {
					if !yield(realtime.SessionReconnected{StateRestored: connection.config.RestoresInFlightState}, nil) {
						return
					}
					continue
				}
				yield(realtime.SessionError{Err: fmt.Errorf("websocket closed: %w", err)}, nil)
				return
			}
			if messageType != websocket.MessageText {
				continue
			}
			events, err := connection.config.Mapper(data)
			if err != nil {
				if !yield(realtime.SessionError{Err: err, Recoverable: true}, nil) {
					return
				}
				continue
			}
			for _, event := range events {
				if call, ok := event.(realtime.ToolCall); ok {
					connection.stateMu.Lock()
					_, duplicate := connection.seenToolCalls[call.ToolCallID]
					connection.seenToolCalls[call.ToolCallID] = struct{}{}
					connection.stateMu.Unlock()
					if duplicate {
						continue
					}
				}
				connection.observe(event)
				if _, internal := event.(ResponseCreated); internal {
					continue
				}
				if !yield(event, nil) {
					return
				}
			}
		}
	}
}

func (connection *Connection) observe(event realtime.CodecEvent) {
	connection.stateMu.Lock()
	defer connection.stateMu.Unlock()
	switch event := event.(type) {
	case realtime.AudioDelta:
		if event.ItemID != "" && event.ItemID != connection.currentItemID {
			connection.currentItemID = event.ItemID
			connection.generatedAudioBytes = 0
		}
		connection.generatedAudioBytes += len(event.Data)
	case ResponseCreated:
		connection.responseActive = true
		connection.activeResponseID = event.ResponseID
	case realtime.ResponseDone:
		connection.responseActive = false
		connection.activeResponseID = ""
	}
}

// ResponseCreated is consumed internally while preserving response state.
type ResponseCreated struct {
	ResponseID string
}

func (connection *Connection) reconnect(ctx context.Context) (bool, error) {
	policy := connection.config.Reconnect
	if policy == nil || connection.config.Dial == nil {
		return false, nil
	}
	connection.stateMu.Lock()
	if connection.reconnects >= policy.MaxReconnects {
		connection.stateMu.Unlock()
		return false, fmt.Errorf("realtime: reconnect limit reached")
	}
	connection.stateMu.Unlock()
	var last error
	for attempt := 0; attempt < policy.MaxAttempts; attempt++ {
		if attempt > 0 || policy.BaseDelay > 0 {
			delay := policy.BaseDelay << attempt
			if delay > policy.MaxDelay {
				delay = policy.MaxDelay
			}
			if policy.Jitter && delay > 0 {
				delay = time.Duration(rand.Int64N(int64(delay) + 1))
			}
			timer := time.NewTimer(delay)
			select {
			case <-timer.C:
			case <-ctx.Done():
				timer.Stop()
				return false, context.Cause(ctx)
			}
		}
		connection.mu.RLock()
		history := connection.history
		connection.mu.RUnlock()
		var messages []ai.ModelMessage
		if history != nil {
			messages = history()
		}
		socket, model, err := connection.config.Dial(ctx, messages)
		if err == nil && socket == nil {
			err = fmt.Errorf("realtime: reconnect dial returned a nil socket")
		}
		if err != nil {
			last = err
			continue
		}
		connection.mu.Lock()
		old := connection.socket
		connection.socket = socket
		connection.model = model
		connection.mu.Unlock()
		_ = old.Close(websocket.StatusGoingAway, "reconnected")
		connection.stateMu.Lock()
		connection.reconnects++
		if !connection.config.RestoresInFlightState {
			connection.responseActive = false
			connection.activeResponseID = ""
			connection.currentItemID = ""
			connection.generatedAudioBytes = 0
		}
		connection.stateMu.Unlock()
		return true, nil
	}
	return false, fmt.Errorf("realtime: reconnect failed: %w", last)
}

// Close releases the websocket. It is idempotent.
func (connection *Connection) Close(context.Context) error {
	connection.mu.Lock()
	if connection.closed {
		connection.mu.Unlock()
		return nil
	}
	connection.closed = true
	socket := connection.socket
	connection.mu.Unlock()
	if err := socket.Close(websocket.StatusNormalClosure, "session closed"); err != nil {
		var closeErr websocket.CloseError
		if !errors.As(err, &closeErr) {
			return err
		}
	}
	return nil
}

// DialSocket opens one websocket with headers and an injected HTTP client.
func DialSocket(
	ctx context.Context, url string, headers http.Header, client *http.Client,
) (*websocket.Conn, *http.Response, error) {
	return websocket.Dial(ctx, url, &websocket.DialOptions{HTTPHeader: headers, HTTPClient: client})
}

func dataURL(content ai.BinaryContent) string {
	return "data:" + content.MediaType + ";base64," + base64.StdEncoding.EncodeToString(content.Data)
}

func userContent(contents []ai.UserContent, supportsImages bool) ([]any, error) {
	parts := make([]any, 0, len(contents))
	for _, content := range contents {
		switch content := content.(type) {
		case ai.TextContent:
			parts = append(parts, map[string]any{"type": "input_text", "text": content.Text})
		case ai.BinaryContent:
			if !supportsImages || len(content.MediaType) < 6 || content.MediaType[:6] != "image/" {
				return nil, fmt.Errorf("realtime: tool result content %q is unsupported", content.MediaType)
			}
			parts = append(parts, map[string]any{"type": "input_image", "image_url": dataURL(content)})
		case ai.CachePoint:
		case nil:
		default:
			return nil, fmt.Errorf("realtime: tool result content %T is unsupported", content)
		}
	}
	return parts, nil
}

// CloneHeader returns a detached header map.
func CloneHeader(headers http.Header) http.Header {
	cloned := make(http.Header, len(headers))
	for key, values := range headers {
		cloned[key] = slices.Clone(values)
	}
	return cloned
}
