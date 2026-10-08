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
	"strconv"
	"strings"
	"sync"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	"github.com/coder/websocket"
)

// Dial opens and fully configures one websocket session.
type Dial func(ctx context.Context, history []ai.ModelMessage) (*websocket.Conn, string, error)

// Mapper translates one provider frame.
type Mapper func(data []byte) ([]realtime.CodecEvent, error)

// Config defines one OpenAI-protocol connection.
type Config struct {
	Provider                   string
	Model                      string
	Socket                     *websocket.Conn
	ServerModel                string
	Dial                       Dial
	Mapper                     Mapper
	Reconnect                  *realtime.ReconnectPolicy
	InputTranscriptionEnabled  bool
	RestoresInFlightState      bool
	InterruptsResponseOnSpeech bool
	SupportsImages             bool
	OutputSampleRate           int
	ManualAudioTurns           bool
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
	inputsReceived      int
	responseInputs      []int
	deferredInputs      []int
	seenToolCalls       map[string]struct{}
	idleTimeoutItems    map[string]bool
	toolBatches         map[string]*toolBatch
	toolResponses       map[string]string
	reconnecting        bool
	gaveUp              bool
	manualMu            sync.Mutex
	commitHeld          bool
	commitAnnounced     bool
	commitListener      func()
	heldAudio           []map[string]any
	heldCommitted       int
	sentAudio           []map[string]any
	audioUncommitted    bool
	audioLatest         bool
	speechDetected      bool
}

type toolBatch struct {
	unanswered map[string]bool
	input      int
	done       bool
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
		toolBatches: map[string]*toolBatch{}, toolResponses: map[string]string{}, idleTimeoutItems: map[string]bool{},
	}, nil
}

// IsReconnecting reports that the current transport is being replaced.
func (connection *Connection) IsReconnecting() bool {
	connection.stateMu.Lock()
	defer connection.stateMu.Unlock()
	return connection.reconnecting
}

// CanReconnect reports whether another transport recovery is available.
func (connection *Connection) CanReconnect() bool {
	connection.stateMu.Lock()
	defer connection.stateMu.Unlock()
	return !connection.gaveUp && connection.config.Reconnect != nil && connection.config.Dial != nil &&
		connection.reconnects < connection.config.Reconnect.MaxReconnects
}

// AnswersToolCallsPerResponse reports that every response's tool results share one reply.
func (*Connection) AnswersToolCallsPerResponse() bool { return true }

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

// InterruptsResponseOnSpeech reports whether server VAD cancels active output.
func (connection *Connection) InterruptsResponseOnSpeech() bool {
	return connection.config.InterruptsResponseOnSpeech
}

// SetMessageHistory installs the live history callback used by reconnect replay.
func (connection *Connection) SetMessageHistory(history func() []ai.ModelMessage) {
	connection.mu.Lock()
	defer connection.mu.Unlock()
	connection.history = history
}

// Send writes one normalized input frame.
func (connection *Connection) Send(ctx context.Context, input realtime.Input) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if connection.config.ManualAudioTurns {
		switch input.(type) {
		case realtime.AudioInput, realtime.CommitAudio, realtime.ClearAudio, realtime.CreateResponse,
			realtime.CancelResponse, realtime.TruncateOutput:
		default:
			connection.manualMu.Lock()
			connection.audioLatest = false
			connection.manualMu.Unlock()
		}
	}
	connection.stateMu.Lock()
	inputIndex := connection.inputsReceived
	connection.inputsReceived++
	connection.stateMu.Unlock()
	switch input := input.(type) {
	case realtime.AudioInput:
		if len(input.Data)%2 != 0 {
			return fmt.Errorf("realtime: PCM16 audio length must be even")
		}
		if connection.IsReconnecting() {
			return nil
		}
		return connection.writeJSON(ctx, map[string]any{
			"type": "input_audio_buffer.append", "audio": base64.StdEncoding.EncodeToString(input.Data),
		})
	case realtime.TextInput:
		return connection.sendText(ctx, input.Text, true, inputIndex)
	case realtime.TextContext:
		return connection.sendText(ctx, input.Text, false, inputIndex)
	case realtime.ImageInput:
		if !connection.config.SupportsImages {
			return fmt.Errorf("realtime: %s does not support image input", connection.config.Provider)
		}
		if err := connection.writeJSON(ctx, map[string]any{
			"type": "conversation.item.create", "event_id": clientEventID("content", []int{inputIndex}),
			"item": map[string]any{
				"type": "message", "role": "user",
				"content": []any{map[string]any{
					"type": "input_image", "image_url": dataURL(input.Content),
				}},
			},
		}); err != nil || !input.Respond {
			return err
		}
		return connection.requestResponse(ctx, []int{inputIndex})
	case realtime.ToolResult:
		parts, err := userContent(input.Content, connection.config.SupportsImages)
		if err != nil {
			return err
		}
		items := []map[string]any{{"type": "function_call_output", "call_id": input.ToolCallID, "output": input.Output}}
		if len(parts) > 0 {
			items = append(items, map[string]any{"type": "message", "role": "user", "content": parts})
		}
		for _, item := range items {
			if err := connection.writeJSON(ctx, map[string]any{"type": "conversation.item.create", "item": item}); err != nil {
				return err
			}
		}
		connection.stateMu.Lock()
		responseID, batched := connection.toolResponses[input.ToolCallID]
		if batched {
			delete(connection.toolResponses, input.ToolCallID)
			batch := connection.toolBatches[responseID]
			delete(batch.unanswered, input.ToolCallID)
			batch.input = inputIndex
		}
		connection.stateMu.Unlock()
		if batched {
			return connection.answerToolBatch(ctx, responseID)
		}
		return connection.requestResponse(ctx, []int{inputIndex})
	case realtime.CommitAudio:
		return connection.writeJSON(ctx, map[string]any{"type": "input_audio_buffer.commit"})
	case realtime.ClearAudio:
		return connection.writeJSON(ctx, map[string]any{"type": "input_audio_buffer.clear"})
	case realtime.CreateResponse:
		return connection.requestResponse(ctx, []int{inputIndex})
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

func (connection *Connection) sendText(ctx context.Context, text string, respond bool, inputIndex int) error {
	if err := connection.writeJSON(ctx, map[string]any{
		"type": "conversation.item.create", "event_id": clientEventID("content", []int{inputIndex}),
		"item": map[string]any{
			"type": "message", "role": "user",
			"content": []any{map[string]any{"type": "input_text", "text": text}},
		},
	}); err != nil || !respond {
		return err
	}
	return connection.requestResponse(ctx, []int{inputIndex})
}

func (connection *Connection) requestResponse(ctx context.Context, inputs []int) error {
	connection.mu.RLock()
	closed := connection.closed
	connection.mu.RUnlock()
	if closed {
		return fmt.Errorf("realtime: connection is closed")
	}
	connection.stateMu.Lock()
	if connection.responseActive {
		connection.deferredInputs = append(connection.deferredInputs, inputs...)
		connection.stateMu.Unlock()
		return nil
	}
	connection.responseActive = true
	connection.responseInputs = slices.Clone(inputs)
	connection.stateMu.Unlock()
	event := map[string]any{"type": "response.create"}
	if len(inputs) > 0 {
		event["event_id"] = clientEventID("response", inputs)
	}
	if err := connection.writeJSON(ctx, event); err != nil {
		connection.stateMu.Lock()
		connection.responseActive = false
		connection.responseInputs = nil
		connection.stateMu.Unlock()
		return err
	}
	return nil
}

func clientEventID(kind string, inputs []int) string {
	parts := make([]string, len(inputs))
	for index, input := range inputs {
		parts[index] = strconv.Itoa(input)
	}
	return "pydantic_ai." + kind + "." + strings.Join(parts, "-")
}

func (connection *Connection) writeJSON(ctx context.Context, value any) error {
	if connection.config.ManualAudioTurns {
		connection.manualMu.Lock()
		defer connection.manualMu.Unlock()
		return connection.writeManual(ctx, value.(map[string]any))
	}
	return connection.writeRaw(ctx, value)
}

func (connection *Connection) writeRaw(ctx context.Context, value any) error {
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
				if !connection.config.RestoresInFlightState {
					connection.stateMu.Lock()
					active, responseID := connection.responseActive, connection.activeResponseID
					connection.stateMu.Unlock()
					if active && !yield(realtime.ResponseDone{Interrupted: true, ProviderResponseID: responseID}, nil) {
						return
					}
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
			var lifecycle struct {
				Type   string `json:"type"`
				ItemID string `json:"item_id"`
			}
			_ = json.Unmarshal(data, &lifecycle)
			if lifecycle.Type == "input_audio_buffer.timeout_triggered" && connection.config.InputTranscriptionEnabled {
				connection.idleTimeoutItems[lifecycle.ItemID] = true
			}
			events, err := connection.config.Mapper(data)
			if err != nil {
				if !yield(realtime.SessionError{Err: err, Recoverable: true}, nil) {
					return
				}
				continue
			}
			for _, event := range events {
				if transcript, ok := event.(realtime.InputTranscript); ok && connection.idleTimeoutItems[transcript.ItemID] {
					if transcript.Final {
						delete(connection.idleTimeoutItems, transcript.ItemID)
					}
					continue
				}
				if failed, ok := event.(realtime.InputTranscriptionError); ok && connection.idleTimeoutItems[failed.ItemID] {
					delete(connection.idleTimeoutItems, failed.ItemID)
					continue
				}
				if call, ok := event.(realtime.ToolCall); ok {
					connection.stateMu.Lock()
					_, duplicate := connection.seenToolCalls[call.ToolCallID]
					connection.seenToolCalls[call.ToolCallID] = struct{}{}
					if !duplicate && call.ResponseID != "" {
						batch := connection.toolBatches[call.ResponseID]
						if batch == nil {
							batch = &toolBatch{unanswered: map[string]bool{}}
							connection.toolBatches[call.ResponseID] = batch
						}
						batch.unanswered[call.ToolCallID] = true
						connection.toolResponses[call.ToolCallID] = call.ResponseID
					}
					connection.stateMu.Unlock()
					if duplicate {
						continue
					}
				}
				if rejected, ok := event.(realtime.InputRejected); ok && rejected.Response {
					if err := connection.rejectResponse(ctx, rejected.InputIndex); err != nil {
						yield(nil, err)
						return
					}
				}
				if _, started := event.(realtime.ResponseStarted); started {
					connection.stateMu.Lock()
					merged := max(0, len(connection.responseInputs)-1)
					connection.stateMu.Unlock()
					if merged > 0 && !yield(realtime.ResponseRequestsMerged{Count: merged}, nil) {
						return
					}
				}
				connection.observe(event)
				if done, ok := event.(realtime.ResponseDone); ok {
					connection.stateMu.Lock()
					batch := connection.toolBatches[done.ProviderResponseID]
					if batch != nil {
						batch.done = true
					}
					connection.stateMu.Unlock()
					if batch != nil {
						if err := connection.answerToolBatch(ctx, done.ProviderResponseID); err != nil {
							yield(nil, err)
							return
						}
					}
					if err := connection.sendDeferredResponse(ctx); err != nil {
						yield(nil, err)
						return
					}
				}
				if !yield(event, nil) {
					return
				}
				if sessionError, ok := event.(realtime.SessionError); ok && !sessionError.Recoverable {
					return
				}
				if _, done := event.(realtime.ResponseDone); done && connection.config.ManualAudioTurns {
					connection.manualMu.Lock()
					err := connection.flushHeldAudio(ctx, false)
					connection.manualMu.Unlock()
					if err != nil {
						yield(nil, err)
						return
					}
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
		connection.currentContentIndex = event.ContentIndex
	case realtime.InputSpeechStarted:
		if connection.config.ManualAudioTurns {
			connection.speechDetected = true
		}
	case realtime.ResponseStarted:
		connection.speechDetected = false
		connection.responseActive = true
		connection.activeResponseID = event.ResponseID
	case realtime.ResponseDone:
		connection.responseActive = false
		connection.activeResponseID = ""
		connection.responseInputs = nil
	}
}

func (connection *Connection) rejectResponse(ctx context.Context, inputIndex int) error {
	connection.stateMu.Lock()
	matched := connection.responseActive && connection.activeResponseID == "" && slices.Contains(connection.responseInputs, inputIndex)
	if matched {
		connection.responseActive = false
		connection.responseInputs = nil
	}
	connection.stateMu.Unlock()
	if matched {
		return connection.sendDeferredResponse(ctx)
	}
	return nil
}

func (connection *Connection) answerToolBatch(ctx context.Context, responseID string) error {
	connection.stateMu.Lock()
	batch := connection.toolBatches[responseID]
	ready := batch != nil && batch.done && len(batch.unanswered) == 0
	input := 0
	if ready {
		input = batch.input
		delete(connection.toolBatches, responseID)
	}
	connection.stateMu.Unlock()
	if ready {
		return connection.requestResponse(ctx, []int{input})
	}
	return nil
}

func (connection *Connection) sendDeferredResponse(ctx context.Context) error {
	connection.stateMu.Lock()
	if connection.responseActive || len(connection.deferredInputs) == 0 {
		connection.stateMu.Unlock()
		return nil
	}
	inputs := slices.Clone(connection.deferredInputs)
	connection.deferredInputs = nil
	connection.stateMu.Unlock()
	return connection.requestResponse(ctx, inputs)
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
	connection.reconnecting = true
	connection.stateMu.Unlock()
	if connection.config.ManualAudioTurns {
		connection.manualMu.Lock()
		if connection.commitHeld {
			connection.heldAudio = append(slices.Clone(connection.sentAudio), connection.heldAudio...)
			connection.heldCommitted += len(connection.sentAudio)
		}
		connection.sentAudio = nil
		connection.audioLatest = false
		connection.audioUncommitted = false
		connection.stateMu.Lock()
		connection.speechDetected = false
		connection.stateMu.Unlock()
		connection.manualMu.Unlock()
	}
	defer func() {
		connection.stateMu.Lock()
		connection.reconnecting = false
		connection.stateMu.Unlock()
	}()
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
			for index, message := range messages {
				switch message := message.(type) {
				case ai.ModelRequest:
					var parts []ai.RequestPart
					for _, part := range message.Parts {
						switch part := part.(type) {
						case ai.SpeechPart:
							if part.Transcript == nil || *part.Transcript == "" {
								parts = append(parts, ai.UserPromptPart{Content: "[The user spoke; no transcript is available.]"})
							} else {
								part.Audio = nil
								parts = append(parts, part)
							}
						case ai.UserPromptPart:
							part.Contents = slices.DeleteFunc(slices.Clone(part.Contents), func(content ai.UserContent) bool {
								_, text := content.(ai.TextContent)
								return !text
							})
							parts = append(parts, part)
						default:
							parts = append(parts, part)
						}
					}
					message.Parts = parts
					messages[index] = message
				case ai.ModelResponse:
					message.Parts = slices.Clone(message.Parts)
					for index, part := range message.Parts {
						if speech, ok := part.(ai.SpeechPart); ok {
							speech.Audio = nil
							message.Parts[index] = speech
						}
					}
					messages[index] = message
				}
			}
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
			connection.responseInputs = nil
			connection.deferredInputs = nil
			clear(connection.toolBatches)
			clear(connection.toolResponses)
			connection.currentItemID = ""
			connection.generatedAudioBytes = 0
		}
		connection.stateMu.Unlock()
		return true, nil
	}
	connection.stateMu.Lock()
	connection.gaveUp = true
	connection.stateMu.Unlock()
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
