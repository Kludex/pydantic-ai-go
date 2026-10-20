package elevenlabs

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
	"github.com/coder/websocket"
)

// audioTailGrace is how long a reply whose text is complete can go without
// audio before it counts as spoken in full. agent_response arrives when the
// LLM finishes writing, while TTS can still be voicing it. Only an agent that
// sends context_usage marks the end of the audio exactly.
const audioTailGrace = time.Second

// toolCallBatchWindow is how long a reply's tool calls wait for a sibling
// before their response ends. Parallel calls arrive as separate frames, back
// to back, and nothing else follows until their results are in.
const toolCallBatchWindow = 50 * time.Millisecond

// Frames that say a reply whose text is complete has no more audio coming.
var tailEndingFrames = map[string]bool{
	"context_usage": true, "user_transcript": true, "client_tool_call": true, "agent_response": true,
	"agent_chat_response_part": true, "mcp_tool_call": true,
}

const mcpApprovalDenied = "The ElevenLabs agent asked to approve a call to one of its MCP server tools, which a " +
	"Pydantic AI session cannot do, so the call was denied. Give the agent the tools through a Pydantic AI MCP " +
	"toolset instead, or turn approval off for the tool on the ElevenLabs side."

type initiationMetadata struct {
	ConversationID         string `json:"conversation_id"`
	AgentOutputAudioFormat string `json:"agent_output_audio_format"`
	UserInputAudioFormat   string `json:"user_input_audio_format"`
}

// handshake sends the initiation payload and reads frames until the
// conversation metadata arrives, answering pings meanwhile.
func handshake(ctx context.Context, socket *websocket.Conn, payload map[string]any) (initiationMetadata, error) {
	if err := writeJSON(ctx, socket, payload); err != nil { // pragma: no cover - a just-accepted socket only fails to write if the peer drops in between
		return initiationMetadata{}, err
	}
	for {
		kind, data, err := socket.Read(ctx)
		if err != nil {
			return initiationMetadata{}, err
		}
		if kind != websocket.MessageText {
			return initiationMetadata{}, errors.New("expected a text frame")
		}
		var frame struct {
			Type     string             `json:"type"`
			Metadata initiationMetadata `json:"conversation_initiation_metadata_event"`
			Ping     json.RawMessage    `json:"ping_event"`
		}
		if err := json.Unmarshal(data, &frame); err != nil {
			return initiationMetadata{}, fmt.Errorf("received a malformed frame: %w", err)
		}
		switch frame.Type {
		case "conversation_initiation_metadata":
			return frame.Metadata, nil
		case "ping":
			if err := pong(ctx, socket, frame.Ping); err != nil { // pragma: no cover - fails only if the peer drops between its ping and our answer
				return initiationMetadata{}, err
			}
		case "client_error", "error":
			return initiationMetadata{}, fmt.Errorf("the server rejected the conversation: %s", data)
		}
	}
}

func pong(ctx context.Context, socket *websocket.Conn, ping json.RawMessage) error {
	var event struct {
		EventID any `json:"event_id"`
	}
	_ = json.Unmarshal(ping, &event)
	return writeJSON(ctx, socket, map[string]any{"type": "pong", "event_id": event.EventID})
}

func writeJSON(ctx context.Context, socket *websocket.Conn, value any) error {
	data, _ := json.Marshal(value)
	return socket.Write(ctx, websocket.MessageText, data)
}

// audioTail is a reply whose whole text arrived while its audio may still stream.
type audioTail struct {
	eventID     *int
	responseID  string
	interrupted bool
	details     map[string]any
}

type frame struct {
	kind websocket.MessageType
	data []byte
	err  error
}

// Connection is one ElevenLabs Agents conversation. Send and Close are safe
// alongside one Events consumer.
type Connection struct {
	socket         *websocket.Conn
	conversationID string
	textOutput     bool
	warn           func(string)

	frames chan frame
	stop   chan struct{}
	once   sync.Once

	mu                 sync.Mutex
	contextLimitTokens int
	tagStripper        *audioTagStripper
	fireAndForget      map[string]bool
	responseOpen       bool
	turnInterrupted    bool
	tail               *audioTail
	callsOpen          bool
	mcpCalls           map[string]bool
	mcpDenied          []ai.NativeToolReturnPart
	warnedMCPApproval  bool
	nativePartIndex    int
	queued             bool
}

func newConnection(socket *websocket.Conn, conversationID string, textOutput, stripTags bool, warn func(string)) *Connection {
	connection := &Connection{
		socket: socket, conversationID: conversationID, textOutput: textOutput, warn: warn,
		frames: make(chan frame, 1), stop: make(chan struct{}),
		fireAndForget: map[string]bool{}, mcpCalls: map[string]bool{},
	}
	if stripTags {
		connection.tagStripper = &audioTagStripper{}
	}
	go func() {
		for {
			kind, data, err := socket.Read(context.Background())
			select {
			case connection.frames <- frame{kind: kind, data: data, err: err}:
			case <-connection.stop:
				return
			}
			if err != nil {
				return
			}
		}
	}()
	return connection
}

// ConversationID returns the server-assigned conversation ID, for looking up
// the conversation's cost through the ElevenLabs conversations API.
func (connection *Connection) ConversationID() string { return connection.conversationID }

// ContextLimitTokens returns the context window of the agent's LLM as last
// reported by context_usage, or zero before the agent reports one.
func (connection *Connection) ContextLimitTokens() int {
	connection.mu.Lock()
	defer connection.mu.Unlock()
	return connection.contextLimitTokens
}

// ModelName returns empty: ElevenLabs reports no realtime model ID, so
// responses carry the agent ID.
func (*Connection) ModelName() string { return "" }

// InputTranscriptionEnabled reports that speech recognition is always on.
func (*Connection) InputTranscriptionEnabled() bool { return true }

// ReconnectRestoresInFlightState reports that dropped conversations are not resumed.
func (*Connection) ReconnectRestoresInFlightState() bool { return false }

// Send writes PCM16 audio, a text turn, context the agent reads without
// replying (TextContext), or a tool result.
func (connection *Connection) Send(ctx context.Context, input realtime.Input) error {
	switch input := input.(type) {
	case realtime.AudioInput:
		if len(input.Data)%2 != 0 {
			return errors.New("elevenlabs: PCM16 audio length must be even")
		}
		// The one client message without a type discriminator.
		return writeJSON(ctx, connection.socket, map[string]any{"user_audio_chunk": base64.StdEncoding.EncodeToString(input.Data)})
	case realtime.TextInput:
		return writeJSON(ctx, connection.socket, map[string]any{"type": "user_message", "text": input.Text})
	case realtime.TextContext:
		return writeJSON(ctx, connection.socket, map[string]any{"type": "contextual_update", "text": input.Text})
	case realtime.ToolResult:
		connection.mu.Lock()
		fireAndForget := connection.fireAndForget[input.ToolCallID]
		delete(connection.fireAndForget, input.ToolCallID)
		connection.mu.Unlock()
		if fireAndForget {
			// The agent asked for no result.
			return nil
		}
		result, err := toolOutput(input)
		if err != nil {
			return err
		}
		return writeJSON(ctx, connection.socket, map[string]any{
			"type": "client_tool_result", "tool_call_id": input.ToolCallID, "result": result, "is_error": false,
		})
	default:
		return fmt.Errorf("elevenlabs: %T input is not supported", input)
	}
}

// toolOutput folds text attachments into the output; the agent takes text only.
func toolOutput(result realtime.ToolResult) (string, error) {
	parts := []string{}
	if result.Output != "" {
		parts = append(parts, result.Output)
	}
	for _, content := range result.Content {
		switch content := content.(type) {
		case ai.TextContent:
			if content.Text != "" {
				parts = append(parts, content.Text)
			}
		case ai.CachePoint:
		default:
			return "", fmt.Errorf(
				"elevenlabs: tool results are text-only, so %T content attached to a tool return cannot be delivered",
				content,
			)
		}
	}
	return strings.Join(parts, "\n\n"), nil
}

// Close ends the conversation. It is idempotent.
func (connection *Connection) Close(context.Context) error {
	var err error
	connection.once.Do(func() {
		close(connection.stop)
		err = connection.socket.CloseNow()
	})
	return err
}

// Events yields normalized events until the conversation ends. A dropped
// conversation is fatal: it ends with a non-recoverable session error.
func (connection *Connection) Events(ctx context.Context) iter.Seq2[realtime.CodecEvent, error] {
	return func(yield func(realtime.CodecEvent, error) bool) {
		emit := func(events []realtime.CodecEvent) bool {
			for _, event := range events {
				if !yield(event, nil) {
					return false
				}
			}
			return true
		}
		for {
			connection.mu.Lock()
			wait := time.Duration(-1)
			if connection.callsOpen {
				wait = toolCallBatchWindow
			} else if connection.tail != nil {
				wait = audioTailGrace
			}
			connection.mu.Unlock()
			var timeout <-chan time.Time
			var timer *time.Timer
			if wait >= 0 {
				timer = time.NewTimer(wait)
				timeout = timer.C
			}
			stopTimer := func() {
				if timer != nil {
					timer.Stop()
				}
			}
			select {
			case <-ctx.Done():
				stopTimer()
				return
			case <-connection.stop:
				stopTimer()
				return
			case <-timeout:
				connection.mu.Lock()
				var events []realtime.CodecEvent
				if connection.callsOpen {
					events = connection.endCallsLocked()
				} else {
					events = connection.endTailLocked()
				}
				connection.mu.Unlock()
				if !emit(events) {
					return
				}
			case received := <-connection.frames:
				stopTimer()
				if received.err != nil {
					emit(connection.connectionLost(received.err))
					return
				}
				if received.kind != websocket.MessageText {
					continue
				}
				events, err := connection.mapFrame(ctx, received.data)
				if err != nil {
					events = append(events, realtime.SessionError{
						Err: fmt.Errorf("elevenlabs: failed to parse event: %w", err), Recoverable: true,
					})
				}
				if !emit(events) {
					return
				}
			}
		}
	}
}

func (connection *Connection) connectionLost(err error) []realtime.CodecEvent {
	connection.mu.Lock()
	defer connection.mu.Unlock()
	message := fmt.Errorf("elevenlabs: connection closed: %w", err)
	if websocket.CloseStatus(err) == 4300 {
		message = fmt.Errorf("elevenlabs: the conversation timed out in the workspace's concurrency queue before " +
			"the agent was free; retry later, or raise the workspace concurrency limit")
	}
	connection.tail = nil
	var events []realtime.CodecEvent
	if connection.callsOpen {
		events = connection.endCallsLocked()
	}
	events = append(events, connection.deniedPartsLocked()...)
	if connection.responseOpen {
		connection.responseOpen = false
		events = append(events, realtime.ResponseDone{Interrupted: true, ProviderDetails: connection.detailsLocked(nil)})
	}
	return append(events, realtime.SessionError{Err: message})
}

type envelope struct {
	Type        string                   `json:"type"`
	Ping        json.RawMessage          `json:"ping_event"`
	QueueStatus *struct{ Status string } `json:"queue_status_event"`
	Audio       *struct {
		Audio   string `json:"audio_base_64"`
		EventID *int   `json:"event_id"`
	} `json:"audio_event"`
	ClientToolCall *struct {
		ToolName        string          `json:"tool_name"`
		ToolCallID      string          `json:"tool_call_id"`
		Parameters      json.RawMessage `json:"parameters"`
		ExpectsResponse *bool           `json:"expects_response"`
	} `json:"client_tool_call"`
	MCPToolCall  *mcpToolCall `json:"mcp_tool_call"`
	ContextUsage *struct {
		ContextTokens      int `json:"context_tokens"`
		ContextLimitTokens int `json:"context_limit_tokens"`
	} `json:"context_usage_event"`
	ClientError *struct {
		ErrorName string `json:"error_name"`
		Message   string `json:"message"`
	} `json:"error_event"`
	TextResponsePart *struct {
		Text string `json:"text"`
		Type string `json:"type"`
	} `json:"text_response_part"`
	AgentResponse *struct {
		AgentResponse string `json:"agent_response"`
		ResponseID    string `json:"response_id"`
		EventID       *int   `json:"event_id"`
	} `json:"agent_response_event"`
	Correction *struct {
		Corrected string `json:"corrected_agent_response"`
	} `json:"agent_response_correction_event"`
	UserTranscript *struct {
		Transcript string `json:"user_transcript"`
	} `json:"user_transcription_event"`
}

type mcpToolCall struct {
	ServiceID    string          `json:"service_id"`
	ToolCallID   string          `json:"tool_call_id"`
	ToolName     string          `json:"tool_name"`
	Parameters   map[string]any  `json:"parameters"`
	State        string          `json:"state"`
	Result       json.RawMessage `json:"result"`
	ErrorMessage string          `json:"error_message"`
}

func (connection *Connection) mapFrame(ctx context.Context, data []byte) ([]realtime.CodecEvent, error) {
	var event envelope
	if err := json.Unmarshal(data, &event); err != nil {
		return nil, err
	}
	if event.Type == "ping" {
		return nil, pong(ctx, connection.socket, event.Ping)
	}
	connection.mu.Lock()
	defer connection.mu.Unlock()
	var ended []realtime.CodecEvent
	if connection.callsOpen && event.Type != "client_tool_call" {
		ended = connection.endCallsLocked()
	}
	if connection.tail != nil && tailEndingFrames[event.Type] {
		ended = append(ended, connection.endTailLocked()...)
	}
	mapped, err := connection.mapEventLocked(ctx, event)
	return append(ended, mapped...), err
}

func (connection *Connection) mapEventLocked(ctx context.Context, event envelope) ([]realtime.CodecEvent, error) {
	switch event.Type {
	case "queue_status":
		if event.QueueStatus == nil {
			return nil, errors.New("queue_status frame without a payload")
		}
		// While waiting for a free slot the server plays hold audio over regular audio frames.
		connection.queued = event.QueueStatus.Status == "waiting"
		return nil, nil
	case "audio":
		if event.Audio == nil {
			return nil, errors.New("audio frame without a payload")
		}
		audio, err := base64.StdEncoding.DecodeString(event.Audio.Audio)
		if err != nil {
			return nil, err
		}
		if connection.queued {
			return nil, nil
		}
		var ended []realtime.CodecEvent
		if connection.tail != nil && !sameEventID(connection.tail.eventID, event.Audio.EventID) {
			// Audio of another turn: the reply whose text is complete was voiced in full.
			ended = connection.endTailLocked()
		}
		connection.responseOpen = true
		return append(ended, realtime.AudioDelta{Data: audio}), nil
	case "client_tool_call":
		call := event.ClientToolCall
		if call == nil {
			return nil, errors.New("client_tool_call frame without a payload")
		}
		// The calls end the response that makes them; the agent answers their results in a response of its own.
		connection.responseOpen = false
		connection.turnInterrupted = false
		if call.ExpectsResponse != nil && !*call.ExpectsResponse {
			connection.fireAndForget[call.ToolCallID] = true
		}
		arguments := "{}"
		var text string
		if json.Unmarshal(call.Parameters, &text) == nil {
			arguments = text
		} else if len(call.Parameters) > 0 && string(call.Parameters) != "null" {
			arguments = string(call.Parameters)
		}
		connection.callsOpen = true
		return []realtime.CodecEvent{realtime.ToolCall{
			ToolCallID: call.ToolCallID, ToolName: call.ToolName, Arguments: arguments,
		}}, nil
	case "mcp_tool_call":
		if event.MCPToolCall == nil {
			return nil, errors.New("mcp_tool_call frame without a payload")
		}
		return connection.mapMCPToolCallLocked(ctx, *event.MCPToolCall)
	case "context_usage":
		usage := event.ContextUsage
		if usage == nil {
			return nil, errors.New("context_usage frame without a payload")
		}
		if usage.ContextLimitTokens > 0 {
			connection.contextLimitTokens = usage.ContextLimitTokens
		}
		// Context consumption only; it belongs to no response, so it adds to the session total.
		result := realtime.SessionUsage{Usage: ai.Usage{InputTokens: usage.ContextTokens}}
		if connection.contextLimitTokens > 0 {
			used := float64(usage.ContextTokens) / float64(connection.contextLimitTokens)
			result.ContextWindowUsed = &used
		}
		return []realtime.CodecEvent{result}, nil
	case "client_error":
		message := "unknown error"
		if event.ClientError != nil {
			message = cmp.Or(event.ClientError.Message, event.ClientError.ErrorName, message)
		}
		return []realtime.CodecEvent{realtime.SessionError{Err: fmt.Errorf("elevenlabs: %s", message), Recoverable: true}}, nil
	case "agent_chat_response_part":
		part := event.TextResponsePart
		if part == nil || part.Type != "delta" || part.Text == "" {
			return nil, nil
		}
		connection.responseOpen = true
		text := part.Text
		if connection.tagStripper != nil {
			text = connection.tagStripper.feed(text)
		}
		if text == "" {
			return nil, nil
		}
		return []realtime.CodecEvent{realtime.OutputTranscript{Text: text, OutputText: connection.textOutput}}, nil
	case "agent_response":
		response := event.AgentResponse
		if response == nil {
			return nil, errors.New("agent_response frame without a payload")
		}
		connection.responseOpen = true
		text, details := response.AgentResponse, map[string]any(nil)
		if connection.tagStripper != nil {
			connection.tagStripper = &audioTagStripper{}
			var tags []string
			text, tags = stripAudioTags(text, false)
			if len(tags) > 0 {
				details = map[string]any{"audio_tags": tags}
			}
		}
		transcript := realtime.OutputTranscript{Text: text, Final: true, OutputText: connection.textOutput}
		if connection.textOutput || connection.turnInterrupted {
			return append([]realtime.CodecEvent{transcript},
				connection.finalizeLocked(connection.turnInterrupted, response.ResponseID, details)...), nil
		}
		// In audio mode this text arrives while TTS can still be voicing it; the reply ends with its last audio.
		connection.tail = &audioTail{eventID: response.EventID, responseID: response.ResponseID, details: details}
		return []realtime.CodecEvent{transcript}, nil
	case "agent_response_correction":
		if event.Correction == nil {
			return nil, errors.New("agent_response_correction frame without a payload")
		}
		corrected := event.Correction.Corrected
		if connection.tagStripper != nil {
			corrected, _ = stripAudioTags(corrected, false)
		}
		// The streamed transcript cannot shrink, so the text that was heard is kept on the response.
		details := map[string]any{"corrected_agent_response": corrected}
		if tail := connection.tail; tail != nil {
			tail.interrupted = true
			tail.details = mergeDetails(tail.details, details)
			return connection.endTailLocked(), nil
		}
		return connection.finalizeLocked(true, "", details), nil
	case "agent_response_complete":
		return connection.finalizeLocked(connection.turnInterrupted, "", nil), nil
	case "user_transcript":
		if event.UserTranscript == nil {
			return nil, errors.New("user_transcript frame without a payload")
		}
		return []realtime.CodecEvent{realtime.InputTranscript{Text: event.UserTranscript.Transcript, Final: true}}, nil
	case "interruption":
		// During playback of a finished reply the response has already ended, so the flag must not leak.
		if connection.responseOpen {
			connection.turnInterrupted = true
		}
		if connection.tail != nil {
			connection.tail.interrupted = true
		}
		return nil, nil
	default:
		// VAD scores, guardrail notices, and newer frames are informational.
		return nil, nil
	}
}

func (connection *Connection) mapMCPToolCallLocked(ctx context.Context, call mcpToolCall) ([]realtime.CodecEvent, error) {
	toolName := "mcp_server"
	if call.ServiceID != "" {
		toolName += ":" + call.ServiceID
	}
	var events []realtime.CodecEvent
	if !connection.mcpCalls[call.ToolCallID] {
		connection.mcpCalls[call.ToolCallID] = true
		arguments, _ := json.Marshal(map[string]any{
			"action": "call_tool", "tool_name": call.ToolName, "tool_args": orEmpty(call.Parameters),
		})
		events = connection.nativePartLocked(ai.NativeToolCallPart{
			ToolName: toolName, Args: arguments, ToolCallID: call.ToolCallID, ProviderName: "elevenlabs",
			ToolKind: ai.ToolPartKindMCPServer,
		})
	}
	switch call.State {
	case "awaiting_approval":
		// Nothing in the session can approve a call ElevenLabs runs itself, and the agent waits for an answer.
		if !connection.warnedMCPApproval {
			connection.warnedMCPApproval = true
			connection.warn("elevenlabs: " + mcpApprovalDenied)
		}
		if err := writeJSON(ctx, connection.socket, map[string]any{
			"type": "mcp_tool_approval_result", "tool_call_id": call.ToolCallID, "is_approved": false,
		}); err != nil { // pragma: no cover - fails only if the peer drops between its request and our answer
			return events, err
		}
		connection.mcpDenied = append(connection.mcpDenied, ai.NativeToolReturnPart{
			ToolName: toolName, Content: mcpApprovalDenied, ToolCallID: call.ToolCallID, ProviderName: "elevenlabs",
			Outcome: ai.ToolReturnOutcomeDenied, ToolKind: ai.ToolPartKindMCPServer,
		})
	case "success", "failure":
		part := ai.NativeToolReturnPart{
			ToolName: toolName, ToolCallID: call.ToolCallID, ProviderName: "elevenlabs", ToolKind: ai.ToolPartKindMCPServer,
			Outcome: ai.ToolReturnOutcomeSuccess,
		}
		denied := -1
		for index, pending := range connection.mcpDenied {
			if pending.ToolCallID == call.ToolCallID {
				denied = index
			}
		}
		switch {
		case call.State == "success":
			var result any
			_ = json.Unmarshal(call.Result, &result)
			part.Content = result
		case denied >= 0:
			// A call denied for want of an approval is reported as failed; it stays the denial it was.
			part.Content, part.Outcome = connection.mcpDenied[denied].Content, ai.ToolReturnOutcomeDenied
		default:
			part.Content, part.Outcome = call.ErrorMessage, ai.ToolReturnOutcomeFailed
		}
		if denied >= 0 {
			connection.mcpDenied = append(connection.mcpDenied[:denied], connection.mcpDenied[denied+1:]...)
		}
		events = append(events, connection.nativePartLocked(part)...)
	}
	return events, nil
}

func (connection *Connection) nativePartLocked(part ai.ResponsePart) []realtime.CodecEvent {
	connection.responseOpen = true
	index := connection.nativePartIndex
	connection.nativePartIndex++
	partID := fmt.Sprintf("elevenlabs-native-%d", index)
	return []realtime.CodecEvent{
		realtime.PartStarted{Event: ai.PartStartEvent{Index: index, PartID: partID, Part: part}},
		realtime.PartEnded{Event: ai.PartEndEvent{Index: index, PartID: partID, Part: part}},
	}
}

func (connection *Connection) deniedPartsLocked() []realtime.CodecEvent {
	var events []realtime.CodecEvent
	for _, part := range connection.mcpDenied {
		events = append(events, connection.nativePartLocked(part)...)
	}
	connection.mcpDenied = nil
	return events
}

// endCallsLocked ends the response whose tool calls are all in.
func (connection *Connection) endCallsLocked() []realtime.CodecEvent {
	connection.callsOpen = false
	return []realtime.CodecEvent{realtime.ResponseDone{
		MoreExpected: true, FinishReason: ai.FinishReasonToolCall, ProviderDetails: connection.detailsLocked(nil),
	}}
}

// endTailLocked ends the reply whose text is complete.
func (connection *Connection) endTailLocked() []realtime.CodecEvent {
	tail := connection.tail
	connection.tail = nil
	return connection.finalizeLocked(tail.interrupted, tail.responseID, tail.details)
}

// finalizeLocked closes the open turn; a boundary with nothing streamed is dropped.
func (connection *Connection) finalizeLocked(interrupted bool, responseID string, details map[string]any) []realtime.CodecEvent {
	connection.turnInterrupted = false
	if !connection.responseOpen {
		return nil
	}
	events := connection.deniedPartsLocked()
	connection.nativePartIndex = 0
	connection.responseOpen = false
	return append(events, realtime.ResponseDone{
		Interrupted: interrupted, ProviderResponseID: responseID, ProviderDetails: connection.detailsLocked(details),
	})
}

// detailsLocked adds the conversation ID, the key for a post-hoc cost lookup,
// since nothing billable reaches the socket.
func (connection *Connection) detailsLocked(details map[string]any) map[string]any {
	if connection.conversationID == "" {
		return details
	}
	return mergeDetails(map[string]any{"conversation_id": connection.conversationID}, details)
}

func mergeDetails(base, extra map[string]any) map[string]any {
	merged := map[string]any{}
	for key, value := range base {
		merged[key] = value
	}
	for key, value := range extra {
		merged[key] = value
	}
	return merged
}

func sameEventID(left, right *int) bool {
	return left == nil && right == nil || left != nil && right != nil && *left == *right
}

func orEmpty(value map[string]any) map[string]any {
	if value == nil {
		return map[string]any{}
	}
	return value
}

var (
	_ realtime.Connection     = (*Connection)(nil)
	_ realtime.ConnectionInfo = (*Connection)(nil)
)
