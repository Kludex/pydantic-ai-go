// Package realtime defines provider-neutral bidirectional model sessions.
package realtime

import (
	"context"
	"iter"

	ai "github.com/Kludex/pydantic-ai-go"
)

// Input is content or a control command sent to a realtime connection.
type Input = any

// TextInput sends one complete text turn.
type TextInput struct {
	Text string
}

// AudioInput sends raw mono PCM16 audio at the model's input sample rate.
type AudioInput struct {
	Data []byte
}

// ImageInput sends one encoded image or video frame.
type ImageInput struct {
	Content ai.BinaryContent
}

// ToolResult returns one completed function call to the provider.
type ToolResult struct {
	ToolCallID string
	Output     string
	Content    []ai.UserContent
}

// CommitAudio commits buffered audio as a user turn.
type CommitAudio struct{}

// ClearAudio discards buffered uncommitted audio.
type ClearAudio struct{}

// CreateResponse asks the model to respond immediately.
type CreateResponse struct{}

// CancelResponse cancels the response currently being generated.
type CancelResponse struct{}

// TruncateOutput removes unheard audio from provider conversation state.
type TruncateOutput struct {
	AudioEndMilliseconds int
}

// CodecEvent is one normalized event received from a provider connection.
type CodecEvent = any

// AudioDelta carries raw PCM16 model output.
type AudioDelta struct {
	Data   []byte
	ItemID string
}

// OutputTranscript updates model speech transcription or plain text output.
type OutputTranscript struct {
	Text       string
	Final      bool
	OutputText bool
	ItemID     string
}

// InputTranscript updates the transcription of one user turn.
type InputTranscript struct {
	Text       string
	Final      bool
	Cumulative bool
	ItemID     string
}

// ToolCall asks the application to execute a function tool.
type ToolCall struct {
	ToolCallID           string
	ToolName             string
	Arguments            string
	ItemID               string
	ResponseUsageFollows bool
}

// ToolCallCancelled reports provider cancellation of in-flight calls.
type ToolCallCancelled struct {
	ToolCallIDs []string
}

// ResponseDone closes the provider's current response.
type ResponseDone struct {
	Interrupted        bool
	ProviderResponseID string
	FinishReason       ai.FinishReason
	ProviderDetails    map[string]any
}

// SessionUsage carries response-scoped or session-scoped provider usage.
type SessionUsage struct {
	Usage              ai.Usage
	ProviderResponseID string
	FinishReason       ai.FinishReason
	ResponseScoped     bool
}

// InputSpeechStarted reports server-side voice activity detection.
type InputSpeechStarted struct {
	ItemID string
}

// InputSpeechEnded reports the end of server-detected user speech.
type InputSpeechEnded struct {
	ItemID string
}

// OutputSpeechStarted reports that provider audio playback started.
type OutputSpeechStarted struct{}

// OutputSpeechEnded reports that provider audio playback stopped.
type OutputSpeechEnded struct{}

// InputTranscriptionError reports a recoverable transcription failure.
type InputTranscriptionError struct {
	ItemID string
	Err    error
}

// SessionReconnected reports a successful transport reconnection.
type SessionReconnected struct {
	StateRestored bool
}

// ConversationCreated carries a provider conversation identifier.
type ConversationCreated struct {
	ConversationID string
}

// ConversationItemCreated identifies a live or replayed provider item.
type ConversationItemCreated struct {
	ItemID     string
	ToolCallID string
	Replayed   bool
}

// SessionError reports a recoverable or terminal provider failure.
type SessionError struct {
	Err         error
	Recoverable bool
}

// Connection is one live provider transport.
type Connection interface {
	// Send writes content or one control command.
	Send(ctx context.Context, input Input) error
	// Events yields normalized provider events until the connection closes.
	Events(ctx context.Context) iter.Seq2[CodecEvent, error]
	// Close releases the transport.
	Close(ctx context.Context) error
}

// ConnectionInfo exposes optional negotiated provider state.
type ConnectionInfo interface {
	// ModelName returns the model reported by the provider.
	ModelName() string
	// InputTranscriptionEnabled reports whether user transcript events will arrive.
	InputTranscriptionEnabled() bool
	// ReconnectRestoresInFlightState reports whether reconnect preserves unfinished work.
	ReconnectRestoresInFlightState() bool
}

// HistoryAwareConnection receives a live history reader for reconnect replay.
type HistoryAwareConnection interface {
	// SetMessageHistory installs a detached history snapshot callback.
	SetMessageHistory(history func() []ai.ModelMessage)
}
