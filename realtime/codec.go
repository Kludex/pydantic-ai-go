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
	// Text is the complete user turn.
	Text string
}

// AudioInput sends raw mono PCM16 audio at the model's input sample rate.
type AudioInput struct {
	// Data contains raw little-endian mono PCM16 samples.
	Data []byte
}

// ImageInput sends one encoded image or video frame.
type ImageInput struct {
	// Content contains one encoded image or video frame.
	Content ai.BinaryContent
}

// ToolResult returns one completed function call to the provider.
type ToolResult struct {
	// ToolCallID identifies the provider call being answered.
	ToolCallID string
	// Output is the flattened result sent through the provider tool channel.
	Output string
	// Content adds supported user content after the tool result.
	Content []ai.UserContent
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
	// AudioEndMilliseconds is the amount of output the user heard.
	AudioEndMilliseconds int
}

// CodecEvent is one normalized event received from a provider connection.
type CodecEvent = any

// AudioDelta carries raw PCM16 model output.
type AudioDelta struct {
	// Data contains raw little-endian mono PCM16 samples.
	Data []byte
	// ItemID identifies the provider output item when available.
	ItemID string
}

// OutputTranscript updates model speech transcription or plain text output.
type OutputTranscript struct {
	// Text is an incremental transcript piece or final snapshot.
	Text string
	// Final reports that the provider finalized this transcript.
	Final bool
	// OutputText distinguishes plain text output from speech transcription.
	OutputText bool
	// ItemID identifies the provider output item when available.
	ItemID string
}

// InputTranscript updates the transcription of one user turn.
type InputTranscript struct {
	// Text is an incremental transcript piece or cumulative snapshot.
	Text string
	// Final reports that the provider finalized this user turn.
	Final bool
	// Cumulative reports that Text replaces the transcript so far.
	Cumulative bool
	// ItemID identifies the provider input item when available.
	ItemID string
}

// ToolCall asks the application to execute a function tool.
type ToolCall struct {
	// ToolCallID is the provider-assigned function call identifier.
	ToolCallID string
	// ToolName is the advertised function name.
	ToolName string
	// Arguments contains raw JSON arguments.
	Arguments string
	// ItemID identifies the provider conversation item when available.
	ItemID string
	// ResponseUsageFollows reports that the current response will provide usage after this call.
	ResponseUsageFollows bool
}

// ToolCallCancelled reports provider cancellation of in-flight calls.
type ToolCallCancelled struct {
	// ToolCallIDs identifies calls whose results must not be returned.
	ToolCallIDs []string
}

// ResponseDone closes the provider's current response.
type ResponseDone struct {
	// Interrupted reports that generation was cancelled before completion.
	Interrupted bool
	// ProviderResponseID is the provider's response identifier.
	ProviderResponseID string
	// FinishReason is the normalized completion reason.
	FinishReason ai.FinishReason
	// ProviderDetails contains detached terminal provider metadata.
	ProviderDetails map[string]any
}

// SessionUsage carries response-scoped or session-scoped provider usage.
type SessionUsage struct {
	// Usage contains normalized provider counters.
	Usage ai.Usage
	// ProviderResponseID identifies the response charged by this usage.
	ProviderResponseID string
	// FinishReason is the response completion reason when supplied with usage.
	FinishReason ai.FinishReason
	// ResponseScoped includes the counters on the current ModelResponse.
	ResponseScoped bool
}

// InputSpeechStarted reports server-side voice activity detection.
type InputSpeechStarted struct {
	// ItemID identifies the detected input segment when available.
	ItemID string
}

// InputSpeechEnded reports the end of server-detected user speech.
type InputSpeechEnded struct {
	// ItemID identifies the detected input segment when available.
	ItemID string
}

// OutputSpeechStarted reports that provider audio playback started.
type OutputSpeechStarted struct{}

// OutputSpeechEnded reports that provider audio playback stopped.
type OutputSpeechEnded struct{}

// InputTranscriptionError reports a recoverable transcription failure.
type InputTranscriptionError struct {
	// ItemID identifies the input segment that failed.
	ItemID string
	// Err describes the recoverable transcription failure.
	Err error
}

// SessionReconnected reports a successful transport reconnection.
type SessionReconnected struct {
	// StateRestored reports that the provider resumed in-flight state.
	StateRestored bool
}

// ConversationCreated carries a provider conversation identifier.
type ConversationCreated struct {
	// ConversationID is the provider-assigned session identifier.
	ConversationID string
}

// ConversationItemCreated identifies a live or replayed provider item.
type ConversationItemCreated struct {
	// ItemID is the provider-assigned conversation item identifier.
	ItemID string
	// ToolCallID identifies a function call or result item.
	ToolCallID string
	// Replayed reports that the provider emitted this item during resumption.
	Replayed bool
}

// SessionError reports a recoverable or terminal provider failure.
type SessionError struct {
	// Err describes the provider or protocol failure.
	Err error
	// Recoverable reports that the connection remains usable.
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
