// Package realtime defines provider-neutral bidirectional model sessions.
package realtime

import (
	"context"
	"iter"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// Input is content or a control command sent to a realtime connection.
type Input interface {
	// RealtimeInputKind returns a stable input discriminator.
	RealtimeInputKind() string
}

// TextInput sends one complete text turn.
type TextInput struct {
	// Text is the complete user turn.
	Text string
}

// RealtimeInputKind identifies text input.
func (TextInput) RealtimeInputKind() string { return "text" }

// AudioInput sends raw mono PCM16 audio at the model's input sample rate.
type AudioInput struct {
	// Data contains raw little-endian mono PCM16 samples.
	Data []byte
}

// RealtimeInputKind identifies audio input.
func (AudioInput) RealtimeInputKind() string { return "audio" }

// ImageInput sends one encoded image or video frame.
type ImageInput struct {
	// Content contains one encoded image or video frame.
	Content ai.BinaryContent
}

// RealtimeInputKind identifies image input.
func (ImageInput) RealtimeInputKind() string { return "image" }

// ToolResult returns one completed function call to the provider.
type ToolResult struct {
	// ToolCallID identifies the provider call being answered.
	ToolCallID string
	// Output is the flattened result sent through the provider tool channel.
	Output string
	// Content adds supported user content after the tool result.
	Content []ai.UserContent
}

// RealtimeInputKind identifies a tool result.
func (ToolResult) RealtimeInputKind() string { return "tool-result" }

// CommitAudio commits buffered audio as a user turn.
type CommitAudio struct{}

// RealtimeInputKind identifies an audio commit command.
func (CommitAudio) RealtimeInputKind() string { return "commit-audio" }

// ClearAudio discards buffered uncommitted audio.
type ClearAudio struct{}

// RealtimeInputKind identifies an audio clear command.
func (ClearAudio) RealtimeInputKind() string { return "clear-audio" }

// CreateResponse asks the model to respond immediately.
type CreateResponse struct{}

// RealtimeInputKind identifies a response creation command.
func (CreateResponse) RealtimeInputKind() string { return "create-response" }

// CancelResponse cancels the response currently being generated.
type CancelResponse struct{}

// RealtimeInputKind identifies a response cancellation command.
func (CancelResponse) RealtimeInputKind() string { return "cancel-response" }

// TruncateOutput removes unheard audio from provider conversation state.
type TruncateOutput struct {
	// AudioEndMilliseconds is the amount of output the user heard.
	AudioEndMilliseconds int
}

// RealtimeInputKind identifies an output truncation command.
func (TruncateOutput) RealtimeInputKind() string { return "truncate-output" }

// CodecEvent is one normalized event received from a provider connection.
type CodecEvent interface {
	// RealtimeCodecEventKind returns a stable event discriminator.
	RealtimeCodecEventKind() string
}

// AudioDelta carries raw PCM16 model output.
type AudioDelta struct {
	// Data contains raw little-endian mono PCM16 samples.
	Data []byte
	// ItemID identifies the provider output item when available.
	ItemID string
}

// RealtimeCodecEventKind identifies an audio delta.
func (AudioDelta) RealtimeCodecEventKind() string { return "audio-delta" }

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

// RealtimeCodecEventKind identifies an output transcript update.
func (OutputTranscript) RealtimeCodecEventKind() string { return "output-transcript" }

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

// RealtimeCodecEventKind identifies an input transcript update.
func (InputTranscript) RealtimeCodecEventKind() string { return "input-transcript" }

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

// RealtimeCodecEventKind identifies a provider function call.
func (ToolCall) RealtimeCodecEventKind() string { return "tool-call" }

// ToolCallCancelled reports provider cancellation of in-flight calls.
type ToolCallCancelled struct {
	// ToolCallIDs identifies calls whose results must not be returned.
	ToolCallIDs []string
}

// RealtimeCodecEventKind identifies cancelled function calls.
func (ToolCallCancelled) RealtimeCodecEventKind() string { return "tool-call-cancelled" }

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

// RealtimeCodecEventKind identifies response completion.
func (ResponseDone) RealtimeCodecEventKind() string { return "response-done" }

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

// RealtimeCodecEventKind identifies provider usage.
func (SessionUsage) RealtimeCodecEventKind() string { return "session-usage" }

// InputSpeechStarted reports server-side voice activity detection.
type InputSpeechStarted struct {
	// ItemID identifies the detected input segment when available.
	ItemID string
}

// RealtimeCodecEventKind identifies the start of user speech.
func (InputSpeechStarted) RealtimeCodecEventKind() string { return "input-speech-started" }

// InputSpeechEnded reports the end of server-detected user speech.
type InputSpeechEnded struct {
	// ItemID identifies the detected input segment when available.
	ItemID string
}

// RealtimeCodecEventKind identifies the end of user speech.
func (InputSpeechEnded) RealtimeCodecEventKind() string { return "input-speech-ended" }

// OutputSpeechStarted reports that provider audio playback started.
type OutputSpeechStarted struct{}

// RealtimeCodecEventKind identifies the start of output playback.
func (OutputSpeechStarted) RealtimeCodecEventKind() string { return "output-speech-started" }

// OutputSpeechEnded reports that provider audio playback stopped.
type OutputSpeechEnded struct{}

// RealtimeCodecEventKind identifies the end of output playback.
func (OutputSpeechEnded) RealtimeCodecEventKind() string { return "output-speech-ended" }

// InputTranscriptionError reports a recoverable transcription failure.
type InputTranscriptionError struct {
	// ItemID identifies the input segment that failed.
	ItemID string
	// Err describes the recoverable transcription failure.
	Err error
}

// RealtimeCodecEventKind identifies a transcription failure.
func (InputTranscriptionError) RealtimeCodecEventKind() string { return "input-transcription-error" }

// SessionReconnected reports a successful transport reconnection.
type SessionReconnected struct {
	// StateRestored reports that the provider resumed in-flight state.
	StateRestored bool
}

// RealtimeCodecEventKind identifies a successful reconnect.
func (SessionReconnected) RealtimeCodecEventKind() string { return "session-reconnected" }

// ConversationCreated carries a provider conversation identifier.
type ConversationCreated struct {
	// ConversationID is the provider-assigned session identifier.
	ConversationID string
}

// RealtimeCodecEventKind identifies provider conversation creation.
func (ConversationCreated) RealtimeCodecEventKind() string { return "conversation-created" }

// ConversationItemCreated identifies a live or replayed provider item.
type ConversationItemCreated struct {
	// ItemID is the provider-assigned conversation item identifier.
	ItemID string
	// ToolCallID identifies a function call or result item.
	ToolCallID string
	// Replayed reports that the provider emitted this item during resumption.
	Replayed bool
}

// RealtimeCodecEventKind identifies a provider conversation item.
func (ConversationItemCreated) RealtimeCodecEventKind() string { return "conversation-item-created" }

// PartStarted carries a complete provider-native response part.
type PartStarted struct {
	// Event is the normalized start event forwarded to session consumers.
	Event ai.PartStartEvent
}

// RealtimeCodecEventKind identifies a provider-native part start.
func (PartStarted) RealtimeCodecEventKind() string { return "part-started" }

// PartEnded closes a provider-native response part.
type PartEnded struct {
	// Event is the normalized end event forwarded to session consumers.
	Event ai.PartEndEvent
}

// RealtimeCodecEventKind identifies a provider-native part end.
func (PartEnded) RealtimeCodecEventKind() string { return "part-ended" }

// ResponseStarted reports the provider identity of a new generation.
type ResponseStarted struct {
	// ResponseID is the provider-assigned response identifier.
	ResponseID string
}

// RealtimeCodecEventKind identifies response creation.
func (ResponseStarted) RealtimeCodecEventKind() string { return "response-started" }

// SessionError reports a recoverable or terminal provider failure.
type SessionError struct {
	// Err describes the provider or protocol failure.
	Err error
	// Recoverable reports that the connection remains usable.
	Recoverable bool
}

// RealtimeCodecEventKind identifies a session error.
func (SessionError) RealtimeCodecEventKind() string { return "session-error" }

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
