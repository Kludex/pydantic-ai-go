package ai

import (
	"encoding/json"
	"fmt"
	"iter"
)

// EventStream is a consumer-facing stream from Agent.RunStream. It may be
// transformed by RunEventStreamWrapper capabilities.
type EventStream iter.Seq2[StreamEvent, error]

// ResponsePartKind identifies a streamed response part.
type ResponsePartKind string

const (
	// ResponsePartKindText identifies text output.
	ResponsePartKindText ResponsePartKind = "text"
	// ResponsePartKindSpeech identifies retained speech output.
	ResponsePartKindSpeech ResponsePartKind = "speech"
	// ResponsePartKindFile identifies generated file output.
	ResponsePartKindFile ResponsePartKind = "file"
	// ResponsePartKindThinking identifies provider reasoning.
	ResponsePartKindThinking ResponsePartKind = "thinking"
	// ResponsePartKindCompaction identifies a durable history boundary.
	ResponsePartKindCompaction ResponsePartKind = "compaction"
	// ResponsePartKindToolCall identifies a locally executed function call.
	ResponsePartKindToolCall ResponsePartKind = "tool-call"
	// ResponsePartKindNativeToolCall identifies a provider-executed call.
	ResponsePartKindNativeToolCall ResponsePartKind = "builtin-tool-call"
	// ResponsePartKindNativeToolReturn identifies a provider-executed result.
	ResponsePartKindNativeToolReturn ResponsePartKind = "builtin-tool-return"
)

// ResponsePartDelta updates one response part.
type ResponsePartDelta interface {
	// Apply returns a copy of part with this delta applied.
	Apply(part ResponsePart) (ResponsePart, error)
	responsePartDeltaKind() ResponsePartKind
}

// TextPartDelta appends content to a TextPart.
type TextPartDelta struct {
	// ContentDelta appends text to the part.
	ContentDelta string
	// ProviderName replaces the part provider when non-empty.
	ProviderName string
	// ProviderDetails merges detached provider-specific data.
	ProviderDetails map[string]any
}

func (TextPartDelta) responsePartDeltaKind() ResponsePartKind { return ResponsePartKindText }

// Apply applies the text delta.
func (d TextPartDelta) Apply(part ResponsePart) (ResponsePart, error) {
	text, ok := part.(TextPart)
	if !ok {
		return nil, fmt.Errorf("ai: cannot apply TextPartDelta to %T", part)
	}
	text.Content += d.ContentDelta
	if d.ProviderName != "" {
		text.ProviderName = d.ProviderName
	}
	text.ProviderDetails = mergeProviderDetails(text.ProviderDetails, d.ProviderDetails)
	return text, nil
}

// SpeechPartDelta updates a speech transcript and appends retained audio.
type SpeechPartDelta struct {
	// Speaker identifies who produced the speech.
	Speaker SpeechSpeaker
	// TranscriptDelta appends an incremental transcript.
	TranscriptDelta string
	// Transcript replaces the accumulated transcript when non-nil.
	Transcript *string
	// AudioChunk appends detached retained audio bytes.
	AudioChunk []byte
}

func (SpeechPartDelta) responsePartDeltaKind() ResponsePartKind { return ResponsePartKindSpeech }

// Apply applies the speech delta without mutating the existing part.
func (delta SpeechPartDelta) Apply(part ResponsePart) (ResponsePart, error) {
	speech, ok := part.(SpeechPart)
	if !ok {
		return nil, fmt.Errorf("ai: cannot apply SpeechPartDelta to %T", part)
	}
	return applySpeechPartDelta(speech, delta), nil
}

func applySpeechPartDelta(speech SpeechPart, delta SpeechPartDelta) SpeechPart {
	speech = cloneSpeechPart(speech)
	if delta.Transcript != nil {
		speech.Transcript = clonePointer(delta.Transcript)
	} else if delta.TranscriptDelta != "" {
		transcript := speech.Content() + delta.TranscriptDelta
		speech.Transcript = &transcript
	}
	if len(delta.AudioChunk) > 0 && speech.Audio != nil {
		speech.Audio.Data = append(speech.Audio.Data, delta.AudioChunk...)
	}
	return speech
}

// ThinkingPartDelta appends content to a ThinkingPart.
type ThinkingPartDelta struct {
	// ContentDelta appends reasoning text.
	ContentDelta string
	// SignatureDelta replaces the reasoning signature when non-empty.
	SignatureDelta string
	// ProviderName replaces the part provider when non-empty.
	ProviderName string
	// ProviderDetails merges detached provider-specific data.
	ProviderDetails map[string]any
}

func (ThinkingPartDelta) responsePartDeltaKind() ResponsePartKind { return ResponsePartKindThinking }

// Apply applies the thinking delta.
func (d ThinkingPartDelta) Apply(part ResponsePart) (ResponsePart, error) {
	thinking, ok := part.(ThinkingPart)
	if !ok {
		return nil, fmt.Errorf("ai: cannot apply ThinkingPartDelta to %T", part)
	}
	thinking.Content += d.ContentDelta
	if d.SignatureDelta != "" {
		thinking.Signature = d.SignatureDelta
	}
	if d.ProviderName != "" {
		thinking.ProviderName = d.ProviderName
	}
	thinking.ProviderDetails = mergeProviderDetails(thinking.ProviderDetails, d.ProviderDetails)
	return thinking, nil
}

// FilePartDelta replaces a complete generated file with a newer snapshot.
type FilePartDelta struct {
	// Part is the detached replacement file snapshot.
	Part FilePart
}

func (FilePartDelta) responsePartDeltaKind() ResponsePartKind { return ResponsePartKindFile }

// Apply applies the file replacement.
func (d FilePartDelta) Apply(part ResponsePart) (ResponsePart, error) {
	if _, ok := part.(FilePart); !ok {
		return nil, fmt.Errorf("ai: cannot apply FilePartDelta to %T", part)
	}
	return cloneResponsePart(d.Part), nil
}

// ToolCallPartDelta updates a ToolCallPart. Names and JSON arguments append;
// a non-empty tool-call ID fills an empty ID and must otherwise match it.
type ToolCallPartDelta struct {
	// ToolNameDelta appends a tool-name fragment.
	ToolNameDelta string
	// ArgsDelta appends a JSON argument fragment.
	ArgsDelta string
	// ToolCallID fills an empty ID and must otherwise match.
	ToolCallID string
	// ProviderName replaces the call provider when non-empty.
	ProviderName string
}

func (ToolCallPartDelta) responsePartDeltaKind() ResponsePartKind { return ResponsePartKindToolCall }

// Apply applies the tool-call delta.
func (d ToolCallPartDelta) Apply(part ResponsePart) (ResponsePart, error) {
	call, ok := part.(ToolCallPart)
	if !ok {
		return nil, fmt.Errorf("ai: cannot apply ToolCallPartDelta to %T", part)
	}
	name, args, callID, providerName, err := applyToolCallDelta(
		call.ToolName, call.Args, call.ToolCallID, call.ProviderName, d,
	)
	if err != nil {
		return nil, err
	}
	call.ToolName, call.Args, call.ToolCallID, call.ProviderName = name, args, callID, providerName
	return call, nil
}

// NativeToolCallPartDelta updates a NativeToolCallPart.
type NativeToolCallPartDelta ToolCallPartDelta

func (NativeToolCallPartDelta) responsePartDeltaKind() ResponsePartKind {
	return ResponsePartKindNativeToolCall
}

// Apply applies the provider-native tool-call delta.
func (d NativeToolCallPartDelta) Apply(part ResponsePart) (ResponsePart, error) {
	call, ok := part.(NativeToolCallPart)
	if !ok {
		return nil, fmt.Errorf("ai: cannot apply NativeToolCallPartDelta to %T", part)
	}
	name, args, callID, providerName, err := applyToolCallDelta(
		call.ToolName, call.Args, call.ToolCallID, call.ProviderName, ToolCallPartDelta(d),
	)
	if err != nil {
		return nil, err
	}
	call.ToolName, call.Args, call.ToolCallID, call.ProviderName = name, args, callID, providerName
	return call, nil
}

func applyToolCallDelta(
	name string,
	args json.RawMessage,
	callID string,
	providerName string,
	delta ToolCallPartDelta,
) (string, json.RawMessage, string, string, error) {
	if delta.ToolCallID != "" && callID != "" && delta.ToolCallID != callID {
		return "", nil, "", "", &UnexpectedModelBehaviorError{Message: fmt.Sprintf(
			"tool call ID changed from %q to %q", callID, delta.ToolCallID,
		)}
	}
	name += delta.ToolNameDelta
	args = json.RawMessage(append(append([]byte(nil), args...), delta.ArgsDelta...))
	if delta.ToolCallID != "" {
		callID = delta.ToolCallID
	}
	if delta.ProviderName != "" {
		providerName = delta.ProviderName
	}
	return name, args, callID, providerName, nil
}

func mergeProviderDetails(base, update map[string]any) map[string]any {
	if len(update) == 0 {
		return cloneSchemaMap(base)
	}
	merged := cloneSchemaMap(base)
	if merged == nil {
		merged = make(map[string]any, len(update))
	}
	for key, value := range update {
		merged[key] = cloneSchemaValue(value)
	}
	return merged
}

// PartStartEvent announces a new response part. Index is stable within the
// model response and follows first-appearance order.
type PartStartEvent struct {
	// Index is the stable first-appearance index within the response.
	Index int
	// PartID is the provider-facing identity used by later updates.
	PartID string
	// Part is the detached initial accumulated part.
	Part ResponsePart
	// PreviousPartKind identifies the preceding grouping boundary.
	PreviousPartKind ResponsePartKind
}

func (PartStartEvent) streamEventKind() string { return "part-start" }

// PartDeltaEvent updates a response part previously announced at Index.
type PartDeltaEvent struct {
	// Index identifies the accumulated response part.
	Index int
	// PartID is the provider-facing identity used by this update.
	PartID string
	// Delta is the detached update applied to the part.
	Delta ResponsePartDelta
}

func (PartDeltaEvent) streamEventKind() string { return "part-delta" }

// PartEndEvent marks the current grouping boundary for a response part. A
// provider may still send keyed deltas for an earlier part after this event.
type PartEndEvent struct {
	// Index identifies the accumulated response part.
	Index int
	// PartID is the provider-facing response-part identity.
	PartID string
	// Part is the detached complete snapshot at this boundary.
	Part ResponsePart
	// NextPartKind identifies the part that caused the grouping boundary.
	NextPartKind ResponsePartKind
}

func (PartEndEvent) streamEventKind() string { return "part-end" }

// FinalResultEvent announces the first response part matching the configured
// output. ToolName is empty for text and native output.
type FinalResultEvent struct {
	// ToolName identifies structured tool output and is empty for text or native output.
	ToolName string
	// ToolCallID identifies the selected output-tool call.
	ToolCallID string
}

func (FinalResultEvent) streamEventKind() string { return "final-result" }

// EnqueuedMessagesEvent announces one queued group when it enters history.
type EnqueuedMessagesEvent struct {
	// EnqueueID identifies one delivered queue group.
	EnqueueID string
	// Messages is the detached group appended to history.
	Messages []ModelMessage
}

func (EnqueuedMessagesEvent) streamEventKind() string { return "enqueued-messages" }

// FunctionToolCallEvent announces a function tool call before execution.
type FunctionToolCallEvent struct {
	// Part is the detached function-tool call.
	Part ToolCallPart
	// ArgsValid reports schema validity when validation has completed.
	ArgsValid *bool
}

func (FunctionToolCallEvent) streamEventKind() string { return "function-tool-call" }

// OutputToolCallEvent announces an output tool call before validation.
type OutputToolCallEvent struct {
	// Part is the detached output-tool call.
	Part ToolCallPart
	// ArgsValid reports schema validity when validation has completed.
	ArgsValid *bool
}

func (OutputToolCallEvent) streamEventKind() string { return "output-tool-call" }

// FunctionToolResultEvent carries the request part produced by a function
// tool. Part is a ToolReturnPart or RetryPromptPart.
type FunctionToolResultEvent struct {
	// Part is the detached tool return or retry prompt added to history.
	Part RequestPart
}

func (FunctionToolResultEvent) streamEventKind() string { return "function-tool-result" }

// OutputToolResultEvent carries the request part produced by an output tool.
// Part is a ToolReturnPart or RetryPromptPart.
type OutputToolResultEvent struct {
	// Part is the detached output return or retry prompt added to history.
	Part RequestPart
}

func (OutputToolResultEvent) streamEventKind() string { return "output-tool-result" }

// ToolAvailabilityDeltaEvent announces tools revealed by a completed tool call.
type ToolAvailabilityDeltaEvent struct {
	// Part is the detached availability change added to history.
	Part ToolAvailabilityDeltaPart
}

func (ToolAvailabilityDeltaEvent) streamEventKind() string { return "tool-availability-delta" }

// DeferredToolRequestsEvent announces the batch of external calls and
// approvals that paused the run.
type DeferredToolRequestsEvent struct {
	// Requests is the detached batch that paused the run.
	Requests DeferredToolRequests
}

func (DeferredToolRequestsEvent) streamEventKind() string { return "deferred-tool-requests" }

// DeferredToolResultsEvent announces results returned by an inline handler.
type DeferredToolResultsEvent struct {
	// Results is the detached batch returned by an inline handler.
	Results DeferredToolResults
}

func (DeferredToolResultsEvent) streamEventKind() string { return "deferred-tool-results" }
