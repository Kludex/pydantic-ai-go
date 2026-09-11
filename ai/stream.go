package ai

import (
	"context"
	"iter"
	"time"
)

// StreamingModel is implemented by providers that support streaming
// responses. The agent discovers it by type assertion; models that do not
// implement it still work with RunStream via a non-streaming fallback.
type StreamingModel interface {
	Model
	// StreamRequest returns provider deltas. The sequence ends after a
	// FinishEvent on success or a non-nil error.
	StreamRequest(
		ctx context.Context, msgs []ModelMessage, params ModelRequestParams,
	) (iter.Seq2[ModelStreamEvent, error], error)
}

// CollectModelStream drains provider events into one detached response. Model
// wrappers use it to adapt streaming-only endpoints to the Model contract.
func CollectModelStream(
	events iter.Seq2[ModelStreamEvent, error], params ModelRequestParams,
) (*ModelResponse, error) {
	return accumulate(events, params, nil, nil)
}

// ModelStreamEvent is a provider-facing response delta. Agent.RunStream
// normalizes these into StreamEvent lifecycle events.
type ModelStreamEvent interface {
	modelStreamEventKind() string
}

// StreamEvent is one normalized event from Agent.RunStream.
type StreamEvent interface {
	streamEventKind() string
}

// ResponseMetadataEvent updates the in-flight response snapshot without
// emitting a consumer StreamEvent. Providers use it to make interrupted or
// detached streams resumable before their terminal event arrives.
type ResponseMetadataEvent struct {
	// Usage replaces the in-flight provider usage snapshot.
	Usage Usage
	// ModelName replaces the response model identity.
	ModelName string
	// Timestamp replaces the response creation time.
	Timestamp time.Time
	// ProviderName replaces the durable provider identity.
	ProviderName string
	// ProviderURL replaces the provider endpoint identity.
	ProviderURL string
	// ProviderDetails replaces detached provider-specific response data.
	ProviderDetails map[string]any
	// Metadata replaces detached application response metadata.
	Metadata map[string]any
	// ProviderResponseID replaces the resumable provider response ID.
	ProviderResponseID string
	// FinishReason replaces the normalized generation stop reason.
	FinishReason FinishReason
	// State replaces the response lifecycle state.
	State ModelResponseState
}

func (ResponseMetadataEvent) modelStreamEventKind() string { return "response-metadata" }

// TextDeltaEvent carries a provider chunk of text output.
type TextDeltaEvent struct {
	// PartID identifies the response part this delta updates. A stable,
	// non-empty ID allows deltas for multiple parts to be interleaved. An
	// empty ID appends to the current text part for sequential streams.
	PartID string
	// Delta is appended to the accumulated text content.
	Delta string
	// ID is the provider's identity for the response part.
	ID string
	// ProviderName identifies the provider that produced this part.
	ProviderName string
	// ProviderDetails merges detached provider-specific part data.
	ProviderDetails map[string]any
}

func (TextDeltaEvent) modelStreamEventKind() string { return "text-delta" }

// SpeechDeltaEvent starts or updates one streamed speech part. Part seeds a
// new PartID; Delta applies to the accumulated part and is emitted to consumers.
type SpeechDeltaEvent struct {
	// PartID identifies the speech part across updates.
	PartID string
	// Part seeds a new accumulated speech part.
	Part SpeechPart
	// Delta updates the accumulated part and is emitted to consumers.
	Delta SpeechPartDelta
}

func (SpeechDeltaEvent) modelStreamEventKind() string { return "speech-delta" }

// ThinkingDeltaEvent carries a provider chunk of reasoning content.
type ThinkingDeltaEvent struct {
	// PartID identifies the response part this delta updates. A stable,
	// non-empty ID allows deltas for multiple parts to be interleaved. An
	// empty ID appends to the current thinking part for sequential streams.
	PartID string
	// Delta is appended to accumulated reasoning content.
	Delta string
	// ID is the provider's identity for the response part.
	ID string
	// SignatureDelta replaces the provider reasoning signature when non-empty.
	SignatureDelta string
	// ProviderName identifies the provider that produced this part.
	ProviderName string
	// ProviderDetails merges detached provider-specific part data.
	ProviderDetails map[string]any
}

func (ThinkingDeltaEvent) modelStreamEventKind() string { return "thinking-delta" }

// CompactionEvent emits one complete provider compaction part.
type CompactionEvent struct {
	// PartID identifies this compaction part within the response.
	PartID string
	// Content is the readable provider summary.
	Content string
	// ID is the provider's identity for the compaction boundary.
	ID string
	// ProviderName identifies the provider that produced the boundary.
	ProviderName string
	// ProviderDetails carries detached opaque compaction state.
	ProviderDetails map[string]any
}

func (CompactionEvent) modelStreamEventKind() string { return "compaction" }

// ToolCallStartEvent begins a provider tool call.
type ToolCallStartEvent struct {
	// PartID identifies this tool-call part. ToolCallDeltaEvent uses the
	// same ID so multiple calls can stream arguments concurrently. An empty
	// ID starts a new sequential tool-call part.
	PartID string
	// ToolName is the complete model-facing tool name when known.
	ToolName string
	// ToolCallID is the provider-assigned call identity when known.
	ToolCallID string
	// ToolKind identifies a portable typed tool surface.
	ToolKind ToolPartKind
	// ID is the provider's response-part identity.
	ID string
	// ProviderName identifies the provider that produced the call.
	ProviderName string
	// ProviderDetails carries detached provider-specific call data.
	ProviderDetails map[string]any
	// Native marks a provider-executed call that the local tool loop must not run.
	Native bool
}

func (ToolCallStartEvent) modelStreamEventKind() string { return "tool-call-start" }

// ToolCallDeltaEvent carries a provider fragment of tool call arguments.
type ToolCallDeltaEvent struct {
	// PartID identifies the tool call started by ToolCallStartEvent. An empty
	// ID targets the current tool call for sequential streams.
	PartID string
	// ToolCallID fills an ID omitted from the start event. A different non-empty
	// ID is rejected after the call has acquired one.
	ToolCallID string
	// ArgsDelta appends one JSON argument fragment.
	ArgsDelta string
}

func (ToolCallDeltaEvent) modelStreamEventKind() string { return "tool-call-delta" }

// NativeToolReturnEvent emits one complete provider-executed tool result.
type NativeToolReturnEvent struct {
	// PartID identifies this return within the response.
	PartID string
	// Part is the complete provider-executed tool result.
	Part NativeToolReturnPart
}

func (NativeToolReturnEvent) modelStreamEventKind() string { return "builtin-tool-return" }

// FileEvent emits one complete model-generated file. Replace updates an
// earlier file with the same PartID, as used by partial image generation.
type FileEvent struct {
	// PartID identifies the generated file across replacements.
	PartID string
	// Part is a detached complete file snapshot.
	Part FilePart
	// Replace updates the earlier file with the same PartID.
	Replace bool
}

func (FileEvent) modelStreamEventKind() string { return "file" }

// FinishEvent ends one streamed model response and carries its usage. It is
// both the provider completion marker and the final normalized response event.
type FinishEvent struct {
	// Parts is an optional authoritative final snapshot. Providers use it when
	// a terminal event contains more complete content than preceding deltas.
	Parts []ResponsePart
	// Usage is the complete usage for this response segment.
	Usage Usage
	// ModelName is the provider's resolved model identity.
	ModelName string
	// Timestamp is the response creation time.
	Timestamp time.Time
	// ProviderName is the durable provider identity.
	ProviderName string
	// ProviderURL is the configured provider endpoint.
	ProviderURL string
	// ProviderDetails contains detached provider-specific response data.
	ProviderDetails map[string]any
	// Metadata contains detached application response metadata.
	Metadata map[string]any
	// ProviderResponseID identifies resumable provider-side state.
	ProviderResponseID string
	// FinishReason is the normalized generation stop reason.
	FinishReason FinishReason
	// State is the final or suspended response lifecycle state.
	State ModelResponseState
}

func (FinishEvent) modelStreamEventKind() string { return "finish" }
func (FinishEvent) streamEventKind() string      { return "finish" }
