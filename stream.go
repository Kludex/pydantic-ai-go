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

// ModelStreamEvent is a provider-facing response delta. Agent.RunStream
// normalizes these into StreamEvent lifecycle events.
type ModelStreamEvent interface {
	modelStreamEventKind() string
}

// StreamEvent is one normalized event from Agent.RunStream.
type StreamEvent interface {
	streamEventKind() string
}

// TextDeltaEvent carries a provider chunk of text output.
type TextDeltaEvent struct {
	// PartID identifies the response part this delta updates. A stable,
	// non-empty ID allows deltas for multiple parts to be interleaved. An
	// empty ID appends to the current text part for sequential streams.
	PartID          string
	Delta           string
	ID              string
	ProviderName    string
	ProviderDetails map[string]any
}

func (TextDeltaEvent) modelStreamEventKind() string { return "text-delta" }

// ThinkingDeltaEvent carries a provider chunk of reasoning content.
type ThinkingDeltaEvent struct {
	// PartID identifies the response part this delta updates. A stable,
	// non-empty ID allows deltas for multiple parts to be interleaved. An
	// empty ID appends to the current thinking part for sequential streams.
	PartID          string
	Delta           string
	ID              string
	SignatureDelta  string
	ProviderName    string
	ProviderDetails map[string]any
}

func (ThinkingDeltaEvent) modelStreamEventKind() string { return "thinking-delta" }

// ToolCallStartEvent begins a provider tool call.
type ToolCallStartEvent struct {
	// PartID identifies this tool-call part. ToolCallDeltaEvent uses the
	// same ID so multiple calls can stream arguments concurrently. An empty
	// ID starts a new sequential tool-call part.
	PartID          string
	ToolName        string
	ToolCallID      string
	ToolKind        ToolPartKind
	ID              string
	ProviderName    string
	ProviderDetails map[string]any
}

func (ToolCallStartEvent) modelStreamEventKind() string { return "tool-call-start" }

// ToolCallDeltaEvent carries a provider fragment of tool call arguments.
type ToolCallDeltaEvent struct {
	// PartID identifies the tool call started by ToolCallStartEvent. An empty
	// ID targets the current tool call for sequential streams.
	PartID    string
	ArgsDelta string
}

func (ToolCallDeltaEvent) modelStreamEventKind() string { return "tool-call-delta" }

// FinishEvent ends one streamed model response and carries its usage. It is
// both the provider completion marker and the final normalized response event.
type FinishEvent struct {
	Usage              Usage
	ModelName          string
	Timestamp          time.Time
	ProviderName       string
	ProviderURL        string
	ProviderDetails    map[string]any
	ProviderResponseID string
	FinishReason       FinishReason
	State              ModelResponseState
}

func (FinishEvent) modelStreamEventKind() string { return "finish" }
func (FinishEvent) streamEventKind() string      { return "finish" }
