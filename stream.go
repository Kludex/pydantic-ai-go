package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
)

// StreamingModel is implemented by providers that support streaming
// responses. The agent discovers it by type assertion; models that do not
// implement it still work with RunStream via a non-streaming fallback.
type StreamingModel interface {
	Model
	// StreamRequest returns the response as a stream of events. The
	// sequence ends after a FinishEvent (success) or a non-nil error.
	StreamRequest(ctx context.Context, msgs []ModelMessage, params ModelRequestParams) (iter.Seq2[StreamEvent, error], error)
}

// StreamEvent is one increment of a streamed model response.
type StreamEvent interface {
	streamEventKind() string
}

// TextDeltaEvent carries a chunk of text output.
type TextDeltaEvent struct {
	// PartID identifies the response part this delta updates. A stable,
	// non-empty ID allows deltas for multiple parts to be interleaved. An
	// empty ID appends to the current text part for sequential streams.
	PartID string
	Delta  string
}

func (TextDeltaEvent) streamEventKind() string { return "text-delta" }

// ThinkingDeltaEvent carries a chunk of reasoning content.
type ThinkingDeltaEvent struct {
	// PartID identifies the response part this delta updates. A stable,
	// non-empty ID allows deltas for multiple parts to be interleaved. An
	// empty ID appends to the current thinking part for sequential streams.
	PartID string
	Delta  string
}

func (ThinkingDeltaEvent) streamEventKind() string { return "thinking-delta" }

// ToolCallStartEvent begins a tool call.
type ToolCallStartEvent struct {
	// PartID identifies this tool-call part. ToolCallDeltaEvent uses the
	// same ID so multiple calls can stream arguments concurrently. An empty
	// ID starts a new sequential tool-call part.
	PartID     string
	ToolName   string
	ToolCallID string
}

func (ToolCallStartEvent) streamEventKind() string { return "tool-call-start" }

// ToolCallDeltaEvent carries a fragment of tool call arguments.
type ToolCallDeltaEvent struct {
	// PartID identifies the tool call started by ToolCallStartEvent. An empty
	// ID targets the current tool call for sequential streams.
	PartID    string
	ArgsDelta string
}

func (ToolCallDeltaEvent) streamEventKind() string { return "tool-call-delta" }

// FinishEvent ends a streamed response, carrying its usage.
type FinishEvent struct {
	Usage     Usage
	ModelName string
}

func (FinishEvent) streamEventKind() string { return "finish" }

type accumulatedPart struct {
	kind       string
	text       string
	toolName   string
	toolCallID string
	toolArgs   string
}

// accumulate replays a stream into a ModelResponse, forwarding each event
// to emit. Parts with IDs are tracked independently, so their deltas may be
// interleaved while response order remains the order each part first appears.
func accumulate(events iter.Seq2[StreamEvent, error], emit func(StreamEvent) bool) (*ModelResponse, error) {
	resp := &ModelResponse{}
	parts := make([]*accumulatedPart, 0)
	partsByID := map[string]*accumulatedPart{}
	var current *accumulatedPart

	newPart := func(id, kind string) *accumulatedPart {
		part := &accumulatedPart{kind: kind}
		parts = append(parts, part)
		if id != "" {
			partsByID[id] = part
		}
		current = part
		return part
	}
	partForDelta := func(id, kind string) (*accumulatedPart, error) {
		if id != "" {
			if part, ok := partsByID[id]; ok {
				if part.kind != kind {
					return nil, &UnexpectedModelBehaviorError{
						Message: fmt.Sprintf("stream part %q changed from %s to %s", id, part.kind, kind),
					}
				}
				current = part
				return part, nil
			}
			return newPart(id, kind), nil
		}
		if current != nil && current.kind == kind {
			return current, nil
		}
		return newPart("", kind), nil
	}
	materialize := func() {
		for _, part := range parts {
			switch part.kind {
			case "text":
				if part.text != "" {
					resp.Parts = append(resp.Parts, TextPart{Content: part.text})
				}
			case "thinking":
				if part.text != "" {
					resp.Parts = append(resp.Parts, ThinkingPart{Content: part.text})
				}
			case "tool-call":
				resp.Parts = append(resp.Parts, ToolCallPart{
					ToolName: part.toolName, Args: json.RawMessage(part.toolArgs), ToolCallID: part.toolCallID,
				})
			}
		}
	}

	for event, err := range events {
		if err != nil {
			return nil, err
		}
		if emit != nil && !emit(event) {
			return nil, context.Canceled
		}
		switch event := event.(type) {
		case TextDeltaEvent:
			part, err := partForDelta(event.PartID, "text")
			if err != nil {
				return nil, err
			}
			part.text += event.Delta
		case ThinkingDeltaEvent:
			part, err := partForDelta(event.PartID, "thinking")
			if err != nil {
				return nil, err
			}
			part.text += event.Delta
		case ToolCallStartEvent:
			if event.PartID != "" {
				if _, exists := partsByID[event.PartID]; exists {
					return nil, &UnexpectedModelBehaviorError{
						Message: fmt.Sprintf("duplicate tool call start for stream part %q", event.PartID),
					}
				}
			}
			part := newPart(event.PartID, "tool-call")
			part.toolName = event.ToolName
			part.toolCallID = event.ToolCallID
		case ToolCallDeltaEvent:
			var part *accumulatedPart
			if event.PartID != "" {
				part = partsByID[event.PartID]
			} else if current != nil && current.kind == "tool-call" {
				part = current
			}
			if part == nil || part.kind != "tool-call" {
				return nil, &UnexpectedModelBehaviorError{Message: "tool call delta before tool call start"}
			}
			current = part
			part.toolArgs += event.ArgsDelta
		case FinishEvent:
			materialize()
			resp.Usage = event.Usage
			resp.ModelName = event.ModelName
			return resp, nil
		default:
			return nil, fmt.Errorf("ai: unknown stream event type %T", event)
		}
	}
	return nil, &UnexpectedModelBehaviorError{Message: "stream ended without a finish event"}
}
