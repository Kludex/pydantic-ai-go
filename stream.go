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
	Delta string
}

func (TextDeltaEvent) streamEventKind() string { return "text-delta" }

// ThinkingDeltaEvent carries a chunk of reasoning content.
type ThinkingDeltaEvent struct {
	Delta string
}

func (ThinkingDeltaEvent) streamEventKind() string { return "thinking-delta" }

// ToolCallStartEvent begins a tool call; ToolCallDeltaEvents follow with
// argument fragments until the next event of a different part.
type ToolCallStartEvent struct {
	ToolName   string
	ToolCallID string
}

func (ToolCallStartEvent) streamEventKind() string { return "tool-call-start" }

// ToolCallDeltaEvent carries a fragment of tool call arguments.
type ToolCallDeltaEvent struct {
	ArgsDelta string
}

func (ToolCallDeltaEvent) streamEventKind() string { return "tool-call-delta" }

// FinishEvent ends a streamed response, carrying its usage.
type FinishEvent struct {
	Usage     Usage
	ModelName string
}

func (FinishEvent) streamEventKind() string { return "finish" }

// accumulate replays a stream into a ModelResponse, forwarding each event
// to emit. It is the single place stream events become response parts, so
// providers only translate wire chunks into events.
func accumulate(events iter.Seq2[StreamEvent, error], emit func(StreamEvent) bool) (*ModelResponse, error) {
	resp := &ModelResponse{}
	var text, thinking string
	var call *ToolCallPart
	var callArgs string

	flushText := func() {
		if text != "" {
			resp.Parts = append(resp.Parts, TextPart{Content: text})
			text = ""
		}
	}
	flushThinking := func() {
		if thinking != "" {
			resp.Parts = append(resp.Parts, ThinkingPart{Content: thinking})
			thinking = ""
		}
	}
	flushCall := func() {
		if call != nil {
			call.Args = json.RawMessage(callArgs)
			resp.Parts = append(resp.Parts, *call)
			call, callArgs = nil, ""
		}
	}

	for event, err := range events {
		if err != nil {
			return nil, err
		}
		if emit != nil && !emit(event) {
			return nil, context.Canceled
		}
		switch e := event.(type) {
		case TextDeltaEvent:
			flushThinking()
			flushCall()
			text += e.Delta
		case ThinkingDeltaEvent:
			flushText()
			flushCall()
			thinking += e.Delta
		case ToolCallStartEvent:
			flushText()
			flushThinking()
			flushCall()
			call = &ToolCallPart{ToolName: e.ToolName, ToolCallID: e.ToolCallID}
		case ToolCallDeltaEvent:
			if call == nil {
				return nil, &UnexpectedModelBehaviorError{Message: "tool call delta before tool call start"}
			}
			callArgs += e.ArgsDelta
		case FinishEvent:
			flushText()
			flushThinking()
			flushCall()
			resp.Usage = e.Usage
			resp.ModelName = e.ModelName
			return resp, nil
		default:
			return nil, fmt.Errorf("ai: unknown stream event type %T", event)
		}
	}
	return nil, &UnexpectedModelBehaviorError{Message: "stream ended without a finish event"}
}
