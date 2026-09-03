package agui

import (
	"encoding/json"
	"errors"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go"
)

var errConsumerStopped = errors.New("agui: stream consumer stopped")

type eventTransformer struct {
	runID       string
	messageID   string
	messageOpen bool
	response    int
	result      int
	calls       map[string]bool
	partCalls   map[string]string
	stopped     bool
}

func (transformer *eventTransformer) emit(yield func(Event, error) bool, event ai.StreamEvent) error {
	downstream := yield
	yield = func(event Event, err error) bool {
		accepted := downstream(event, err)
		transformer.stopped = !accepted
		return accepted
	}
	switch value := event.(type) {
	case ai.PartStartEvent:
		switch part := value.Part.(type) {
		case ai.TextPart:
			if !transformer.ensureMessage(yield) {
				return errConsumerStopped
			}
			if part.Content != "" {
				yield(Event{Type: EventTextMessageContent, MessageID: transformer.messageID, Delta: part.Content}, nil)
			}
		case ai.ToolCallPart:
			transformer.partCalls[value.PartID] = part.ToolCallID
			transformer.startToolCall(yield, part.ToolCallID, part.ToolName, string(part.Args))
		case ai.NativeToolCallPart:
			transformer.partCalls[value.PartID] = part.ToolCallID
			transformer.startToolCall(yield, part.ToolCallID, part.ToolName, string(part.Args))
		case ai.NativeToolReturnPart:
			content, err := encodeResult(part.Content)
			if err != nil {
				return err
			}
			transformer.toolResult(yield, part.ToolCallID, content)
		}
	case ai.PartDeltaEvent:
		switch delta := value.Delta.(type) {
		case ai.TextPartDelta:
			if !transformer.ensureMessage(yield) {
				return errConsumerStopped
			}
			if delta.ContentDelta != "" {
				yield(Event{Type: EventTextMessageContent, MessageID: transformer.messageID, Delta: delta.ContentDelta}, nil)
			}
		case ai.ToolCallPartDelta:
			toolCallID := delta.ToolCallID
			if toolCallID == "" {
				toolCallID = transformer.partCalls[value.PartID]
			}
			if delta.ArgsDelta != "" {
				yield(Event{Type: EventToolCallArgs, ToolCallID: toolCallID, Delta: delta.ArgsDelta}, nil)
			}
		case ai.NativeToolCallPartDelta:
			converted := ai.ToolCallPartDelta(delta)
			toolCallID := converted.ToolCallID
			if toolCallID == "" {
				toolCallID = transformer.partCalls[value.PartID]
			}
			if converted.ArgsDelta != "" {
				yield(Event{Type: EventToolCallArgs, ToolCallID: toolCallID, Delta: converted.ArgsDelta}, nil)
			}
		}
	case ai.FunctionToolCallEvent:
		transformer.endToolCall(yield, value.Part.ToolCallID, value.Part.ToolName, string(value.Part.Args))
	case ai.OutputToolCallEvent:
		transformer.endToolCall(yield, value.Part.ToolCallID, value.Part.ToolName, string(value.Part.Args))
	case ai.FunctionToolResultEvent:
		toolCallID, content, err := resultContent(value.Part)
		if err != nil {
			return err
		}
		transformer.toolResult(yield, toolCallID, content)
	case ai.OutputToolResultEvent:
		toolCallID, content, err := resultContent(value.Part)
		if err != nil {
			return err
		}
		transformer.toolResult(yield, toolCallID, content)
	case ai.FinishEvent:
		transformer.closeMessage(yield)
		transformer.response++
	}
	if transformer.stopped {
		return errConsumerStopped
	}
	return nil
}

func (transformer *eventTransformer) ensureMessage(yield func(Event, error) bool) bool {
	if transformer.messageOpen {
		return true
	}
	transformer.messageID = fmt.Sprintf("%s:message:%d", transformer.runID, transformer.response)
	transformer.messageOpen = yield(Event{
		Type: EventTextMessageStart, MessageID: transformer.messageID, Role: "assistant",
	}, nil)
	return transformer.messageOpen
}

func (transformer *eventTransformer) closeMessage(yield func(Event, error) bool) bool {
	if !transformer.messageOpen {
		return true
	}
	messageID := transformer.messageID
	transformer.messageOpen = false
	transformer.messageID = ""
	return yield(Event{Type: EventTextMessageEnd, MessageID: messageID}, nil)
}

func (transformer *eventTransformer) startToolCall(
	yield func(Event, error) bool, toolCallID string, name string, args string,
) {
	if transformer.calls[toolCallID] {
		return
	}
	if !transformer.ensureMessage(yield) {
		return
	}
	transformer.calls[toolCallID] = false
	if !yield(Event{
		Type: EventToolCallStart, ToolCallID: toolCallID, ToolCallName: name,
		ParentMessageID: transformer.messageID,
	}, nil) {
		return
	}
	if args != "" {
		yield(Event{Type: EventToolCallArgs, ToolCallID: toolCallID, Delta: args}, nil)
	}
}

func (transformer *eventTransformer) endToolCall(
	yield func(Event, error) bool, toolCallID string, name string, args string,
) {
	if _, exists := transformer.calls[toolCallID]; !exists {
		transformer.startToolCall(yield, toolCallID, name, args)
	}
	if transformer.calls[toolCallID] {
		return
	}
	transformer.calls[toolCallID] = true
	yield(Event{Type: EventToolCallEnd, ToolCallID: toolCallID}, nil)
}

func (transformer *eventTransformer) toolResult(yield func(Event, error) bool, toolCallID string, content string) {
	transformer.result++
	yield(Event{
		Type: EventToolCallResult, MessageID: fmt.Sprintf("%s:tool:%d", transformer.runID, transformer.result),
		Role: "tool", ToolCallID: toolCallID, Content: content,
	}, nil)
}

func encodeResult(content any) (string, error) {
	if text, ok := content.(string); ok {
		return text, nil
	}
	encoded, err := json.Marshal(content)
	if err != nil {
		return "", fmt.Errorf("agui: encode native tool result: %w", err)
	}
	return string(encoded), nil
}
