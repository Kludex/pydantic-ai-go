package agui

import (
	"encoding/json"
	"errors"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go"
)

var errConsumerStopped = errors.New("agui: stream consumer stopped")

type eventTransformer struct {
	runID            string
	version          protocolVersion
	messageID        string
	messageOpen      bool
	response         int
	result           int
	reasoning        int
	reasoningID      string
	reasoningStarted bool
	reasoningText    bool
	calls            map[string]bool
	partCalls        map[string]string
	outcome          RunOutcome
	stopped          bool
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
		case ai.ThinkingPart:
			transformer.startReasoning(yield, part)
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
		case ai.ThinkingPartDelta:
			if delta.ContentDelta != "" {
				transformer.reasoningDelta(yield, delta.ContentDelta)
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
	case ai.PartEndEvent:
		if part, ok := value.Part.(ai.ThinkingPart); ok {
			transformer.endReasoning(yield, part)
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
	case ai.DeferredToolRequestsEvent:
		transformer.outcome = RunOutcome{Type: "interrupt"}
		for _, call := range value.Requests.Approvals {
			transformer.outcome.Interrupts = append(transformer.outcome.Interrupts, Interrupt{
				ID: "int-" + call.ToolCallID, Reason: "tool_call", ToolCallID: call.ToolCallID,
				Message:        fmt.Sprintf("Approve %s(%s)?", call.ToolName, call.Args),
				ResponseSchema: approvalResponseSchema(),
				Metadata:       cloneMap(value.Requests.Metadata[call.ToolCallID]),
			})
		}
	case ai.FinishEvent:
		transformer.closeMessage(yield)
		transformer.response++
	}
	if transformer.stopped {
		return errConsumerStopped
	}
	return nil
}

func (transformer *eventTransformer) startReasoning(yield func(Event, error) bool, part ai.ThinkingPart) {
	transformer.reasoning++
	transformer.reasoningID = fmt.Sprintf("%s:reasoning:%d", transformer.runID, transformer.reasoning)
	transformer.reasoningStarted = false
	transformer.reasoningText = false
	if part.Content == "" {
		return
	}
	transformer.openReasoning(yield)
	yield(Event{Type: transformer.reasoningContentType(), MessageID: transformer.reasoningMessageID(), Delta: part.Content}, nil)
}

func (transformer *eventTransformer) reasoningDelta(yield func(Event, error) bool, delta string) {
	if transformer.reasoningID == "" {
		transformer.reasoning++
		transformer.reasoningID = fmt.Sprintf("%s:reasoning:%d", transformer.runID, transformer.reasoning)
	}
	transformer.openReasoning(yield)
	yield(Event{Type: transformer.reasoningContentType(), MessageID: transformer.reasoningMessageID(), Delta: delta}, nil)
}

func (transformer *eventTransformer) openReasoning(yield func(Event, error) bool) {
	modern := transformer.version.atLeast(0, 1, 13)
	if !transformer.reasoningStarted {
		kind := EventThinkingStart
		if modern {
			kind = EventReasoningStart
		}
		yield(Event{Type: kind, MessageID: transformer.reasoningMessageID()}, nil)
		transformer.reasoningStarted = true
	}
	if transformer.reasoningText {
		return
	}
	kind := EventThinkingTextMessageStart
	role := ""
	if modern {
		kind = EventReasoningMessageStart
		role = "assistant"
		if transformer.version.atLeast(0, 1, 14) {
			role = "reasoning"
		}
	}
	yield(Event{Type: kind, MessageID: transformer.reasoningMessageID(), Role: role}, nil)
	transformer.reasoningText = true
}

func (transformer *eventTransformer) endReasoning(yield func(Event, error) bool, part ai.ThinkingPart) {
	modern := transformer.version.atLeast(0, 1, 13)
	metadata := reasoningMetadata(part)
	if !transformer.reasoningStarted && (!modern || len(metadata) == 0) {
		transformer.reasoningID = ""
		return
	}
	if !transformer.reasoningStarted {
		kind := EventThinkingStart
		if modern {
			kind = EventReasoningStart
		}
		yield(Event{Type: kind, MessageID: transformer.reasoningMessageID()}, nil)
	}
	if transformer.reasoningText {
		kind := EventThinkingTextMessageEnd
		if modern {
			kind = EventReasoningMessageEnd
		}
		yield(Event{Type: kind, MessageID: transformer.reasoningMessageID()}, nil)
	}
	if modern && len(metadata) > 0 {
		encoded, _ := json.Marshal(metadata)
		yield(Event{
			Type: EventReasoningEncryptedValue, Subtype: "message", EntityID: transformer.reasoningID,
			EncryptedValue: string(encoded),
		}, nil)
	}
	kind := EventThinkingEnd
	if modern {
		kind = EventReasoningEnd
	}
	yield(Event{Type: kind, MessageID: transformer.reasoningMessageID()}, nil)
	transformer.reasoningID = ""
	transformer.reasoningStarted = false
	transformer.reasoningText = false
}

func (transformer *eventTransformer) reasoningMessageID() string {
	if transformer.version.atLeast(0, 1, 13) {
		return transformer.reasoningID
	}
	return ""
}

func (transformer *eventTransformer) reasoningContentType() EventType {
	if transformer.version.atLeast(0, 1, 13) {
		return EventReasoningMessageContent
	}
	return EventThinkingTextMessageContent
}

func reasoningMetadata(part ai.ThinkingPart) map[string]any {
	metadata := map[string]any{}
	if part.ID != "" {
		metadata["id"] = part.ID
	}
	if part.Signature != "" {
		metadata["signature"] = part.Signature
	}
	if part.ProviderName != "" {
		metadata["provider_name"] = part.ProviderName
	}
	if len(part.ProviderDetails) > 0 {
		metadata["provider_details"] = cloneMap(part.ProviderDetails)
	}
	return metadata
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

func approvalResponseSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"approved":   map[string]any{"type": "boolean"},
			"editedArgs": map[string]any{"type": "object"},
			"reason":     map[string]any{"type": "string"},
		},
		"required": []string{"approved"},
	}
}

func cloneMap(value map[string]any) map[string]any {
	encoded, _ := json.Marshal(value)
	var cloned map[string]any
	_ = json.Unmarshal(encoded, &cloned)
	return cloned
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
