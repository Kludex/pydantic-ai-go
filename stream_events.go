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
	ResponsePartKindText     ResponsePartKind = "text"
	ResponsePartKindThinking ResponsePartKind = "thinking"
	ResponsePartKindToolCall ResponsePartKind = "tool-call"
)

// ResponsePartDelta updates one response part.
type ResponsePartDelta interface {
	// Apply returns a copy of part with this delta applied.
	Apply(part ResponsePart) (ResponsePart, error)
	responsePartDeltaKind() ResponsePartKind
}

// TextPartDelta appends content to a TextPart.
type TextPartDelta struct {
	ContentDelta string
	ProviderName string
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
	return text, nil
}

// ThinkingPartDelta appends content to a ThinkingPart.
type ThinkingPartDelta struct {
	ContentDelta   string
	SignatureDelta string
	ProviderName   string
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
	return thinking, nil
}

// ToolCallPartDelta updates a ToolCallPart. Names and JSON arguments append;
// a non-empty tool-call ID fills an empty ID and must otherwise match it.
type ToolCallPartDelta struct {
	ToolNameDelta string
	ArgsDelta     string
	ToolCallID    string
	ProviderName  string
}

func (ToolCallPartDelta) responsePartDeltaKind() ResponsePartKind { return ResponsePartKindToolCall }

// Apply applies the tool-call delta.
func (d ToolCallPartDelta) Apply(part ResponsePart) (ResponsePart, error) {
	call, ok := part.(ToolCallPart)
	if !ok {
		return nil, fmt.Errorf("ai: cannot apply ToolCallPartDelta to %T", part)
	}
	if d.ToolCallID != "" && call.ToolCallID != "" && d.ToolCallID != call.ToolCallID {
		return nil, &UnexpectedModelBehaviorError{Message: fmt.Sprintf(
			"tool call ID changed from %q to %q", call.ToolCallID, d.ToolCallID,
		)}
	}
	call.ToolName += d.ToolNameDelta
	call.Args = json.RawMessage(append(append([]byte(nil), call.Args...), d.ArgsDelta...))
	if d.ToolCallID != "" {
		call.ToolCallID = d.ToolCallID
	}
	if d.ProviderName != "" {
		call.ProviderName = d.ProviderName
	}
	return call, nil
}

func mergeProviderDetails(base, update map[string]any) map[string]any {
	if len(update) == 0 {
		return base
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
	Index            int
	PartID           string
	Part             ResponsePart
	PreviousPartKind ResponsePartKind
}

func (PartStartEvent) streamEventKind() string { return "part-start" }

// PartDeltaEvent updates a response part previously announced at Index.
type PartDeltaEvent struct {
	Index  int
	PartID string
	Delta  ResponsePartDelta
}

func (PartDeltaEvent) streamEventKind() string { return "part-delta" }

// PartEndEvent marks the current grouping boundary for a response part. A
// provider may still send keyed deltas for an earlier part after this event.
type PartEndEvent struct {
	Index        int
	PartID       string
	Part         ResponsePart
	NextPartKind ResponsePartKind
}

func (PartEndEvent) streamEventKind() string { return "part-end" }

// FinalResultEvent announces the first response part matching the configured
// output. ToolName is empty for text and native output.
type FinalResultEvent struct {
	ToolName   string
	ToolCallID string
}

func (FinalResultEvent) streamEventKind() string { return "final-result" }

// FunctionToolCallEvent announces a function tool call before execution.
type FunctionToolCallEvent struct {
	Part      ToolCallPart
	ArgsValid *bool
}

func (FunctionToolCallEvent) streamEventKind() string { return "function-tool-call" }

// OutputToolCallEvent announces an output tool call before validation.
type OutputToolCallEvent struct {
	Part      ToolCallPart
	ArgsValid *bool
}

func (OutputToolCallEvent) streamEventKind() string { return "output-tool-call" }

// FunctionToolResultEvent carries the request part produced by a function
// tool. Part is a ToolReturnPart or RetryPromptPart.
type FunctionToolResultEvent struct {
	Part RequestPart
}

func (FunctionToolResultEvent) streamEventKind() string { return "function-tool-result" }

// OutputToolResultEvent carries the request part produced by an output tool.
// Part is a ToolReturnPart or RetryPromptPart.
type OutputToolResultEvent struct {
	Part RequestPart
}

func (OutputToolResultEvent) streamEventKind() string { return "output-tool-result" }

// DeferredToolRequestsEvent announces the batch of external calls and
// approvals that paused the run.
type DeferredToolRequestsEvent struct {
	Requests DeferredToolRequests
}

func (DeferredToolRequestsEvent) streamEventKind() string { return "deferred-tool-requests" }
