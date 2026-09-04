// Package agui adapts typed agents to the Agent User Interaction protocol.
package agui

import (
	"encoding/json"

	ai "github.com/Kludex/pydantic-ai-go"
)

// EventType identifies one AG-UI stream event.
type EventType string

const (
	// EventRunStarted begins one protocol run.
	EventRunStarted EventType = "RUN_STARTED"
	// EventRunFinished completes one protocol run.
	EventRunFinished EventType = "RUN_FINISHED"
	// EventRunError reports a failed protocol run.
	EventRunError EventType = "RUN_ERROR"
	// EventTextMessageStart begins one assistant message.
	EventTextMessageStart EventType = "TEXT_MESSAGE_START"
	// EventTextMessageContent appends assistant text.
	EventTextMessageContent EventType = "TEXT_MESSAGE_CONTENT"
	// EventTextMessageEnd closes one assistant message.
	EventTextMessageEnd EventType = "TEXT_MESSAGE_END"
	// EventToolCallStart begins one model tool call.
	EventToolCallStart EventType = "TOOL_CALL_START"
	// EventToolCallArgs appends generated JSON arguments.
	EventToolCallArgs EventType = "TOOL_CALL_ARGS"
	// EventToolCallEnd closes one model tool call.
	EventToolCallEnd EventType = "TOOL_CALL_END"
	// EventToolCallResult returns one local tool result.
	EventToolCallResult EventType = "TOOL_CALL_RESULT"
	// EventThinkingStart begins legacy reasoning output before AG-UI 0.1.13.
	EventThinkingStart EventType = "THINKING_START"
	// EventThinkingEnd closes legacy reasoning output.
	EventThinkingEnd EventType = "THINKING_END"
	// EventThinkingTextMessageStart begins legacy reasoning text.
	EventThinkingTextMessageStart EventType = "THINKING_TEXT_MESSAGE_START"
	// EventThinkingTextMessageContent appends legacy reasoning text.
	EventThinkingTextMessageContent EventType = "THINKING_TEXT_MESSAGE_CONTENT"
	// EventThinkingTextMessageEnd closes legacy reasoning text.
	EventThinkingTextMessageEnd EventType = "THINKING_TEXT_MESSAGE_END"
	// EventReasoningStart begins AG-UI 0.1.13+ reasoning output.
	EventReasoningStart EventType = "REASONING_START"
	// EventReasoningEnd closes AG-UI 0.1.13+ reasoning output.
	EventReasoningEnd EventType = "REASONING_END"
	// EventReasoningMessageStart begins AG-UI 0.1.13+ reasoning text.
	EventReasoningMessageStart EventType = "REASONING_MESSAGE_START"
	// EventReasoningMessageContent appends AG-UI 0.1.13+ reasoning text.
	EventReasoningMessageContent EventType = "REASONING_MESSAGE_CONTENT"
	// EventReasoningMessageEnd closes AG-UI 0.1.13+ reasoning text.
	EventReasoningMessageEnd EventType = "REASONING_MESSAGE_END"
	// EventReasoningEncryptedValue preserves opaque reasoning metadata.
	EventReasoningEncryptedValue EventType = "REASONING_ENCRYPTED_VALUE"
	// EventActivitySnapshot replaces one AG-UI activity value.
	EventActivitySnapshot EventType = "ACTIVITY_SNAPSHOT"
	// EventActivityDelta applies a JSON Patch to one AG-UI activity value.
	EventActivityDelta EventType = "ACTIVITY_DELTA"
	// EventStateSnapshot replaces frontend-managed state.
	EventStateSnapshot EventType = "STATE_SNAPSHOT"
	// EventStateDelta applies a JSON Patch to frontend-managed state.
	EventStateDelta EventType = "STATE_DELTA"
	// EventCustom carries application-defined data.
	EventCustom EventType = "CUSTOM"
)

// Event is one JSON-encoded AG-UI event. Fields not used by Type are omitted.
type Event struct {
	// Type is the event discriminator.
	Type EventType `json:"type"`
	// Timestamp is Unix time in milliseconds.
	Timestamp int64 `json:"timestamp,omitempty"`
	// ThreadID is the protocol conversation identity.
	ThreadID string `json:"threadId,omitempty"`
	// RunID is the client-provided protocol run identity.
	RunID string `json:"runId,omitempty"`
	// MessageID identifies an assistant or tool-result message.
	MessageID string `json:"messageId,omitempty"`
	// Role is assistant or tool when the event starts a message.
	Role string `json:"role,omitempty"`
	// Delta contains incremental text, JSON arguments, or a state patch array.
	Delta any `json:"delta,omitempty"`
	// ToolCallID identifies one model tool call.
	ToolCallID string `json:"toolCallId,omitempty"`
	// ToolCallName is the model-facing function name.
	ToolCallName string `json:"toolCallName,omitempty"`
	// ParentMessageID attaches a tool call to its assistant response.
	ParentMessageID string `json:"parentMessageId,omitempty"`
	// Content contains a tool result or complete activity snapshot.
	Content any `json:"content,omitempty"`
	// Snapshot contains a complete application state value.
	Snapshot any `json:"snapshot,omitempty"`
	// Name identifies a custom event.
	Name string `json:"name,omitempty"`
	// Value contains custom event data.
	Value any `json:"value,omitempty"`
	// Subtype identifies a message or tool-call encrypted value.
	Subtype string `json:"subtype,omitempty"`
	// EntityID identifies the message or tool call carrying an encrypted value.
	EntityID string `json:"entityId,omitempty"`
	// EncryptedValue preserves opaque reasoning or typed-tool metadata.
	EncryptedValue string `json:"encryptedValue,omitempty"`
	// ActivityType identifies one activity data contract.
	ActivityType string `json:"activityType,omitempty"`
	// Patch contains an activity JSON Patch delta.
	Patch []any `json:"patch,omitempty"`
	// Replace controls whether an activity snapshot replaces prior content.
	Replace *bool `json:"replace,omitempty"`
	// Message describes a run error.
	Message string `json:"message,omitempty"`
	// Outcome describes a successful or interrupted run.
	Outcome *RunOutcome `json:"outcome,omitempty"`
}

// RunAgentInput is the provider-neutral subset of an AG-UI run request.
type RunAgentInput struct {
	// ThreadID identifies the frontend conversation.
	ThreadID string `json:"threadId"`
	// RunID identifies this frontend run.
	RunID string `json:"runId"`
	// State contains frontend-managed application state.
	State any `json:"state,omitempty"`
	// Messages is untrusted client-held history.
	Messages []Message `json:"messages"`
	// Tools contains client-executed frontend tool definitions.
	Tools []FrontendTool `json:"tools,omitempty"`
	// Context contains frontend-provided contextual values.
	Context []Context `json:"context,omitempty"`
	// ForwardedProps contains application-specific request data.
	ForwardedProps any `json:"forwardedProps,omitempty"`
	// Resume contains approval decisions for prior interrupts.
	Resume []ResumeEntry `json:"resume,omitempty"`
}

// RunInputReceiver accepts detached AG-UI state, context, and forwarded properties.
type RunInputReceiver interface {
	SetAGUIRunInput(input ForwardedInput) error
}

// ForwardedInput contains application data supplied with one AG-UI run.
type ForwardedInput struct {
	// ThreadID identifies the frontend conversation.
	ThreadID string
	// RunID identifies the frontend run.
	RunID string
	// State contains frontend-managed application state.
	State any
	// Context contains frontend-provided contextual values.
	Context []Context
	// ForwardedProps contains application-specific request data.
	ForwardedProps any
	// Events emits state and custom events for this run.
	Events *EventQueue
}

// Context is one frontend-provided description and value pair.
type Context struct {
	// Description explains the contextual value to the application.
	Description string `json:"description"`
	// Value contains the contextual data.
	Value string `json:"value"`
}

// FrontendTool describes one client-executed tool available for this run.
type FrontendTool struct {
	// Name is the model-facing tool name.
	Name string `json:"name"`
	// Description explains when the model should call the tool.
	Description string `json:"description"`
	// Parameters is the tool's JSON Schema.
	Parameters map[string]any `json:"parameters,omitempty"`
}

// ResumeEntry resolves one prior approval interrupt.
type ResumeEntry struct {
	// InterruptID is the int-prefixed tool-call identity.
	InterruptID string `json:"interruptId"`
	// Status may be completed or cancelled.
	Status string `json:"status"`
	// Payload contains the strict approval decision.
	Payload json.RawMessage `json:"payload"`
}

// RunOutcome is the terminal AG-UI run outcome.
type RunOutcome struct {
	// Type is success or interrupt.
	Type string `json:"type"`
	// Interrupts contains pending approval requests.
	Interrupts []Interrupt `json:"interrupts,omitempty"`
}

// Interrupt describes one pending tool approval.
type Interrupt struct {
	// ID is the int-prefixed tool-call identity.
	ID string `json:"id"`
	// Reason is tool_call.
	Reason string `json:"reason"`
	// ToolCallID is the original model call identity.
	ToolCallID string `json:"toolCallId"`
	// Message is a human-readable approval question.
	Message string `json:"message"`
	// ResponseSchema describes the accepted ResumeEntry payload.
	ResponseSchema map[string]any `json:"responseSchema"`
	// Metadata contains detached approval metadata.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Message is one inbound AG-UI text or tool message.
type Message struct {
	// ID is the client-assigned message identity.
	ID string `json:"id"`
	// Role is system, user, assistant, or tool.
	Role string `json:"role"`
	// Content is text or typed multimodal user content.
	Content any `json:"content,omitempty"`
	// ToolCalls contains assistant function calls.
	ToolCalls []ToolCall `json:"toolCalls,omitempty"`
	// ToolCallID associates a tool result with its call.
	ToolCallID string `json:"toolCallId,omitempty"`
	// Name is the function name for a tool result.
	Name string `json:"name,omitempty"`
	// EncryptedValue preserves reasoning or typed-tool metadata.
	EncryptedValue string `json:"encryptedValue,omitempty"`
	// ActivityType identifies an activity message contract.
	ActivityType string `json:"activityType,omitempty"`
}

// InputContent is one AG-UI text, binary, image, audio, video, or document input.
type InputContent struct {
	// Type identifies text, binary, image, audio, video, or document content.
	Type string `json:"type"`
	// Text contains text input.
	Text string `json:"text,omitempty"`
	// URL contains legacy binary input by URL.
	URL string `json:"url,omitempty"`
	// Data contains legacy binary input as base64.
	Data string `json:"data,omitempty"`
	// MimeType identifies legacy binary input.
	MimeType string `json:"mimeType,omitempty"`
	// Source contains typed multimodal URL or base64 data.
	Source *InputContentSource `json:"source,omitempty"`
	// Metadata carries application data plus reserved file options.
	Metadata any `json:"metadata,omitempty"`
}

// InputContentSource identifies typed multimodal URL or base64 data.
type InputContentSource struct {
	// Type is url or data.
	Type string `json:"type"`
	// Value is a URL or base64 data.
	Value string `json:"value"`
	// MimeType identifies the content.
	MimeType string `json:"mimeType"`
}

// ToolCall is one OpenAI-shaped AG-UI assistant tool call.
type ToolCall struct {
	// ID is the model tool-call identity.
	ID string `json:"id"`
	// Type is normally function.
	Type string `json:"type"`
	// Function contains the name and encoded arguments.
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction describes one AG-UI function call.
type ToolCallFunction struct {
	// Name is the model-facing function name.
	Name string `json:"name"`
	// Arguments contains a complete JSON value.
	Arguments string `json:"arguments"`
}

// Config controls inbound trust boundaries.
type Config struct {
	// Version selects AG-UI protocol behavior. Zero defaults to 0.1.19.
	Version string
	// Sanitization controls untrusted history. The zero value is secure.
	Sanitization ai.MessageSanitizationOptions
	// PreserveFileData round-trips generated and provider-hosted files through reserved activities.
	PreserveFileData bool
	// MaxRequestBytes bounds an HTTP request body. Zero defaults to 10 MiB.
	MaxRequestBytes int64
}

// StreamConfig controls standalone AG-UI stream transformation.
type StreamConfig struct {
	// Version selects AG-UI protocol behavior. Zero defaults to 0.1.19.
	Version string
	// ThreadID identifies the frontend conversation.
	ThreadID string
	// RunID identifies the frontend run.
	RunID string
	// PreserveFileData emits generated files through reserved activity snapshots.
	PreserveFileData bool
}
