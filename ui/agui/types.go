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
)

// Event is one JSON-encoded AG-UI event. Fields not used by Type are omitted.
type Event struct {
	// Type is the event discriminator.
	Type EventType `json:"type"`
	// ThreadID is the protocol conversation identity.
	ThreadID string `json:"threadId,omitempty"`
	// RunID is the client-provided protocol run identity.
	RunID string `json:"runId,omitempty"`
	// MessageID identifies an assistant or tool-result message.
	MessageID string `json:"messageId,omitempty"`
	// Role is assistant or tool when the event starts a message.
	Role string `json:"role,omitempty"`
	// Delta contains incremental text or JSON arguments.
	Delta string `json:"delta,omitempty"`
	// ToolCallID identifies one model tool call.
	ToolCallID string `json:"toolCallId,omitempty"`
	// ToolCallName is the model-facing function name.
	ToolCallName string `json:"toolCallName,omitempty"`
	// ParentMessageID attaches a tool call to its assistant response.
	ParentMessageID string `json:"parentMessageId,omitempty"`
	// Content is a JSON string containing a tool result.
	Content string `json:"content,omitempty"`
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
	// Messages is untrusted client-held history.
	Messages []Message `json:"messages"`
	// Resume contains approval decisions for prior interrupts.
	Resume []ResumeEntry `json:"resume,omitempty"`
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
	// Sanitization controls untrusted history. The zero value is secure.
	Sanitization ai.MessageSanitizationOptions
	// MaxRequestBytes bounds an HTTP request body. Zero defaults to 10 MiB.
	MaxRequestBytes int64
}
