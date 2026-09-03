// Package vercel adapts typed agents to the Vercel AI UI message stream protocol.
package vercel

import (
	"encoding/json"

	ai "github.com/Kludex/pydantic-ai-go"
)

// ChunkType identifies one Vercel AI UI message stream chunk.
type ChunkType string

const (
	// ChunkStart begins one assistant message stream.
	ChunkStart ChunkType = "start"
	// ChunkStartStep begins one model response step.
	ChunkStartStep ChunkType = "start-step"
	// ChunkFinishStep ends one model response step.
	ChunkFinishStep ChunkType = "finish-step"
	// ChunkFinish reports the normalized run finish reason.
	ChunkFinish ChunkType = "finish"
	// ChunkDone emits the terminal [DONE] SSE record.
	ChunkDone ChunkType = "done"
	// ChunkError reports a failed run or transformation.
	ChunkError ChunkType = "error"
	// ChunkTextStart begins one text part.
	ChunkTextStart ChunkType = "text-start"
	// ChunkTextDelta appends text content.
	ChunkTextDelta ChunkType = "text-delta"
	// ChunkTextEnd closes one text part.
	ChunkTextEnd ChunkType = "text-end"
	// ChunkReasoningStart begins one reasoning part.
	ChunkReasoningStart ChunkType = "reasoning-start"
	// ChunkReasoningDelta appends reasoning content.
	ChunkReasoningDelta ChunkType = "reasoning-delta"
	// ChunkReasoningEnd closes one reasoning part.
	ChunkReasoningEnd ChunkType = "reasoning-end"
	// ChunkToolInputStart begins streamed tool arguments.
	ChunkToolInputStart ChunkType = "tool-input-start"
	// ChunkToolInputDelta appends a tool argument fragment.
	ChunkToolInputDelta ChunkType = "tool-input-delta"
	// ChunkToolInputAvailable announces complete validated arguments.
	ChunkToolInputAvailable ChunkType = "tool-input-available"
	// ChunkToolOutputAvailable returns a successful tool result.
	ChunkToolOutputAvailable ChunkType = "tool-output-available"
	// ChunkToolOutputError returns a failed, denied, or interrupted result.
	ChunkToolOutputError ChunkType = "tool-output-error"
)

// Chunk is one JSON Vercel AI UI message stream value.
type Chunk struct {
	// Type is the chunk discriminator.
	Type ChunkType `json:"type"`
	// ID identifies one text or reasoning part.
	ID string `json:"id,omitempty"`
	// MessageID identifies the server-generated assistant message.
	MessageID string `json:"messageId,omitempty"`
	// Delta contains incremental text or reasoning.
	Delta string `json:"delta,omitempty"`
	// ToolCallID identifies one model tool call.
	ToolCallID string `json:"toolCallId,omitempty"`
	// ToolName is the model-facing function name.
	ToolName string `json:"toolName,omitempty"`
	// InputTextDelta contains incremental JSON arguments.
	InputTextDelta string `json:"inputTextDelta,omitempty"`
	// Input contains complete decoded tool arguments.
	Input any `json:"input,omitempty"`
	// Output contains a normalized successful tool result.
	Output any `json:"output,omitempty"`
	// ErrorText describes a run, transformation, or tool failure.
	ErrorText string `json:"errorText,omitempty"`
	// FinishReason is the Vercel AI normalized stop reason.
	FinishReason string `json:"finishReason,omitempty"`
	// ProviderExecuted marks provider-native tool calls and results.
	ProviderExecuted *bool `json:"providerExecuted,omitempty"`
}

// RequestData is a Vercel AI submit-message or regenerate-message request.
type RequestData struct {
	// Trigger is submit-message or regenerate-message.
	Trigger string `json:"trigger"`
	// ID is the frontend chat identity mapped to the agent conversation.
	ID string `json:"id"`
	// Messages is untrusted client-held UI history.
	Messages []UIMessage `json:"messages"`
	// MessageID selects an assistant message for regeneration.
	MessageID string `json:"messageId,omitempty"`
}

// UIMessage is one client-held Vercel AI message.
type UIMessage struct {
	// ID is the client-assigned message identity.
	ID string `json:"id"`
	// Role is system, user, or assistant.
	Role string `json:"role"`
	// Parts contains text, reasoning, or tool state.
	Parts []UIMessagePart `json:"parts"`
}

// UIMessagePart is the supported subset of a Vercel AI message part.
type UIMessagePart struct {
	// Type is text, reasoning, or a tool-name-prefixed discriminator.
	Type string `json:"type"`
	// Text contains text or reasoning content.
	Text string `json:"text,omitempty"`
	// ToolCallID identifies one tool state machine.
	ToolCallID string `json:"toolCallId,omitempty"`
	// State identifies input or output availability.
	State string `json:"state,omitempty"`
	// Input contains complete JSON tool arguments.
	Input json.RawMessage `json:"input,omitempty"`
	// Output contains a complete JSON tool result.
	Output json.RawMessage `json:"output,omitempty"`
	// ErrorText describes an output-error tool result.
	ErrorText string `json:"errorText,omitempty"`
}

// Config controls Vercel AI protocol and trust behavior.
type Config struct {
	// Sanitization controls untrusted history. The zero value is secure.
	Sanitization ai.MessageSanitizationOptions
	// SDKVersion targets AI SDK UI major 5, 6, or 7. Zero defaults to 5.
	SDKVersion int
	// ServerMessageID overrides the generated assistant message ID.
	ServerMessageID string
	// MaxRequestBytes bounds an HTTP request body. Zero defaults to 10 MiB.
	MaxRequestBytes int64
}
