// Package vercel adapts typed agents to the Vercel AI UI message stream protocol.
package vercel

import (
	"encoding/json"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
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
	// ChunkMessageMetadata merges final response metadata into the assistant message.
	ChunkMessageMetadata ChunkType = "message-metadata"
	// ChunkDone emits the terminal [DONE] SSE record.
	ChunkDone ChunkType = "done"
	// ChunkError reports a failed run or transformation.
	ChunkError ChunkType = "error"
	// ChunkAbort reports a canceled run.
	ChunkAbort ChunkType = "abort"
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
	// ChunkToolInputError reports invalid arguments to an AI SDK v6+ client.
	ChunkToolInputError ChunkType = "tool-input-error"
	// ChunkToolOutputAvailable returns a successful tool result.
	ChunkToolOutputAvailable ChunkType = "tool-output-available"
	// ChunkToolOutputError returns a failed tool result.
	ChunkToolOutputError ChunkType = "tool-output-error"
	// ChunkToolOutputDenied reports a denied tool to an AI SDK v6+ client.
	ChunkToolOutputDenied ChunkType = "tool-output-denied"
	// ChunkToolApprovalRequest asks an AI SDK v6+ client to approve a tool.
	ChunkToolApprovalRequest ChunkType = "tool-approval-request"
	// ChunkFile carries one model-generated file as a data URL.
	ChunkFile ChunkType = "file"
	// ChunkSourceURL carries one cited URL.
	ChunkSourceURL ChunkType = "source-url"
	// ChunkSourceDocument carries one cited document.
	ChunkSourceDocument ChunkType = "source-document"
	// ChunkDataCompaction carries a durable compaction boundary.
	ChunkDataCompaction ChunkType = "data-compaction"
	// ChunkDataToolAvailability carries tools revealed by an earlier result.
	ChunkDataToolAvailability ChunkType = "data-tool-availability-delta"
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
	// Dynamic marks a tool whose schema is not statically known by the client.
	Dynamic *bool `json:"dynamic,omitempty"`
	// Preliminary marks an intermediate provider-native result.
	Preliminary *bool `json:"preliminary,omitempty"`
	// ApprovalID identifies one tool approval request.
	ApprovalID string `json:"approvalId,omitempty"`
	// URL contains a generated file or cited source URL.
	URL string `json:"url,omitempty"`
	// MediaType is the IANA media type of a file or document source.
	MediaType string `json:"mediaType,omitempty"`
	// Filename is the optional name of a document source.
	Filename string `json:"filename,omitempty"`
	// SourceID identifies one cited source.
	SourceID string `json:"sourceId,omitempty"`
	// Title is the display title of a cited source.
	Title string `json:"title,omitempty"`
	// Data contains a protocol-specific data-part payload.
	Data any `json:"data,omitempty"`
	// Transient prevents a custom data part from entering UI message history.
	Transient bool `json:"transient,omitempty"`
	// ProviderMetadata preserves provider-specific part state.
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
	// MessageMetadata preserves application metadata and the response timestamp.
	MessageMetadata map[string]any `json:"messageMetadata,omitempty"`
	// Reason explains why a run was aborted.
	Reason string `json:"reason,omitempty"`
}

// MarshalJSON preserves the required data field on custom data chunks, including JSON null.
func (c Chunk) MarshalJSON() ([]byte, error) {
	type chunk Chunk
	if !strings.HasPrefix(string(c.Type), "data-") {
		return json.Marshal(chunk(c))
	}
	encoded, err := json.Marshal(chunk(c))
	if err != nil || c.Data != nil {
		return encoded, err
	}
	return append(encoded[:len(encoded)-1], `,"data":null}`...), nil
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
	// Model selects an application-approved model in a web chat request.
	Model string `json:"model,omitempty"`
	// BuiltinTools selects application-approved provider-native tools in a web chat request.
	BuiltinTools []string `json:"builtinTools,omitempty"`
}

// UIMessage is one client-held Vercel AI message.
type UIMessage struct {
	// ID is the client-assigned message identity.
	ID string `json:"id"`
	// Role is system, user, or assistant.
	Role string `json:"role"`
	// Parts contains text, file, reasoning, or tool state.
	Parts []UIMessagePart `json:"parts"`
	// Metadata contains application data and framework-owned timestamp state.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// UIMessagePart is the supported subset of a Vercel AI message part.
type UIMessagePart struct {
	// Type identifies a text, reasoning, file, source, data, or tool part.
	Type string `json:"type"`
	// Text contains text or reasoning content.
	Text string `json:"text,omitempty"`
	// ToolName names a dynamic tool part.
	ToolName string `json:"toolName,omitempty"`
	// ToolCallID identifies one tool state machine.
	ToolCallID string `json:"toolCallId,omitempty"`
	// State identifies input or output availability.
	State string `json:"state,omitempty"`
	// Input contains complete JSON tool arguments.
	Input json.RawMessage `json:"input,omitempty"`
	// RawInput preserves malformed input on an output-error part.
	RawInput json.RawMessage `json:"rawInput,omitempty"`
	// Output contains a complete JSON tool result.
	Output json.RawMessage `json:"output,omitempty"`
	// ErrorText describes an output-error tool result.
	ErrorText string `json:"errorText,omitempty"`
	// Approval contains a requested or completed tool decision.
	Approval *ToolApproval `json:"approval,omitempty"`
	// URL contains a hosted file URL or an inline data URL.
	URL string `json:"url,omitempty"`
	// MediaType is the IANA media type of a file.
	MediaType string `json:"mediaType,omitempty"`
	// Filename is the optional display name supplied by the client.
	Filename string `json:"filename,omitempty"`
	// SourceID identifies one cited source.
	SourceID string `json:"sourceId,omitempty"`
	// Title is the display title of a cited source.
	Title string `json:"title,omitempty"`
	// Data contains a data-name-prefixed part payload.
	Data any `json:"data,omitempty"`
	// ProviderMetadata preserves provider-specific text, reasoning, file, or source state.
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
	// CallProviderMetadata preserves provider-specific tool-call state.
	CallProviderMetadata map[string]any `json:"callProviderMetadata,omitempty"`
	// ProviderExecuted marks a provider-native tool call.
	ProviderExecuted *bool `json:"providerExecuted,omitempty"`
	// Preliminary marks an intermediate provider-native result.
	Preliminary *bool `json:"preliminary,omitempty"`
}

// ToolApproval is one Vercel AI tool approval state.
type ToolApproval struct {
	// ID identifies the approval request.
	ID string `json:"id"`
	// Approved is a strict decision when the client has responded.
	Approved *bool `json:"approved,omitempty"`
	// Reason explains a denial.
	Reason string `json:"reason,omitempty"`
}

// StreamConfig controls standalone stream transformation.
type StreamConfig struct {
	// SDKVersion targets AI SDK UI major 5, 6, or 7. Zero defaults to 5.
	SDKVersion int
	// ServerMessageID overrides the generated assistant message ID.
	ServerMessageID string
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
