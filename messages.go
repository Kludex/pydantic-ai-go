package ai

import (
	"encoding/json"
	"time"
)

// ModelMessage is one message in a conversation: either a ModelRequest
// sent to the model or a ModelResponse received from it.
//
// The JSON encoding matches PydanticAI's message format, so serialized
// histories interoperate with PydanticAI, pydantic-evals-go, and Logfire.
type ModelMessage interface {
	messageKind() string
}

// ModelRequest is a message sent to the model.
type ModelRequest struct {
	Parts []RequestPart
}

func (ModelRequest) messageKind() string { return "request" }

// ModelResponse is a message received from the model.
type ModelResponse struct {
	Parts     []ResponsePart
	Usage     Usage
	ModelName string
	Timestamp time.Time
}

func (ModelResponse) messageKind() string { return "response" }

// ToolCalls returns the tool calls in the response.
func (r ModelResponse) ToolCalls() []ToolCallPart {
	var calls []ToolCallPart
	for _, p := range r.Parts {
		if c, ok := p.(ToolCallPart); ok {
			calls = append(calls, c)
		}
	}
	return calls
}

// Text returns the concatenated text content of the response.
func (r ModelResponse) Text() string {
	var s string
	for _, p := range r.Parts {
		if t, ok := p.(TextPart); ok {
			s += t.Content
		}
	}
	return s
}

// RequestPart is one part of a ModelRequest.
type RequestPart interface {
	requestPartKind() string
}

// SystemPromptPart carries the system prompt / instructions.
type SystemPromptPart struct {
	Content string
}

func (SystemPromptPart) requestPartKind() string { return "system-prompt" }

// UserPromptPart carries user input.
type UserPromptPart struct {
	Content string
}

func (UserPromptPart) requestPartKind() string { return "user-prompt" }

// ToolReturnPart carries the result of a tool call back to the model.
type ToolReturnPart struct {
	ToolName   string
	Content    any
	ToolCallID string
}

func (ToolReturnPart) requestPartKind() string { return "tool-return" }

// RetryPromptPart asks the model to try again, carrying the reason.
// It is produced by tool argument validation failures, tools returning
// Retryf errors, and output validation failures.
type RetryPromptPart struct {
	Content    string
	ToolName   string
	ToolCallID string
}

func (RetryPromptPart) requestPartKind() string { return "retry-prompt" }

// ResponsePart is one part of a ModelResponse.
type ResponsePart interface {
	responsePartKind() string
}

// TextPart is plain text produced by the model.
type TextPart struct {
	Content string
}

func (TextPart) responsePartKind() string { return "text" }

// ToolCallPart is a tool call requested by the model.
type ToolCallPart struct {
	ToolName   string
	Args       json.RawMessage
	ToolCallID string
}

func (ToolCallPart) responsePartKind() string { return "tool-call" }

// ThinkingPart is reasoning content produced by the model.
type ThinkingPart struct {
	Content string
}

func (ThinkingPart) responsePartKind() string { return "thinking" }
