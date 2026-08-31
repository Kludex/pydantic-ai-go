package ai

import (
	"encoding/json"
	"fmt"
	"time"
)

// ModelMessage is one message in a conversation: either a ModelRequest
// sent to the model or a ModelResponse received from it.
//
// The JSON encoding matches PydanticAI's message format, so serialized
// histories interoperate with PydanticAI, pydantic-evals-go, and Logfire.
type ModelMessage interface {
	EnqueueItem
	messageKind() string
}

// RequestState reports whether construction of a model request completed.
type RequestState string

const (
	// RequestStateComplete is the default state for an ordinary request.
	RequestStateComplete RequestState = "complete"
	// RequestStateInterrupted marks partial tool results retained during cancellation.
	RequestStateInterrupted RequestState = "interrupted"
)

// ModelRequest is a message sent to the model.
type ModelRequest struct {
	Parts          []RequestPart
	Timestamp      time.Time
	Instructions   string
	RunID          string
	ConversationID string
	Metadata       map[string]any
	State          RequestState
}

func (ModelRequest) messageKind() string     { return "request" }
func (ModelRequest) enqueueItemKind() string { return "message" }

// FinishReason is the normalized reason generation stopped.
type FinishReason string

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonLength        FinishReason = "length"
	FinishReasonContentFilter FinishReason = "content_filter"
	FinishReasonToolCall      FinishReason = "tool_call"
	FinishReasonError         FinishReason = "error"
)

// ModelResponseState describes the lifecycle of a model response.
type ModelResponseState string

const (
	ModelResponseStateComplete    ModelResponseState = "complete"
	ModelResponseStateIncomplete  ModelResponseState = "incomplete"
	ModelResponseStateSuspended   ModelResponseState = "suspended"
	ModelResponseStateInterrupted ModelResponseState = "interrupted"
)

// ModelResponse is a message received from the model.
type ModelResponse struct {
	Parts              []ResponsePart
	Usage              Usage
	ModelName          string
	Timestamp          time.Time
	ProviderName       string
	ProviderURL        string
	ProviderDetails    map[string]any
	ProviderResponseID string
	FinishReason       FinishReason
	RunID              string
	ConversationID     string
	Metadata           map[string]any
	State              ModelResponseState

	pricingAttempted bool
}

func (ModelResponse) messageKind() string     { return "response" }
func (ModelResponse) enqueueItemKind() string { return "message" }

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
	EnqueueItem
	requestPartKind() string
}

// SystemPromptPart carries a legacy system prompt. Prefer instructions for
// new applications. DynamicRef identifies a registered dynamic prompt that
// is reevaluated when serialized history is resumed.
type SystemPromptPart struct {
	Content    string
	Timestamp  time.Time
	DynamicRef string
}

func (SystemPromptPart) requestPartKind() string { return "system-prompt" }
func (SystemPromptPart) enqueueItemKind() string { return "request-part" }

// UserPromptPart carries user input. Content holds plain text; Contents,
// when non-empty, holds multimodal items instead and Content is ignored.
type UserPromptPart struct {
	Content   string
	Contents  []UserContent
	Timestamp time.Time
}

func (UserPromptPart) requestPartKind() string { return "user-prompt" }
func (UserPromptPart) enqueueItemKind() string { return "request-part" }

// UserContent is one multimodal item in a user prompt.
type UserContent interface {
	EnqueueItem
	userContentKind() string
}

// TextContent is a text item in a multimodal prompt.
type TextContent struct {
	Text string
}

func (TextContent) userContentKind() string { return "text-content" }
func (TextContent) enqueueItemKind() string { return "user-content" }

// ImageURL references an image by URL.
type ImageURL struct {
	URL string
}

func (ImageURL) userContentKind() string { return "image-url" }
func (ImageURL) enqueueItemKind() string { return "user-content" }

// BinaryContent carries inline binary data, such as an image or document.
type BinaryContent struct {
	Data      []byte
	MediaType string // e.g. "image/png"
}

func (BinaryContent) userContentKind() string { return "binary" }
func (BinaryContent) enqueueItemKind() string { return "user-content" }

// ToolReturnOutcome reports whether a tool completed successfully.
type ToolReturnOutcome string

const (
	// ToolReturnOutcomeSuccess is the default for ordinary return values.
	ToolReturnOutcomeSuccess ToolReturnOutcome = "success"
	// ToolReturnOutcomeFailed marks a terminal failure the model should adapt to.
	ToolReturnOutcomeFailed ToolReturnOutcome = "failed"
	// ToolReturnOutcomeDenied marks a call rejected by an approval policy.
	ToolReturnOutcomeDenied ToolReturnOutcome = "denied"
	// ToolReturnOutcomeInterrupted marks a synthesized result for an interrupted call.
	ToolReturnOutcomeInterrupted ToolReturnOutcome = "interrupted"
)

// SynthesizedToolReturnMetadataKey marks tool returns created while repairing
// incomplete history rather than produced by tool execution.
const SynthesizedToolReturnMetadataKey = "pydantic_ai_synthesized_tool_return"

// ToolReturn separates the value sent as a tool result from additional user
// content and application-only metadata. Return it directly from a function
// tool when the result needs this richer shape.
type ToolReturn struct {
	ReturnValue any
	Content     []UserContent
	Metadata    map[string]any
	Tools       []string
}

// ToolReturnPart carries the result of a tool call back to the model.
type ToolReturnPart struct {
	ToolName   string
	Content    any
	ToolCallID string
	ToolKind   ToolPartKind
	Outcome    ToolReturnOutcome
	Metadata   map[string]any
	Timestamp  time.Time
}

func (ToolReturnPart) requestPartKind() string { return "tool-return" }
func (ToolReturnPart) enqueueItemKind() string { return "request-part" }

// ToolAvailabilityDeltaPart records deferred tools revealed at one point in
// history. ToolsAdded contains model-facing tool names in reveal order.
type ToolAvailabilityDeltaPart struct {
	ToolsAdded []string
	ToolCallID string
}

func (ToolAvailabilityDeltaPart) requestPartKind() string { return "tool-availability-delta" }
func (ToolAvailabilityDeltaPart) enqueueItemKind() string { return "request-part" }

// ValidationError is one structured JSON Schema validation failure.
type ValidationError struct {
	Type     string         `json:"type"`
	Location []any          `json:"loc"`
	Message  string         `json:"msg"`
	Input    any            `json:"input,omitempty"`
	Context  map[string]any `json:"ctx,omitempty"`
	URL      string         `json:"url,omitempty"`
}

// RetryPromptPart asks the model to try again, carrying either plain content
// or structured validation errors. Errors takes precedence when non-nil.
type RetryPromptPart struct {
	Content    string
	Errors     []ValidationError
	ToolName   string
	ToolCallID string
	Timestamp  time.Time
}

func (RetryPromptPart) requestPartKind() string { return "retry-prompt" }
func (RetryPromptPart) enqueueItemKind() string { return "request-part" }

// ModelResponse formats retry feedback for a model.
func (p RetryPromptPart) ModelResponse() string {
	description := p.Content
	if p.Errors != nil {
		errors := make([]ValidationError, len(p.Errors))
		copy(errors, p.Errors)
		for index := range errors {
			errors[index].Context = nil
			if p.ToolName == "" && len(errors[index].Location) <= 1 {
				errors[index].Input = nil
			}
		}
		content, _ := json.MarshalIndent(errors, "", "  ")
		noun := "errors"
		if len(errors) == 1 {
			noun = "error"
		}
		description = fmt.Sprintf("%d validation %s:\n```json\n%s\n```", len(errors), noun, content)
	} else if p.ToolName == "" {
		description = "Validation feedback:\n" + description
	}
	return description + "\n\nFix the errors and try again."
}

// ResponsePart is one part of a ModelResponse.
type ResponsePart interface {
	responsePartKind() string
}

// TextPart is plain text produced by the model.
type TextPart struct {
	Content         string
	ID              string
	ProviderName    string
	ProviderDetails map[string]any
}

func (TextPart) responsePartKind() string { return "text" }

// ToolPartKind identifies a typed cross-provider tool part.
type ToolPartKind string

const (
	ToolPartKindToolSearch     ToolPartKind = "tool-search"
	ToolPartKindCapabilityLoad ToolPartKind = "capability-load"
)

// ToolCallPart is a tool call requested by the model.
type ToolCallPart struct {
	ToolName        string
	Args            json.RawMessage
	ToolCallID      string
	ToolKind        ToolPartKind
	ID              string
	ProviderName    string
	ProviderDetails map[string]any
}

func (ToolCallPart) responsePartKind() string { return "tool-call" }

// NativeToolCallPart records a provider-executed tool call. The agent does not
// execute it locally. ToolKind identifies a portable typed shape when available.
type NativeToolCallPart struct {
	ToolName        string
	Args            json.RawMessage
	ToolCallID      string
	ToolKind        ToolPartKind
	ID              string
	ProviderName    string
	ProviderDetails map[string]any
}

func (NativeToolCallPart) responsePartKind() string { return "builtin-tool-call" }

// NativeToolReturnPart records the provider's result for a native tool call.
type NativeToolReturnPart struct {
	ToolName        string
	Content         any
	ToolCallID      string
	ToolKind        ToolPartKind
	Metadata        map[string]any
	Timestamp       time.Time
	Outcome         ToolReturnOutcome
	ProviderName    string
	ProviderDetails map[string]any
}

func (NativeToolReturnPart) responsePartKind() string { return "builtin-tool-return" }

// ThinkingPart is reasoning content produced by the model.
type ThinkingPart struct {
	Content         string
	ID              string
	Signature       string
	ProviderName    string
	ProviderDetails map[string]any
}

func (ThinkingPart) responsePartKind() string { return "thinking" }

// StandingPromptPlantedKey marks provider details for a compaction built from
// a window that already contained the run's standing prompt.
const StandingPromptPlantedKey = "pydantic_ai_standing_prompt_planted"

// CompactionPart summarizes history that a provider compacted. ProviderDetails
// may contain opaque data required when sending the part back to that provider.
type CompactionPart struct {
	Content         string
	ID              string
	ProviderName    string
	ProviderDetails map[string]any
}

// HasContent reports whether the provider supplied a readable summary.
func (part CompactionPart) HasContent() bool { return part.Content != "" }

func (CompactionPart) responsePartKind() string { return "compaction" }
