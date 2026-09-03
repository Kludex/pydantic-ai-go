package ai

import (
	"encoding/json"
	"fmt"
	"strings"
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
	// Parts contains user, system, tool-return, and retry content.
	Parts []RequestPart
	// Timestamp is the request creation time.
	Timestamp time.Time
	// Instructions records the standing instructions used for this request.
	Instructions string
	// RunID identifies the run that created the request.
	RunID string
	// ConversationID identifies related runs in one conversation.
	ConversationID string
	// Metadata contains detached application request data.
	Metadata map[string]any
	// State reports complete or interrupted request construction.
	State RequestState
}

func (ModelRequest) messageKind() string     { return "request" }
func (ModelRequest) enqueueItemKind() string { return "message" }

// FinishReason is the normalized reason generation stopped.
type FinishReason string

const (
	// FinishReasonStop reports an ordinary model stop.
	FinishReasonStop FinishReason = "stop"
	// FinishReasonLength reports a token or context limit.
	FinishReasonLength FinishReason = "length"
	// FinishReasonContentFilter reports provider safety filtering.
	FinishReasonContentFilter FinishReason = "content_filter"
	// FinishReasonToolCall reports that generation requested tools.
	FinishReasonToolCall FinishReason = "tool_call"
	// FinishReasonError reports malformed or failed generation.
	FinishReasonError FinishReason = "error"
)

// ModelResponseState describes the lifecycle of a model response.
type ModelResponseState string

const (
	// ModelResponseStateComplete marks a finished response.
	ModelResponseStateComplete ModelResponseState = "complete"
	// ModelResponseStateIncomplete marks provider-truncated output.
	ModelResponseStateIncomplete ModelResponseState = "incomplete"
	// ModelResponseStateSuspended marks provider work that can continue.
	ModelResponseStateSuspended ModelResponseState = "suspended"
	// ModelResponseStateInterrupted marks locally interrupted streaming output.
	ModelResponseStateInterrupted ModelResponseState = "interrupted"
)

// ModelResponse is a message received from the model.
type ModelResponse struct {
	// Parts contains detached model output in provider order.
	Parts []ResponsePart
	// Usage contains usage for this response segment.
	Usage Usage
	// ModelName is the provider's resolved model identity.
	ModelName string
	// Timestamp is the response creation time.
	Timestamp time.Time
	// ProviderName is the durable provider identity.
	ProviderName string
	// ProviderURL is the configured provider endpoint.
	ProviderURL string
	// ProviderDetails contains detached provider-specific response data.
	ProviderDetails map[string]any
	// ProviderResponseID identifies resumable provider-side state.
	ProviderResponseID string
	// FinishReason is the normalized generation stop reason.
	FinishReason FinishReason
	// RunID identifies the run that received the response.
	RunID string
	// ConversationID identifies related runs in one conversation.
	ConversationID string
	// Metadata contains detached application response data.
	Metadata map[string]any
	// State reports the response lifecycle state.
	State ModelResponseState

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

// Text returns text content, including speech transcripts. Adjacent text and
// speech parts are joined; non-text parts separate groups with blank lines.
func (r ModelResponse) Text() string {
	var groups []string
	adjacent := false
	for _, responsePart := range r.Parts {
		content := ""
		switch part := responsePart.(type) {
		case TextPart:
			content = part.Content
		case SpeechPart:
			content = part.Content()
		}
		if content == "" {
			adjacent = false
			continue
		}
		if adjacent {
			groups[len(groups)-1] += content
		} else {
			groups = append(groups, content)
		}
		adjacent = true
	}
	return strings.Join(groups, "\n\n")
}

// SpeechSpeaker identifies who produced a SpeechPart.
type SpeechSpeaker string

const (
	// SpeechSpeakerUser identifies user-produced audio.
	SpeechSpeakerUser SpeechSpeaker = "user"
	// SpeechSpeakerAssistant identifies model-produced audio.
	SpeechSpeakerAssistant SpeechSpeaker = "assistant"
)

// SpeechPart is spoken audio from a realtime session with its optional transcript.
// Request messages require SpeechSpeakerUser. Response messages require SpeechSpeakerAssistant.
type SpeechPart struct {
	// Speaker identifies who produced the audio.
	Speaker SpeechSpeaker
	// Transcript is the retained current transcript when available.
	Transcript *string
	// Audio is detached retained audio when enabled.
	Audio *BinaryContent
	// InterruptedAtMS is the playback interruption offset.
	InterruptedAtMS *int
	// ID is the provider's response-part identity.
	ID string
	// ProviderName identifies the provider that produced the speech.
	ProviderName string
	// ProviderDetails contains detached provider-specific speech data.
	ProviderDetails map[string]any
}

// Content returns the transcript or an empty string when no transcript is available.
func (part SpeechPart) Content() string {
	if part.Transcript == nil {
		return ""
	}
	return *part.Transcript
}

// HasContent reports whether the part has a non-empty transcript or retained audio.
func (part SpeechPart) HasContent() bool { return part.Content() != "" || part.Audio != nil }

func (SpeechPart) requestPartKind() string  { return "speech" }
func (SpeechPart) responsePartKind() string { return "speech" }
func (SpeechPart) enqueueItemKind() string  { return "request-part" }

// RequestPart is one part of a ModelRequest.
type RequestPart interface {
	EnqueueItem
	requestPartKind() string
}

// SystemPromptPart carries a legacy system prompt. Prefer instructions for
// new applications. DynamicRef identifies a registered dynamic prompt that
// is reevaluated when serialized history is resumed.
type SystemPromptPart struct {
	// Content is the legacy system prompt text.
	Content string
	// Timestamp is the part creation time.
	Timestamp time.Time
	// DynamicRef identifies a registered prompt reevaluated during history reuse.
	DynamicRef string
}

func (SystemPromptPart) requestPartKind() string { return "system-prompt" }
func (SystemPromptPart) enqueueItemKind() string { return "request-part" }

// UserPromptPart carries user input. Content holds plain text; Contents,
// when non-empty, holds multimodal items instead and Content is ignored.
type UserPromptPart struct {
	// Content is plain text used when Contents is empty.
	Content string
	// Contents contains detached multimodal input and takes precedence over Content.
	Contents []UserContent
	// Timestamp is the prompt creation time.
	Timestamp time.Time
}

func (UserPromptPart) requestPartKind() string { return "user-prompt" }
func (UserPromptPart) enqueueItemKind() string { return "request-part" }

// UserContent is one multimodal item in a user prompt.
type UserContent interface {
	EnqueueItem
	userContentKind() string
}

// TextContent is a text item with application metadata that providers do not receive.
type TextContent struct {
	// Text is model-visible prompt content.
	Text string
	// Metadata is application-only data not sent to providers.
	Metadata any
}

func (TextContent) userContentKind() string { return "text-content" }
func (TextContent) enqueueItemKind() string { return "user-content" }

// CachePointTTL selects the lifetime of an explicit prompt-cache boundary.
type CachePointTTL string

const (
	// CachePointTTL5Minutes requests five-minute prompt retention.
	CachePointTTL5Minutes CachePointTTL = "5m"
	// CachePointTTL1Hour requests one-hour prompt retention.
	CachePointTTL1Hour CachePointTTL = "1h"
)

// CachePoint marks the preceding user-content item as a prompt-cache boundary.
// The zero value uses a five-minute lifetime. Unsupported providers omit the marker.
type CachePoint struct {
	// TTL selects prompt retention. Empty defaults to five minutes.
	TTL CachePointTTL
}

// ResolvedTTL returns the explicit lifetime, applying and validating the default.
func (point CachePoint) ResolvedTTL() (CachePointTTL, error) {
	if point.TTL == "" {
		return CachePointTTL5Minutes, nil
	}
	if point.TTL != CachePointTTL5Minutes && point.TTL != CachePointTTL1Hour {
		return "", fmt.Errorf("ai: invalid cache point TTL %q", point.TTL)
	}
	return point.TTL, nil
}

func (CachePoint) userContentKind() string { return "cache-point" }
func (CachePoint) enqueueItemKind() string { return "user-content" }

// UploadedFile references a file already hosted by a model provider.
type UploadedFile struct {
	// FileID is the provider-hosted file identifier or URI.
	FileID string
	// ProviderName identifies the provider that owns the file.
	ProviderName string
	// MediaType identifies the hosted content when known.
	MediaType string
	// Identifier is the stable application content identity.
	Identifier string
	// VendorMetadata contains detached provider-specific file data.
	VendorMetadata map[string]any
}

func (UploadedFile) userContentKind() string { return "uploaded-file" }
func (UploadedFile) enqueueItemKind() string { return "user-content" }

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
	// ReturnValue is the primary model-visible tool result.
	ReturnValue any
	// Content adds model-visible rich user content beside the result.
	Content []UserContent
	// Metadata is detached application-only result data.
	Metadata map[string]any
	// Tools lists deferred tool names revealed by this result.
	Tools []string
	// Usage adds delegated model work to the parent run's totals and limits.
	Usage Usage
}

// ToolReturnPart carries the result of a tool call back to the model.
type ToolReturnPart struct {
	// ToolName identifies the called function.
	ToolName string
	// Content is the model-visible return value.
	Content any
	// ToolCallID associates the result with a model call.
	ToolCallID string
	// ToolKind identifies a typed framework-managed tool surface.
	ToolKind ToolPartKind
	// Outcome reports success, failure, denial, or interruption.
	Outcome ToolReturnOutcome
	// Metadata contains detached application-only result data.
	Metadata map[string]any
	// Timestamp is the result creation time.
	Timestamp time.Time
}

func (ToolReturnPart) requestPartKind() string { return "tool-return" }
func (ToolReturnPart) enqueueItemKind() string { return "request-part" }

// ToolAvailabilityDeltaPart records deferred tools revealed at one point in
// history. ToolsAdded contains model-facing tool names in reveal order.
type ToolAvailabilityDeltaPart struct {
	// ToolsAdded contains revealed model-facing names in order.
	ToolsAdded []string
	// ToolCallID identifies the result that revealed the tools.
	ToolCallID string
}

func (ToolAvailabilityDeltaPart) requestPartKind() string { return "tool-availability-delta" }
func (ToolAvailabilityDeltaPart) enqueueItemKind() string { return "request-part" }

// ValidationError is one structured JSON Schema validation failure.
type ValidationError struct {
	// Type is the stable validation error discriminator.
	Type string `json:"type"`
	// Location is the path to the invalid value.
	Location []any `json:"loc"`
	// Message describes the validation failure.
	Message string `json:"msg"`
	// Input is the rejected value when safe to expose.
	Input any `json:"input,omitempty"`
	// Context contains detached validator-specific details.
	Context map[string]any `json:"ctx,omitempty"`
	// URL links to additional error documentation.
	URL string `json:"url,omitempty"`
}

// RetryPromptPart asks the model to try again, carrying either plain content
// or structured validation errors. Errors takes precedence when non-nil.
type RetryPromptPart struct {
	// Content is plain corrective feedback used when Errors is nil.
	Content string
	// Errors is structured validation feedback and takes precedence over Content.
	Errors []ValidationError
	// ToolName identifies the tool whose arguments or result failed.
	ToolName string
	// ToolCallID associates feedback with a model call.
	ToolCallID string
	// Timestamp is the feedback creation time.
	Timestamp time.Time
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
	// Content is model-generated text.
	Content string
	// ID is the provider's response-part identity.
	ID string
	// ProviderName identifies the provider that produced the text.
	ProviderName string
	// ProviderDetails contains detached provider-specific part data.
	ProviderDetails map[string]any
}

func (TextPart) responsePartKind() string { return "text" }

// FilePart is binary content produced by a model or provider-native tool.
type FilePart struct {
	// Content contains detached generated bytes and media type.
	Content BinaryContent
	// ID is the provider's response-part identity.
	ID string
	// ProviderName identifies the provider that produced the file.
	ProviderName string
	// ProviderDetails contains detached provider-specific part data.
	ProviderDetails map[string]any
}

func (FilePart) responsePartKind() string { return "file" }

// ToolPartKind identifies a typed cross-provider tool part.
type ToolPartKind string

const (
	// ToolPartKindToolSearch identifies deferred tool discovery.
	ToolPartKindToolSearch ToolPartKind = "tool-search"
	// ToolPartKindCapabilityLoad identifies capability discovery.
	ToolPartKindCapabilityLoad ToolPartKind = "capability-load"
	// ToolPartKindWebSearch identifies web search.
	ToolPartKindWebSearch ToolPartKind = "web-search"
	// ToolPartKindWebFetch identifies URL retrieval.
	ToolPartKindWebFetch ToolPartKind = "web-fetch"
	// ToolPartKindCodeExecution identifies provider code execution.
	ToolPartKindCodeExecution ToolPartKind = "code-execution"
	// ToolPartKindImageGeneration identifies image generation.
	ToolPartKindImageGeneration ToolPartKind = "image-generation"
	// ToolPartKindFileSearch identifies managed file retrieval.
	ToolPartKindFileSearch ToolPartKind = "file-search"
	// ToolPartKindMCPServer identifies provider-hosted MCP calls.
	ToolPartKindMCPServer ToolPartKind = "mcp-server"
	// ToolPartKindAdvisor identifies provider advisor calls.
	ToolPartKindAdvisor ToolPartKind = "advisor"
)

// ToolCallPart is a tool call requested by the model.
type ToolCallPart struct {
	// ToolName is the model-facing function name.
	ToolName string
	// Args contains generated JSON arguments.
	Args json.RawMessage
	// ToolCallID is the provider-assigned call identity.
	ToolCallID string
	// ToolKind identifies a typed framework-managed surface.
	ToolKind ToolPartKind
	// ID is the provider's response-part identity.
	ID string
	// ProviderName identifies the provider that produced the call.
	ProviderName string
	// ProviderDetails contains detached provider-specific call data.
	ProviderDetails map[string]any
}

func (ToolCallPart) responsePartKind() string { return "tool-call" }

// NativeToolCallPart records a provider-executed tool call. The agent does not
// execute it locally. ToolKind identifies a portable typed shape when available.
type NativeToolCallPart struct {
	// ToolName is the provider-native tool name.
	ToolName string
	// Args contains generated JSON arguments.
	Args json.RawMessage
	// ToolCallID is the provider-assigned call identity.
	ToolCallID string
	// ToolKind identifies the portable typed surface.
	ToolKind ToolPartKind
	// ID is the provider's response-part identity.
	ID string
	// ProviderName identifies the provider that executed the call.
	ProviderName string
	// ProviderDetails contains detached provider-specific call data.
	ProviderDetails map[string]any
}

func (NativeToolCallPart) responsePartKind() string { return "builtin-tool-call" }

// NativeToolReturnPart records the provider's result for a native tool call.
type NativeToolReturnPart struct {
	// ToolName is the provider-native tool name.
	ToolName string
	// Content is the normalized provider result.
	Content any
	// ToolCallID associates the result with a provider call.
	ToolCallID string
	// ToolKind identifies the portable typed surface.
	ToolKind ToolPartKind
	// Metadata contains detached application-only result data.
	Metadata map[string]any
	// Timestamp is the result creation time.
	Timestamp time.Time
	// Outcome reports success or a normalized failure.
	Outcome ToolReturnOutcome
	// ProviderName identifies the provider that executed the tool.
	ProviderName string
	// ProviderDetails contains detached provider-specific result data.
	ProviderDetails map[string]any
}

func (NativeToolReturnPart) responsePartKind() string { return "builtin-tool-return" }

// ThinkingPart is reasoning content produced by the model.
type ThinkingPart struct {
	// Content is readable provider reasoning when available.
	Content string
	// ID is the provider's response-part identity.
	ID string
	// Signature is opaque verification state required for replay.
	Signature string
	// ProviderName identifies the provider that produced the reasoning.
	ProviderName string
	// ProviderDetails contains detached provider-specific reasoning data.
	ProviderDetails map[string]any
}

func (ThinkingPart) responsePartKind() string { return "thinking" }

// StandingPromptPlantedKey marks provider details for a compaction built from
// a window that already contained the run's standing prompt.
const StandingPromptPlantedKey = "pydantic_ai_standing_prompt_planted"

// CompactionPart summarizes history that a provider compacted. ProviderDetails
// may contain opaque data required when sending the part back to that provider.
type CompactionPart struct {
	// Content is the readable compacted summary when available.
	Content string
	// ID is the provider's compaction identity.
	ID string
	// ProviderName identifies the provider that created the boundary.
	ProviderName string
	// ProviderDetails contains detached opaque state required for replay.
	ProviderDetails map[string]any
}

// HasContent reports whether the provider supplied a readable summary.
func (part CompactionPart) HasContent() bool { return part.Content != "" }

func (CompactionPart) responsePartKind() string { return "compaction" }
