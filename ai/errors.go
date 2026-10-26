package ai

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// ErrUsageLimitExceeded is returned by Agent.Run when a UsageLimits bound is hit.
var ErrUsageLimitExceeded = errors.New("ai: usage limit exceeded")

// ErrMaxRetriesExceeded is returned by Agent.Run when a tool or the output
// validator keeps asking the model to retry past the configured cap.
var ErrMaxRetriesExceeded = errors.New("ai: max retries exceeded")

// ErrRunCancelled identifies cancellation requested through RunContext.Cancel,
// AgentRun.Cancel, or an expired run context deadline.
var ErrRunCancelled = errors.New("ai: run cancelled")

// ErrUnknownModelID identifies an application model ID no resolver accepted.
var ErrUnknownModelID = errors.New("ai: unknown model ID")

// ErrNoModel is returned when no default or selector supplies a model.
var ErrNoModel = errors.New("ai: no model selected")

// ErrTokenCountingUnsupported is returned when a model has no token-counting API.
var ErrTokenCountingUnsupported = errors.New("ai: token counting is not supported")

// ErrCompactionUnsupported is returned when a model has no explicit compaction API.
var ErrCompactionUnsupported = errors.New("ai: compaction is not supported")

// ErrUnpreparedSpeech reports realtime speech passed directly to a standard provider adapter.
// Use PrepareModelMessages or the agent and direct request APIs to convert it first.
var ErrUnpreparedSpeech = errors.New("ai: SpeechPart cannot be sent to a standard model as-is")

// ErrNoSuspendedResponse is returned when Resume receives history that does
// not end with a suspended model response.
var ErrNoSuspendedResponse = errors.New("ai: message history does not end with a suspended model response")

// ErrOutputTypeOverrideWithValidators is returned when RunAs or RunStreamAs
// would replace the type expected by an agent-level output validator.
var ErrOutputTypeOverrideWithValidators = errors.New(
	"ai: per-run output type cannot be used when the agent has output validators",
)

// ErrOutputTypeOverrideWithCustomOutput is returned when RunAs or RunStreamAs
// would discard an agent's custom output processing.
var ErrOutputTypeOverrideWithCustomOutput = errors.New(
	"ai: per-run output type cannot be used when the agent has custom output processing",
)

// ErrOutputTypeOverrideWithUnion is returned when RunAs or RunStreamAs would
// discard an agent's registered union-output alternatives.
var ErrOutputTypeOverrideWithUnion = errors.New(
	"ai: per-run output type cannot be used when the agent has union output",
)

// ErrOutputTypeOverrideWithImageOutput is returned when RunAs or RunStreamAs
// would discard an agent's image-output contract.
var ErrOutputTypeOverrideWithImageOutput = errors.New(
	"ai: per-run output type cannot be used when the agent has image output",
)

// ModelAPIError identifies a provider API response error suitable for model fallback.
type ModelAPIError interface {
	error
	// IsModelAPIError marks errors eligible for provider fallback.
	IsModelAPIError() bool
}

// ModelTransportError reports a provider connection or response-read failure.
type ModelTransportError struct {
	// ModelName is the model active when the operation failed.
	ModelName string
	// ProviderName is the provider identity when available.
	ProviderName string
	// Operation identifies the failed transport action.
	Operation string
	// Err is the underlying transport failure.
	Err error
}

// NewModelTransportError preserves caller cancellation and classifies other transport failures for model fallback.
func NewModelTransportError(ctx context.Context, model Model, operation string, err error) error {
	if err == nil {
		return nil
	}
	modelName, providerName := "", ""
	if !modelIsNil(model) {
		modelName = model.Name()
		if identity, ok := model.(ModelProviderIdentity); ok {
			providerName = identity.ProviderName()
		}
	}
	if ctx.Err() != nil {
		return fmt.Errorf("%s: %s: %w", transportErrorSource(modelName, providerName), transportOperation(operation), err)
	}
	return &ModelTransportError{
		ModelName: modelName, ProviderName: providerName, Operation: operation, Err: err,
	}
}

// Error describes the provider operation and underlying failure.
func (e *ModelTransportError) Error() string {
	source := transportErrorSource(e.ModelName, e.ProviderName)
	if e.Err == nil {
		return fmt.Sprintf("%s: %s failed", source, transportOperation(e.Operation))
	}
	return fmt.Sprintf("%s: %s: %v", source, transportOperation(e.Operation), e.Err)
}

// Unwrap returns the underlying transport failure.
func (e *ModelTransportError) Unwrap() error { return e.Err }

// IsModelAPIError marks the provider transport failure as eligible for default model fallback.
func (*ModelTransportError) IsModelAPIError() bool { return true }

func transportErrorSource(modelName string, providerName string) string {
	if providerName != "" {
		return providerName
	}
	if modelName != "" {
		return modelName
	}
	return "model"
}

func transportOperation(operation string) string {
	if operation == "" {
		return "request"
	}
	return operation
}

// UnknownModelIDError reports an unresolved application model ID.
type UnknownModelIDError struct {
	// ID is the unresolved application model identifier.
	ID string
}

// Error describes the unresolved identifier.
func (e *UnknownModelIDError) Error() string {
	return fmt.Sprintf("%s %q", ErrUnknownModelID, e.ID)
}

// Unwrap supports errors.Is(err, ErrUnknownModelID).
func (e *UnknownModelIDError) Unwrap() error { return ErrUnknownModelID }

// RunCancelledError reports a run cancelled through RunContext.Cancel or
// AgentRun.Cancel, or stopped by the deadline of the context passed to the
// run. It retains the history and usage completed before cancellation took
// effect, so you can resume from Messages.
//
// When the deadline stopped the run, the error also matches the context's
// cause, context.DeadlineExceeded unless the deadline was set with a custom
// cause.
type RunCancelledError struct {
	messages []ModelMessage
	usage    Usage
	metadata map[string]any
	deadline error
}

// Error identifies run cancellation.
func (e *RunCancelledError) Error() string {
	if e.deadline != nil {
		return ErrRunCancelled.Error() + ": " + e.deadline.Error()
	}
	return ErrRunCancelled.Error()
}

// Unwrap supports errors.Is(err, ErrRunCancelled) and, after a deadline,
// errors.Is(err, context.DeadlineExceeded).
func (e *RunCancelledError) Unwrap() []error {
	if e.deadline != nil {
		return []error{ErrRunCancelled, e.deadline}
	}
	return []error{ErrRunCancelled}
}

// Messages returns a copy of the history retained at cancellation.
func (e *RunCancelledError) Messages() []ModelMessage {
	return append([]ModelMessage(nil), e.messages...)
}

// Usage returns usage accumulated before cancellation.
func (e *RunCancelledError) Usage() Usage { return e.usage.Clone() }

// Metadata returns application metadata resolved before cancellation.
func (e *RunCancelledError) Metadata() map[string]any { return cloneSchemaMap(e.metadata) }

// UnexpectedModelBehaviorError is returned when the model produces a
// response the loop cannot interpret or recover from.
type UnexpectedModelBehaviorError struct {
	// Message describes the invalid or unsupported model response.
	Message string
}

// Error describes the unexpected model response.
func (e *UnexpectedModelBehaviorError) Error() string {
	return "ai: unexpected model behavior: " + e.Message
}

// ContentFilterError reports a provider content filter. It retains the full
// response so callers can inspect partial output and provider details.
type ContentFilterError struct {
	// Message describes the provider content-filter decision.
	Message  string
	response ModelResponse
	body     []byte
}

// NewContentFilterError returns an inspectable content-filter error with a detached response.
func NewContentFilterError(message string, response ModelResponse) *ContentFilterError {
	cloned := cloneModelResponse(&response)
	body, _ := MarshalMessages([]ModelMessage{*cloned})
	return &ContentFilterError{Message: message, response: *cloned, body: body}
}

// Error describes the content-filter decision.
func (e *ContentFilterError) Error() string { return "ai: " + e.Message }

// Response returns a detached filtered response.
func (e *ContentFilterError) Response() ModelResponse { return *cloneModelResponse(&e.response) }

// Body returns the filtered response serialized as an interoperable message array.
func (e *ContentFilterError) Body() []byte { return append([]byte(nil), e.body...) }

// RetryError asks the model to try again. Return one from a tool or output
// validator with Retryf; the message is sent back to the model as a retry
// prompt instead of failing the run.
type RetryError struct {
	// Message is returned to the model as corrective feedback.
	Message string
}

// Error describes the requested model retry.
func (e *RetryError) Error() string { return "ai: model retry: " + e.Message }

// Retryf returns a RetryError with a formatted message.
func Retryf(format string, args ...any) error {
	return &RetryError{Message: fmt.Sprintf(format, args...)}
}

// ToolFailedError reports a completed but unsuccessful tool call. The
// message is returned to the model without consuming the retry budget.
type ToolFailedError struct {
	// Message is returned to the model as the terminal tool result.
	Message string
}

// Error describes the failed tool call.
func (e *ToolFailedError) Error() string { return "ai: tool failed: " + e.Message }

// ToolFailedf returns a terminal tool failure with a formatted message.
func ToolFailedf(format string, args ...any) error {
	return &ToolFailedError{Message: fmt.Sprintf(format, args...)}
}

// Provider model error categories. Each is a sentinel that wraps the same
// underlying cause with a precise classification callers can match with
// errors.Is. The wrapping types below also implement ModelAPIError, so a
// FallbackModel chain configured without explicit predicates will try the
// next candidate on any of them.

var (
	// ErrModelRateLimit reports a rate-limited provider response that named no retry interval.
	ErrModelRateLimit = errors.New("ai: model rate limited")
	// ErrModelQuotaExceeded reports an exhausted billing or quota block the provider announced explicitly.
	ErrModelQuotaExceeded = errors.New("ai: model quota exceeded")
	// ErrModelUnavailable reports the provider refusing requests without a recoverable cause.
	ErrModelUnavailable = errors.New("ai: model unavailable")
	// ErrModelOverloaded reports an explicit overload response from the provider (HTTP 529).
	ErrModelOverloaded = errors.New("ai: model overloaded")
	// ErrModelServerError reports an unrecoverable provider 5xx response.
	ErrModelServerError = errors.New("ai: model server error")
	// ErrModelConnection reports a transport failure on the way to the provider.
	ErrModelConnection = errors.New("ai: model connection error")
	// ErrModelTimeout reports a client-side timeout reaching the provider.
	ErrModelTimeout = errors.New("ai: model timeout")
	// ErrModelContextWindowExceeded reports that the configured inputs exceeded the model's context window.
	ErrModelContextWindowExceeded = errors.New("ai: model context window exceeded")
)

// ModelRateLimitError reports a rate-limited response with the optional retry
// hint the provider advertised.
type ModelRateLimitError struct {
	// ModelName is the model active when the limit was hit.
	ModelName string
	// ProviderName is the provider identity when available.
	ProviderName string
	// RetryAfter is the wait the provider asked for, or zero when none was reported.
	RetryAfter time.Duration
	// Cause is the underlying provider error.
	Cause error
}

// Error describes the rate limit and retry interval when known.
func (e *ModelRateLimitError) Error() string {
	if e.RetryAfter > 0 {
		return fmt.Sprintf("%s: retry after %s", ErrModelRateLimit, e.RetryAfter)
	}
	return ErrModelRateLimit.Error()
}

// IsModelAPIError marks the rate limit as eligible for default model fallback.
func (*ModelRateLimitError) IsModelAPIError() bool { return true }

// Unwrap exposes the underlying cause to errors.Is and errors.As.
func (e *ModelRateLimitError) Unwrap() error { return e.Cause }

// Is matches the ModelRateLimit sentinel and any wrapped cause.
func (e *ModelRateLimitError) Is(target error) bool {
	return target == ErrModelRateLimit
}

// ModelContextWindowExceededError reports inputs that the model's profile
// classified as larger than its context window. Token counts are reported
// when the provider supplied them.
type ModelContextWindowExceededError struct {
	// ModelName is the model active when the limit was hit.
	ModelName string
	// ProviderName is the provider identity when available.
	ProviderName string
	// MaxTokens is the model's known context window when the provider reported it.
	MaxTokens int
	// Tokens is the attempted input size when the provider reported it.
	Tokens int
	// Cause is the underlying provider error.
	Cause error
}

// Error describes the exceeded window.
func (e *ModelContextWindowExceededError) Error() string {
	if e.MaxTokens > 0 && e.Tokens > 0 {
		return fmt.Sprintf("%s: %d tokens > %d window", ErrModelContextWindowExceeded, e.Tokens, e.MaxTokens)
	}
	return ErrModelContextWindowExceeded.Error()
}

// IsModelAPIError marks the overflow as eligible for default model fallback.
func (*ModelContextWindowExceededError) IsModelAPIError() bool { return true }

// Unwrap exposes the underlying cause to errors.Is and errors.As.
func (e *ModelContextWindowExceededError) Unwrap() error { return e.Cause }

// Is matches the ModelContextWindowExceeded sentinel and any wrapped cause.
func (e *ModelContextWindowExceededError) Is(target error) bool {
	return target == ErrModelContextWindowExceeded
}

// ConnectionErrorPhase says where a transport failure occurred.
type ConnectionErrorPhase string

const (
	// ConnectionPhasePool reports failing to obtain a connection from the pool.
	ConnectionPhasePool ConnectionErrorPhase = "pool"
	// ConnectionPhaseConnect reports failing to open a TCP/TLS connection.
	ConnectionPhaseConnect ConnectionErrorPhase = "connect"
	// ConnectionPhaseWrite reports failing to send the request.
	ConnectionPhaseWrite ConnectionErrorPhase = "write"
	// ConnectionPhaseRead reports failing to receive the response.
	ConnectionPhaseRead ConnectionErrorPhase = "read"
)

// ModelConnectionError reports a transport failure classified by phase.
type ModelConnectionError struct {
	// ModelName is the model active when the failure occurred.
	ModelName string
	// ProviderName is the provider identity when available.
	ProviderName string
	// Phase identifies where the failure happened.
	Phase ConnectionErrorPhase
	// InStream reports whether the failure occurred after the stream opened.
	InStream bool
	// Cause is the underlying transport error.
	Cause error
}

// Error describes the failed phase.
func (e *ModelConnectionError) Error() string {
	phase := string(e.Phase)
	if phase == "" {
		phase = "transport"
	}
	source := transportErrorSource(e.ModelName, e.ProviderName)
	if e.Cause != nil {
		return fmt.Sprintf("%s: %s: %s: %v", source, ErrModelConnection, phase, e.Cause)
	}
	return fmt.Sprintf("%s: %s: %s", source, ErrModelConnection, phase)
}

// IsModelAPIError marks the connection failure as eligible for default model fallback.
func (*ModelConnectionError) IsModelAPIError() bool { return true }

// Unwrap exposes the underlying cause to errors.Is and errors.As.
func (e *ModelConnectionError) Unwrap() error { return e.Cause }

// Is matches the ModelConnection sentinel and any wrapped cause.
func (e *ModelConnectionError) Is(target error) bool {
	return target == ErrModelConnection
}

// UsageLimitUnavailableWarning is a sentinel callers can match with errors.Is
// when a configured limit could not be evaluated because the model reported
// no usage. Combine with errors.As on ErrUsageLimitExceeded to decide whether
// to treat the warning as terminal.
var UsageLimitUnavailableWarning = errors.New("ai: usage limit unavailable")
