package ai

import (
	"context"
	"errors"
	"fmt"
)

// ErrUsageLimitExceeded is returned by Agent.Run when a UsageLimits bound is hit.
var ErrUsageLimitExceeded = errors.New("ai: usage limit exceeded")

// ErrMaxRetriesExceeded is returned by Agent.Run when a tool or the output
// validator keeps asking the model to retry past the configured cap.
var ErrMaxRetriesExceeded = errors.New("ai: max retries exceeded")

// ErrRunCancelled identifies cancellation requested through RunContext.Cancel.
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

// ModelAPIError identifies a provider API response error suitable for model fallback.
type ModelAPIError interface {
	error
	IsModelAPIError() bool
}

// ModelTransportError reports a provider connection or response-read failure.
type ModelTransportError struct {
	ModelName    string
	ProviderName string
	Operation    string
	Err          error
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
	ID string
}

func (e *UnknownModelIDError) Error() string {
	return fmt.Sprintf("%s %q", ErrUnknownModelID, e.ID)
}

// Unwrap supports errors.Is(err, ErrUnknownModelID).
func (e *UnknownModelIDError) Unwrap() error { return ErrUnknownModelID }

// RunCancelledError reports a run cancelled through RunContext.Cancel. It
// retains the history and usage completed before cancellation took effect.
type RunCancelledError struct {
	messages []ModelMessage
	usage    Usage
	metadata map[string]any
}

func (e *RunCancelledError) Error() string { return ErrRunCancelled.Error() }

// Unwrap supports errors.Is(err, ErrRunCancelled).
func (e *RunCancelledError) Unwrap() error { return ErrRunCancelled }

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
	Message string
}

func (e *UnexpectedModelBehaviorError) Error() string {
	return "ai: unexpected model behavior: " + e.Message
}

// ContentFilterError reports a provider content filter. It retains the full
// response so callers can inspect partial output and provider details.
type ContentFilterError struct {
	Message  string
	response ModelResponse
	body     []byte
}

func (e *ContentFilterError) Error() string { return "ai: " + e.Message }

// Response returns a detached filtered response.
func (e *ContentFilterError) Response() ModelResponse { return *cloneModelResponse(&e.response) }

// Body returns the filtered response serialized as an interoperable message array.
func (e *ContentFilterError) Body() []byte { return append([]byte(nil), e.body...) }

// RetryError asks the model to try again. Return one from a tool or output
// validator with Retryf; the message is sent back to the model as a retry
// prompt instead of failing the run.
type RetryError struct {
	Message string
}

func (e *RetryError) Error() string { return "ai: model retry: " + e.Message }

// Retryf returns a RetryError with a formatted message.
func Retryf(format string, args ...any) error {
	return &RetryError{Message: fmt.Sprintf(format, args...)}
}

// ToolFailedError reports a completed but unsuccessful tool call. The
// message is returned to the model without consuming the retry budget.
type ToolFailedError struct {
	Message string
}

func (e *ToolFailedError) Error() string { return "ai: tool failed: " + e.Message }

// ToolFailedf returns a terminal tool failure with a formatted message.
func ToolFailedf(format string, args ...any) error {
	return &ToolFailedError{Message: fmt.Sprintf(format, args...)}
}
