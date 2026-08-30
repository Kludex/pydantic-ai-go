package ai

import (
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

// RunCancelledError reports a run cancelled through RunContext.Cancel. It
// retains the history and usage completed before cancellation took effect.
type RunCancelledError struct {
	messages []ModelMessage
	usage    Usage
}

func (e *RunCancelledError) Error() string { return ErrRunCancelled.Error() }

// Unwrap supports errors.Is(err, ErrRunCancelled).
func (e *RunCancelledError) Unwrap() error { return ErrRunCancelled }

// Messages returns a copy of the history retained at cancellation.
func (e *RunCancelledError) Messages() []ModelMessage {
	return append([]ModelMessage(nil), e.messages...)
}

// Usage returns usage accumulated before cancellation.
func (e *RunCancelledError) Usage() Usage { return e.usage }

// UnexpectedModelBehaviorError is returned when the model produces a
// response the loop cannot interpret or recover from.
type UnexpectedModelBehaviorError struct {
	Message string
}

func (e *UnexpectedModelBehaviorError) Error() string {
	return "ai: unexpected model behavior: " + e.Message
}

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
