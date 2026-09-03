package ai

import "context"

// ToolExecutionFunc continues a function-tool execution wrapper.
type ToolExecutionFunc func(ctx context.Context, args any) (any, error)

// BeforeToolExecutionHook modifies validated arguments before local execution.
// Return RequestToolApproval or RequestExternalToolExecution to defer without executing.
// Independent calls invoke hooks concurrently.
type BeforeToolExecutionHook interface {
	// BeforeToolExecution returns arguments passed to the local function.
	BeforeToolExecution(ctx context.Context, ri *RunInfo, hook ToolHookContext, args any) (any, error)
}

// AfterToolExecutionHook modifies a successful function-tool return value.
// A deferred request returned here discards the result after the tool has already run.
type AfterToolExecutionHook interface {
	// AfterToolExecution returns the successful result passed onward.
	AfterToolExecution(
		ctx context.Context, ri *RunInfo, hook ToolHookContext, args any, result any,
	) (any, error)
}

// ToolExecutionErrorHook may replace an ordinary execution error with a result.
// Retry, failure, cancellation, timeout, and direct before/after hook errors bypass it.
type ToolExecutionErrorHook interface {
	// OnToolExecutionError returns a recovered result or replacement error.
	OnToolExecutionError(
		ctx context.Context, ri *RunInfo, hook ToolHookContext, args any, executionErr error,
	) (any, error)
}

// ToolExecutionWrapper wraps only local execution, after argument validation and deferral checks.
type ToolExecutionWrapper interface {
	// WrapToolExecution wraps only local function execution.
	WrapToolExecution(
		ctx context.Context, ri *RunInfo, hook ToolHookContext, args any, next ToolExecutionFunc,
	) (any, error)
}

// BeforeToolExecutionFunc adapts a function into an execution hook capability.
type BeforeToolExecutionFunc func(
	ctx context.Context, ri *RunInfo, hook ToolHookContext, args any,
) (any, error)

// Setup implements Capability.
func (BeforeToolExecutionFunc) Setup(*CapabilityRegistry) error { return nil }

// BeforeToolExecution calls the adapted function.
func (fn BeforeToolExecutionFunc) BeforeToolExecution(
	ctx context.Context, ri *RunInfo, hook ToolHookContext, args any,
) (any, error) {
	return fn(ctx, ri, hook, args)
}

// AfterToolExecutionFunc adapts a function into a post-execution hook capability.
type AfterToolExecutionFunc func(
	ctx context.Context, ri *RunInfo, hook ToolHookContext, args any, result any,
) (any, error)

// Setup implements Capability.
func (AfterToolExecutionFunc) Setup(*CapabilityRegistry) error { return nil }

// AfterToolExecution calls the adapted function.
func (fn AfterToolExecutionFunc) AfterToolExecution(
	ctx context.Context, ri *RunInfo, hook ToolHookContext, args any, result any,
) (any, error) {
	return fn(ctx, ri, hook, args, result)
}

// ToolExecutionErrorFunc adapts an execution recovery function into a capability.
type ToolExecutionErrorFunc func(
	ctx context.Context, ri *RunInfo, hook ToolHookContext, args any, executionErr error,
) (any, error)

// Setup implements Capability.
func (ToolExecutionErrorFunc) Setup(*CapabilityRegistry) error { return nil }

// OnToolExecutionError calls the adapted function.
func (fn ToolExecutionErrorFunc) OnToolExecutionError(
	ctx context.Context, ri *RunInfo, hook ToolHookContext, args any, executionErr error,
) (any, error) {
	return fn(ctx, ri, hook, args, executionErr)
}
