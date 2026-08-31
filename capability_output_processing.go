package ai

import "context"

// OutputProcessingFunc continues an output processing wrapper.
type OutputProcessingFunc func(ctx context.Context, output any) (any, error)

// BeforeOutputProcessingHook modifies validated output before final extraction.
type BeforeOutputProcessingHook interface {
	BeforeOutputProcessing(ctx context.Context, ri *RunInfo, hook OutputHookContext, output any) (any, error)
}

// AfterOutputProcessingHook modifies the final output value.
type AfterOutputProcessingHook interface {
	AfterOutputProcessing(ctx context.Context, ri *RunInfo, hook OutputHookContext, output any) (any, error)
}

// OutputProcessingErrorHook may replace an ordinary output processing error.
type OutputProcessingErrorHook interface {
	OnOutputProcessingError(
		ctx context.Context, ri *RunInfo, hook OutputHookContext, output any, processingErr error,
	) (any, error)
}

// OutputProcessingWrapper wraps semantic validators and final output extraction.
type OutputProcessingWrapper interface {
	WrapOutputProcessing(
		ctx context.Context, ri *RunInfo, hook OutputHookContext, output any, next OutputProcessingFunc,
	) (any, error)
}

// BeforeOutputProcessingFunc adapts a function into an output processing capability.
type BeforeOutputProcessingFunc func(
	ctx context.Context, ri *RunInfo, hook OutputHookContext, output any,
) (any, error)

// Setup implements Capability.
func (BeforeOutputProcessingFunc) Setup(*CapabilityRegistry) error { return nil }

// BeforeOutputProcessing calls the adapted function.
func (fn BeforeOutputProcessingFunc) BeforeOutputProcessing(
	ctx context.Context, ri *RunInfo, hook OutputHookContext, output any,
) (any, error) {
	return fn(ctx, ri, hook, output)
}

// AfterOutputProcessingFunc adapts a function into a post-processing capability.
type AfterOutputProcessingFunc func(
	ctx context.Context, ri *RunInfo, hook OutputHookContext, output any,
) (any, error)

// Setup implements Capability.
func (AfterOutputProcessingFunc) Setup(*CapabilityRegistry) error { return nil }

// AfterOutputProcessing calls the adapted function.
func (fn AfterOutputProcessingFunc) AfterOutputProcessing(
	ctx context.Context, ri *RunInfo, hook OutputHookContext, output any,
) (any, error) {
	return fn(ctx, ri, hook, output)
}

// OutputProcessingErrorFunc adapts an output processing recovery function into a capability.
type OutputProcessingErrorFunc func(
	ctx context.Context, ri *RunInfo, hook OutputHookContext, output any, processingErr error,
) (any, error)

// Setup implements Capability.
func (OutputProcessingErrorFunc) Setup(*CapabilityRegistry) error { return nil }

// OnOutputProcessingError calls the adapted function.
func (fn OutputProcessingErrorFunc) OnOutputProcessingError(
	ctx context.Context, ri *RunInfo, hook OutputHookContext, output any, processingErr error,
) (any, error) {
	return fn(ctx, ri, hook, output, processingErr)
}
