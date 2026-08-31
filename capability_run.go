package ai

import "context"

// RunOutcome is the untyped result passed through run lifecycle middleware.
// Deferred takes precedence over Output when it is non-nil.
type RunOutcome struct {
	Output   any
	Deferred *DeferredToolRequests
}

// CompletedRunOutcome creates a completed outcome for wrappers and recovery hooks.
func CompletedRunOutcome(output any) RunOutcome {
	return RunOutcome{Output: output}
}

// PendingRunOutcome creates an outcome that pauses for deferred tool results.
func PendingRunOutcome(requests DeferredToolRequests) RunOutcome {
	cloned := requests.Clone()
	return RunOutcome{Deferred: &cloned}
}

// Clone detaches deferred requests. Output remains the caller-owned semantic value.
func (outcome RunOutcome) Clone() RunOutcome {
	if outcome.Deferred != nil {
		cloned := outcome.Deferred.Clone()
		outcome.Deferred = &cloned
	}
	return outcome
}

// RunFunc continues the run lifecycle chain.
type RunFunc func(ctx context.Context) (RunOutcome, error)

// RunWrapper wraps the entire run. A wrapper may transform or recover a result,
// or skip next and return a short-circuit outcome. Cancellation remains terminal.
type RunWrapper interface {
	WrapRun(ctx context.Context, ri *RunInfo, next RunFunc) (RunOutcome, error)
}

// BeforeRunHook observes a run after its enclosing wrappers have entered and before the loop starts.
type BeforeRunHook interface {
	BeforeRun(ctx context.Context, ri *RunInfo) error
}

// AfterRunHook transforms a successful, recovered, or short-circuit outcome.
type AfterRunHook interface {
	AfterRun(ctx context.Context, ri *RunInfo, outcome RunOutcome) (RunOutcome, error)
}

// RunErrorHook may replace an error left by run wrappers with an outcome.
type RunErrorHook interface {
	OnRunError(ctx context.Context, ri *RunInfo, runErr error) (RunOutcome, error)
}

// BeforeRunFunc adapts a function into a before-run capability.
type BeforeRunFunc func(ctx context.Context, ri *RunInfo) error

// Setup implements Capability.
func (BeforeRunFunc) Setup(*CapabilityRegistry) error { return nil }

// BeforeRun calls the adapted function.
func (fn BeforeRunFunc) BeforeRun(ctx context.Context, ri *RunInfo) error {
	return fn(ctx, ri)
}

// AfterRunFunc adapts a function into an after-run capability.
type AfterRunFunc func(ctx context.Context, ri *RunInfo, outcome RunOutcome) (RunOutcome, error)

// Setup implements Capability.
func (AfterRunFunc) Setup(*CapabilityRegistry) error { return nil }

// AfterRun calls the adapted function.
func (fn AfterRunFunc) AfterRun(
	ctx context.Context, ri *RunInfo, outcome RunOutcome,
) (RunOutcome, error) {
	return fn(ctx, ri, outcome)
}

// RunErrorFunc adapts a run recovery function into a capability.
type RunErrorFunc func(ctx context.Context, ri *RunInfo, runErr error) (RunOutcome, error)

// Setup implements Capability.
func (RunErrorFunc) Setup(*CapabilityRegistry) error { return nil }

// OnRunError calls the adapted function.
func (fn RunErrorFunc) OnRunError(
	ctx context.Context, ri *RunInfo, runErr error,
) (RunOutcome, error) {
	return fn(ctx, ri, runErr)
}
