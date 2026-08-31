package ai

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"sync"
)

// ErrConcurrencyLimitExceeded reports that a bounded concurrency queue is full.
var ErrConcurrencyLimitExceeded = errors.New("ai: concurrency limit exceeded")

// ConcurrencyLimitExceededError describes a rejected queued operation.
type ConcurrencyLimitExceededError struct {
	Name       string
	QueueDepth int
	MaxQueued  int
}

func (err *ConcurrencyLimitExceededError) Error() string {
	message := fmt.Sprintf(
		"%s: queue depth %d exceeds maximum %d", ErrConcurrencyLimitExceeded, err.QueueDepth, err.MaxQueued,
	)
	if err.Name != "" {
		message += " for " + err.Name
	}
	return message
}

// Unwrap supports errors.Is(err, ErrConcurrencyLimitExceeded).
func (*ConcurrencyLimitExceededError) Unwrap() error { return ErrConcurrencyLimitExceeded }

// ConcurrencyGate controls admission to a shared concurrency pool.
type ConcurrencyGate interface {
	Acquire(ctx context.Context, source string) error
	Release()
}

// ConcurrencyLimiterOption configures a ConcurrencyLimiter.
type ConcurrencyLimiterOption func(*ConcurrencyLimiter)

// WithMaxQueued bounds waiting operations. Zero rejects instead of waiting.
func WithMaxQueued(maxQueued int) ConcurrencyLimiterOption {
	if maxQueued < 0 {
		panic(fmt.Sprintf("ai: max queued must be non-negative, got %d", maxQueued))
	}
	return func(limiter *ConcurrencyLimiter) { limiter.maxQueued = &maxQueued }
}

// WithConcurrencyLimiterName sets the name included in queue-limit errors.
func WithConcurrencyLimiterName(name string) ConcurrencyLimiterOption {
	return func(limiter *ConcurrencyLimiter) { limiter.name = name }
}

// ConcurrencyLimiter is an in-process concurrency gate with optional queue backpressure.
type ConcurrencyLimiter struct {
	slots chan struct{}

	mu        sync.RWMutex
	waiting   int
	maxQueued *int
	name      string
}

// NewConcurrencyLimiter creates a limiter for maxRunning concurrent operations.
func NewConcurrencyLimiter(maxRunning int, options ...ConcurrencyLimiterOption) *ConcurrencyLimiter {
	if maxRunning < 1 {
		panic(fmt.Sprintf("ai: max running must be at least 1, got %d", maxRunning))
	}
	limiter := &ConcurrencyLimiter{slots: make(chan struct{}, maxRunning)}
	for _, option := range options {
		option(limiter)
	}
	return limiter
}

// Acquire waits for a slot or returns when ctx is canceled.
func (limiter *ConcurrencyLimiter) Acquire(ctx context.Context, source string) error {
	if err := context.Cause(ctx); err != nil {
		return err
	}
	select {
	case limiter.slots <- struct{}{}:
		return nil
	default:
	}
	limiter.mu.Lock()
	if limiter.maxQueued != nil && limiter.waiting >= *limiter.maxQueued {
		depth := limiter.waiting + 1
		maximum := *limiter.maxQueued
		name := limiter.name
		if name == "" {
			name = source
		}
		limiter.mu.Unlock()
		return &ConcurrencyLimitExceededError{Name: name, QueueDepth: depth, MaxQueued: maximum}
	}
	limiter.waiting++
	limiter.mu.Unlock()
	defer func() {
		limiter.mu.Lock()
		limiter.waiting--
		limiter.mu.Unlock()
	}()
	select {
	case limiter.slots <- struct{}{}:
		return nil
	case <-ctx.Done():
		return context.Cause(ctx)
	}
}

// Release returns one acquired slot. It panics when no operation owns a slot.
func (limiter *ConcurrencyLimiter) Release() {
	select {
	case <-limiter.slots:
	default:
		panic("ai: release called without an acquired concurrency slot")
	}
}

// Name returns the optional shared-pool name.
func (limiter *ConcurrencyLimiter) Name() string {
	limiter.mu.RLock()
	defer limiter.mu.RUnlock()
	return limiter.name
}

// MaxRunning returns the number of concurrent slots.
func (limiter *ConcurrencyLimiter) MaxRunning() int { return cap(limiter.slots) }

// Running returns the number of acquired slots.
func (limiter *ConcurrencyLimiter) Running() int { return len(limiter.slots) }

// Waiting returns the number of operations waiting for a slot.
func (limiter *ConcurrencyLimiter) Waiting() int {
	limiter.mu.RLock()
	defer limiter.mu.RUnlock()
	return limiter.waiting
}

// Available returns the number of immediately available slots.
func (limiter *ConcurrencyLimiter) Available() int { return cap(limiter.slots) - len(limiter.slots) }

// ConcurrencyCapability limits complete agent runs with a shared gate.
type ConcurrencyCapability struct {
	limiter ConcurrencyGate
	source  string
}

// NewConcurrencyCapability creates a run-level concurrency capability.
func NewConcurrencyCapability(limiter ConcurrencyGate, source string) *ConcurrencyCapability {
	if limiter == nil {
		panic("ai: concurrency limiter must not be nil")
	}
	return &ConcurrencyCapability{limiter: limiter, source: source}
}

// Setup implements Capability.
func (*ConcurrencyCapability) Setup(*CapabilityRegistry) error { return nil }

// WrapRun holds one concurrency slot for the full agent run.
func (capability *ConcurrencyCapability) WrapRun(
	ctx context.Context, _ *RunInfo, next RunFunc,
) (RunOutcome, error) {
	source := capability.source
	if source == "" {
		source = "agent"
	}
	if err := capability.limiter.Acquire(ctx, source); err != nil {
		return RunOutcome{}, err
	}
	defer capability.limiter.Release()
	return next(ctx)
}

// ConcurrencyLimitedModel applies a shared concurrency gate to complete model requests.
type ConcurrencyLimitedModel struct {
	*ModelWrapper
	limiter ConcurrencyGate
}

// LimitModelConcurrency wraps model with limiter. A nil limiter returns model unchanged.
func LimitModelConcurrency(model Model, limiter ConcurrencyGate) Model {
	if limiter == nil {
		return model
	}
	return NewConcurrencyLimitedModel(model, limiter)
}

// NewConcurrencyLimitedModel wraps model with a non-nil concurrency gate.
func NewConcurrencyLimitedModel(model Model, limiter ConcurrencyGate) *ConcurrencyLimitedModel {
	if limiter == nil {
		panic("ai: concurrency limiter must not be nil")
	}
	return &ConcurrencyLimitedModel{ModelWrapper: WrapModel(model), limiter: limiter}
}

// Request holds one concurrency slot for the request.
func (model *ConcurrencyLimitedModel) Request(
	ctx context.Context, messages []ModelMessage, params ModelRequestParams,
) (*ModelResponse, error) {
	if err := model.limiter.Acquire(ctx, "model:"+model.Name()); err != nil {
		return nil, err
	}
	defer model.limiter.Release()
	return model.UnwrapModel().Request(ctx, messages, params)
}

// StreamRequest holds one concurrency slot until the returned sequence ends.
func (model *ConcurrencyLimitedModel) StreamRequest(
	ctx context.Context, messages []ModelMessage, params ModelRequestParams,
) (iter.Seq2[ModelStreamEvent, error], error) {
	if err := model.limiter.Acquire(ctx, "model:"+model.Name()); err != nil {
		return nil, err
	}
	events, err := model.ModelWrapper.StreamRequest(ctx, messages, params)
	if err != nil {
		model.limiter.Release()
		return nil, err
	}
	return func(yield func(ModelStreamEvent, error) bool) {
		defer model.limiter.Release()
		for event, err := range events {
			if !yield(event, err) {
				return
			}
			if err != nil {
				return
			}
		}
	}, nil
}
