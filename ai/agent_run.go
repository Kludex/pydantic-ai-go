package ai

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
)

type agentRunEvent struct {
	event StreamEvent
	err   error
}

// AgentRun is a manually driven agent run. Next advances execution to the
// next normalized event. Call Close when abandoning a run before completion.
type AgentRun[Deps, Output any] struct {
	run      *run[Deps, Output]
	worker   func()
	events   chan agentRunEvent
	advance  chan struct{}
	stop     chan struct{}
	done     chan struct{}
	start    sync.Once
	stopOnce sync.Once
	nextMu   sync.Mutex

	stateMu        sync.RWMutex
	ended          bool
	pendingAdvance bool
	result         *RunResult[Output]
	terminalErr    error
	closeErr       error
	usage          Usage
}

// StartRun prepares a manually driven run with a text prompt.
func (a *Agent[Deps, Output]) StartRun(
	ctx context.Context, prompt string, deps Deps, opts ...RunOption,
) (*AgentRun[Deps, Output], error) {
	return a.iterPrompt(ctx, UserPromptPart{Content: prompt}, deps, opts)
}

// StartRunParts prepares a manually driven run with multimodal user content.
func (a *Agent[Deps, Output]) StartRunParts(
	ctx context.Context, contents []UserContent, deps Deps, opts ...RunOption,
) (*AgentRun[Deps, Output], error) {
	return a.iterPrompt(ctx, UserPromptPart{Contents: contents}, deps, opts)
}

// ResumeRun prepares a manually driven continuation of a suspended response.
func (a *Agent[Deps, Output]) ResumeRun(
	ctx context.Context, history []ModelMessage, deps Deps, opts ...RunOption,
) (*AgentRun[Deps, Output], error) {
	return a.iterPrompt(ctx, UserPromptPart{}, deps, suspendedRunOptions(history, opts))
}

func (a *Agent[Deps, Output]) iterPrompt(
	ctx context.Context, prompt UserPromptPart, deps Deps, opts []RunOption,
) (*AgentRun[Deps, Output], error) {
	cfg := buildRunConfig(opts)
	model := a.model
	if cfg.model != nil {
		model = cfg.model
	}
	capabilities := append(slices.Clone(a.capabilities), cfg.capabilities...)
	ctx, span := startRunSpan(ctx, modelName(model), !hasInstrumentationCapability(capabilities))
	r, err := a.newRun(ctx, prompt, deps, cfg)
	if err != nil {
		endSpan(span, err)
		return nil, err
	}
	driver := &AgentRun[Deps, Output]{
		run: r, events: make(chan agentRunEvent), advance: make(chan struct{}),
		stop: make(chan struct{}), done: make(chan struct{}),
	}
	r.recordSelectedModel = func(name string) { recordRunModel(span, name) }
	r.observeUsage = driver.setUsage
	r.publishUsage(nil)
	driver.worker = func() {
		var result *RunResult[Output]
		var runErr error
		stopped := false
		core := EventStream(func(yield func(StreamEvent, error) bool) {
			if !r.setEventEmitter(func(event StreamEvent) bool {
				if !yield(event, nil) {
					stopped = true
					r.cancellation.stopStream()
					return false
				}
				return true
			}) {
				runErr = r.streamEventError()
				if runErr != nil {
					yield(nil, runErr)
				}
				return
			}
			result, runErr = r.wrappedLoop(r.ctx)
			if !stopped && runErr != nil {
				yield(nil, runErr)
			}
		})
		stream := wrapEventStream(r.ctx, r.info, core, r.capabilities, r.capabilityIDs)
		for event, eventErr := range stream {
			if eventErr != nil {
				runErr = eventErr
			}
			if !driver.send(agentRunEvent{event: event, err: eventErr}) {
				break
			}
		}
		closeErr := r.closeRunResources(context.WithoutCancel(ctx))
		r.cancellation.finish()
		if closeErr != nil {
			if runErr == nil {
				_ = driver.send(agentRunEvent{err: closeErr})
			}
			runErr = errors.Join(runErr, closeErr)
			result = nil
		}
		if runErr != nil {
			result = nil
		}
		if result != nil {
			recordUsage(span, result.usage)
		}
		endSpan(span, runErr)
		driver.finish(result, runErr, closeErr)
	}
	return driver, nil
}

// Next advances the run to its next normalized event. ok is false after the
// run completes. An execution error is returned once with ok false.
func (r *AgentRun[Deps, Output]) Next() (event StreamEvent, ok bool, err error) {
	r.nextMu.Lock()
	defer r.nextMu.Unlock()
	r.startWorker()

	r.stateMu.Lock()
	pendingAdvance := r.pendingAdvance
	r.pendingAdvance = false
	r.stateMu.Unlock()
	if pendingAdvance {
		select {
		case r.advance <- struct{}{}:
		case <-r.done:
		}
	}
	item, open := <-r.events
	if !open {
		return nil, false, nil
	}
	if item.err != nil {
		<-r.done
		return nil, false, item.err
	}
	r.stateMu.Lock()
	r.pendingAdvance = true
	r.stateMu.Unlock()
	return item.event, true, nil
}

// Events advances the run event by event. Consume either Events or Next, not
// both concurrently.
func (r *AgentRun[Deps, Output]) Events() EventStream {
	return func(yield func(StreamEvent, error) bool) {
		for {
			event, ok, err := r.Next()
			if err != nil {
				yield(nil, err)
				return
			}
			if !ok {
				return
			}
			if !yield(event, nil) {
				_ = r.Close()
				return
			}
		}
	}
}

// Enqueue adds content for delivery at the next model-request boundary.
func (r *AgentRun[Deps, Output]) Enqueue(items ...EnqueueItem) (string, error) {
	return r.EnqueueWithPriority(PendingMessageASAP, items...)
}

// EnqueueWhenIdle adds content only when the run would otherwise finish.
func (r *AgentRun[Deps, Output]) EnqueueWhenIdle(items ...EnqueueItem) (string, error) {
	return r.EnqueueWithPriority(PendingMessageWhenIdle, items...)
}

// EnqueueWithPriority adds content from outside tools and capability hooks.
func (r *AgentRun[Deps, Output]) EnqueueWithPriority(
	priority PendingMessagePriority, items ...EnqueueItem,
) (string, error) {
	r.stateMu.Lock()
	defer r.stateMu.Unlock()
	if r.ended {
		return "", fmt.Errorf("ai: agent run has ended")
	}
	return enqueuePendingMessage(r.run.pendingMessages, priority, items)
}

// Emit adds a custom event before the run's next generated event. It may be
// called before the first call to Next.
func (r *AgentRun[Deps, Output]) Emit(event StreamEvent) error {
	r.stateMu.RLock()
	ended := r.ended
	r.stateMu.RUnlock()
	if ended {
		return fmt.Errorf("ai: agent run has ended")
	}
	return r.run.emitEvent(event, "", "", "")
}

// Cancel requests terminal run cancellation. The next progression observes
// ErrRunCancelled after in-flight work has drained.
func (r *AgentRun[Deps, Output]) Cancel() { r.run.cancellation.cancelRun() }

// Metadata returns detached application metadata resolved for the run.
func (r *AgentRun[Deps, Output]) Metadata() map[string]any {
	if r.run == nil || r.run.info == nil {
		return nil
	}
	return r.run.info.Metadata()
}

// Usage returns a detached snapshot accumulated through the current event.
func (r *AgentRun[Deps, Output]) Usage() Usage {
	r.stateMu.RLock()
	defer r.stateMu.RUnlock()
	return r.usage.Clone()
}

// AgentName returns the configured application agent name.
func (r *AgentRun[Deps, Output]) AgentName() string { return r.run.info.AgentName() }

// AgentDescription returns the description resolved for this run.
func (r *AgentRun[Deps, Output]) AgentDescription() string { return r.run.info.AgentDescription() }

// RunID returns the immutable run identifier.
func (r *AgentRun[Deps, Output]) RunID() string { return r.run.rc.RunID }

// ConversationID returns the immutable conversation identifier.
func (r *AgentRun[Deps, Output]) ConversationID() string { return r.run.rc.ConversationID }

// Result returns the completed result, or nil before successful completion.
func (r *AgentRun[Deps, Output]) Result() *RunResult[Output] {
	r.stateMu.RLock()
	defer r.stateMu.RUnlock()
	return r.result
}

// Err returns the terminal execution or cleanup error after the run ends.
func (r *AgentRun[Deps, Output]) Err() error {
	r.stateMu.RLock()
	defer r.stateMu.RUnlock()
	return r.terminalErr
}

// Close cancels unfinished work, drains it, and releases model and toolset
// resources. It returns cleanup errors but not the expected cancellation.
func (r *AgentRun[Deps, Output]) Close() error {
	r.startWorker()
	r.stopOnce.Do(func() {
		close(r.stop)
		r.run.cancellation.cancelRun()
	})
	<-r.done
	r.stateMu.RLock()
	defer r.stateMu.RUnlock()
	return r.closeErr
}

func (r *AgentRun[Deps, Output]) startWorker() {
	r.start.Do(func() { go r.worker() })
}

func (r *AgentRun[Deps, Output]) send(item agentRunEvent) bool {
	select {
	case r.events <- item:
	case <-r.stop:
		return false
	}
	if item.err != nil {
		return true
	}
	select {
	case <-r.advance:
		return true
	case <-r.stop:
		return false
	}
}

func (r *AgentRun[Deps, Output]) setUsage(usage Usage) {
	r.stateMu.Lock()
	defer r.stateMu.Unlock()
	r.usage = usage.Clone()
}

func (r *AgentRun[Deps, Output]) finish(result *RunResult[Output], terminalErr, closeErr error) {
	r.stateMu.Lock()
	r.ended = true
	r.pendingAdvance = false
	r.result = result
	r.terminalErr = terminalErr
	r.closeErr = closeErr
	r.stateMu.Unlock()
	close(r.events)
	close(r.done)
}
