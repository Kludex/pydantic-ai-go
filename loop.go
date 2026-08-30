package ai

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/Kludex/pydantic-ai-go/internal/schema"
)

const outputToolName = "final_result"

type toolFunc[Deps any] func(ctx context.Context, rc *RunContext[Deps], rawArgs json.RawMessage) (any, error)

// Run executes the agent loop: send the conversation to the model, execute
// any tool calls, repeat until the model produces a final output.
func (a *Agent[Deps, Output]) Run(ctx context.Context, prompt string, deps Deps, opts ...RunOption) (*RunResult[Output], error) {
	return a.runPrompt(ctx, UserPromptPart{Content: prompt}, deps, opts)
}

// RunParts is Run with a multimodal prompt: text, image URLs, and inline
// binary data. Providers reject content kinds they do not support.
func (a *Agent[Deps, Output]) RunParts(ctx context.Context, contents []UserContent, deps Deps, opts ...RunOption) (*RunResult[Output], error) {
	return a.runPrompt(ctx, UserPromptPart{Contents: contents}, deps, opts)
}

func (a *Agent[Deps, Output]) runPrompt(ctx context.Context, prompt UserPromptPart, deps Deps, opts []RunOption) (result *RunResult[Output], err error) {
	ctx, span := startRunSpan(ctx, a.model.Name())
	defer func() {
		if result != nil {
			recordUsage(span, result.usage)
		}
		endSpan(span, err)
	}()
	a.started.Store(true)
	r, err := a.newRun(ctx, prompt, deps, opts)
	if err != nil {
		return nil, err
	}
	return r.wrappedLoop(ctx)
}

func (a *Agent[Deps, Output]) newRun(ctx context.Context, prompt UserPromptPart, deps Deps, opts []RunOption) (*run[Deps, Output], error) {
	a.started.Store(true)
	var cfg runConfig
	for _, opt := range opts {
		opt(&cfg)
	}
	r := &run[Deps, Output]{agent: a}
	r.messages = append(r.messages, cfg.history...)
	r.newMessages = len(r.messages)
	r.rc = &RunContext[Deps]{Deps: deps, RunID: newRunID(), usage: &r.usage, messages: &r.messages}
	r.info = &RunInfo{RunID: r.rc.RunID, usage: &r.usage, messages: &r.messages}
	instructions, err := a.buildInstructions(ctx, r.rc, r.info)
	if err != nil {
		return nil, err
	}
	r.params, err = a.buildParams(instructions)
	if err != nil {
		return nil, err
	}
	r.messages = append(r.messages, ModelRequest{Parts: []RequestPart{prompt}})
	return r, nil
}

type run[Deps, Output any] struct {
	agent       *Agent[Deps, Output]
	rc          *RunContext[Deps]
	info        *RunInfo
	params      ModelRequestParams
	messages    []ModelMessage
	newMessages int
	usage       Usage
	retries     int
	retriesMu   sync.Mutex
	// emit forwards stream events during RunStream; nil for plain runs.
	emit func(StreamEvent) bool
}

// modelRequest is the model-request interception point: tracing plus
// capability middleware (ModelRequestWrapper), outermost first.
func (r *run[Deps, Output]) modelRequest(ctx context.Context) (*ModelResponse, error) {
	a := r.agent
	inner := func(ctx context.Context, msgs []ModelMessage, params ModelRequestParams) (*ModelResponse, error) {
		reqCtx, reqSpan := startRequestSpan(ctx, a.model.Name())
		resp, err := r.doModelRequest(reqCtx, msgs, params)
		if err != nil {
			endSpan(reqSpan, err)
			return nil, err
		}
		recordUsage(reqSpan, resp.Usage)
		endSpan(reqSpan, nil)
		if resp.Timestamp.IsZero() {
			resp.Timestamp = time.Now().UTC()
		}
		if resp.ModelName == "" {
			resp.ModelName = a.model.Name()
		}
		return resp, nil
	}
	next := inner
	for i := len(a.capabilities) - 1; i >= 0; i-- {
		if wrapper, ok := a.capabilities[i].(ModelRequestWrapper); ok {
			innerNext := next
			next = func(ctx context.Context, msgs []ModelMessage, params ModelRequestParams) (*ModelResponse, error) {
				return wrapper.WrapModelRequest(ctx, r.info, msgs, params, innerNext)
			}
		}
	}
	return next(ctx, r.messages, r.params)
}

// doModelRequest streams when the run has an emit callback and the model
// supports it; otherwise it falls back to a plain request, replaying the
// response as events so RunStream works with every Model.
func (r *run[Deps, Output]) doModelRequest(ctx context.Context, msgs []ModelMessage, params ModelRequestParams) (*ModelResponse, error) {
	if r.emit == nil {
		return r.agent.model.Request(ctx, msgs, params)
	}
	if sm, ok := r.agent.model.(StreamingModel); ok {
		events, err := sm.StreamRequest(ctx, msgs, params)
		if err != nil {
			return nil, err
		}
		return accumulate(events, r.emit)
	}
	resp, err := r.agent.model.Request(ctx, msgs, params)
	if err != nil {
		return nil, err
	}
	return accumulate(replayAsEvents(resp), r.emit)
}

func (r *run[Deps, Output]) loop(ctx context.Context) (*RunResult[Output], error) {
	for {
		resp, err := r.modelRequest(ctx)
		if err != nil {
			return nil, err
		}
		r.usage.Add(resp.Usage)
		r.messages = append(r.messages, *resp)

		calls := resp.ToolCalls()
		if len(calls) == 0 {
			result, retry, err := r.finalizeText(ctx, resp)
			if err != nil {
				return nil, err
			}
			if retry != nil {
				r.recordRetry(*retry)
				continue
			}
			return result, nil
		}

		parts, final, err := r.executeCalls(ctx, calls)
		if err != nil {
			return nil, err
		}
		if final != nil {
			return final, nil
		}
		r.messages = append(r.messages, ModelRequest{Parts: parts})
	}
}

type callOutcome[Output any] struct {
	part   RequestPart
	result *RunResult[Output]
	err    error
}

// executeCalls runs independent tool calls concurrently and preserves their
// original order in the request sent back to the model. Sequential tools and
// output tools form barriers.
func (r *run[Deps, Output]) executeCalls(
	ctx context.Context, calls []ToolCallPart,
) ([]RequestPart, *RunResult[Output], error) {
	outcomes := make([]callOutcome[Output], len(calls))
	batchStart := 0
	for i, call := range calls {
		if !r.callIsSequential(call) {
			continue
		}
		r.executeCallBatch(ctx, calls, outcomes, batchStart, i)
		if err := firstCallError(outcomes[batchStart:i]); err != nil {
			return nil, nil, err
		}
		part, result, err := r.executeCall(ctx, call)
		outcomes[i] = callOutcome[Output]{part: part, result: result, err: err}
		if err != nil {
			return nil, nil, err
		}
		batchStart = i + 1
	}
	r.executeCallBatch(ctx, calls, outcomes, batchStart, len(calls))
	if err := firstCallError(outcomes[batchStart:]); err != nil {
		return nil, nil, err
	}

	parts := make([]RequestPart, 0, len(calls))
	var final *RunResult[Output]
	for _, outcome := range outcomes {
		if outcome.part != nil {
			parts = append(parts, outcome.part)
		}
		if final == nil && outcome.result != nil {
			final = outcome.result
		}
	}
	return parts, final, nil
}

func (r *run[Deps, Output]) executeCallBatch(
	ctx context.Context, calls []ToolCallPart, outcomes []callOutcome[Output], start, end int,
) {
	if start >= end {
		return
	}
	if end-start == 1 {
		part, result, err := r.executeCall(ctx, calls[start])
		outcomes[start] = callOutcome[Output]{part: part, result: result, err: err}
		return
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var wg sync.WaitGroup
	for i := start; i < end; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			part, result, err := r.executeCall(ctx, calls[i])
			outcomes[i] = callOutcome[Output]{part: part, result: result, err: err}
			if err != nil {
				cancel()
			}
		}()
	}
	wg.Wait()
}

func (r *run[Deps, Output]) callIsSequential(call ToolCallPart) bool {
	if r.params.OutputTool != nil && call.ToolName == outputToolName {
		return true
	}
	entry, ok := r.agent.findTool(call.ToolName)
	return ok && entry.def.Sequential
}

func firstCallError[Output any](outcomes []callOutcome[Output]) error {
	for _, outcome := range outcomes {
		if outcome.err != nil {
			return outcome.err
		}
	}
	return nil
}

// executeCall runs one tool call. It returns the request part to send back
// to the model, or the final result if the call was the output tool.
func (r *run[Deps, Output]) executeCall(ctx context.Context, call ToolCallPart) (RequestPart, *RunResult[Output], error) {
	if r.params.OutputTool != nil && call.ToolName == outputToolName {
		return r.finalizeOutputCall(ctx, call)
	}
	entry, ok := r.agent.findTool(call.ToolName)
	if !ok {
		return nil, nil, &UnexpectedModelBehaviorError{Message: fmt.Sprintf("model called unknown tool %q", call.ToolName)}
	}
	toolRC := *r.rc
	toolRC.ToolCallID = call.ToolCallID
	toolRC.Retry = r.retryCount()
	toolCtx, toolSpan := startToolSpan(ctx, call.ToolName, call.ToolCallID)
	content, err := r.callTool(toolCtx, &toolRC, entry, call)
	endSpan(toolSpan, err)
	var retry *RetryError
	switch {
	case errors.As(err, &retry):
		if err := r.countRetry(); err != nil {
			return nil, nil, err
		}
		return RetryPromptPart{Content: retry.Message, ToolName: call.ToolName, ToolCallID: call.ToolCallID}, nil, nil
	case err != nil:
		return nil, nil, fmt.Errorf("ai: tool %q: %w", call.ToolName, err)
	}
	return ToolReturnPart{ToolName: call.ToolName, Content: content, ToolCallID: call.ToolCallID}, nil, nil
}

func (r *run[Deps, Output]) finalizeOutputCall(ctx context.Context, call ToolCallPart) (RequestPart, *RunResult[Output], error) {
	var out Output
	if err := json.Unmarshal(call.Args, &out); err != nil {
		if err := r.countRetry(); err != nil {
			return nil, nil, err
		}
		msg := fmt.Sprintf("invalid final result: %v", err)
		return RetryPromptPart{Content: msg, ToolName: call.ToolName, ToolCallID: call.ToolCallID}, nil, nil
	}
	if retry, err := r.validate(ctx, out); err != nil {
		return nil, nil, err
	} else if retry != nil {
		return RetryPromptPart{Content: retry.Message, ToolName: call.ToolName, ToolCallID: call.ToolCallID}, nil, nil
	}
	r.messages = append(r.messages, ModelRequest{Parts: []RequestPart{
		ToolReturnPart{ToolName: call.ToolName, Content: "Final result processed.", ToolCallID: call.ToolCallID},
	}})
	return nil, r.result(out), nil
}

// finalizeText handles a response with no tool calls. String outputs take
// the text as-is; native-mode structured outputs unmarshal it; tool-mode
// structured outputs must call the output tool, so text triggers a retry.
func (r *run[Deps, Output]) finalizeText(ctx context.Context, resp *ModelResponse) (*RunResult[Output], *RetryPromptPart, error) {
	if !r.params.AllowText {
		if err := r.countRetry(); err != nil {
			return nil, nil, err
		}
		return nil, &RetryPromptPart{Content: fmt.Sprintf("Respond by calling the %s tool to provide the final result.", outputToolName)}, nil
	}
	var out Output
	if r.params.OutputSchema != nil {
		if err := json.Unmarshal([]byte(resp.Text()), &out); err != nil {
			if err := r.countRetry(); err != nil {
				return nil, nil, err
			}
			return nil, &RetryPromptPart{Content: fmt.Sprintf("invalid JSON output: %v", err)}, nil
		}
	} else {
		// buildParams sets AllowText without a schema only when Output is
		// string, so this assertion cannot fail.
		out = any(resp.Text()).(Output)
	}
	if retry, err := r.validate(ctx, out); err != nil {
		return nil, nil, err
	} else if retry != nil {
		return nil, &RetryPromptPart{Content: retry.Message}, nil
	}
	return r.result(out), nil, nil
}

func (r *run[Deps, Output]) validate(ctx context.Context, out Output) (*RetryError, error) {
	for _, validate := range r.agent.outputValidators {
		err := validate(ctx, r.rc, out)
		var retry *RetryError
		switch {
		case errors.As(err, &retry):
			if err := r.countRetry(); err != nil {
				return nil, err
			}
			return retry, nil
		case err != nil:
			return nil, fmt.Errorf("ai: output validation: %w", err)
		}
	}
	return nil, nil
}

func (r *run[Deps, Output]) recordRetry(part RetryPromptPart) {
	r.messages = append(r.messages, ModelRequest{Parts: []RequestPart{part}})
}

func (r *run[Deps, Output]) countRetry() error {
	r.retriesMu.Lock()
	defer r.retriesMu.Unlock()
	r.retries++
	if r.retries > r.agent.maxRetries {
		return fmt.Errorf("%w: %d retries", ErrMaxRetriesExceeded, r.retries)
	}
	return nil
}

func (r *run[Deps, Output]) retryCount() int {
	r.retriesMu.Lock()
	defer r.retriesMu.Unlock()
	return r.retries
}

func (r *run[Deps, Output]) result(out Output) *RunResult[Output] {
	return &RunResult[Output]{Output: out, usage: r.usage, messages: r.messages, newMessages: r.newMessages}
}

func (a *Agent[Deps, Output]) findTool(name string) (toolEntry[Deps], bool) {
	for _, entry := range a.tools {
		if entry.def.Name == name {
			return entry, true
		}
	}
	return toolEntry[Deps]{}, false
}

func (a *Agent[Deps, Output]) buildInstructions(ctx context.Context, rc *RunContext[Deps], info *RunInfo) (string, error) {
	parts := make([]string, 0, len(a.instructionsFuncs)+len(a.capInstructions)+1)
	if a.instructions != "" {
		parts = append(parts, a.instructions)
	}
	parts = append(parts, a.capInstructions...)
	for _, capability := range a.capabilities {
		provider, ok := capability.(InstructionsProvider)
		if !ok {
			continue
		}
		s, err := provider.Instructions(ctx, info)
		if err != nil {
			return "", fmt.Errorf("ai: instructions: %w", err)
		}
		if s != "" {
			parts = append(parts, s)
		}
	}
	for _, fn := range a.instructionsFuncs {
		s, err := fn(ctx, rc)
		if err != nil {
			return "", fmt.Errorf("ai: instructions: %w", err)
		}
		if s != "" {
			parts = append(parts, s)
		}
	}
	return strings.Join(parts, "\n\n"), nil
}

func (a *Agent[Deps, Output]) buildParams(instructions string) (ModelRequestParams, error) {
	params := ModelRequestParams{Instructions: instructions, Settings: a.settings}
	for _, entry := range a.tools {
		params.Tools = append(params.Tools, entry.def)
	}
	var out Output
	if _, isString := any(out).(string); isString {
		params.AllowText = true
		return params, nil
	}
	s, err := schema.For(reflect.TypeFor[Output]())
	if err != nil {
		return params, fmt.Errorf("ai: output type: %w", err)
	}
	if a.outputMode == OutputModeNative {
		params.OutputSchema = s
		params.AllowText = true
		return params, nil
	}
	params.OutputTool = &ToolDefinition{
		Name:        outputToolName,
		Description: "The final result of the run.",
		Schema:      s,
	}
	return params, nil
}

func newRunID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// callTool is the tool-call interception point: capability middleware
// (ToolCallWrapper) around the tool itself, outermost first.
func (r *run[Deps, Output]) callTool(
	ctx context.Context, rc *RunContext[Deps], entry toolEntry[Deps], call ToolCallPart,
) (any, error) {
	next := ToolCallFunc(func(ctx context.Context, call ToolCallPart) (any, error) {
		return entry.call(ctx, rc, call.Args)
	})
	for i := len(r.agent.capabilities) - 1; i >= 0; i-- {
		if wrapper, ok := r.agent.capabilities[i].(ToolCallWrapper); ok {
			innerNext := next
			next = func(ctx context.Context, call ToolCallPart) (any, error) {
				return wrapper.WrapToolCall(ctx, r.info, call, innerNext)
			}
		}
	}
	return next(ctx, call)
}

// wrappedLoop is the run interception point: capability middleware
// (RunWrapper) around the whole loop, outermost first.
func (r *run[Deps, Output]) wrappedLoop(ctx context.Context) (*RunResult[Output], error) {
	var result *RunResult[Output]
	next := RunFunc(func(ctx context.Context) error {
		var err error
		result, err = r.loop(ctx)
		return err
	})
	for i := len(r.agent.capabilities) - 1; i >= 0; i-- {
		if wrapper, ok := r.agent.capabilities[i].(RunWrapper); ok {
			innerNext := next
			next = func(ctx context.Context) error {
				return wrapper.WrapRun(ctx, r.info, innerNext)
			}
		}
	}
	if err := next(ctx); err != nil {
		return nil, err
	}
	return result, nil
}
