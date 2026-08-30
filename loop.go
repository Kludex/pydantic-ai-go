package ai

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
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
	if hasEventStreamCapability(a.capabilities) {
		stream := a.runStreamPrompt(ctx, prompt, deps, opts, false)
		for _, streamErr := range stream.Events() {
			if streamErr != nil {
				return nil, streamErr
			}
		}
		if stream.Result() == nil {
			return nil, &UnexpectedModelBehaviorError{Message: "event stream ended before the run completed"}
		}
		return stream.Result(), nil
	}
	cfg := buildRunConfig(opts)
	model := a.model
	if cfg.model != nil {
		model = cfg.model
	}
	ctx, span := startRunSpan(ctx, model.Name())
	defer func() {
		if result != nil {
			recordUsage(span, result.usage)
		}
		endSpan(span, err)
	}()
	a.started.Store(true)
	r, err := a.newRun(ctx, prompt, deps, cfg)
	if err != nil {
		return nil, err
	}
	defer r.cancellation.finish()
	return r.wrappedLoop(r.ctx)
}

func buildRunConfig(opts []RunOption) runConfig {
	var cfg runConfig
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

func (a *Agent[Deps, Output]) newRun(
	ctx context.Context, prompt UserPromptPart, deps Deps, cfg runConfig,
) (*run[Deps, Output], error) {
	a.started.Store(true)
	runCtx, cancel := context.WithCancelCause(ctx)
	cancellation := &runCancellation{cancel: cancel, active: true}
	model := a.model
	if cfg.model != nil {
		model = cfg.model
	}
	capabilities := slices.Clone(a.capabilities)
	limits := a.usageLimits
	if cfg.usageLimits != nil {
		limits = *cfg.usageLimits
	}
	if limits != (UsageLimits{}) {
		limitCapability := usageLimitsCapability{limits: limits}
		_ = limitCapability.Setup(&CapabilityRegistry{})
		capabilities = append([]Capability{limitCapability}, capabilities...)
	}
	r := &run[Deps, Output]{
		agent: a, model: model, capabilities: capabilities, ctx: runCtx, cancellation: cancellation,
		retryLimits: a.retryLimits, toolRetries: make(map[string]int),
	}
	if cfg.retryLimits != nil {
		validateRetryLimits(*cfg.retryLimits)
		r.retryLimits = *cfg.retryLimits
	}
	history, interruptedReturns := repairDanglingToolCalls(dropOrphanedToolResults(cfg.history))
	r.messages = append(r.messages, history...)
	r.newMessages = len(r.messages)
	settings := mergeModelSettings(a.settings, cfg.settings)
	r.rc = &RunContext[Deps]{
		Deps: deps, MaxRetries: r.retryLimits.Output, RunID: newRunID(),
		Model: model, ModelSettings: settings, UsageLimits: limits,
		usage: &r.usage, toolCalls: &r.toolCalls, messages: &r.messages, cancellation: cancellation,
	}
	r.info = &RunInfo{RunID: r.rc.RunID, usage: &r.usage, toolCalls: &r.toolCalls, messages: &r.messages}
	instructionParts, err := a.buildInstructions(runCtx, r.rc, r.info, cfg.instructions)
	if err != nil {
		cancellation.finish()
		return nil, err
	}
	outputMode := a.outputMode
	if cfg.outputMode != nil {
		outputMode = *cfg.outputMode
	}
	validateOutputMode(outputMode)
	r.params, err = a.buildParams(instructionParts, settings, outputMode)
	if err != nil {
		cancellation.finish()
		return nil, err
	}
	requestParts := slices.Clone(interruptedReturns)
	requestParts = append(requestParts, prompt)
	r.messages = append(r.messages, ModelRequest{Parts: requestParts})
	return r, nil
}

func dropOrphanedToolResults(messages []ModelMessage) []ModelMessage {
	seenCalls := map[string]struct{}{}
	repaired := make([]ModelMessage, 0, len(messages))
	for index, message := range messages {
		switch message := message.(type) {
		case ModelResponse:
			for _, call := range message.ToolCalls() {
				seenCalls[toolCallMatchKey(call.ToolName, call.ToolCallID)] = struct{}{}
			}
			repaired = append(repaired, message)
		case ModelRequest:
			parts := make([]RequestPart, 0, len(message.Parts))
			for _, part := range message.Parts {
				toolName, toolCallID, isResult := toolResultIdentity(part)
				if isResult {
					if _, seen := seenCalls[toolCallMatchKey(toolName, toolCallID)]; !seen {
						continue
					}
				}
				parts = append(parts, part)
			}
			if len(parts) > 0 || index == len(messages)-1 {
				message.Parts = parts
				repaired = append(repaired, message)
			}
		}
	}
	return repaired
}

type trackedToolCall struct {
	responseIndex int
	call          ToolCallPart
	active        bool
	dangling      bool
}

func repairDanglingToolCalls(messages []ModelMessage) ([]ModelMessage, []RequestPart) {
	tracked := make([]*trackedToolCall, 0)
	open := map[string]*trackedToolCall{}
	for index, message := range messages {
		switch message := message.(type) {
		case ModelResponse:
			for _, call := range message.ToolCalls() {
				key := toolCallMatchKey(call.ToolName, call.ToolCallID)
				if shadowed := open[key]; shadowed != nil {
					shadowed.active = false
					shadowed.dangling = true
				}
				item := &trackedToolCall{responseIndex: index, call: call, active: true}
				tracked = append(tracked, item)
				open[key] = item
			}
		case ModelRequest:
			for _, part := range message.Parts {
				toolName, toolCallID, ok := toolResultIdentity(part)
				if !ok {
					continue
				}
				key := toolCallMatchKey(toolName, toolCallID)
				if item := open[key]; item != nil {
					item.active = false
					delete(open, key)
				}
			}
		}
	}
	dangling := map[int][]RequestPart{}
	for _, item := range tracked {
		if item.active || item.dangling {
			call := item.call
			dangling[item.responseIndex] = append(dangling[item.responseIndex], ToolReturnPart{
				ToolName: call.ToolName, ToolCallID: call.ToolCallID,
				Content:  "The tool call was interrupted before a result was produced.",
				Outcome:  ToolReturnOutcomeInterrupted,
				Metadata: map[string]any{SynthesizedToolReturnMetadataKey: true},
			})
		}
	}
	if len(dangling) == 0 {
		return slices.Clone(messages), nil
	}
	repaired := make([]ModelMessage, 0, len(messages)+len(dangling))
	var pending []RequestPart
	for index, message := range messages {
		switch message := message.(type) {
		case ModelResponse:
			if len(pending) > 0 {
				repaired = append(repaired, ModelRequest{Parts: pending})
			}
			repaired = append(repaired, message)
			pending = dangling[index]
		case ModelRequest:
			if len(pending) > 0 {
				parts := slices.Clone(message.Parts)
				insertAt := 0
				for partIndex, part := range parts {
					if _, _, ok := toolResultIdentity(part); ok {
						insertAt = partIndex + 1
					}
				}
				parts = slices.Insert(parts, insertAt, pending...)
				message.Parts = parts
				pending = nil
			}
			repaired = append(repaired, message)
		}
	}
	return repaired, pending
}

func toolCallMatchKey(toolName, toolCallID string) string {
	if toolCallID != "" {
		return "id:" + toolCallID
	}
	return "name:" + toolName
}

func toolResultIdentity(part RequestPart) (toolName, toolCallID string, ok bool) {
	switch part := part.(type) {
	case ToolReturnPart:
		return part.ToolName, part.ToolCallID, true
	case RetryPromptPart:
		return part.ToolName, part.ToolCallID, part.ToolName != ""
	default:
		return "", "", false
	}
}

type run[Deps, Output any] struct {
	agent        *Agent[Deps, Output]
	model        Model
	capabilities []Capability
	ctx          context.Context
	cancellation *runCancellation
	rc           *RunContext[Deps]
	info         *RunInfo
	params       ModelRequestParams
	messages     []ModelMessage
	newMessages  int
	usage        Usage
	toolCalls    atomic.Int64
	retryLimits  RetryLimits
	toolRetries  map[string]int
	outputRetry  int
	retriesMu    sync.Mutex
	currentTools map[string]struct{}
	// emit forwards stream events during streamed model execution.
	emit                 func(StreamEvent) bool
	emitMu               sync.Mutex
	commitStreamedOutput bool
}

// modelRequest is the model-request interception point: tracing plus
// capability middleware (ModelRequestWrapper), outermost first.
func (r *run[Deps, Output]) modelRequest(ctx context.Context) (*ModelResponse, error) {
	inner := func(ctx context.Context, msgs []ModelMessage, params ModelRequestParams) (*ModelResponse, error) {
		r.setCurrentTools(params)
		reqCtx, reqSpan := startRequestSpan(ctx, r.model.Name())
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
			resp.ModelName = r.model.Name()
		}
		return resp, nil
	}
	params, err := r.prepareModelParams(ctx)
	if err != nil {
		return nil, err
	}
	next := inner
	for i := len(r.capabilities) - 1; i >= 0; i-- {
		if wrapper, ok := r.capabilities[i].(ModelRequestWrapper); ok {
			innerNext := next
			next = func(ctx context.Context, msgs []ModelMessage, params ModelRequestParams) (*ModelResponse, error) {
				return wrapper.WrapModelRequest(ctx, r.info, msgs, params, innerNext)
			}
		}
	}
	r.setCurrentTools(params)
	return next(ctx, r.messages, params)
}

func (r *run[Deps, Output]) setCurrentTools(params ModelRequestParams) {
	r.currentTools = make(map[string]struct{}, len(params.Tools))
	for _, def := range params.Tools {
		r.currentTools[def.Name] = struct{}{}
	}
}

func (r *run[Deps, Output]) prepareModelParams(ctx context.Context) (ModelRequestParams, error) {
	params := r.params
	params.InstructionParts = slices.Clone(params.InstructionParts)
	tools := make([]ToolDefinition, 0, len(params.Tools))
	rc := *r.rc
	rc.Retry = r.outputRetryCount()
	rc.MaxRetries = r.retryLimits.Output
	for _, def := range params.Tools {
		prepared := cloneToolDefinition(def)
		entry, _ := r.agent.findTool(def.Name)
		if entry.prepare != nil {
			result, err := entry.prepare(ctx, &rc, prepared)
			if err != nil {
				return ModelRequestParams{}, fmt.Errorf("ai: prepare tool %q: %w", def.Name, err)
			}
			if result == nil {
				continue
			}
			prepared = *result
		}
		tools = append(tools, prepared)
	}
	for _, prepare := range r.agent.toolsPrepareFuncs {
		var err error
		tools, err = prepare(ctx, &rc, tools)
		if err != nil {
			return ModelRequestParams{}, fmt.Errorf("ai: prepare tools: %w", err)
		}
	}
	known := make(map[string]struct{}, len(r.agent.tools))
	for _, entry := range r.agent.tools {
		known[entry.def.Name] = struct{}{}
	}
	seen := make(map[string]struct{}, len(tools))
	for _, def := range tools {
		if _, ok := known[def.Name]; !ok {
			return ModelRequestParams{}, fmt.Errorf("ai: prepare tools returned unknown tool %q", def.Name)
		}
		if _, ok := seen[def.Name]; ok {
			return ModelRequestParams{}, fmt.Errorf("ai: prepare tools returned duplicate tool %q", def.Name)
		}
		seen[def.Name] = struct{}{}
	}
	params.Tools = tools
	return params, nil
}

func cloneToolDefinition(def ToolDefinition) ToolDefinition {
	def.Schema = cloneSchemaMap(def.Schema)
	if def.Strict != nil {
		strict := *def.Strict
		def.Strict = &strict
	}
	return def
}

func cloneSchemaMap(schema map[string]any) map[string]any {
	if schema == nil {
		return nil
	}
	cloned := make(map[string]any, len(schema))
	for key, value := range schema {
		cloned[key] = cloneSchemaValue(value)
	}
	return cloned
}

func cloneSchemaValue(value any) any {
	switch value := value.(type) {
	case map[string]any:
		return cloneSchemaMap(value)
	case []any:
		cloned := make([]any, len(value))
		for i, item := range value {
			cloned[i] = cloneSchemaValue(item)
		}
		return cloned
	default:
		return value
	}
}

// doModelRequest streams when the run has an emit callback and the model
// supports it; otherwise it falls back to a plain request, replaying the
// response as events so RunStream works with every Model.
func (r *run[Deps, Output]) doModelRequest(ctx context.Context, msgs []ModelMessage, params ModelRequestParams) (*ModelResponse, error) {
	if r.emit == nil {
		return r.model.Request(ctx, msgs, params)
	}
	if sm, ok := r.model.(StreamingModel); ok {
		events, err := sm.StreamRequest(ctx, msgs, params)
		if err != nil {
			return nil, err
		}
		return accumulate(events, params, r.emitStreamEvent)
	}
	resp, err := r.model.Request(ctx, msgs, params)
	if err != nil {
		return nil, err
	}
	return accumulate(replayAsEvents(resp), params, r.emitStreamEvent)
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
		if r.emit != nil && r.commitStreamedOutput {
			output, winningCall, committed, err := r.streamedOutput(ctx, resp)
			if err != nil {
				return nil, err
			}
			if committed {
				parts, err := r.executeCallsWithCommittedOutput(ctx, calls, winningCall)
				if errors.Is(context.Cause(r.ctx), ErrRunCancelled) {
					if len(parts) > 0 {
						r.messages = append(r.messages, ModelRequest{Parts: parts, State: RequestStateInterrupted})
					}
					return nil, ErrRunCancelled
				}
				if err != nil {
					return nil, err
				}
				if len(parts) > 0 {
					r.messages = append(r.messages, ModelRequest{Parts: parts})
				}
				return r.result(*output), nil
			}
		}
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

		if output, ok, err := r.earlyNativeOutput(ctx, resp); err != nil {
			return nil, err
		} else if ok {
			if !r.emitToolCallEvents(calls) {
				return nil, context.Canceled
			}
			parts := make([]RequestPart, 0, len(calls))
			for _, call := range calls {
				part := ToolReturnPart{
					ToolName: call.ToolName, Content: toolSkipped, ToolCallID: call.ToolCallID,
					Outcome: ToolReturnOutcomeSuccess,
				}
				parts = append(parts, part)
				if !r.emitStreamEvent(FunctionToolResultEvent{Part: part}) {
					return nil, context.Canceled
				}
			}
			r.messages = append(r.messages, ModelRequest{Parts: parts})
			return r.result(*output), nil
		}

		parts, final, err := r.executeCalls(ctx, calls)
		if errors.Is(context.Cause(r.ctx), ErrRunCancelled) {
			if len(parts) > 0 {
				r.messages = append(r.messages, ModelRequest{Parts: parts, State: RequestStateInterrupted})
			}
			return nil, ErrRunCancelled
		}
		if err != nil {
			return nil, err
		}
		r.messages = append(r.messages, ModelRequest{Parts: parts})
		if final != nil {
			return r.result(*final), nil
		}
	}
}

func (r *run[Deps, Output]) earlyNativeOutput(
	ctx context.Context, resp *ModelResponse,
) (*Output, bool, error) {
	if r.agent.endStrategy != EndStrategyEarly || r.params.OutputSchema == nil || resp.Text() == "" {
		return nil, false, nil
	}
	var out Output
	if err := json.Unmarshal([]byte(resp.Text()), &out); err != nil {
		return nil, false, nil
	}
	outputRC := r.outputRunContext("")
	for _, validate := range r.agent.outputValidators {
		err := validate(ctx, outputRC, out)
		var retry *RetryError
		switch {
		case errors.As(err, &retry):
			return nil, false, nil
		case err != nil:
			return nil, false, fmt.Errorf("ai: output validation: %w", err)
		}
	}
	return &out, true, nil
}

type callOutcome[Output any] struct {
	part          RequestPart
	output        *Output
	outputCall    bool
	functionCall  bool
	resultEmitted bool
	err           error
}

const (
	finalResultProcessed = "Final result processed."
	retryWins            = "Output not used as the final result - addressing tool retries from this round first."
	outputSkipped        = "Output tool not used - a final result was already processed."
	outputNotFinal       = "Output tool processed, but its value will not be the final result of the agent run."
	toolSkipped          = "Tool not executed - a final result was already processed."
)

func (r *run[Deps, Output]) emitStreamEvent(event StreamEvent) bool {
	if r.emit == nil {
		return true
	}
	r.emitMu.Lock()
	defer r.emitMu.Unlock()
	return r.emit(event)
}

func (r *run[Deps, Output]) emitToolCallEvents(calls []ToolCallPart) bool {
	for _, call := range calls {
		var event StreamEvent = FunctionToolCallEvent{Part: call}
		if r.isOutputCall(call) {
			event = OutputToolCallEvent{Part: call}
		}
		if !r.emitStreamEvent(event) {
			return false
		}
	}
	return true
}

// executeCalls honors the configured end strategy while preserving emission
// order in the request sent back to the model.
func (r *run[Deps, Output]) executeCalls(
	ctx context.Context, calls []ToolCallPart,
) ([]RequestPart, *Output, error) {
	if !r.emitToolCallEvents(calls) {
		return nil, nil, context.Canceled
	}
	if r.agent.endStrategy == EndStrategyEarly {
		return r.executeCallsEarly(ctx, calls)
	}
	if r.agent.endStrategy == EndStrategyGraceful {
		return r.executeCallsGraceful(ctx, calls)
	}
	return r.executeCallsExhaustive(ctx, calls)
}

func (r *run[Deps, Output]) checkToolCallLimit(calls []ToolCallPart) error {
	if r.rc.UsageLimits.ToolCallLimit == nil {
		return nil
	}
	pending := 0
	for _, call := range calls {
		_, registered := r.agent.findTool(call.ToolName)
		_, available := r.currentTools[call.ToolName]
		if !r.isOutputCall(call) && registered && available {
			pending++
		}
	}
	projected := int(r.toolCalls.Load()) + pending
	if projected > *r.rc.UsageLimits.ToolCallLimit {
		return fmt.Errorf(
			"%w: tool call count %d exceeds limit %d",
			ErrUsageLimitExceeded, projected, *r.rc.UsageLimits.ToolCallLimit,
		)
	}
	return nil
}

func (r *run[Deps, Output]) executeCallsEarly(
	ctx context.Context, calls []ToolCallPart,
) ([]RequestPart, *Output, error) {
	outcomes := make([]callOutcome[Output], len(calls))
	var winner *Output
	for i, call := range calls {
		if !r.isOutputCall(call) {
			continue
		}
		if winner != nil {
			outcomes[i].part = ToolReturnPart{
				ToolName: call.ToolName, Content: outputSkipped, ToolCallID: call.ToolCallID,
				Outcome: ToolReturnOutcomeSuccess,
			}
			continue
		}
		outcomes[i] = r.executeOne(ctx, call)
		if outcomes[i].err != nil {
			return completedCallParts(outcomes), nil, outcomes[i].err
		}
		winner = outcomes[i].output
	}
	if winner != nil {
		for i, call := range calls {
			if !r.isOutputCall(call) {
				outcomes[i].part = ToolReturnPart{
					ToolName: call.ToolName, Content: toolSkipped, ToolCallID: call.ToolCallID,
					Outcome: ToolReturnOutcomeSuccess,
				}
			}
		}
		return r.collectCallOutcomes(outcomes, false)
	}
	if err := r.checkToolCallLimit(calls); err != nil {
		return completedCallParts(outcomes), nil, err
	}
	if err := r.executeSelected(ctx, calls, outcomes, r.functionCallIndexes(calls), false); err != nil {
		return completedCallParts(outcomes), nil, err
	}
	return r.collectCallOutcomes(outcomes, false)
}

func (r *run[Deps, Output]) executeCallsGraceful(
	ctx context.Context, calls []ToolCallPart,
) ([]RequestPart, *Output, error) {
	if err := r.checkToolCallLimit(calls); err != nil {
		return nil, nil, err
	}
	outcomes := make([]callOutcome[Output], len(calls))
	batch := make([]int, 0, len(calls))
	var winner *Output
	for i, call := range calls {
		if !r.isOutputCall(call) && !r.callIsBarrier(call, true) {
			batch = append(batch, i)
			continue
		}
		if err := r.executeIndexBatch(ctx, calls, outcomes, batch); err != nil {
			return completedCallParts(outcomes), nil, err
		}
		batch = batch[:0]
		if r.isOutputCall(call) && winner != nil {
			outcomes[i].part = ToolReturnPart{
				ToolName: call.ToolName, Content: outputSkipped, ToolCallID: call.ToolCallID,
				Outcome: ToolReturnOutcomeSuccess,
			}
			continue
		}
		outcomes[i] = r.executeOne(ctx, call)
		if outcomes[i].err != nil {
			return completedCallParts(outcomes), nil, outcomes[i].err
		}
		if outcomes[i].output != nil {
			winner = outcomes[i].output
		}
	}
	if err := r.executeIndexBatch(ctx, calls, outcomes, batch); err != nil {
		return completedCallParts(outcomes), nil, err
	}
	return r.collectCallOutcomes(outcomes, true)
}

func (r *run[Deps, Output]) executeCallsExhaustive(
	ctx context.Context, calls []ToolCallPart,
) ([]RequestPart, *Output, error) {
	if err := r.checkToolCallLimit(calls); err != nil {
		return nil, nil, err
	}
	outcomes := make([]callOutcome[Output], len(calls))
	indexes := make([]int, len(calls))
	for i := range calls {
		indexes[i] = i
	}
	if err := r.executeSelected(ctx, calls, outcomes, indexes, true); err != nil {
		return completedCallParts(outcomes), nil, err
	}
	return r.collectCallOutcomes(outcomes, true)
}

func (r *run[Deps, Output]) executeSelected(
	ctx context.Context,
	calls []ToolCallPart,
	outcomes []callOutcome[Output],
	indexes []int,
	outputToolsConcurrent bool,
) error {
	batch := make([]int, 0, len(indexes))
	for _, i := range indexes {
		if !r.callIsBarrier(calls[i], outputToolsConcurrent) {
			batch = append(batch, i)
			continue
		}
		if err := r.executeIndexBatch(ctx, calls, outcomes, batch); err != nil {
			return err
		}
		batch = batch[:0]
		outcomes[i] = r.executeOne(ctx, calls[i])
		if outcomes[i].err != nil {
			return outcomes[i].err
		}
	}
	return r.executeIndexBatch(ctx, calls, outcomes, batch)
}

func (r *run[Deps, Output]) executeIndexBatch(
	ctx context.Context, calls []ToolCallPart, outcomes []callOutcome[Output], indexes []int,
) error {
	if len(indexes) == 0 {
		return nil
	}
	if len(indexes) == 1 {
		outcomes[indexes[0]] = r.executeOne(ctx, calls[indexes[0]])
		return outcomes[indexes[0]].err
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var wg sync.WaitGroup
	for _, i := range indexes {
		wg.Add(1)
		go func() {
			defer wg.Done()
			outcomes[i] = r.executeOne(ctx, calls[i])
			if outcomes[i].err != nil {
				cancel()
			}
		}()
	}
	wg.Wait()
	for _, i := range indexes {
		if outcomes[i].err != nil {
			return outcomes[i].err
		}
	}
	return nil
}

func (r *run[Deps, Output]) executeOne(ctx context.Context, call ToolCallPart) callOutcome[Output] {
	part, output, err := r.executeCall(ctx, call)
	_, registered := r.agent.findTool(call.ToolName)
	_, available := r.currentTools[call.ToolName]
	outcome := callOutcome[Output]{
		part: part, output: output, outputCall: r.isOutputCall(call), functionCall: registered && available, err: err,
	}
	if err == nil && errors.Is(context.Cause(r.ctx), ErrRunCancelled) {
		outcome.part = nil
		outcome.output = nil
	}
	if outcome.err == nil && outcome.part != nil && outcome.functionCall {
		if part, ok := outcome.part.(ToolReturnPart); ok && part.Outcome == ToolReturnOutcomeSuccess {
			r.toolCalls.Add(1)
		}
	}
	if outcome.err == nil && outcome.part != nil && !outcome.outputCall {
		if !r.emitStreamEvent(FunctionToolResultEvent{Part: outcome.part}) {
			outcome.err = context.Canceled
		} else {
			outcome.resultEmitted = true
		}
	}
	return outcome
}

func (r *run[Deps, Output]) functionCallIndexes(calls []ToolCallPart) []int {
	indexes := make([]int, 0, len(calls))
	for i, call := range calls {
		if !r.isOutputCall(call) {
			indexes = append(indexes, i)
		}
	}
	return indexes
}

func (r *run[Deps, Output]) isOutputCall(call ToolCallPart) bool {
	return r.params.OutputTool != nil && call.ToolName == outputToolName
}

func (r *run[Deps, Output]) callIsBarrier(call ToolCallPart, outputToolsConcurrent bool) bool {
	if r.agent.sequentialTools {
		return true
	}
	if r.isOutputCall(call) {
		return !outputToolsConcurrent
	}
	entry, ok := r.agent.findTool(call.ToolName)
	return ok && entry.def.Sequential
}

func completedCallParts[Output any](outcomes []callOutcome[Output]) []RequestPart {
	parts := make([]RequestPart, 0, len(outcomes))
	for _, outcome := range outcomes {
		if outcome.err == nil && outcome.part != nil {
			parts = append(parts, outcome.part)
		}
	}
	return parts
}

func (r *run[Deps, Output]) collectCallOutcomes(
	outcomes []callOutcome[Output], retryCanWin bool,
) ([]RequestPart, *Output, error) {
	parts := make([]RequestPart, 0, len(outcomes))
	partPositions := make([]int, len(outcomes))
	for index := range partPositions {
		partPositions[index] = -1
	}
	var winner *Output
	winningPart := -1
	functionRetry := false
	for index, outcome := range outcomes {
		if outcome.part != nil {
			parts = append(parts, outcome.part)
			partPositions[index] = len(parts) - 1
		}
		if outcome.output != nil {
			if winner == nil {
				winner = outcome.output
				winningPart = len(parts) - 1
			} else {
				part := outcome.part.(ToolReturnPart)
				part.Content = outputNotFinal
				parts[partPositions[index]] = part
			}
		} else if _, ok := outcome.part.(RetryPromptPart); ok && outcome.functionCall {
			functionRetry = true
		}
	}
	if retryCanWin && functionRetry && winner != nil {
		part := parts[winningPart].(ToolReturnPart)
		part.Content = retryWins
		parts[winningPart] = part
		winner = nil
	}
	for index := range outcomes {
		if partPositions[index] >= 0 {
			outcomes[index].part = parts[partPositions[index]]
		}
	}
	if err := r.emitPendingCallResults(outcomes); err != nil {
		return nil, nil, err
	}
	return parts, winner, nil
}

func (r *run[Deps, Output]) emitPendingCallResults(outcomes []callOutcome[Output]) error {
	for index := range outcomes {
		if outcomes[index].resultEmitted || outcomes[index].part == nil {
			continue
		}
		var event StreamEvent = FunctionToolResultEvent{Part: outcomes[index].part}
		if outcomes[index].outputCall {
			event = OutputToolResultEvent{Part: outcomes[index].part}
		}
		if !r.emitStreamEvent(event) {
			return context.Canceled
		}
		outcomes[index].resultEmitted = true
	}
	return nil
}

// executeCall runs one tool call. It returns the request part to send back
// to the model and a validated value for a successful output tool.
func (r *run[Deps, Output]) executeCall(ctx context.Context, call ToolCallPart) (RequestPart, *Output, error) {
	if r.params.OutputTool != nil && call.ToolName == outputToolName {
		return r.finalizeOutputCall(ctx, call)
	}
	entry, registered := r.agent.findTool(call.ToolName)
	_, available := r.currentTools[call.ToolName]
	if !registered || !available {
		if err := r.countToolRetry(call.ToolName); err != nil {
			return nil, nil, err
		}
		return RetryPromptPart{
			Content: r.unknownToolMessage(call.ToolName), ToolName: call.ToolName, ToolCallID: call.ToolCallID,
		}, nil, nil
	}
	toolRC := *r.rc
	toolRC.ToolCallID = call.ToolCallID
	toolRC.Retry, toolRC.MaxRetries = r.toolRetryInfo(call.ToolName)
	spanCtx, toolSpan := startToolSpan(ctx, call.ToolName, call.ToolCallID)
	toolCtx := spanCtx
	var cancel context.CancelFunc = func() {}
	if entry.def.timeout > 0 {
		toolCtx, cancel = context.WithTimeout(spanCtx, entry.def.timeout)
	}
	content, err := r.callTool(toolCtx, &toolRC, entry, call)
	if ctx.Err() == nil && errors.Is(toolCtx.Err(), context.DeadlineExceeded) {
		err = Retryf("Timed out after %s.", entry.def.timeout)
	}
	cancel()
	endSpan(toolSpan, err)
	var failed *ToolFailedError
	var retry *RetryError
	switch {
	case errors.As(err, &failed):
		return ToolReturnPart{
			ToolName: call.ToolName, Content: failed.Message, ToolCallID: call.ToolCallID,
			Outcome: ToolReturnOutcomeFailed,
		}, nil, nil
	case errors.As(err, &retry):
		if err := r.countToolRetry(call.ToolName); err != nil {
			return nil, nil, err
		}
		return RetryPromptPart{Content: retry.Message, ToolName: call.ToolName, ToolCallID: call.ToolCallID}, nil, nil
	case err != nil:
		return nil, nil, fmt.Errorf("ai: tool %q: %w", call.ToolName, err)
	}
	return ToolReturnPart{
		ToolName: call.ToolName, Content: content, ToolCallID: call.ToolCallID, Outcome: ToolReturnOutcomeSuccess,
	}, nil, nil
}

func (r *run[Deps, Output]) unknownToolMessage(name string) string {
	available := make([]string, 0, len(r.currentTools)+1)
	for toolName := range r.currentTools {
		available = append(available, toolName)
	}
	if r.params.OutputTool != nil {
		available = append(available, r.params.OutputTool.Name)
	}
	slices.Sort(available)
	if len(available) == 0 {
		return fmt.Sprintf("Unknown tool name: %s. No tools available.", quoteToolName(name))
	}
	for i, toolName := range available {
		available[i] = quoteToolName(toolName)
	}
	return fmt.Sprintf("Unknown tool name: %s. Available tools: %s", quoteToolName(name), strings.Join(available, ", "))
}

func quoteToolName(name string) string {
	return "'" + strings.ReplaceAll(name, "'", `\'`) + "'"
}

func (r *run[Deps, Output]) finalizeOutputCall(ctx context.Context, call ToolCallPart) (RequestPart, *Output, error) {
	var out Output
	if err := json.Unmarshal(call.Args, &out); err != nil {
		if err := r.countOutputRetry(); err != nil {
			return nil, nil, err
		}
		msg := fmt.Sprintf("invalid final result: %v", err)
		return RetryPromptPart{Content: msg, ToolName: call.ToolName, ToolCallID: call.ToolCallID}, nil, nil
	}
	if retry, err := r.validate(ctx, r.outputRunContext(call.ToolCallID), out); err != nil {
		return nil, nil, err
	} else if retry != nil {
		return RetryPromptPart{Content: retry.Message, ToolName: call.ToolName, ToolCallID: call.ToolCallID}, nil, nil
	}
	return ToolReturnPart{
		ToolName: call.ToolName, Content: finalResultProcessed, ToolCallID: call.ToolCallID,
		Outcome: ToolReturnOutcomeSuccess,
	}, &out, nil
}

// finalizeText handles a response with no tool calls. String outputs take
// the text as-is; native-mode structured outputs unmarshal it; tool-mode
// structured outputs must call the output tool, so text triggers a retry.
func (r *run[Deps, Output]) finalizeText(ctx context.Context, resp *ModelResponse) (*RunResult[Output], *RetryPromptPart, error) {
	if !r.params.AllowText {
		if err := r.countOutputRetry(); err != nil {
			return nil, nil, err
		}
		return nil, &RetryPromptPart{Content: fmt.Sprintf("Respond by calling the %s tool to provide the final result.", outputToolName)}, nil
	}
	var out Output
	if r.params.OutputSchema != nil {
		if err := json.Unmarshal([]byte(resp.Text()), &out); err != nil {
			if err := r.countOutputRetry(); err != nil {
				return nil, nil, err
			}
			return nil, &RetryPromptPart{Content: fmt.Sprintf("invalid JSON output: %v", err)}, nil
		}
	} else {
		// buildParams sets AllowText without a schema only when Output is
		// string, so this assertion cannot fail.
		out = any(resp.Text()).(Output)
	}
	if retry, err := r.validate(ctx, r.outputRunContext(""), out); err != nil {
		return nil, nil, err
	} else if retry != nil {
		return nil, &RetryPromptPart{Content: retry.Message}, nil
	}
	return r.result(out), nil, nil
}

func (r *run[Deps, Output]) validate(
	ctx context.Context, rc *RunContext[Deps], out Output,
) (*RetryError, error) {
	for _, validate := range r.agent.outputValidators {
		err := validate(ctx, rc, out)
		var retry *RetryError
		switch {
		case errors.As(err, &retry):
			if err := r.countOutputRetry(); err != nil {
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

func (r *run[Deps, Output]) outputRunContext(toolCallID string) *RunContext[Deps] {
	rc := *r.rc
	rc.Retry = r.outputRetryCount()
	rc.MaxRetries = r.retryLimits.Output
	rc.ToolCallID = toolCallID
	return &rc
}

func (r *run[Deps, Output]) countToolRetry(name string) error {
	r.retriesMu.Lock()
	defer r.retriesMu.Unlock()
	r.toolRetries[name]++
	_, maxRetries := r.toolRetryInfoLocked(name)
	if r.toolRetries[name] > maxRetries {
		return fmt.Errorf("%w: tool %q exceeded %d retries", ErrMaxRetriesExceeded, name, maxRetries)
	}
	return nil
}

func (r *run[Deps, Output]) toolRetryInfo(name string) (int, int) {
	r.retriesMu.Lock()
	defer r.retriesMu.Unlock()
	return r.toolRetryInfoLocked(name)
}

func (r *run[Deps, Output]) toolRetryInfoLocked(name string) (int, int) {
	maxRetries := r.retryLimits.Tools
	if entry, ok := r.agent.findTool(name); ok && entry.def.maxRetries != nil {
		maxRetries = *entry.def.maxRetries
	}
	return r.toolRetries[name], maxRetries
}

func (r *run[Deps, Output]) countOutputRetry() error {
	r.retriesMu.Lock()
	defer r.retriesMu.Unlock()
	r.outputRetry++
	if r.outputRetry > r.retryLimits.Output {
		return fmt.Errorf("%w: output exceeded %d retries", ErrMaxRetriesExceeded, r.retryLimits.Output)
	}
	return nil
}

func (r *run[Deps, Output]) outputRetryCount() int {
	r.retriesMu.Lock()
	defer r.retriesMu.Unlock()
	return r.outputRetry
}

func (r *run[Deps, Output]) result(out Output) *RunResult[Output] {
	usage := r.usage
	usage.ToolCalls = int(r.toolCalls.Load())
	return &RunResult[Output]{Output: out, usage: usage, messages: r.messages, newMessages: r.newMessages}
}

func (a *Agent[Deps, Output]) findTool(name string) (toolEntry[Deps], bool) {
	for _, entry := range a.tools {
		if entry.def.Name == name {
			return entry, true
		}
	}
	return toolEntry[Deps]{}, false
}

func (a *Agent[Deps, Output]) buildInstructions(
	ctx context.Context, rc *RunContext[Deps], info *RunInfo, additional string,
) ([]InstructionPart, error) {
	parts := make([]InstructionPart, 0, len(a.instructionsFuncs)+len(a.capInstructions)+2)
	if a.instructions != "" {
		parts = append(parts, InstructionPart{Content: a.instructions})
	}
	for _, instructions := range a.capInstructions {
		parts = append(parts, InstructionPart{Content: instructions})
	}
	if additional != "" {
		parts = append(parts, InstructionPart{Content: additional})
	}
	for _, capability := range a.capabilities {
		provider, ok := capability.(InstructionsProvider)
		if !ok {
			continue
		}
		instructions, err := provider.Instructions(ctx, info)
		if err != nil {
			return nil, fmt.Errorf("ai: instructions: %w", err)
		}
		if instructions != "" {
			parts = append(parts, InstructionPart{Content: instructions, Dynamic: true})
		}
	}
	for _, fn := range a.instructionsFuncs {
		instructions, err := fn(ctx, rc)
		if err != nil {
			return nil, fmt.Errorf("ai: instructions: %w", err)
		}
		if instructions != "" {
			parts = append(parts, InstructionPart{Content: instructions, Dynamic: true})
		}
	}
	return parts, nil
}

func (a *Agent[Deps, Output]) buildParams(
	instructionParts []InstructionPart, settings ModelSettings, outputMode OutputMode,
) (ModelRequestParams, error) {
	instructions := make([]string, 0, len(instructionParts))
	for _, part := range instructionParts {
		instructions = append(instructions, part.Content)
	}
	params := ModelRequestParams{
		Instructions: strings.Join(instructions, "\n\n"), InstructionParts: instructionParts, Settings: settings,
	}
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
	if outputMode == OutputModeNative {
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
	for i := len(r.capabilities) - 1; i >= 0; i-- {
		if wrapper, ok := r.capabilities[i].(ToolCallWrapper); ok {
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
	for i := len(r.capabilities) - 1; i >= 0; i-- {
		if wrapper, ok := r.capabilities[i].(RunWrapper); ok {
			innerNext := next
			next = func(ctx context.Context) error {
				return wrapper.WrapRun(ctx, r.info, innerNext)
			}
		}
	}
	err := next(ctx)
	if errors.Is(context.Cause(r.ctx), ErrRunCancelled) {
		usage := r.usage
		usage.ToolCalls = int(r.toolCalls.Load())
		return nil, &RunCancelledError{messages: slices.Clone(r.messages), usage: usage}
	}
	if err != nil {
		return nil, err
	}
	return result, nil
}
