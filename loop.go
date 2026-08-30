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
	return r.loop(ctx)
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
	instructions, err := a.buildInstructions(ctx, r.rc)
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
	params      ModelRequestParams
	messages    []ModelMessage
	newMessages int
	usage       Usage
	retries     int
	// emit forwards stream events during RunStream; nil for plain runs.
	emit func(StreamEvent) bool
}

// modelRequest is the model-request interception point: tracing today,
// capability middleware (WrapModelRequest) in v0.3.
func (r *run[Deps, Output]) modelRequest(ctx context.Context) (*ModelResponse, error) {
	a := r.agent
	reqCtx, reqSpan := startRequestSpan(ctx, a.model.Name())
	resp, err := r.doModelRequest(reqCtx)
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

// doModelRequest streams when the run has an emit callback and the model
// supports it; otherwise it falls back to a plain request, replaying the
// response as events so RunStream works with every Model.
func (r *run[Deps, Output]) doModelRequest(ctx context.Context) (*ModelResponse, error) {
	if r.emit == nil {
		return r.agent.model.Request(ctx, r.messages, r.params)
	}
	if sm, ok := r.agent.model.(StreamingModel); ok {
		events, err := sm.StreamRequest(ctx, r.messages, r.params)
		if err != nil {
			return nil, err
		}
		return accumulate(events, r.emit)
	}
	resp, err := r.agent.model.Request(ctx, r.messages, r.params)
	if err != nil {
		return nil, err
	}
	return accumulate(replayAsEvents(resp), r.emit)
}

func (r *run[Deps, Output]) loop(ctx context.Context) (*RunResult[Output], error) {
	a := r.agent
	for {
		resp, err := r.modelRequest(ctx)
		if err != nil {
			return nil, err
		}
		r.usage.Add(resp.Usage)
		if err := a.limits.check(r.usage); err != nil {
			return nil, err
		}
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

		var parts []RequestPart
		var final *RunResult[Output]
		for _, call := range calls {
			part, result, err := r.executeCall(ctx, call)
			if err != nil {
				return nil, err
			}
			if result != nil && final == nil {
				final = result
			}
			if part != nil {
				parts = append(parts, part)
			}
		}
		if final != nil {
			return final, nil
		}
		r.messages = append(r.messages, ModelRequest{Parts: parts})
	}
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
	r.rc.ToolCallID = call.ToolCallID
	r.rc.Retry = r.retries
	toolCtx, toolSpan := startToolSpan(ctx, call.ToolName, call.ToolCallID)
	content, err := entry.call(toolCtx, r.rc, call.Args)
	endSpan(toolSpan, err)
	r.rc.ToolCallID = ""
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

// finalizeText handles a response with no tool calls. For string outputs the
// text is the result; for structured outputs the model must call the output
// tool, so text-only responses trigger a retry.
func (r *run[Deps, Output]) finalizeText(ctx context.Context, resp *ModelResponse) (*RunResult[Output], *RetryPromptPart, error) {
	if !r.params.AllowText {
		if err := r.countRetry(); err != nil {
			return nil, nil, err
		}
		return nil, &RetryPromptPart{Content: fmt.Sprintf("Respond by calling the %s tool to provide the final result.", outputToolName)}, nil
	}
	// buildParams sets AllowText only when Output is string, so this
	// assertion cannot fail.
	out := any(resp.Text()).(Output)
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
	r.retries++
	if r.retries > r.agent.maxRetries {
		return fmt.Errorf("%w: %d retries", ErrMaxRetriesExceeded, r.retries)
	}
	return nil
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

func (a *Agent[Deps, Output]) buildInstructions(ctx context.Context, rc *RunContext[Deps]) (string, error) {
	parts := make([]string, 0, len(a.instructionsFuncs)+1)
	if a.instructions != "" {
		parts = append(parts, a.instructions)
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
