package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
)

type toolArgsSchemaValidationError struct{ err error }

func (err *toolArgsSchemaValidationError) Error() string { return err.err.Error() }

func (r *run[Deps, Output]) validateToolCall(
	ctx context.Context, rc *RunContext[Deps], entry toolEntry[Deps], call ToolCallPart,
) (ToolCallPart, any, error) {
	hookContext := ToolHookContext{
		Call: call, Definition: entry.def, Approved: rc.ToolCallApproved,
		CallMetadata: cloneSchemaMap(rc.ToolCallMetadata),
	}.Clone()
	rawArgs := slices.Clone(call.Args)
	for _, capability := range r.capabilities {
		hook, ok := capability.(BeforeToolValidationHook)
		if !ok {
			continue
		}
		var err error
		rawArgs, err = hook.BeforeToolValidation(ctx, r.info, hookContext.Clone(), slices.Clone(rawArgs))
		if err != nil {
			call.Args = slices.Clone(rawArgs)
			return call, nil, err
		}
	}
	call.Args = slices.Clone(rawArgs)
	next := ToolValidationFunc(func(ctx context.Context, rawArgs json.RawMessage) (any, error) {
		if validator := r.currentToolValidators[call.ToolName]; validator != nil {
			if err := validator.ValidateJSON(rawArgs); err != nil {
				return nil, &toolArgsSchemaValidationError{err: err}
			}
		}
		return entry.validate(ctx, rc, rawArgs)
	})
	for index := len(r.capabilities) - 1; index >= 0; index-- {
		if wrapper, ok := r.capabilities[index].(ToolValidationWrapper); ok {
			innerNext := next
			next = func(ctx context.Context, rawArgs json.RawMessage) (any, error) {
				return wrapper.WrapToolValidation(
					ctx, r.info, hookContext.Clone(), slices.Clone(rawArgs), innerNext,
				)
			}
		}
	}
	validated, err := next(ctx, rawArgs)
	if err != nil {
		for index := len(r.capabilities) - 1; index >= 0; index-- {
			hook, ok := r.capabilities[index].(ToolValidationErrorHook)
			if !ok {
				continue
			}
			validated, err = hook.OnToolValidationError(
				ctx, r.info, hookContext.Clone(), slices.Clone(rawArgs), err,
			)
			if err == nil {
				break
			}
		}
	}
	if err != nil {
		return call, nil, err
	}
	if _, deferred := asToolHookDeferral(validated, nil); deferred {
		return call, nil, fmt.Errorf("tool validation error hooks cannot defer a call")
	}
	for index := len(r.capabilities) - 1; index >= 0; index-- {
		hook, ok := r.capabilities[index].(AfterToolValidationHook)
		if !ok {
			continue
		}
		previous := validated
		validated, err = hook.AfterToolValidation(ctx, r.info, hookContext.Clone(), validated)
		if err != nil {
			return call, nil, err
		}
		if deferred, ok := asToolHookDeferral(validated, previous); ok {
			call.Args, err = validatedToolArgsJSON(previous)
			if err != nil {
				return call, nil, fmt.Errorf("marshal validated arguments: %w", err)
			}
			return call, deferred, nil
		}
	}
	call.Args, err = validatedToolArgsJSON(validated)
	if err != nil {
		return call, nil, fmt.Errorf("marshal validated arguments: %w", err)
	}
	return call, validated, nil
}

func validatedToolArgsJSON(args any) (json.RawMessage, error) {
	if raw, ok := args.(json.RawMessage); ok {
		return slices.Clone(raw), nil
	}
	return json.Marshal(args)
}
