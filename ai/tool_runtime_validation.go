package ai

import (
	"context"
	"encoding/json"
	"errors"
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
	validatedOnce := false
	var validatedArgs any
	next := ToolValidationFunc(func(ctx context.Context, rawArgs json.RawMessage) (any, error) {
		for index, capability := range r.capabilities {
			hook, ok := capability.(BeforeToolValidationHook)
			if !ok {
				continue
			}
			var err error
			rawArgs, err = hook.BeforeToolValidation(ctx, r.capabilityInfo(index), hookContext.Clone(), slices.Clone(rawArgs))
			call.Args = slices.Clone(rawArgs)
			if err != nil {
				return nil, err
			}
		}
		call.Args = slices.Clone(rawArgs)
		var validated any
		var err error
		if validator := r.currentToolValidators[call.ToolName]; validator != nil {
			if validationErr := validator.ValidateJSON(rawArgs); validationErr != nil {
				err = &toolArgsSchemaValidationError{err: validationErr}
			}
		}
		if err == nil {
			validated, err = entry.validate(ctx, rc, rawArgs)
		}
		var failed *ToolFailedError
		if err != nil && !errors.As(err, &failed) {
			for index := len(r.capabilities) - 1; index >= 0; index-- {
				hook, ok := r.capabilities[index].(ToolValidationErrorHook)
				if !ok {
					continue
				}
				validated, err = hook.OnToolValidationError(
					ctx, r.capabilityInfo(index), hookContext.Clone(), slices.Clone(rawArgs), err,
				)
				if err == nil {
					break
				}
			}
		}
		if err != nil {
			return nil, err
		}
		if _, deferred := asToolHookDeferral(validated, nil); deferred {
			return nil, fmt.Errorf("tool validation error hooks cannot defer a call")
		}
		validatedOnce = true
		for index := len(r.capabilities) - 1; index >= 0; index-- {
			hook, ok := r.capabilities[index].(AfterToolValidationHook)
			if !ok {
				continue
			}
			previous := validated
			validated, err = hook.AfterToolValidation(ctx, r.capabilityInfo(index), hookContext.Clone(), validated)
			if err != nil {
				return nil, err
			}
			if deferred, ok := asToolHookDeferral(validated, previous); ok {
				return deferred, nil
			}
		}
		validatedArgs = validated
		return validated, nil
	})
	for index := len(r.capabilities) - 1; index >= 0; index-- {
		if wrapper, ok := r.capabilities[index].(ToolValidationWrapper); ok {
			innerNext := next
			info := r.capabilityInfo(index)
			next = func(ctx context.Context, rawArgs json.RawMessage) (any, error) {
				return wrapper.WrapToolValidation(
					ctx, info, hookContext.Clone(), slices.Clone(rawArgs), innerNext,
				)
			}
		}
	}
	validated, err := next(ctx, slices.Clone(call.Args))
	if err != nil {
		return call, nil, err
	}
	args := validated
	if deferred, ok := asToolHookDeferral(validated, validatedArgs); ok {
		if !validatedOnce {
			return call, nil, fmt.Errorf("tool validation wrappers cannot defer before validation")
		}
		args = deferred.args
		validated = deferred
	}
	if deferred, ok := validated.(toolHookDeferral); ok {
		args = deferred.args
	}
	call.Args, err = validatedToolArgsJSON(args)
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
