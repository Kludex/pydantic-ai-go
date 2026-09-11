package ai

import (
	"bytes"
	"context"
	"errors"
)

type toolHookDeferral struct {
	args     any
	approval *ToolApprovalRequest
	external *ExternalToolRequest
}

func asToolHookDeferral(value any, args any) (toolHookDeferral, bool) {
	switch request := value.(type) {
	case ToolApprovalRequest:
		return toolHookDeferral{args: args, approval: &request}, true
	case *ToolApprovalRequest:
		return toolHookDeferral{args: args, approval: request}, request != nil
	case ExternalToolRequest:
		return toolHookDeferral{args: args, external: &request}, true
	case *ExternalToolRequest:
		return toolHookDeferral{args: args, external: request}, request != nil
	default:
		return toolHookDeferral{}, false
	}
}

func toolHookDeferredRequest(call ToolCallPart, deferred toolHookDeferral) deferredRequestPart {
	if deferred.approval != nil {
		return deferredRequestPart{
			call: call, kind: deferredCallApproval, metadata: cloneSchemaMap(deferred.approval.Metadata),
		}
	}
	return deferredRequestPart{
		call: call, kind: deferredCallExternal, metadata: cloneSchemaMap(deferred.external.Metadata),
	}
}

// callTool is the legacy middleware point around local execution.
// Wrapper-modified arguments are validated again before execution.
func (r *run[Deps, Output]) callTool(
	ctx context.Context, rc *RunContext[Deps], entry toolEntry[Deps], call ToolCallPart, validated any,
) (any, error) {
	validatedArgs := append([]byte(nil), call.Args...)
	next := ToolCallFunc(func(ctx context.Context, wrappedCall ToolCallPart) (any, error) {
		args := validated
		if !bytes.Equal(wrappedCall.Args, validatedArgs) {
			var err error
			wrappedCall, args, err = r.validateToolCall(ctx, rc, entry, wrappedCall)
			if err != nil {
				var schemaValidation *toolArgsSchemaValidationError
				if errors.As(err, &schemaValidation) {
					return nil, Retryf("invalid arguments for tool %q: %v", call.ToolName, schemaValidation.err)
				}
				return nil, err
			}
		}
		return r.executeTool(ctx, rc, entry, wrappedCall, args)
	})
	for index := len(r.capabilities) - 1; index >= 0; index-- {
		if wrapper, ok := r.capabilities[index].(ToolCallWrapper); ok {
			innerNext := next
			info := r.capabilityInfo(index)
			next = func(ctx context.Context, call ToolCallPart) (any, error) {
				return wrapper.WrapToolCall(ctx, info, call, innerNext)
			}
		}
	}
	return next(ctx, call)
}

func (r *run[Deps, Output]) executeTool(
	ctx context.Context, rc *RunContext[Deps], entry toolEntry[Deps], call ToolCallPart, args any,
) (any, error) {
	hookContext := ToolHookContext{
		Call: call, Definition: entry.def, Approved: rc.ToolCallApproved,
		CallMetadata: cloneSchemaMap(rc.ToolCallMetadata),
	}.Clone()
	var err error
	for index, capability := range r.capabilities {
		hook, ok := capability.(BeforeToolExecutionHook)
		if !ok {
			continue
		}
		previous := args
		args, err = hook.BeforeToolExecution(ctx, r.capabilityInfo(index), hookContext.Clone(), args)
		if err != nil {
			return nil, err
		}
		if deferred, ok := asToolHookDeferral(args, previous); ok {
			return deferred, nil
		}
	}
	next := ToolExecutionFunc(func(ctx context.Context, args any) (any, error) {
		return entry.execute(ctx, rc, args)
	})
	for index := len(r.capabilities) - 1; index >= 0; index-- {
		if wrapper, ok := r.capabilities[index].(ToolExecutionWrapper); ok {
			innerNext := next
			info := r.capabilityInfo(index)
			next = func(ctx context.Context, args any) (any, error) {
				return wrapper.WrapToolExecution(ctx, info, hookContext.Clone(), args, innerNext)
			}
		}
	}
	result, err := next(ctx, args)
	var failed *ToolFailedError
	var retry *RetryError
	if err != nil && !errors.As(err, &failed) && !errors.As(err, &retry) &&
		!errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
		for index := len(r.capabilities) - 1; index >= 0; index-- {
			hook, ok := r.capabilities[index].(ToolExecutionErrorHook)
			if !ok {
				continue
			}
			result, err = hook.OnToolExecutionError(ctx, r.capabilityInfo(index), hookContext.Clone(), args, err)
			if err == nil {
				break
			}
		}
	}
	if err != nil {
		return result, err
	}
	for index := len(r.capabilities) - 1; index >= 0; index-- {
		hook, ok := r.capabilities[index].(AfterToolExecutionHook)
		if !ok {
			continue
		}
		result, err = hook.AfterToolExecution(ctx, r.capabilityInfo(index), hookContext.Clone(), args, result)
		if err != nil {
			return result, err
		}
		if deferred, ok := asToolHookDeferral(result, args); ok {
			return deferred, nil
		}
	}
	return result, nil
}
