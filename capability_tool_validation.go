package ai

import (
	"context"
	"encoding/json"
	"slices"
)

// ToolHookContext identifies one prepared function-tool call.
type ToolHookContext struct {
	Call         ToolCallPart
	Definition   ToolDefinition
	Approved     bool
	CallMetadata map[string]any
}

// Clone returns a context detached from call arguments, provider details, and the definition.
func (hook ToolHookContext) Clone() ToolHookContext {
	hook.Call.Args = slices.Clone(hook.Call.Args)
	hook.Call.ProviderDetails = cloneSchemaMap(hook.Call.ProviderDetails)
	hook.Definition = cloneToolDefinition(hook.Definition)
	hook.CallMetadata = cloneSchemaMap(hook.CallMetadata)
	return hook
}

// ToolValidationFunc continues a tool-argument validation wrapper.
type ToolValidationFunc func(ctx context.Context, rawArgs json.RawMessage) (any, error)

// BeforeToolValidationHook modifies raw JSON before schema, decoding, and semantic validation.
// Independent calls invoke hooks concurrently.
type BeforeToolValidationHook interface {
	BeforeToolValidation(
		ctx context.Context, ri *RunInfo, hook ToolHookContext, rawArgs json.RawMessage,
	) (json.RawMessage, error)
}

// AfterToolValidationHook modifies successfully decoded and validated arguments.
// Return RequestToolApproval or RequestExternalToolExecution to defer the validated call.
type AfterToolValidationHook interface {
	AfterToolValidation(ctx context.Context, ri *RunInfo, hook ToolHookContext, args any) (any, error)
}

// ToolValidationErrorHook may replace a core or wrapper validation error with validated arguments.
// Errors returned directly by before and after hooks bypass error hooks.
type ToolValidationErrorHook interface {
	OnToolValidationError(
		ctx context.Context, ri *RunInfo, hook ToolHookContext, rawArgs json.RawMessage, validationErr error,
	) (any, error)
}

// ToolValidationWrapper wraps schema, decoding, and semantic argument validation.
type ToolValidationWrapper interface {
	WrapToolValidation(
		ctx context.Context, ri *RunInfo, hook ToolHookContext, rawArgs json.RawMessage, next ToolValidationFunc,
	) (any, error)
}

// BeforeToolValidationFunc adapts a function into a validation hook capability.
type BeforeToolValidationFunc func(
	ctx context.Context, ri *RunInfo, hook ToolHookContext, rawArgs json.RawMessage,
) (json.RawMessage, error)

// Setup implements Capability.
func (BeforeToolValidationFunc) Setup(*CapabilityRegistry) error { return nil }

// BeforeToolValidation calls the adapted function.
func (fn BeforeToolValidationFunc) BeforeToolValidation(
	ctx context.Context, ri *RunInfo, hook ToolHookContext, rawArgs json.RawMessage,
) (json.RawMessage, error) {
	return fn(ctx, ri, hook, rawArgs)
}

// AfterToolValidationFunc adapts a function into a post-validation hook capability.
type AfterToolValidationFunc func(
	ctx context.Context, ri *RunInfo, hook ToolHookContext, args any,
) (any, error)

// Setup implements Capability.
func (AfterToolValidationFunc) Setup(*CapabilityRegistry) error { return nil }

// AfterToolValidation calls the adapted function.
func (fn AfterToolValidationFunc) AfterToolValidation(
	ctx context.Context, ri *RunInfo, hook ToolHookContext, args any,
) (any, error) {
	return fn(ctx, ri, hook, args)
}

// ToolValidationErrorFunc adapts a validation recovery function into a capability.
type ToolValidationErrorFunc func(
	ctx context.Context, ri *RunInfo, hook ToolHookContext, rawArgs json.RawMessage, validationErr error,
) (any, error)

// Setup implements Capability.
func (ToolValidationErrorFunc) Setup(*CapabilityRegistry) error { return nil }

// OnToolValidationError calls the adapted function.
func (fn ToolValidationErrorFunc) OnToolValidationError(
	ctx context.Context, ri *RunInfo, hook ToolHookContext, rawArgs json.RawMessage, validationErr error,
) (any, error) {
	return fn(ctx, ri, hook, rawArgs, validationErr)
}
