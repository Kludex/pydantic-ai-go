package ai

import (
	"context"
	"reflect"
)

// OutputHookMode identifies how the current output candidate was delivered.
type OutputHookMode string

const (
	// OutputHookModeText identifies unstructured text output.
	OutputHookModeText OutputHookMode = "text"
	// OutputHookModeTool identifies arguments from an output tool call.
	OutputHookModeTool OutputHookMode = "tool"
	// OutputHookModeNative identifies provider-native structured text.
	OutputHookModeNative OutputHookMode = "native"
	// OutputHookModePrompted identifies structured text requested through instructions.
	OutputHookModePrompted OutputHookMode = "prompted"
)

// OutputHookContext describes one final or partial output candidate.
type OutputHookContext struct {
	Mode           OutputHookMode
	OutputType     reflect.Type
	Schema         map[string]any
	ToolCall       *ToolCallPart
	ToolDefinition *ToolDefinition
	AllowsText     bool
	Structured     bool
	Partial        bool
}

// Clone returns a context detached from schemas, tool calls, and definitions.
func (hook OutputHookContext) Clone() OutputHookContext {
	hook.Schema = cloneSchemaMap(hook.Schema)
	if hook.ToolCall != nil {
		call := *hook.ToolCall
		call.Args = append([]byte(nil), call.Args...)
		call.ProviderDetails = cloneSchemaMap(call.ProviderDetails)
		hook.ToolCall = &call
	}
	if hook.ToolDefinition != nil {
		definition := cloneToolDefinition(*hook.ToolDefinition)
		hook.ToolDefinition = &definition
	}
	return hook
}

// OutputValidationFunc continues a structured-output parsing wrapper.
type OutputValidationFunc func(ctx context.Context, rawOutput any) (any, error)

// BeforeOutputValidationHook modifies raw structured output before parsing.
// Raw output is a string, json.RawMessage, []byte, or map[string]any. It runs
// for tool, native, prompted, and partial structured outputs, but not plain text.
type BeforeOutputValidationHook interface {
	BeforeOutputValidation(
		ctx context.Context, ri *RunInfo, hook OutputHookContext, rawOutput any,
	) (any, error)
}

// AfterOutputValidationHook modifies a successfully parsed semantic output value.
type AfterOutputValidationHook interface {
	AfterOutputValidation(ctx context.Context, ri *RunInfo, hook OutputHookContext, output any) (any, error)
}

// OutputValidationErrorHook may replace a structured parsing or schema error.
type OutputValidationErrorHook interface {
	OnOutputValidationError(
		ctx context.Context, ri *RunInfo, hook OutputHookContext, rawOutput any, validationErr error,
	) (any, error)
}

// OutputValidationWrapper wraps structured parsing and schema validation.
type OutputValidationWrapper interface {
	WrapOutputValidation(
		ctx context.Context, ri *RunInfo, hook OutputHookContext, rawOutput any, next OutputValidationFunc,
	) (any, error)
}

// BeforeOutputValidationFunc adapts a function into a validation hook capability.
type BeforeOutputValidationFunc func(
	ctx context.Context, ri *RunInfo, hook OutputHookContext, rawOutput any,
) (any, error)

// Setup implements Capability.
func (BeforeOutputValidationFunc) Setup(*CapabilityRegistry) error { return nil }

// BeforeOutputValidation calls the adapted function.
func (fn BeforeOutputValidationFunc) BeforeOutputValidation(
	ctx context.Context, ri *RunInfo, hook OutputHookContext, rawOutput any,
) (any, error) {
	return fn(ctx, ri, hook, rawOutput)
}

// AfterOutputValidationFunc adapts a function into a post-validation hook capability.
type AfterOutputValidationFunc func(
	ctx context.Context, ri *RunInfo, hook OutputHookContext, output any,
) (any, error)

// Setup implements Capability.
func (AfterOutputValidationFunc) Setup(*CapabilityRegistry) error { return nil }

// AfterOutputValidation calls the adapted function.
func (fn AfterOutputValidationFunc) AfterOutputValidation(
	ctx context.Context, ri *RunInfo, hook OutputHookContext, output any,
) (any, error) {
	return fn(ctx, ri, hook, output)
}

// OutputValidationErrorFunc adapts an output validation recovery function into a capability.
type OutputValidationErrorFunc func(
	ctx context.Context, ri *RunInfo, hook OutputHookContext, rawOutput any, validationErr error,
) (any, error)

// Setup implements Capability.
func (OutputValidationErrorFunc) Setup(*CapabilityRegistry) error { return nil }

// OnOutputValidationError calls the adapted function.
func (fn OutputValidationErrorFunc) OnOutputValidationError(
	ctx context.Context, ri *RunInfo, hook OutputHookContext, rawOutput any, validationErr error,
) (any, error) {
	return fn(ctx, ri, hook, rawOutput, validationErr)
}
