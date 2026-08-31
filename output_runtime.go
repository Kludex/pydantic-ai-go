package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
)

type outputSchemaValidationError struct {
	err error
	raw []byte
}

func (err *outputSchemaValidationError) Error() string { return err.err.Error() }

type outputDecodeError struct{ err error }

func (err *outputDecodeError) Error() string { return err.err.Error() }
func (err *outputDecodeError) Unwrap() error { return err.err }

type outputValidationHookError struct{ err error }

func (err *outputValidationHookError) Error() string {
	return "ai: output validation: " + err.err.Error()
}
func (err *outputValidationHookError) Unwrap() error { return err.err }

type outputProcessingHookError struct{ err error }

func (err *outputProcessingHookError) Error() string {
	return "ai: output processing: " + err.err.Error()
}
func (err *outputProcessingHookError) Unwrap() error { return err.err }

func (r *run[Deps, Output]) outputHookContext(
	call *ToolCallPart, structured bool, partial bool,
) OutputHookContext {
	mode := OutputHookModeText
	if call != nil {
		mode = OutputHookModeTool
	} else if structured {
		switch r.params.OutputMode {
		case OutputModeNative:
			mode = OutputHookModeNative
		case OutputModePrompted:
			mode = OutputHookModePrompted
		}
	}
	context := OutputHookContext{
		Mode: mode, OutputType: reflect.TypeFor[Output](), Schema: r.params.OutputSchema,
		AllowsText: r.params.AllowText, Structured: structured, Partial: partial,
	}
	if call != nil {
		cloned := *call
		context.ToolCall = &cloned
		if r.currentOutputTool != nil {
			definition := cloneToolDefinition(*r.currentOutputTool)
			context.ToolDefinition = &definition
			context.Schema = definition.Schema
		}
	}
	return context.Clone()
}

func (r *run[Deps, Output]) validateAndProcessOutput(
	ctx context.Context,
	runContext *RunContext[Deps],
	hookContext OutputHookContext,
	rawOutput any,
	decode func(any) (Output, error),
) (Output, error) {
	var output Output
	var err error
	if hookContext.Structured {
		output, err = r.validateOutputWithHooks(ctx, hookContext, rawOutput, decode)
	} else {
		output, err = decode(rawOutput)
	}
	if err != nil {
		return output, &outputValidationHookError{err: err}
	}
	output, err = r.processOutputWithHooks(ctx, runContext, hookContext, output)
	if err != nil {
		return output, &outputProcessingHookError{err: err}
	}
	return output, nil
}

func (r *run[Deps, Output]) validateOutputWithHooks(
	ctx context.Context,
	hookContext OutputHookContext,
	rawOutput any,
	decode func(any) (Output, error),
) (Output, error) {
	var zero Output
	rawOutput = cloneRawOutput(rawOutput)
	for _, capability := range r.capabilities {
		hook, ok := capability.(BeforeOutputValidationHook)
		if !ok {
			continue
		}
		var err error
		rawOutput, err = hook.BeforeOutputValidation(
			ctx, r.info, hookContext.Clone(), cloneRawOutput(rawOutput),
		)
		rawOutput = cloneRawOutput(rawOutput)
		if err != nil {
			return zero, err
		}
	}
	next := OutputValidationFunc(func(_ context.Context, rawOutput any) (any, error) {
		output, err := decode(rawOutput)
		if err != nil {
			return nil, err
		}
		return output, nil
	})
	for index := len(r.capabilities) - 1; index >= 0; index-- {
		if wrapper, ok := r.capabilities[index].(OutputValidationWrapper); ok {
			innerNext := next
			next = func(ctx context.Context, rawOutput any) (any, error) {
				return wrapper.WrapOutputValidation(
					ctx, r.info, hookContext.Clone(), cloneRawOutput(rawOutput), innerNext,
				)
			}
		}
	}
	validated, err := next(ctx, rawOutput)
	if err != nil && !hookContext.Partial {
		for index := len(r.capabilities) - 1; index >= 0; index-- {
			hook, ok := r.capabilities[index].(OutputValidationErrorHook)
			if !ok {
				continue
			}
			validated, err = hook.OnOutputValidationError(
				ctx, r.info, hookContext.Clone(), cloneRawOutput(rawOutput), err,
			)
			if err == nil {
				break
			}
		}
	}
	if err != nil {
		return zero, err
	}
	for index := len(r.capabilities) - 1; index >= 0; index-- {
		hook, ok := r.capabilities[index].(AfterOutputValidationHook)
		if !ok {
			continue
		}
		validated, err = hook.AfterOutputValidation(ctx, r.info, hookContext.Clone(), validated)
		if err != nil {
			return zero, err
		}
	}
	output, ok := validated.(Output)
	if !ok {
		return zero, fmt.Errorf("validated output has type %T, expected %v", validated, reflect.TypeFor[Output]())
	}
	return output, nil
}

func (r *run[Deps, Output]) processOutputWithHooks(
	ctx context.Context,
	runContext *RunContext[Deps],
	hookContext OutputHookContext,
	output Output,
) (Output, error) {
	var zero Output
	processed := any(output)
	var err error
	for _, capability := range r.capabilities {
		hook, ok := capability.(BeforeOutputProcessingHook)
		if !ok {
			continue
		}
		processed, err = hook.BeforeOutputProcessing(ctx, r.info, hookContext.Clone(), processed)
		if err != nil {
			return zero, err
		}
	}
	next := OutputProcessingFunc(func(_ context.Context, processed any) (any, error) {
		value, ok := processed.(Output)
		if !ok {
			return nil, fmt.Errorf(
				"output processing input has type %T, expected %v", processed, reflect.TypeFor[Output](),
			)
		}
		for _, validate := range r.agent.outputValidators {
			if err := validate(ctx, runContext, value); err != nil {
				return value, err
			}
		}
		return value, nil
	})
	for index := len(r.capabilities) - 1; index >= 0; index-- {
		if wrapper, ok := r.capabilities[index].(OutputProcessingWrapper); ok {
			innerNext := next
			next = func(ctx context.Context, output any) (any, error) {
				return wrapper.WrapOutputProcessing(ctx, r.info, hookContext.Clone(), output, innerNext)
			}
		}
	}
	processed, err = next(ctx, processed)
	var retry *RetryError
	if err != nil && !errors.As(err, &retry) {
		for index := len(r.capabilities) - 1; index >= 0; index-- {
			hook, ok := r.capabilities[index].(OutputProcessingErrorHook)
			if !ok {
				continue
			}
			processed, err = hook.OnOutputProcessingError(
				ctx, r.info, hookContext.Clone(), processed, err,
			)
			if err == nil {
				break
			}
		}
	}
	if err != nil {
		return zero, err
	}
	for index := len(r.capabilities) - 1; index >= 0; index-- {
		hook, ok := r.capabilities[index].(AfterOutputProcessingHook)
		if !ok {
			continue
		}
		processed, err = hook.AfterOutputProcessing(ctx, r.info, hookContext.Clone(), processed)
		if err != nil {
			return zero, err
		}
	}
	final, ok := processed.(Output)
	if !ok {
		return zero, fmt.Errorf("processed output has type %T, expected %v", processed, reflect.TypeFor[Output]())
	}
	return final, nil
}

func (r *run[Deps, Output]) decodeOutput(rawOutput any, structured bool) (Output, error) {
	var output Output
	if !structured {
		return rawOutput.(Output), nil
	}
	raw, err := outputBytes(rawOutput)
	if err != nil {
		return output, err
	}
	if r.currentOutputValidator != nil {
		if err := r.currentOutputValidator.ValidateJSON(raw); err != nil {
			return output, &outputSchemaValidationError{err: err, raw: append([]byte(nil), raw...)}
		}
	}
	if err := json.Unmarshal(raw, &output); err != nil {
		return output, &outputDecodeError{err: err}
	}
	return output, nil
}

func cloneRawOutput(rawOutput any) any {
	switch raw := rawOutput.(type) {
	case json.RawMessage:
		return append(json.RawMessage(nil), raw...)
	case []byte:
		return append([]byte(nil), raw...)
	case map[string]any:
		return cloneSchemaMap(raw)
	default:
		return rawOutput
	}
}

func outputBytes(rawOutput any) ([]byte, error) {
	switch raw := rawOutput.(type) {
	case string:
		return []byte(raw), nil
	case json.RawMessage:
		return append([]byte(nil), raw...), nil
	case []byte:
		return append([]byte(nil), raw...), nil
	case map[string]any:
		return json.Marshal(raw)
	default:
		return nil, fmt.Errorf(
			"raw structured output has type %T, expected string, JSON bytes, or map[string]any", rawOutput,
		)
	}
}
