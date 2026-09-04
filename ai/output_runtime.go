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

type decodedOutput struct {
	value any
	state any
}

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
	outputType := reflect.TypeFor[Output]()
	if r.agent.outputInputType != nil {
		outputType = r.agent.outputInputType
	}
	context := OutputHookContext{
		Mode: mode, OutputType: outputType, Schema: r.params.OutputSchema,
		AllowsText: r.params.AllowText, AllowsImage: r.params.AllowImageOutput,
		Structured: structured, Partial: partial,
		HasFunction: r.agent.outputHasFunction, FunctionName: r.agent.outputFunctionName,
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
	decode func(any) (decodedOutput, error),
) (Output, error) {
	var candidate decodedOutput
	var err error
	if hookContext.Structured {
		candidate, err = r.validateOutputWithHooks(ctx, hookContext, rawOutput, decode)
	} else {
		candidate, err = decode(rawOutput)
	}
	if err != nil {
		var zero Output
		return zero, &outputValidationHookError{err: err}
	}
	output, err := r.processOutputWithHooks(ctx, runContext, hookContext, candidate)
	if err != nil {
		return output, &outputProcessingHookError{err: err}
	}
	return output, nil
}

func (r *run[Deps, Output]) validateOutputWithHooks(
	ctx context.Context,
	hookContext OutputHookContext,
	rawOutput any,
	decode func(any) (decodedOutput, error),
) (decodedOutput, error) {
	var zero decodedOutput
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
	var state any
	next := OutputValidationFunc(func(_ context.Context, rawOutput any) (any, error) {
		output, err := decode(rawOutput)
		if err != nil {
			return nil, err
		}
		state = output.state
		return output.value, nil
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
	if r.agent.outputProcessor == nil {
		if _, ok := validated.(Output); !ok {
			return zero, fmt.Errorf("validated output has type %T, expected %v", validated, reflect.TypeFor[Output]())
		}
	}
	return decodedOutput{value: validated, state: state}, nil
}

func (r *run[Deps, Output]) processOutputWithHooks(
	ctx context.Context,
	runContext *RunContext[Deps],
	hookContext OutputHookContext,
	candidate decodedOutput,
) (Output, error) {
	var zero Output
	processed := candidate.value
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
	next := OutputProcessingFunc(func(processingCtx context.Context, processed any) (any, error) {
		var value Output
		var err error
		if r.agent.outputProcessor != nil {
			value, err = r.agent.outputProcessor(processingCtx, runContext, processed, candidate.state)
			if err != nil {
				return value, err
			}
		} else {
			var ok bool
			value, ok = processed.(Output)
			if !ok {
				return nil, fmt.Errorf(
					"output processing input has type %T, expected %v", processed, reflect.TypeFor[Output](),
				)
			}
		}
		for _, validate := range r.agent.outputValidators {
			if err := validate(processingCtx, runContext, value); err != nil {
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

func (r *run[Deps, Output]) decodeOutput(rawOutput any, structured bool) (decodedOutput, error) {
	if !structured {
		if r.agent.outputProcessor != nil {
			return decodedOutput{value: rawOutput}, nil
		}
		return decodedOutput{value: rawOutput.(Output)}, nil
	}
	raw, err := outputBytes(rawOutput)
	if err != nil {
		return decodedOutput{}, err
	}
	if r.currentOutputValidator != nil {
		if err := r.currentOutputValidator.ValidateJSON(raw); err != nil {
			return decodedOutput{}, &outputSchemaValidationError{err: err, raw: append([]byte(nil), raw...)}
		}
	}
	return r.decodeOutputBytes(raw)
}

func (r *run[Deps, Output]) decodeOutputBytes(raw []byte) (decodedOutput, error) {
	if r.agent.outputDecoder != nil {
		decoded, err := r.agent.outputDecoder(raw)
		if err != nil {
			return decodedOutput{}, &outputDecodeError{err: err}
		}
		return decoded, nil
	}
	var output Output
	if err := json.Unmarshal(raw, &output); err != nil {
		return decodedOutput{}, &outputDecodeError{err: err}
	}
	return decodedOutput{value: output}, nil
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
