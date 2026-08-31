package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/Kludex/pydantic-ai-go/internal/schema"
)

// OutputFunction describes structured data produced by the model and a
// function that converts it into the agent's final output.
type OutputFunction[Deps, Output any] struct {
	name      string
	schema    map[string]any
	inputType reflect.Type
	decode    func([]byte) (any, error)
	process   func(context.Context, *RunContext[Deps], any) (Output, error)
}

// NewOutputFunction reflects Value's schema and registers fn as the final
// output processor. Concurrent exhaustive and streaming output processing may
// call fn concurrently, so fn must synchronize shared mutable state.
func NewOutputFunction[Deps, Value, Output any](
	name string,
	fn func(context.Context, *RunContext[Deps], Value) (Output, error),
) OutputFunction[Deps, Output] {
	if name == "" {
		panic("ai: output function name must not be empty")
	}
	if fn == nil {
		panic("ai: output function must not be nil")
	}
	valueSchema, err := schema.For(reflect.TypeFor[Value]())
	if err != nil {
		panic(fmt.Sprintf("ai: output function %q: %v", name, err))
	}
	return OutputFunction[Deps, Output]{
		name: name, schema: cloneSchemaMap(valueSchema), inputType: reflect.TypeFor[Value](),
		decode: func(raw []byte) (any, error) {
			var value Value
			if err := json.Unmarshal(raw, &value); err != nil {
				return nil, err
			}
			return value, nil
		},
		process: func(ctx context.Context, runContext *RunContext[Deps], value any) (Output, error) {
			typed, ok := value.(Value)
			if !ok {
				var zero Output
				return zero, fmt.Errorf("output function %q received %T, expected %v", name, value, reflect.TypeFor[Value]())
			}
			return fn(ctx, runContext, typed)
		},
	}
}

// Name returns the stable output-function name.
func (output OutputFunction[Deps, Output]) Name() string { return output.name }

// Schema returns a detached copy of the model-produced value's JSON Schema.
func (output OutputFunction[Deps, Output]) Schema() map[string]any {
	return cloneSchemaMap(output.schema)
}

// NewOutputFunctionAgent creates an agent that validates model output as the
// function's input type before calling the function. Function failures use the
// output retry and error handling pipeline.
func NewOutputFunctionAgent[Deps, Output any](
	model Model,
	output OutputFunction[Deps, Output],
	opts ...Option,
) *Agent[Deps, Output] {
	if output.name == "" || output.schema == nil || output.decode == nil || output.process == nil || output.inputType == nil {
		panic("ai: invalid output function")
	}
	agent := NewAgent[Deps, Output](model, opts...)
	agent.outputSchema = cloneSchemaMap(output.schema)
	agent.outputDecoder = func(raw []byte) (decodedOutput, error) {
		value, err := output.decode(raw)
		if err != nil {
			return decodedOutput{}, err
		}
		return decodedOutput{value: value}, nil
	}
	agent.outputProcessor = func(
		ctx context.Context, runContext *RunContext[Deps], value, _ any,
	) (Output, error) {
		return output.process(ctx, runContext, value)
	}
	agent.outputHasFunction = true
	agent.outputFunctionName = output.name
	agent.outputInputType = output.inputType
	agent.outputOverrideErr = ErrOutputTypeOverrideWithCustomOutput
	if agent.outputTool.Name == "" {
		agent.outputTool.Name = output.name
	}
	return agent
}
