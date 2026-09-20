package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
)

// Choice describes one runtime output option and the value it resolves to.
type Choice[Output any] struct {
	// Key is the string the model selects.
	Key string
	// Description explains this option to the model.
	Description string
	// Value is returned when the model selects this option and Action is nil.
	Value Output
	// Action runs once after the final choice. It may return Retryf to ask the model again.
	Action func(context.Context) (Output, error)
}

// Choices is a detached runtime set of described output options.
type Choices[Output any] struct {
	name        string
	description string
	choices     map[string]Choice[Output]
	order       []string
	schema      map[string]any
}

// NewChoices creates a runtime output set. Keys must be unique and non-empty.
func NewChoices[Output any](name, description string, choices ...Choice[Output]) Choices[Output] {
	if name == "" {
		name = "choices"
	}
	if len(choices) == 0 {
		panic("ai: choices requires at least one option")
	}
	result := Choices[Output]{
		name: name, description: description, choices: make(map[string]Choice[Output], len(choices)),
		order: make([]string, 0, len(choices)),
	}
	options := make([]any, 0, len(choices))
	for _, choice := range choices {
		if choice.Key == "" {
			panic("ai: choice key must not be empty")
		}
		if _, exists := result.choices[choice.Key]; exists {
			panic(fmt.Sprintf("ai: duplicate choice key %q", choice.Key))
		}
		result.choices[choice.Key] = choice
		result.order = append(result.order, choice.Key)
		option := map[string]any{"const": choice.Key}
		if choice.Description != "" {
			option["description"] = choice.Description
		}
		options = append(options, option)
	}
	result.schema = map[string]any{"type": "string", "anyOf": options}
	if description != "" {
		result.schema["description"] = description
	}
	return result
}

// NewStringChoices creates options that resolve to their own keys.
func NewStringChoices(name, description string, choices ...string) Choices[string] {
	options := make([]Choice[string], len(choices))
	for index, choice := range choices {
		options[index] = Choice[string]{Key: choice, Value: choice}
	}
	return NewChoices(name, description, options...)
}

// Schema returns the detached JSON Schema sent to the model.
func (choices Choices[Output]) Schema() map[string]any { return cloneSchemaMap(choices.schema) }

// NewChoicesAgent creates an agent that resolves the model's selected key.
func NewChoicesAgent[Deps, Output any](
	model Model, choices Choices[Output], opts ...Option,
) *Agent[Deps, Output] {
	if len(choices.choices) == 0 || choices.schema == nil {
		panic("ai: invalid choices")
	}
	agent := NewAgent[Deps, Output](model, opts...)
	agent.outputSchema = cloneSchemaMap(choices.schema)
	agent.outputDecoder = func(raw []byte) (decodedOutput, error) {
		var key string
		if err := json.Unmarshal(raw, &key); err != nil {
			return decodedOutput{}, err
		}
		choice, exists := choices.choices[key]
		if !exists {
			return decodedOutput{}, fmt.Errorf("unknown choice %q", key)
		}
		return decodedOutput{value: choice}, nil
	}
	agent.outputProcessor = func(
		ctx context.Context, _ *RunContext[Deps], value, _ any,
	) (Output, error) {
		choice := value.(Choice[Output])
		if choice.Action != nil {
			return choice.Action(ctx)
		}
		return choice.Value, nil
	}
	agent.outputHasFunction = true
	agent.outputFunctionName = choices.name
	agent.outputInputType = reflect.TypeFor[string]()
	agent.outputOverrideErr = ErrOutputTypeOverrideWithCustomOutput
	agent.outputTool.Name = choices.name
	return agent
}
