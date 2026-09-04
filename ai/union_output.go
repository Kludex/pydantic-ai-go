package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/Kludex/pydantic-ai-go/ai/internal/schema"
)

// OutputAlternative describes one member of a UnionOutput. Kind is the stable
// discriminator sent to the model.
type OutputAlternative[Output any] struct {
	kind        string
	schema      map[string]any
	decode      func(json.RawMessage) (any, error)
	process     func(any) (Output, error)
	inputType   reflect.Type
	hasFunction bool
}

// NewOutputAlternative reflects Value's schema and converts a decoded Value to
// Output. Kind must be unique within the union.
func NewOutputAlternative[Output, Value any](
	kind string, convert func(Value) Output,
) OutputAlternative[Output] {
	if kind == "" {
		panic("ai: output alternative kind must not be empty")
	}
	if convert == nil {
		panic("ai: output alternative converter must not be nil")
	}
	valueSchema, err := schema.For(reflect.TypeFor[Value]())
	if err != nil {
		panic(fmt.Sprintf("ai: output alternative %q: %v", kind, err))
	}
	return newOutputAlternative(kind, valueSchema, func(value Value) (Output, error) {
		return convert(value), nil
	})
}

// NewOutputAlternativeFunc reflects Value's schema and converts a decoded
// Value with a function that may return an error or request a retry.
func NewOutputAlternativeFunc[Output, Value any](
	kind string, convert func(Value) (Output, error),
) OutputAlternative[Output] {
	if kind == "" {
		panic("ai: output alternative kind must not be empty")
	}
	if convert == nil {
		panic("ai: output alternative converter must not be nil")
	}
	valueSchema, err := schema.For(reflect.TypeFor[Value]())
	if err != nil {
		panic(fmt.Sprintf("ai: output alternative %q: %v", kind, err))
	}
	return newOutputAlternative(kind, valueSchema, convert)
}

func newOutputAlternative[Output, Value any](
	kind string, valueSchema map[string]any, convert func(Value) (Output, error),
) OutputAlternative[Output] {
	return OutputAlternative[Output]{
		kind: kind, schema: cloneSchemaMap(valueSchema), inputType: reflect.TypeFor[Value](), hasFunction: true,
		decode: func(raw json.RawMessage) (any, error) {
			var value Value
			if err := json.Unmarshal(raw, &value); err != nil {
				return nil, err
			}
			return value, nil
		},
		process: func(value any) (Output, error) {
			typed, ok := value.(Value)
			if !ok {
				var zero Output
				return zero, fmt.Errorf("output alternative %q received %T, expected %v", kind, value, reflect.TypeFor[Value]())
			}
			return convert(typed)
		},
	}
}

// NewRawOutputAlternative creates an alternative from an explicit JSON Schema
// and decoder. The schema is copied before it is stored.
func NewRawOutputAlternative[Output any](
	kind string, valueSchema map[string]any, decode func(json.RawMessage) (Output, error),
) OutputAlternative[Output] {
	if kind == "" {
		panic("ai: output alternative kind must not be empty")
	}
	if valueSchema == nil {
		panic("ai: output alternative schema must not be nil")
	}
	if decode == nil {
		panic("ai: output alternative decoder must not be nil")
	}
	return OutputAlternative[Output]{
		kind: kind, schema: cloneSchemaMap(valueSchema), inputType: reflect.TypeFor[Output](),
		decode: func(raw json.RawMessage) (any, error) { return decode(raw) },
		process: func(value any) (Output, error) {
			output, ok := value.(Output)
			if !ok {
				var zero Output
				return zero, fmt.Errorf("raw output alternative %q decoded %T, expected %v", kind, value, reflect.TypeFor[Output]())
			}
			return output, nil
		},
	}
}

// UnionOutput is a discriminated structured-output specification. It uses one
// portable envelope in every structured-output mode:
// {"result":{"kind":"...","data":...}}.
type UnionOutput[Output any] struct {
	schema       map[string]any
	alternatives map[string]OutputAlternative[Output]
}

// NewUnionOutput creates a detached union specification. At least two
// alternatives are required.
func NewUnionOutput[Output any](alternatives ...OutputAlternative[Output]) UnionOutput[Output] {
	if len(alternatives) < 2 {
		panic("ai: union output requires at least two alternatives")
	}
	union := UnionOutput[Output]{alternatives: make(map[string]OutputAlternative[Output], len(alternatives))}
	variants := make([]any, 0, len(alternatives))
	definitions := map[string]any{}
	for index, alternative := range alternatives {
		if alternative.kind == "" || alternative.decode == nil || alternative.process == nil || alternative.schema == nil {
			panic("ai: invalid output alternative")
		}
		if _, duplicate := union.alternatives[alternative.kind]; duplicate {
			panic(fmt.Sprintf("ai: duplicate output alternative kind %q", alternative.kind))
		}
		alternative.schema = cloneSchemaMap(alternative.schema)
		preparedSchema := prepareUnionAlternativeSchema(alternative.schema, index, definitions)
		union.alternatives[alternative.kind] = alternative
		variants = append(variants, map[string]any{
			"type": "object",
			"properties": map[string]any{
				"kind": map[string]any{"type": "string", "const": alternative.kind},
				"data": preparedSchema,
			},
			"required":             []any{"kind", "data"},
			"additionalProperties": false,
		})
	}
	union.schema = map[string]any{
		"type": "object",
		"properties": map[string]any{
			"result": map[string]any{"anyOf": variants},
		},
		"required":             []any{"result"},
		"additionalProperties": false,
	}
	if len(definitions) > 0 {
		union.schema["$defs"] = definitions
	}
	return union
}

// Schema returns a detached copy of the discriminated union's JSON Schema.
func (output UnionOutput[Output]) Schema() map[string]any { return cloneSchemaMap(output.schema) }

// Decode resolves the union discriminator and decodes its selected value. An
// agent applies full JSON Schema validation before calling Decode.
func (output UnionOutput[Output]) Decode(raw []byte) (Output, error) {
	candidate, err := output.decodeCandidate(raw)
	if err != nil {
		var zero Output
		return zero, err
	}
	return output.processCandidate(candidate.value, candidate.state)
}

// NewUnionAgent creates an agent whose Output can be one of the registered
// alternatives. Output validators receive the converted union value.
func NewUnionAgent[Deps, Output any](
	model Model, output UnionOutput[Output], opts ...Option,
) *Agent[Deps, Output] {
	if output.schema == nil || len(output.alternatives) < 2 {
		panic("ai: invalid union output")
	}
	agent := NewAgent[Deps, Output](model, opts...)
	agent.outputSchema = cloneSchemaMap(output.schema)
	alternatives := make(map[string]OutputAlternative[Output], len(output.alternatives))
	for kind, alternative := range output.alternatives {
		alternative.schema = cloneSchemaMap(alternative.schema)
		alternatives[kind] = alternative
	}
	union := UnionOutput[Output]{schema: cloneSchemaMap(output.schema), alternatives: alternatives}
	agent.outputDecoder = union.decodeCandidate
	agent.outputProcessor = func(_ context.Context, _ *RunContext[Deps], value, state any) (Output, error) {
		return union.processCandidate(value, state)
	}
	for _, alternative := range alternatives {
		agent.outputHasFunction = agent.outputHasFunction || alternative.hasFunction
	}
	agent.outputOverrideErr = ErrOutputTypeOverrideWithUnion
	return agent
}

func (output UnionOutput[Output]) decodeCandidate(raw []byte) (decodedOutput, error) {
	var envelope struct {
		Result struct {
			Kind string          `json:"kind"`
			Data json.RawMessage `json:"data"`
		} `json:"result"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return decodedOutput{}, err
	}
	alternative, ok := output.alternatives[envelope.Result.Kind]
	if !ok {
		return decodedOutput{}, fmt.Errorf("unknown output alternative kind %q", envelope.Result.Kind)
	}
	value, err := alternative.decode(envelope.Result.Data)
	if err != nil {
		return decodedOutput{}, err
	}
	return decodedOutput{value: value, state: alternative}, nil
}

func (output UnionOutput[Output]) processCandidate(value, state any) (Output, error) {
	original, hasOriginal := state.(OutputAlternative[Output])
	if hasOriginal && original.accepts(value) {
		return original.process(value)
	}
	var matched OutputAlternative[Output]
	matches := 0
	for _, alternative := range output.alternatives {
		if alternative.accepts(value) {
			matched = alternative
			matches++
		}
	}
	if matches == 1 {
		return matched.process(value)
	}
	if hasOriginal {
		return original.process(value)
	}
	var zero Output
	return zero, fmt.Errorf("union output cannot select an alternative for %T", value)
}

func (output OutputAlternative[Output]) accepts(value any) bool {
	if output.inputType == nil || value == nil {
		return false
	}
	return reflect.TypeOf(value).AssignableTo(output.inputType)
}

func prepareUnionAlternativeSchema(value map[string]any, index int, definitions map[string]any) map[string]any {
	value = cloneSchemaMap(value)
	prefix := fmt.Sprintf("#/properties/result/anyOf/%d/properties/data", index)
	rawDefinitions, _ := value["$defs"].(map[string]any)
	delete(value, "$defs")
	replacements := make(map[string]string, len(rawDefinitions))
	for name := range rawDefinitions {
		newName := fmt.Sprintf("alternative_%d_%s", index, name)
		replacements["#/$defs/"+escapeJSONPointer(name)] = "#/$defs/" + escapeJSONPointer(newName)
	}
	value = rewriteSchemaReferences(value, replacements).(map[string]any)
	value = prefixSchemaReferences(value, prefix).(map[string]any)
	for name, definition := range rawDefinitions {
		newName := fmt.Sprintf("alternative_%d_%s", index, name)
		definition = rewriteSchemaReferences(definition, replacements)
		definitions[newName] = prefixSchemaReferences(definition, prefix)
	}
	return value
}

func rewriteSchemaReferences(value any, replacements map[string]string) any {
	switch value := value.(type) {
	case map[string]any:
		result := make(map[string]any, len(value))
		for key, item := range value {
			if key == "$ref" {
				if reference, ok := item.(string); ok {
					if replacement, exists := replacements[reference]; exists {
						item = replacement
					}
				}
			}
			result[key] = rewriteSchemaReferences(item, replacements)
		}
		return result
	case []any:
		result := make([]any, len(value))
		for index, item := range value {
			result[index] = rewriteSchemaReferences(item, replacements)
		}
		return result
	default:
		return value
	}
}

func prefixSchemaReferences(value any, prefix string) any {
	switch value := value.(type) {
	case map[string]any:
		result := make(map[string]any, len(value))
		for key, item := range value {
			if key == "$ref" {
				if reference, ok := item.(string); ok && strings.HasPrefix(reference, "#") &&
					!strings.HasPrefix(reference, "#/$defs/alternative_") {
					item = prefix + strings.TrimPrefix(reference, "#")
				}
			}
			result[key] = prefixSchemaReferences(item, prefix)
		}
		return result
	case []any:
		result := make([]any, len(value))
		for index, item := range value {
			result[index] = prefixSchemaReferences(item, prefix)
		}
		return result
	default:
		return value
	}
}

func escapeJSONPointer(value string) string {
	return strings.ReplaceAll(strings.ReplaceAll(value, "~", "~0"), "/", "~1")
}
