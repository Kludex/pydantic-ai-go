package schema

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"

	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"
)

// Validator is a compiled JSON Schema.
type Validator struct {
	schema *jsonschema.Schema
}

// Compile compiles a JSON Schema for repeated validation.
func Compile(document map[string]any) (*Validator, error) {
	return compileAt(document, "schema.json")
}

func compileAt(document map[string]any, location string) (*Validator, error) {
	compiler := jsonschema.NewCompiler()
	compiler.AssertFormat()
	if err := compiler.AddResource(location, normalize(document)); err != nil {
		return nil, fmt.Errorf("add schema resource: %w", err)
	}
	compiled, err := compiler.Compile(location)
	if err != nil {
		return nil, fmt.Errorf("compile schema: %w", err)
	}
	return &Validator{schema: compiled}, nil
}

func normalize(value any) any {
	switch value := value.(type) {
	case map[string]any:
		normalized := make(map[string]any, len(value))
		for key, item := range value {
			normalized[key] = normalize(item)
		}
		return normalized
	case []any:
		normalized := make([]any, len(value))
		for index, item := range value {
			normalized[index] = normalize(item)
		}
		return normalized
	case []string:
		normalized := make([]any, len(value))
		for index, item := range value {
			normalized[index] = item
		}
		return normalized
	default:
		return value
	}
}

// Validate validates a decoded JSON value.
func (v *Validator) Validate(value any) error {
	return v.schema.Validate(value)
}

// ValidateJSON decodes and validates a JSON value without losing number precision.
func (v *Validator) ValidateJSON(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return fmt.Errorf("decode JSON: %w", err)
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("decode JSON: multiple values")
		}
		return fmt.Errorf("decode trailing JSON: %w", err)
	}
	return v.Validate(value)
}
