// Package schema reflects JSON Schemas from Go structs.
package schema

import (
	"encoding"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"time"
)

// ForType returns a JSON Schema for any supported Go type.
func ForType(t reflect.Type) (map[string]any, error) {
	return forType(t, "#", make(map[reflect.Type]string))
}

// For returns a JSON Schema (draft 2020-12 compatible object schema) for
// the struct type T, derived from `json` and `jsonschema` field tags.
func For(t reflect.Type) (map[string]any, error) {
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("schema: expected a struct type, got %s", t.Kind())
	}
	return forStruct(t, "#", make(map[reflect.Type]string))
}

var (
	jsonRawMessageType = reflect.TypeFor[json.RawMessage]()
	textMarshalerType  = reflect.TypeFor[encoding.TextMarshaler]()
	timeType           = reflect.TypeFor[time.Time]()
)

func forStruct(t reflect.Type, path string, active map[reflect.Type]string) (map[string]any, error) {
	if reference, recursive := active[t]; recursive {
		return map[string]any{"$ref": reference}, nil
	}
	active[t] = path
	defer delete(active, t)

	properties := map[string]any{}
	var required []string
	for i := range t.NumField() {
		f := t.Field(i)
		if !f.IsExported() {
			continue
		}
		name, omitempty, skip := jsonName(f)
		if skip {
			continue
		}
		if f.Anonymous && strings.Split(f.Tag.Get("json"), ",")[0] == "" {
			embeddedType := f.Type
			if embeddedType.Kind() == reflect.Pointer {
				embeddedType = embeddedType.Elem()
				omitempty = true
			}
			if embeddedType.Kind() == reflect.Struct {
				embedded, err := forStruct(embeddedType, path, active)
				if err != nil {
					return nil, fmt.Errorf("schema: field %s: %w", f.Name, err)
				}
				for propertyName, property := range embedded["properties"].(map[string]any) {
					properties[propertyName] = property
				}
				if !omitempty {
					if names, ok := embedded["required"].([]string); ok {
						required = append(required, names...)
					}
				}
				continue
			}
		}
		fieldSchema, err := forType(f.Type, path+"/properties/"+escapeJSONPointer(name), active)
		if err != nil {
			return nil, fmt.Errorf("schema: field %s: %w", f.Name, err)
		}
		if err := applyTag(fieldSchema, f.Tag.Get("jsonschema")); err != nil {
			return nil, fmt.Errorf("schema: field %s: %w", f.Name, err)
		}
		properties[name] = fieldSchema
		if !omitempty || hasSchemaFlag(f.Tag.Get("jsonschema"), "required") {
			required = append(required, name)
		}
	}
	s := map[string]any{
		"type":                 "object",
		"properties":           properties,
		"additionalProperties": false,
	}
	if len(required) > 0 {
		s["required"] = required
	}
	return s, nil
}

func forType(t reflect.Type, path string, active map[reflect.Type]string) (map[string]any, error) {
	if t.Kind() == reflect.Pointer {
		inner, err := forType(t.Elem(), path, active)
		if err != nil {
			return nil, err
		}
		return map[string]any{"anyOf": []any{inner, map[string]any{"type": "null"}}}, nil
	}
	if t == jsonRawMessageType {
		return map[string]any{}, nil
	}
	if t == timeType {
		return map[string]any{"type": "string", "format": "date-time"}, nil
	}
	if t.Implements(textMarshalerType) || reflect.PointerTo(t).Implements(textMarshalerType) {
		return map[string]any{"type": "string"}, nil
	}
	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}, nil
	case reflect.Bool:
		return map[string]any{"type": "boolean"}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}, nil
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}, nil
	case reflect.Slice, reflect.Array:
		if t.Elem().Kind() == reflect.Uint8 {
			return map[string]any{"type": "string", "contentEncoding": "base64"}, nil
		}
		items, err := forType(t.Elem(), path+"/items", active)
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "array", "items": items}, nil
	case reflect.Map:
		if t.Key().Kind() != reflect.String {
			return nil, fmt.Errorf("unsupported map key type %s", t.Key())
		}
		values, err := forType(t.Elem(), path+"/additionalProperties", active)
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "object", "additionalProperties": values}, nil
	case reflect.Struct:
		return forStruct(t, path, active)
	case reflect.Interface:
		return map[string]any{}, nil
	default:
		return nil, fmt.Errorf("unsupported type %s", t)
	}
}

func jsonName(f reflect.StructField) (name string, omitempty, skip bool) {
	tag := f.Tag.Get("json")
	if tag == "-" {
		return "", false, true
	}
	name = f.Name
	parts := strings.Split(tag, ",")
	if parts[0] != "" {
		name = parts[0]
	}
	for _, opt := range parts[1:] {
		if opt == "omitempty" || opt == "omitzero" {
			omitempty = true
		}
	}
	return name, omitempty, false
}

func applyTag(schema map[string]any, tag string) error {
	if tag == "" {
		return nil
	}
	entries, err := splitSchemaTag(tag)
	if err != nil {
		return err
	}
	var enum []any
	for _, entry := range entries {
		key, value, hasValue := strings.Cut(entry, "=")
		if !hasValue {
			switch key {
			case "required":
				continue
			case "uniqueItems", "readOnly", "writeOnly", "deprecated":
				schema[key] = true
			default:
				if _, exists := schema["description"]; exists {
					return fmt.Errorf("unknown annotation %q", key)
				}
				schema["description"] = key
			}
			continue
		}
		switch key {
		case "$id", "$anchor", "$dynamicAnchor", "$comment", "description", "title", "format", "pattern",
			"contentEncoding", "contentMediaType":
			schema[key] = value
		case "enum":
			enum = append(enum, parseTagValue(value))
		case "default", "example", "const":
			schema[key] = parseTagValue(value)
		case "examples":
			parsed := parseTagValue(value)
			if values, ok := parsed.([]any); ok {
				schema[key] = values
			} else {
				schema[key] = []any{parsed}
			}
		case "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf":
			number, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return fmt.Errorf("invalid %s value %q", key, value)
			}
			schema[key] = number
		case "minLength", "maxLength", "minItems", "maxItems", "minContains", "maxContains", "minProperties",
			"maxProperties":
			number, err := strconv.Atoi(value)
			if err != nil || number < 0 {
				return fmt.Errorf("invalid %s value %q", key, value)
			}
			schema[key] = number
		case "readOnly", "writeOnly", "deprecated", "uniqueItems", "additionalProperties",
			"unevaluatedProperties":
			boolean, err := strconv.ParseBool(value)
			if err != nil {
				return fmt.Errorf("invalid %s value %q", key, value)
			}
			schema[key] = boolean
		case "allOf", "anyOf", "oneOf", "not", "if", "then", "else", "contains", "prefixItems", "propertyNames",
			"dependentRequired", "dependentSchemas":
			parsed := parseTagValue(value)
			if _, unparsed := parsed.(string); unparsed {
				return fmt.Errorf("invalid %s JSON value %q", key, value)
			}
			schema[key] = parsed
		default:
			return fmt.Errorf("unknown annotation %q", key)
		}
	}
	if len(enum) > 0 {
		schema["enum"] = enum
	}
	return nil
}

func hasSchemaFlag(tag, flag string) bool {
	entries, _ := splitSchemaTag(tag)
	return slices.Contains(entries, flag)
}

func splitSchemaTag(tag string) ([]string, error) {
	var entries []string
	start := 0
	depth := 0
	quoted := false
	escaped := false
	for index, character := range tag {
		switch {
		case escaped:
			escaped = false
		case character == '\\' && quoted:
			escaped = true
		case character == '"':
			quoted = !quoted
		case !quoted && (character == '[' || character == '{'):
			depth++
		case !quoted && (character == ']' || character == '}'):
			depth--
			if depth < 0 {
				return nil, fmt.Errorf("invalid annotation nesting")
			}
		case !quoted && depth == 0 && character == ',':
			entries = append(entries, strings.TrimSpace(tag[start:index]))
			start = index + 1
		}
	}
	if quoted || depth != 0 {
		return nil, fmt.Errorf("invalid annotation nesting")
	}
	entries = append(entries, strings.TrimSpace(tag[start:]))
	return entries, nil
}

func parseTagValue(value string) any {
	var parsed any
	if json.Unmarshal([]byte(value), &parsed) == nil {
		return parsed
	}
	return value
}

func escapeJSONPointer(value string) string {
	return strings.ReplaceAll(strings.ReplaceAll(value, "~", "~0"), "/", "~1")
}
