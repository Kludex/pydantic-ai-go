// Package schema reflects JSON Schemas from Go structs.
package schema

import (
	"cmp"
	"encoding"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode"
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
	jsonNumberType     = reflect.TypeFor[json.Number]()
	jsonMarshalerType  = reflect.TypeFor[json.Marshaler]()
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
	for _, selected := range fieldsForStruct(t) {
		field := selected.field
		fieldSchema, err := forType(field.Type, path+"/properties/"+escapeJSONPointer(selected.name), active)
		if err != nil {
			return nil, fmt.Errorf("schema: field %s: %w", field.Name, err)
		}
		if jsonStringOption(field) {
			fieldSchema, err = stringEncodedSchema(field.Type)
			if err != nil {
				return nil, fmt.Errorf("schema: field %s: %w", field.Name, err)
			}
		}
		if err := applyTag(fieldSchema, field.Tag.Get("jsonschema")); err != nil {
			return nil, fmt.Errorf("schema: field %s: %w", field.Name, err)
		}
		properties[selected.name] = fieldSchema
		if !selected.optional || hasSchemaFlag(field.Tag.Get("jsonschema"), "required") {
			required = append(required, selected.name)
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
	if t == jsonNumberType {
		return map[string]any{"type": "number"}, nil
	}
	if t.Implements(textMarshalerType) || reflect.PointerTo(t).Implements(textMarshalerType) {
		return map[string]any{"type": "string"}, nil
	}
	if t.Implements(jsonMarshalerType) || reflect.PointerTo(t).Implements(jsonMarshalerType) {
		return map[string]any{}, nil
	}
	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}, nil
	case reflect.Bool:
		return map[string]any{"type": "boolean"}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return map[string]any{"type": "integer"}, nil
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}, nil
	case reflect.Slice, reflect.Array:
		if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8 {
			return map[string]any{"type": "string", "contentEncoding": "base64"}, nil
		}
		items, err := forType(t.Elem(), path+"/items", active)
		if err != nil {
			return nil, err
		}
		schema := map[string]any{"type": "array", "items": items}
		if t.Kind() == reflect.Array {
			schema["minItems"] = t.Len()
			schema["maxItems"] = t.Len()
		}
		return schema, nil
	case reflect.Map:
		if !supportedMapKey(t.Key()) {
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

func supportedMapKey(key reflect.Type) bool {
	if key.Kind() == reflect.String || key.Kind() >= reflect.Int && key.Kind() <= reflect.Int64 ||
		key.Kind() >= reflect.Uint && key.Kind() <= reflect.Uintptr {
		return true
	}
	return key.Implements(textMarshalerType) || reflect.PointerTo(key).Implements(textMarshalerType)
}

func jsonStringOption(field reflect.StructField) bool {
	for _, option := range strings.Split(field.Tag.Get("json"), ",")[1:] {
		if option == "string" {
			return true
		}
	}
	return false
}

func stringEncodedSchema(t reflect.Type) (map[string]any, error) {
	if t.Kind() == reflect.Pointer {
		inner, err := stringEncodedSchema(t.Elem())
		if err != nil {
			return nil, err
		}
		return map[string]any{"anyOf": []any{inner, map[string]any{"type": "null"}}}, nil
	}
	switch t.Kind() {
	case reflect.Bool, reflect.String, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64:
		return map[string]any{"type": "string"}, nil
	default:
		return nil, fmt.Errorf("json string option is not supported for %s", t)
	}
}

type selectedField struct {
	field       reflect.StructField
	name        string
	index       []int
	tagged      bool
	tagPriority int
	optional    bool
}

func fieldsForStruct(root reflect.Type) []selectedField {
	type embedded struct {
		typeOf   reflect.Type
		index    []int
		optional bool
	}
	current := []embedded{}
	next := []embedded{{typeOf: root}}
	var count, nextCount map[reflect.Type]int
	visited := map[reflect.Type]bool{}
	var fields []selectedField
	for len(next) > 0 {
		current, next = next, current[:0]
		count, nextCount = nextCount, map[reflect.Type]int{}
		for _, parent := range current {
			if visited[parent.typeOf] {
				continue
			}
			visited[parent.typeOf] = true
			for index := range parent.typeOf.NumField() {
				field := parent.typeOf.Field(index)
				fieldType := field.Type
				if fieldType.Kind() == reflect.Pointer {
					fieldType = fieldType.Elem()
				}
				if !field.IsExported() && (!field.Anonymous || fieldType.Kind() != reflect.Struct) {
					continue
				}
				name, optional, skip, tagged := jsonName(field)
				if skip {
					continue
				}
				fieldIndex := append(slices.Clone(parent.index), index)
				if tagged || !field.Anonymous || fieldType.Kind() != reflect.Struct {
					tagPriority := 1
					if tagged {
						tagPriority = 0
					}
					candidate := selectedField{
						field: field, name: name, index: fieldIndex, tagged: tagged, tagPriority: tagPriority,
						optional: optional || parent.optional,
					}
					fields = append(fields, candidate)
					if count[parent.typeOf] > 1 {
						fields = append(fields, candidate)
					}
					continue
				}
				nextCount[fieldType]++
				if nextCount[fieldType] == 1 {
					next = append(next, embedded{
						typeOf: fieldType, index: fieldIndex,
						optional: parent.optional || field.Type.Kind() == reflect.Pointer,
					})
				}
			}
		}
	}
	slices.SortFunc(fields, func(first, second selectedField) int {
		if order := strings.Compare(first.name, second.name); order != 0 {
			return order
		}
		if order := cmp.Compare(len(first.index), len(second.index)); order != 0 {
			return order
		}
		if order := cmp.Compare(first.tagPriority, second.tagPriority); order != 0 {
			return order
		}
		return slices.Compare(first.index, second.index)
	})
	selected := fields[:0]
	for start := 0; start < len(fields); {
		end := start + 1
		for end < len(fields) && fields[end].name == fields[start].name {
			end++
		}
		if end-start == 1 || len(fields[start].index) != len(fields[start+1].index) ||
			fields[start].tagged != fields[start+1].tagged {
			selected = append(selected, fields[start])
		}
		start = end
	}
	slices.SortFunc(selected, func(first, second selectedField) int {
		return slices.Compare(first.index, second.index)
	})
	return selected
}

func jsonName(field reflect.StructField) (name string, optional, skip, tagged bool) {
	tag := field.Tag.Get("json")
	if tag == "-" {
		return "", false, true, false
	}
	name = field.Name
	parts := strings.Split(tag, ",")
	if validJSONTagName(parts[0]) {
		name = parts[0]
		tagged = true
	}
	for _, option := range parts[1:] {
		if option == "omitempty" || option == "omitzero" {
			optional = true
		}
	}
	return name, optional, false, tagged
}

func validJSONTagName(name string) bool {
	if name == "" {
		return false
	}
	for _, character := range name {
		if strings.ContainsRune("!#$%&()*+-./:;<=>?@[]^_{|}~ ", character) {
			continue
		}
		if !unicode.IsLetter(character) && !unicode.IsDigit(character) {
			return false
		}
	}
	return true
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
		case "$id", "$schema", "$anchor", "$dynamicAnchor", "$ref", "$dynamicRef", "$comment", "description",
			"title", "format", "pattern", "contentEncoding", "contentMediaType":
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
		case "readOnly", "writeOnly", "deprecated", "uniqueItems":
			boolean, err := strconv.ParseBool(value)
			if err != nil {
				return fmt.Errorf("invalid %s value %q", key, value)
			}
			schema[key] = boolean
		case "type":
			parsed := parseTagValue(value)
			switch parsed.(type) {
			case string, []any:
				schema[key] = parsed
			default:
				return fmt.Errorf("invalid %s JSON value %q", key, value)
			}
		case "additionalProperties", "unevaluatedProperties", "unevaluatedItems", "items", "contains", "not", "if",
			"then", "else", "propertyNames", "contentSchema":
			parsed := parseTagValue(value)
			switch parsed.(type) {
			case bool, map[string]any:
				schema[key] = parsed
			default:
				return fmt.Errorf("invalid %s JSON value %q", key, value)
			}
		case "$defs", "$vocabulary", "definitions", "properties", "patternProperties", "dependentRequired",
			"dependentSchemas", "discriminator":
			parsed := parseTagValue(value)
			object, ok := parsed.(map[string]any)
			if !ok {
				return fmt.Errorf("invalid %s JSON value %q", key, value)
			}
			schema[key] = object
		case "allOf", "anyOf", "oneOf", "prefixItems", "required":
			parsed := parseTagValue(value)
			array, ok := parsed.([]any)
			if !ok {
				return fmt.Errorf("invalid %s JSON value %q", key, value)
			}
			schema[key] = array
		case "schema", "schemaOverride":
			parsed := parseTagValue(value)
			object, ok := parsed.(map[string]any)
			if !ok {
				return fmt.Errorf("invalid %s JSON value %q", key, value)
			}
			if key == "schemaOverride" {
				clear(schema)
			}
			for field, fieldValue := range object {
				schema[field] = fieldValue
			}
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
