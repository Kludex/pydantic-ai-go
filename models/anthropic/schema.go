package anthropic

import (
	"fmt"
	"sort"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	jsonschema "github.com/Kludex/pydantic-ai-go/internal/schema"
)

var anthropicStrictFormats = map[string]bool{
	"date-time": true,
	"time":      true,
	"date":      true,
	"duration":  true,
	"email":     true,
	"hostname":  true,
	"ipv4":      true,
	"ipv6":      true,
	"uri":       true,
	"uuid":      true,
}

func prepareAnthropicTool(
	def ai.ToolDefinition, supportsStrict bool, warningHandler func(SchemaWarning),
) (toolParam, error) {
	def, err := ai.PrepareToolReturnSchema(def, false)
	if err != nil {
		return toolParam{}, err
	}
	strict := def.Strict != nil && *def.Strict
	schema := jsonschema.Transform(def.Schema, func(node map[string]any) {
		delete(node, "title")
		delete(node, "$schema")
	})
	if strict {
		if warningHandler != nil && hasDynamicMapSchema(schema) {
			warningHandler(SchemaWarning{
				ToolName: def.Name,
				Message: "`dict` fields are not supported by Anthropic in strict mode. " +
					"Anthropic sets `additionalProperties` to `false`, which forces the model to return `{}`. " +
					"Use entries with explicit `key` and `value` fields, or disable strict mode.",
			})
		}
		var err error
		schema, err = transformAnthropicStrictSchema(schema)
		if err != nil {
			return toolParam{}, fmt.Errorf("anthropic: tool %q: %w", def.Name, err)
		}
	}
	var strictFlag *bool
	if strict && supportsStrict {
		strictFlag = def.Strict
	}
	return toolParam{
		Name: def.Name, Description: def.Description, InputSchema: schema, Strict: strictFlag,
	}, nil
}

func hasDynamicMapSchema(source map[string]any) bool {
	found := false
	jsonschema.Transform(source, func(node map[string]any) {
		additionalProperties := node["additionalProperties"]
		if allowed, ok := additionalProperties.(bool); ok && allowed {
			found = true
			return
		}
		if _, ok := additionalProperties.(map[string]any); ok {
			found = true
		}
	})
	return found
}

func transformAnthropicStrictSchema(source map[string]any) (map[string]any, error) {
	remaining := copyMap(source)
	strict := map[string]any{}

	if definitions, ok := remaining["$defs"].(map[string]any); ok {
		delete(remaining, "$defs")
		transformed := make(map[string]any, len(definitions))
		for name, definition := range definitions {
			definition, ok := definition.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("definition %q must be a schema object", name)
			}
			var err error
			transformed[name], err = transformAnthropicStrictSchema(definition)
			if err != nil {
				return nil, err
			}
		}
		strict["$defs"] = transformed
	}
	if ref, ok := remaining["$ref"]; ok {
		strict["$ref"] = ref
		return strict, nil
	}

	typeName, _ := remaining["type"].(string)
	delete(remaining, "type")
	if variants, key, ok := schemaVariants(remaining); ok {
		delete(remaining, key)
		transformed := make([]any, 0, len(variants))
		for _, variant := range variants {
			variant, ok := variant.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("%s variants must be schema objects", key)
			}
			item, err := transformAnthropicStrictSchema(variant)
			if err != nil {
				return nil, err
			}
			transformed = append(transformed, item)
		}
		if key == "oneOf" {
			key = "anyOf"
		}
		strict[key] = transformed
	} else {
		if typeName == "" {
			return nil, fmt.Errorf("schema must have type, anyOf, oneOf, or allOf")
		}
		strict["type"] = typeName
	}

	if enum, ok := remaining["enum"].([]any); ok {
		strict["enum"] = enum
		delete(remaining, "enum")
	} else if enum, ok := remaining["enum"].([]string); ok {
		strict["enum"] = enum
		delete(remaining, "enum")
	}
	if description, ok := remaining["description"]; ok {
		strict["description"] = description
		delete(remaining, "description")
	}

	switch typeName {
	case "object":
		properties, _ := remaining["properties"].(map[string]any)
		delete(remaining, "properties")
		transformed := make(map[string]any, len(properties))
		for name, property := range properties {
			property, ok := property.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("property %q must be a schema object", name)
			}
			var err error
			transformed[name], err = transformAnthropicStrictSchema(property)
			if err != nil {
				return nil, err
			}
		}
		strict["properties"] = transformed
		delete(remaining, "additionalProperties")
		strict["additionalProperties"] = false
		if required, ok := remaining["required"]; ok {
			strict["required"] = required
			delete(remaining, "required")
		}
	case "string":
		if format, ok := remaining["format"].(string); ok && anthropicStrictFormats[format] {
			strict["format"] = format
			delete(remaining, "format")
		}
	case "array":
		if items, ok := remaining["items"]; ok {
			delete(remaining, "items")
			items, ok := items.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("array items must be a schema object")
			}
			var err error
			strict["items"], err = transformAnthropicStrictSchema(items)
			if err != nil {
				return nil, err
			}
		}
		if minItems, ok := remaining["minItems"]; ok && (numberIs(minItems, 0) || numberIs(minItems, 1)) {
			strict["minItems"] = minItems
			delete(remaining, "minItems")
		}
	case "boolean", "integer", "number", "null", "":
	default:
		return nil, fmt.Errorf("unsupported schema type %q", typeName)
	}

	if len(remaining) > 0 {
		keys := make([]string, 0, len(remaining))
		for key := range remaining {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		notes := make([]string, 0, len(keys))
		for _, key := range keys {
			notes = append(notes, fmt.Sprintf("%s: %v", key, remaining[key]))
		}
		description, _ := strict["description"].(string)
		if description != "" {
			description += "\n\n"
		}
		strict["description"] = description + "{" + strings.Join(notes, ", ") + "}"
	}
	return strict, nil
}

func schemaVariants(schema map[string]any) ([]any, string, bool) {
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if variants, ok := schema[key].([]any); ok {
			return variants, key, true
		}
	}
	return nil, "", false
}

func copyMap(source map[string]any) map[string]any {
	result := make(map[string]any, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}

func numberIs(value any, target int64) bool {
	switch value := value.(type) {
	case int:
		return int64(value) == target
	case int64:
		return value == target
	case float64:
		return value == float64(target)
	default:
		return false
	}
}
