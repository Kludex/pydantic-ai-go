package vllm

import (
	"fmt"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	jsonschema "github.com/Kludex/pydantic-ai-go/ai/internal/schema"
)

func transformSchemas(params ai.ModelRequestParams, profile schemaProfile) ai.ModelRequestParams {
	var transform func(map[string]any) map[string]any
	switch profile {
	case schemaInline:
		transform = inlineDefinitions
	case schemaGoogle:
		transform = transformGoogleSchema
	default:
		return params
	}
	params.Tools = transformDefinitions(params.Tools, transform)
	params.DeferredTools = transformDefinitions(params.DeferredTools, transform)
	if params.OutputTool != nil {
		output := *params.OutputTool
		output.Schema = transform(output.Schema)
		if output.ReturnSchema != nil {
			output.ReturnSchema = transform(output.ReturnSchema)
		}
		params.OutputTool = &output
	}
	if params.OutputSchema != nil {
		params.OutputSchema = transform(params.OutputSchema)
	}
	return params
}

func transformDefinitions(
	definitions []ai.ToolDefinition, transform func(map[string]any) map[string]any,
) []ai.ToolDefinition {
	if definitions == nil {
		return nil
	}
	transformed := make([]ai.ToolDefinition, len(definitions))
	for index, definition := range definitions {
		transformed[index] = definition
		transformed[index].Schema = transform(definition.Schema)
		if definition.ReturnSchema != nil {
			transformed[index].ReturnSchema = transform(definition.ReturnSchema)
		}
	}
	return transformed
}

func transformGoogleSchema(source map[string]any) map[string]any {
	result := inlineDefinitions(source)
	return jsonschema.Transform(result, func(schema map[string]any) {
		for _, key := range []string{
			"$schema", "title", "discriminator", "examples", "exclusiveMaximum", "exclusiveMinimum",
		} {
			delete(schema, key)
		}
		if value, exists := schema["const"]; exists {
			delete(schema, "const")
			schema["enum"] = []any{value}
		}
		if values, ok := schema["enum"].([]any); ok && len(values) > 0 {
			converted := make([]any, len(values))
			for index, value := range values {
				switch value := value.(type) {
				case nil:
					converted[index] = "None"
				case bool:
					if value {
						converted[index] = "True"
					} else {
						converted[index] = "False"
					}
				default:
					converted[index] = fmt.Sprint(value)
				}
			}
			schema["type"] = "string"
			schema["enum"] = converted
		}
		for _, unionName := range []string{"anyOf", "oneOf"} {
			members, ok := schema[unionName].([]any)
			if !ok || len(members) != 2 {
				continue
			}
			var nonNull map[string]any
			nullCount := 0
			for _, member := range members {
				member, ok := member.(map[string]any)
				if !ok {
					continue
				}
				if member["type"] == "null" && len(member) == 1 {
					nullCount++
				} else {
					nonNull = member
				}
			}
			if nullCount == 1 && nonNull != nil {
				delete(schema, unionName)
				for key, value := range nonNull {
					if _, exists := schema[key]; !exists {
						schema[key] = value
					}
				}
				schema["nullable"] = true
			}
		}
		if _, typed := schema["type"]; !typed {
			if oneOf, exists := schema["oneOf"]; exists {
				delete(schema, "oneOf")
				schema["anyOf"] = oneOf
			}
		}
		if schema["type"] == "string" {
			if format, ok := schema["format"].(string); ok && format != "" {
				delete(schema, "format")
				if description, ok := schema["description"].(string); ok && description != "" {
					schema["description"] = description + " (format: " + format + ")"
				} else {
					schema["description"] = "Format: " + format
				}
			}
		}
	})
}

func inlineDefinitions(source map[string]any) map[string]any {
	cloned := jsonschema.Transform(source, func(map[string]any) {})
	definitions, ok := cloned["$defs"].(map[string]any)
	if !ok || len(definitions) == 0 || hasRecursiveDefinitions(definitions) {
		return cloned
	}
	delete(cloned, "$defs")
	var expand func(any) any
	expand = func(value any) any {
		switch value := value.(type) {
		case map[string]any:
			if reference, ok := value["$ref"].(string); ok {
				key := strings.TrimPrefix(reference, "#/$defs/")
				if key != reference {
					key = strings.ReplaceAll(strings.ReplaceAll(key, "~1", "/"), "~0", "~")
					if definition, ok := definitions[key].(map[string]any); ok {
						merged := jsonschema.Transform(definition, func(map[string]any) {})
						for sibling, siblingValue := range value {
							if sibling != "$ref" {
								merged[sibling] = siblingValue
							}
						}
						return expand(merged)
					}
				}
			}
			for key, child := range value {
				value[key] = expand(child)
			}
			return value
		case []any:
			for index, child := range value {
				value[index] = expand(child)
			}
			return value
		default:
			return value
		}
	}
	return expand(cloned).(map[string]any)
}

func hasRecursiveDefinitions(definitions map[string]any) bool {
	var visit func(string, map[string]bool) bool
	visit = func(name string, stack map[string]bool) bool {
		if stack[name] {
			return true
		}
		definition, ok := definitions[name]
		if !ok {
			return false
		}
		next := make(map[string]bool, len(stack)+1)
		for key, value := range stack {
			next[key] = value
		}
		next[name] = true
		var scan func(any) bool
		scan = func(value any) bool {
			switch value := value.(type) {
			case map[string]any:
				if reference, ok := value["$ref"].(string); ok {
					key := strings.TrimPrefix(reference, "#/$defs/")
					key = strings.ReplaceAll(strings.ReplaceAll(key, "~1", "/"), "~0", "~")
					if key != reference && visit(key, next) {
						return true
					}
				}
				for _, child := range value {
					if scan(child) {
						return true
					}
				}
			case []any:
				for _, child := range value {
					if scan(child) {
						return true
					}
				}
			}
			return false
		}
		return scan(definition)
	}
	for name := range definitions {
		if visit(name, nil) {
			return true
		}
	}
	return false
}
