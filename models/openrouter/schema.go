package openrouter

import (
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go"
	jsonschema "github.com/Kludex/pydantic-ai-go/internal/schema"
)

func transformSchemas(params ai.ModelRequestParams, provider string) ai.ModelRequestParams {
	var transform func(map[string]any) map[string]any
	switch provider {
	case "google":
		transform = transformGoogleSchema
	case "amazon", "meta-llama", "qwen":
		transform = inlineSchemaDefinitions
	default:
		return params
	}
	params.Tools = transformToolDefinitions(params.Tools, transform)
	params.DeferredTools = transformToolDefinitions(params.DeferredTools, transform)
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

func transformToolDefinitions(
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
	result := inlineSchemaDefinitions(source)
	transform := func(schema map[string]any) {
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
	}
	result = jsonschema.Transform(result, transform)
	return jsonschema.Transform(result, transform)
}
