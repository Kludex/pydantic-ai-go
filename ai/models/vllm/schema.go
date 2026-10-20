package vllm

import (
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/internal/requestschema"
	jsonschema "github.com/Kludex/pydantic-ai-go/ai/internal/schema"
)

func transformSchemas(params ai.ModelRequestParams, profile schemaProfile) ai.ModelRequestParams {
	var transform func(map[string]any) map[string]any
	switch profile {
	case schemaInline:
		transform = jsonschema.InlineDefinitions
	case schemaGoogle:
		transform = transformGoogleSchema
	default:
		return params
	}
	return requestschema.Transform(params, transform)
}

func transformGoogleSchema(source map[string]any) map[string]any {
	result := jsonschema.InlineDefinitions(source)
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
