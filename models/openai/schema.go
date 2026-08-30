package openai

import (
	"fmt"
	"sort"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
	jsonschema "github.com/Kludex/pydantic-ai-go/internal/schema"
)

var openAIStrictIncompatibleKeys = []string{
	"minLength",
	"maxLength",
	"patternProperties",
	"unevaluatedProperties",
	"propertyNames",
	"minProperties",
	"maxProperties",
	"unevaluatedItems",
	"contains",
	"minContains",
	"maxContains",
	"uniqueItems",
}

var openAIStrictFormats = map[string]bool{
	"date-time": true,
	"time":      true,
	"date":      true,
	"duration":  true,
	"email":     true,
	"hostname":  true,
	"ipv4":      true,
	"ipv6":      true,
	"uuid":      true,
}

func prepareOpenAITool(def ai.ToolDefinition, supportsStrict bool) (map[string]any, *bool, error) {
	schema, strict, err := prepareOpenAISchema(def.Schema, def.Strict)
	if err != nil {
		return nil, nil, fmt.Errorf("openai: tool %q: %w", def.Name, err)
	}
	if !strict || !supportsStrict {
		return schema, nil, nil
	}
	return schema, &strict, nil
}

func prepareOpenAISchema(source map[string]any, requested *bool) (map[string]any, bool, error) {
	if source["type"] != "object" {
		if _, recursive := source["$ref"]; !recursive {
			return nil, false, fmt.Errorf("tool schema root must have type object")
		}
	}
	compatible := true
	var transformErr error
	result := jsonschema.Transform(source, func(schema map[string]any) {
		delete(schema, "title")
		delete(schema, "$schema")
		delete(schema, "discriminator")

		if _, ok := schema["default"]; ok {
			if requested != nil && *requested {
				delete(schema, "default")
			} else if requested == nil {
				compatible = false
			}
		}

		var incompatible []string
		for _, key := range openAIStrictIncompatibleKeys {
			if value, ok := schema[key]; ok {
				incompatible = append(incompatible, fmt.Sprintf("%s=%v", key, value))
			}
		}
		if format, ok := schema["format"].(string); ok && !openAIStrictFormats[format] {
			incompatible = append(incompatible, "format="+format)
		}
		if pattern, ok := schema["pattern"].(string); ok && containsRegexLookaround(pattern) {
			incompatible = append(incompatible, "pattern="+pattern)
		}
		if len(incompatible) > 0 {
			if requested != nil && *requested {
				for _, note := range incompatible {
					key, _, _ := strings.Cut(note, "=")
					delete(schema, key)
				}
				details := strings.Join(incompatible, ", ")
				if description, ok := schema["description"].(string); ok && description != "" {
					schema["description"] = description + " (" + details + ")"
				} else {
					schema["description"] = details
				}
			} else if requested == nil {
				compatible = false
			}
		}

		if oneOf, ok := schema["oneOf"]; ok {
			if requested != nil && *requested {
				delete(schema, "oneOf")
				schema["anyOf"] = oneOf
			} else {
				compatible = false
			}
		}

		switch schema["type"] {
		case "object":
			properties, ok := schema["properties"].(map[string]any)
			if !ok {
				properties = map[string]any{}
				schema["properties"] = properties
			}
			if requested != nil && *requested {
				schema["additionalProperties"] = false
				required := make([]string, 0, len(properties))
				for name := range properties {
					required = append(required, name)
				}
				sort.Strings(required)
				schema["required"] = required
			} else if requested == nil {
				if additional, ok := schema["additionalProperties"]; ok && additional != false {
					compatible = false
				} else {
					schema["additionalProperties"] = false
				}
				required, requiredOK := schema["required"].([]string)
				if !requiredOK {
					if values, valuesOK := schema["required"].([]any); valuesOK {
						requiredOK = true
						required = make([]string, 0, len(values))
						for _, value := range values {
							if value, valueOK := value.(string); valueOK {
								required = append(required, value)
							}
						}
					}
				}
				if !requiredOK {
					compatible = false
				}
				for name := range properties {
					if !containsString(required, name) {
						compatible = false
					}
				}
			}
		case "array":
			items, hasItems := schema["items"].(map[string]any)
			typedItems := hasItems && hasTypeBearingKey(items)
			prefixItems, hasPrefix := schema["prefixItems"].([]any)
			if !typedItems && (!hasPrefix || len(prefixItems) == 0) {
				if requested != nil && *requested {
					transformErr = fmt.Errorf(
						"strict mode requires array items to have a type; add an item type or disable strict mode",
					)
				} else if requested == nil {
					compatible = false
				}
			}
		}
	})
	if transformErr != nil {
		return nil, false, transformErr
	}
	if requested != nil {
		return result, *requested, nil
	}
	return result, compatible, nil
}

func hasTypeBearingKey(schema map[string]any) bool {
	for _, key := range []string{"type", "$ref", "anyOf", "oneOf", "allOf", "enum", "const"} {
		if _, ok := schema[key]; ok {
			return true
		}
	}
	return false
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func containsRegexLookaround(pattern string) bool {
	escaped := false
	for index := 0; index < len(pattern); index++ {
		if escaped {
			escaped = false
			continue
		}
		if pattern[index] == '\\' {
			escaped = true
			continue
		}
		remaining := pattern[index:]
		for _, token := range []string{"(?=", "(?!", "(?<=", "(?<!"} {
			if strings.HasPrefix(remaining, token) {
				return true
			}
		}
	}
	return false
}
