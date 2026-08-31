package openrouter

import (
	"strings"

	jsonschema "github.com/Kludex/pydantic-ai-go/internal/schema"
)

func inlineSchemaDefinitions(source map[string]any) map[string]any {
	cloned := jsonschema.Transform(source, func(map[string]any) {})
	definitions, ok := cloned["$defs"].(map[string]any)
	if !ok || len(definitions) == 0 || recursiveSchemaDefinitions(definitions) {
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

func recursiveSchemaDefinitions(definitions map[string]any) bool {
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
