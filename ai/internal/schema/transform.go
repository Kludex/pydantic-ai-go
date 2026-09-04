package schema

// Transform deep-copies a JSON Schema and calls fn once for each schema node.
// Maps that contain property names or definition names are containers, not
// schema nodes, so their keys are preserved.
func Transform(source map[string]any, fn func(map[string]any)) map[string]any {
	result := cloneMap(source)
	transformNode(result, fn)
	return result
}

func transformNode(node map[string]any, fn func(map[string]any)) {
	fn(node)
	for _, key := range []string{
		"additionalProperties", "contains", "contentSchema", "else", "if", "items", "not",
		"propertyNames", "then", "unevaluatedProperties",
	} {
		if child, ok := node[key].(map[string]any); ok {
			transformNode(child, fn)
		}
	}
	for _, key := range []string{"allOf", "anyOf", "oneOf", "prefixItems"} {
		if children, ok := node[key].([]any); ok {
			for _, child := range children {
				if child, ok := child.(map[string]any); ok {
					transformNode(child, fn)
				}
			}
		}
	}
	for _, key := range []string{"$defs", "definitions", "dependentSchemas", "patternProperties", "properties"} {
		if children, ok := node[key].(map[string]any); ok {
			for _, child := range children {
				if child, ok := child.(map[string]any); ok {
					transformNode(child, fn)
				}
			}
		}
	}
}

func cloneMap(source map[string]any) map[string]any {
	result := make(map[string]any, len(source))
	for key, value := range source {
		result[key] = cloneValue(value)
	}
	return result
}

func cloneValue(value any) any {
	switch value := value.(type) {
	case map[string]any:
		return cloneMap(value)
	case []any:
		result := make([]any, len(value))
		for i, item := range value {
			result[i] = cloneValue(item)
		}
		return result
	default:
		return value
	}
}
