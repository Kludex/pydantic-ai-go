package ai

import (
	"bytes"
	"encoding/json"
)

func decodePartialJSON[Output any](raw string, schema map[string]any) (Output, bool) {
	var zero Output
	data := bytes.TrimSpace([]byte(raw))
	for end := len(data); end > 0; end-- {
		candidate, ok := closeJSONPrefix(data[:end])
		if !ok {
			continue
		}
		var value any
		if err := json.Unmarshal(candidate, &value); err != nil || !requiredFieldsPresent(value, schema) {
			continue
		}
		var output Output
		if err := json.Unmarshal(candidate, &output); err != nil {
			continue
		}
		return output, true
	}
	return zero, false
}

func closeJSONPrefix(prefix []byte) ([]byte, bool) {
	candidate := append([]byte(nil), bytes.TrimSpace(prefix)...)
	stack := make([]byte, 0)
	inString := false
	escaped := false
	for _, char := range candidate {
		if inString {
			if escaped {
				escaped = false
				continue
			}
			switch char {
			case '\\':
				escaped = true
			case '"':
				inString = false
			}
			continue
		}
		switch char {
		case '"':
			inString = true
		case '{', '[':
			stack = append(stack, char)
		case '}':
			if len(stack) == 0 || stack[len(stack)-1] != '{' {
				return nil, false
			}
			stack = stack[:len(stack)-1]
		case ']':
			if len(stack) == 0 || stack[len(stack)-1] != '[' {
				return nil, false
			}
			stack = stack[:len(stack)-1]
		}
	}
	if inString {
		if escaped {
			candidate = append(candidate, '\\')
		}
		candidate = append(candidate, '"')
	}
	for index := len(stack) - 1; index >= 0; index-- {
		if stack[index] == '{' {
			candidate = append(candidate, '}')
		} else {
			candidate = append(candidate, ']')
		}
	}
	return candidate, true
}

func requiredFieldsPresent(value any, schema map[string]any) bool {
	switch schema["type"] {
	case "object":
		object, ok := value.(map[string]any)
		if !ok {
			return false
		}
		required, _ := schema["required"].([]string)
		for _, name := range required {
			if _, ok := object[name]; !ok {
				return false
			}
		}
		properties, _ := schema["properties"].(map[string]any)
		for name, property := range properties {
			child, present := object[name]
			if present && !requiredFieldsPresent(child, property.(map[string]any)) {
				return false
			}
		}
	case "array":
		array, ok := value.([]any)
		if !ok {
			return false
		}
		items := schema["items"].(map[string]any)
		for _, item := range array {
			if !requiredFieldsPresent(item, items) {
				return false
			}
		}
	}
	return true
}
