package ai

import (
	"bytes"
	"encoding/json"

	"github.com/Kludex/pydantic-ai-go/ai/internal/schema"
)

func decodePartialJSON(
	raw string,
	validator *schema.Validator,
	decode func([]byte) (decodedOutput, error),
) (decodedOutput, bool) {
	data := bytes.TrimSpace([]byte(raw))
	for end := len(data); end > 0; end-- {
		candidate, ok := closeJSONPrefix(data[:end])
		if !ok {
			continue
		}
		var value any
		if err := json.Unmarshal(candidate, &value); err != nil {
			continue
		}
		if validator != nil {
			if err := validator.Validate(value); err != nil {
				continue
			}
		}
		output, err := decode(candidate)
		if err != nil {
			continue
		}
		return output, true
	}
	return decodedOutput{}, false
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
