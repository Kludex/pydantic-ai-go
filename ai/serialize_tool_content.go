package ai

import "encoding/json"

// NormalizeToolReturnContent rebuilds valid nested multimodal wire values without
// coercing application maps that only collide with a reserved kind.
func NormalizeToolReturnContent(content any) any {
	return narrowToolReturnContent(content)
}

func narrowToolReturnContent(content any) any {
	switch content := content.(type) {
	case map[string]any:
		if value, ok := nestedFileContent(content); ok {
			return value
		}
		result := make(map[string]any, len(content))
		for key, value := range content {
			result[key] = narrowToolReturnContent(value)
		}
		return result
	case []any:
		result := make([]any, len(content))
		for index, value := range content {
			result[index] = narrowToolReturnContent(value)
		}
		return result
	default:
		return content
	}
}

func nestedFileContent(content map[string]any) (UserContent, bool) {
	kind, _ := content["kind"].(string)
	if !isCompleteNestedFileContent(kind, content) {
		return nil, false
	}
	encoded, err := json.Marshal(content)
	if err != nil {
		return nil, false
	}
	var wire wireUserContent
	if err := json.Unmarshal(encoded, &wire); err != nil {
		return nil, false
	}
	value, err := unmarshalUserContentItem(wire)
	return value, err == nil
}

func isCompleteNestedFileContent(kind string, content map[string]any) bool {
	nonEmptyString := func(key string) bool {
		value, ok := content[key].(string)
		return ok && value != ""
	}
	switch kind {
	case "image-url", "video-url", "audio-url", "document-url":
		return nonEmptyString("url") && nonEmptyString("media_type")
	case "binary":
		return nonEmptyString("media_type") && content["data"] != nil
	case "uploaded-file":
		return nonEmptyString("file_id") && nonEmptyString("provider_name")
	default:
		return false
	}
}
