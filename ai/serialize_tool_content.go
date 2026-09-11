package ai

import "encoding/json"

func narrowToolReturnContent(content any) (any, error) {
	switch content := content.(type) {
	case map[string]any:
		if kind, _ := content["kind"].(string); isNestedFileContent(kind, content) {
			encoded, _ := json.Marshal(content)
			var wire wireUserContent
			_ = json.Unmarshal(encoded, &wire)
			value, err := unmarshalUserContentItem(wire)
			if err != nil {
				return nil, err
			}
			return value, nil
		}
		result := make(map[string]any, len(content))
		for key, value := range content {
			narrowed, err := narrowToolReturnContent(value)
			if err != nil {
				return nil, err
			}
			result[key] = narrowed
		}
		return result, nil
	case []any:
		result := make([]any, len(content))
		for index, value := range content {
			narrowed, err := narrowToolReturnContent(value)
			if err != nil {
				return nil, err
			}
			result[index] = narrowed
		}
		return result, nil
	default:
		return content, nil
	}
}

func isNestedFileContent(kind string, content map[string]any) bool {
	switch kind {
	case "image-url", "video-url", "audio-url", "document-url":
		_, hasURL := content["url"]
		mediaType, hasMediaType := content["media_type"].(string)
		return hasURL && hasMediaType && mediaType != ""
	case "binary":
		_, ok := content["media_type"]
		return ok
	case "uploaded-file":
		_, ok := content["file_id"]
		return ok
	default:
		return false
	}
}
