package bedrock

import (
	"encoding/json"
	"strings"
)

func bedrockMetadataValue(value any) any {
	encoded, _ := json.Marshal(value)
	var decoded any
	_ = json.Unmarshal(encoded, &decoded)
	return normalizeBedrockMetadata(decoded)
}

func normalizeBedrockMetadata(value any) any {
	switch value := value.(type) {
	case map[string]any:
		normalized := make(map[string]any, len(value))
		for name, field := range value {
			if field == nil || name == "" {
				continue
			}
			name = strings.ToLower(name[:1]) + name[1:]
			normalized[name] = normalizeBedrockMetadata(field)
		}
		if len(normalized) == 1 {
			if union, exists := normalized["value"]; exists {
				return union
			}
		}
		return normalized
	case []any:
		for index := range value {
			value[index] = normalizeBedrockMetadata(value[index])
		}
		return value
	default:
		return value
	}
}
