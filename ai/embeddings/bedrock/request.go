package bedrock

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
)

type family string

const (
	familyTitan  family = "titan"
	familyCohere family = "cohere"
	familyNova   family = "nova"
)

var geoPrefixes = []string{"us.", "eu.", "apac.", "jp.", "au.", "ca.", "global.", "us-gov."}

func normalizedModelName(modelName string) string {
	for _, prefix := range geoPrefixes {
		if strings.HasPrefix(modelName, prefix) {
			return strings.TrimPrefix(modelName, prefix)
		}
	}
	return modelName
}

func modelFamily(modelName string) (family, error) {
	normalized := normalizedModelName(modelName)
	switch {
	case strings.HasPrefix(normalized, "amazon.titan-embed"):
		return familyTitan, nil
	case strings.HasPrefix(normalized, "cohere.embed"):
		return familyCohere, nil
	case strings.HasPrefix(normalized, "amazon.nova"):
		return familyNova, nil
	default:
		return "", fmt.Errorf(
			"bedrock embeddings: unsupported model %q; expected amazon.titan-embed, cohere.embed, or amazon.nova", modelName,
		)
	}
}

func (model *Model) requestBody(
	inputs []string, inputType embeddings.InputType, settings requestSettings, extra map[string]any,
) ([]byte, error) {
	var body map[string]any
	normalized := normalizedModelName(model.modelName)
	switch model.family {
	case familyTitan:
		body = map[string]any{"inputText": inputs[0]}
		if modelVersion(normalized) != 1 {
			if settings.dimensions != nil {
				body["dimensions"] = *settings.dimensions
			}
			if settings.titanNormalize == nil {
				body["normalize"] = true
			} else {
				body["normalize"] = *settings.titanNormalize
			}
		}
	case familyCohere:
		cohereInputType := settings.cohereInputType
		if cohereInputType == "" {
			cohereInputType = CohereInputSearchQuery
			if inputType == embeddings.InputTypeDocument {
				cohereInputType = CohereInputSearchDocument
			}
		}
		body = map[string]any{"texts": inputs, "input_type": cohereInputType, "truncate": TruncationNone}
		if settings.cohereTruncate != "" {
			body["truncate"] = settings.cohereTruncate
		} else if settings.truncate {
			body["truncate"] = TruncationEnd
		}
		if modelVersion(normalized) != 3 {
			if settings.cohereMaxTokens != nil {
				body["max_tokens"] = *settings.cohereMaxTokens
			}
			if settings.dimensions != nil {
				body["output_dimension"] = *settings.dimensions
			}
		}
	default:
		truncation := settings.novaTruncate
		if truncation == "" {
			truncation = TruncationNone
			if settings.truncate {
				truncation = TruncationEnd
			}
		}
		purpose := settings.novaPurpose
		if purpose == "" {
			purpose = NovaPurposeGenericRetrieval
			if inputType == embeddings.InputTypeDocument {
				purpose = NovaPurposeGenericIndex
			}
		}
		params := map[string]any{
			"embeddingPurpose": purpose,
			"text":             map[string]any{"value": inputs[0], "truncationMode": truncation},
		}
		if settings.dimensions != nil {
			params["embeddingDimension"] = *settings.dimensions
		}
		body = map[string]any{"taskType": "SINGLE_EMBEDDING", "singleEmbeddingParams": params}
	}
	for key, value := range extra {
		if _, exists := body[key]; exists {
			return nil, fmt.Errorf("bedrock embeddings: extra body field %q conflicts with typed request field", key)
		}
		body[key] = value
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("bedrock embeddings: encode request: %w", err)
	}
	return encoded, nil
}

var versionPattern = regexp.MustCompile(`v(\d+)`)

func modelVersion(modelName string) int {
	match := versionPattern.FindStringSubmatch(modelName)
	if len(match) != 2 {
		return 0
	}
	version, _ := strconv.Atoi(match[1])
	return version
}
