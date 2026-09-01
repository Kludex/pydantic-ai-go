package bedrock

import (
	"fmt"
	"maps"

	"github.com/Kludex/pydantic-ai-go/embeddings"
)

type requestSettings struct {
	titanNormalize   *bool
	cohereMaxTokens  *int
	cohereInputType  CohereInputType
	cohereTruncate   Truncation
	novaTruncate     Truncation
	novaPurpose      NovaPurpose
	inferenceProfile string
	maxConcurrency   int
	dimensions       *int
	truncate         bool
	headers          map[string]string
}

func extractSettings(settings embeddings.Settings) (requestSettings, map[string]any, error) {
	body := maps.Clone(settings.ExtraBody)
	values := requestSettings{
		dimensions: settings.Dimensions, maxConcurrency: 5, headers: maps.Clone(settings.ExtraHeaders),
	}
	var err error
	values.titanNormalize, err = takePointer[bool](body, titanNormalizeKey)
	if err != nil {
		return requestSettings{}, nil, err
	}
	values.cohereMaxTokens, err = takePointer[int](body, cohereMaxTokensKey)
	if err != nil {
		return requestSettings{}, nil, err
	}
	if values.cohereMaxTokens != nil && *values.cohereMaxTokens <= 0 {
		return requestSettings{}, nil, fmt.Errorf("bedrock embeddings: %q must be a positive integer", cohereMaxTokensKey)
	}
	values.cohereInputType, err = takeValue[CohereInputType](body, cohereInputTypeKey)
	if err != nil {
		return requestSettings{}, nil, err
	}
	values.cohereTruncate, err = takeValue[Truncation](body, cohereTruncateKey)
	if err != nil {
		return requestSettings{}, nil, err
	}
	values.novaTruncate, err = takeValue[Truncation](body, novaTruncateKey)
	if err != nil {
		return requestSettings{}, nil, err
	}
	values.novaPurpose, err = takeValue[NovaPurpose](body, novaPurposeKey)
	if err != nil {
		return requestSettings{}, nil, err
	}
	values.inferenceProfile, err = takeValue[string](body, inferenceProfileKey)
	if err != nil {
		return requestSettings{}, nil, err
	}
	if values.cohereInputType != "" && !validCohereInputType(values.cohereInputType) {
		return requestSettings{}, nil, fmt.Errorf("bedrock embeddings: invalid Cohere input type %q", values.cohereInputType)
	}
	if values.cohereTruncate != "" && !validTruncation(values.cohereTruncate) {
		return requestSettings{}, nil, fmt.Errorf("bedrock embeddings: invalid Cohere truncation %q", values.cohereTruncate)
	}
	if values.novaTruncate != "" && !validTruncation(values.novaTruncate) {
		return requestSettings{}, nil, fmt.Errorf("bedrock embeddings: invalid Nova truncation %q", values.novaTruncate)
	}
	if values.novaPurpose != "" && !validNovaPurpose(values.novaPurpose) {
		return requestSettings{}, nil, fmt.Errorf("bedrock embeddings: invalid Nova purpose %q", values.novaPurpose)
	}
	if maxConcurrency, exists := body[maxConcurrencyKey]; exists {
		delete(body, maxConcurrencyKey)
		value, ok := maxConcurrency.(int)
		if !ok || value < 1 {
			return requestSettings{}, nil, fmt.Errorf("bedrock embeddings: %q must be an integer of at least one", maxConcurrencyKey)
		}
		values.maxConcurrency = value
	}
	if settings.Truncate != nil {
		values.truncate = *settings.Truncate
	}
	return values, body, nil
}

func mergeSettings(base, override embeddings.Settings) embeddings.Settings {
	merged := embeddings.MergeSettings(base, override)
	if override.ExtraBody == nil || base.ExtraBody == nil {
		return merged
	}
	for _, key := range []string{
		titanNormalizeKey, cohereMaxTokensKey, cohereInputTypeKey, cohereTruncateKey,
		novaTruncateKey, novaPurposeKey, inferenceProfileKey, maxConcurrencyKey,
	} {
		if _, overridden := override.ExtraBody[key]; overridden {
			continue
		}
		if value, exists := base.ExtraBody[key]; exists {
			merged.ExtraBody[key] = value
		}
	}
	return merged
}

func takePointer[T any](body map[string]any, key string) (*T, error) {
	value, exists := body[key]
	if !exists {
		return nil, nil
	}
	delete(body, key)
	typed, ok := value.(T)
	if !ok {
		return nil, fmt.Errorf("bedrock embeddings: %q has invalid type %T", key, value)
	}
	return &typed, nil
}

func takeValue[T ~string](body map[string]any, key string) (T, error) {
	value, exists := body[key]
	if !exists {
		return "", nil
	}
	delete(body, key)
	if typed, ok := value.(T); ok {
		return typed, nil
	}
	if text, ok := value.(string); ok {
		return T(text), nil
	}
	return "", fmt.Errorf("bedrock embeddings: %q has invalid type %T", key, value)
}

func validCohereInputType(value CohereInputType) bool {
	return value == CohereInputSearchDocument || value == CohereInputSearchQuery ||
		value == CohereInputClassification || value == CohereInputClustering
}

func validTruncation(value Truncation) bool {
	return value == TruncationNone || value == TruncationStart || value == TruncationEnd
}

func validNovaPurpose(value NovaPurpose) bool {
	return value == NovaPurposeGenericIndex || value == NovaPurposeGenericRetrieval ||
		value == NovaPurposeTextRetrieval || value == NovaPurposeClassification || value == NovaPurposeClustering
}

func pointerValue[T any](value *T) any {
	if value == nil {
		return nil
	}
	return *value
}
