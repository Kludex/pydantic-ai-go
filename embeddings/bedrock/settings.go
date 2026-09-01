package bedrock

import (
	"fmt"

	"github.com/Kludex/pydantic-ai-go/embeddings"
)

const (
	titanNormalizeKey   = "bedrock_titan_normalize"
	cohereMaxTokensKey  = "bedrock_cohere_max_tokens"
	cohereInputTypeKey  = "bedrock_cohere_input_type"
	cohereTruncateKey   = "bedrock_cohere_truncate"
	novaTruncateKey     = "bedrock_nova_truncate"
	novaPurposeKey      = "bedrock_nova_embedding_purpose"
	inferenceProfileKey = "bedrock_inference_profile"
	maxConcurrencyKey   = "bedrock_max_concurrency"
)

// CohereInputType controls Cohere's embedding optimization.
type CohereInputType string

const (
	// CohereInputSearchDocument identifies content stored for search.
	CohereInputSearchDocument CohereInputType = "search_document"
	// CohereInputSearchQuery identifies a query used for search.
	CohereInputSearchQuery CohereInputType = "search_query"
	// CohereInputClassification identifies classification input.
	CohereInputClassification CohereInputType = "classification"
	// CohereInputClustering identifies clustering input.
	CohereInputClustering CohereInputType = "clustering"
)

// Truncation controls which side of oversized input a model removes.
type Truncation string

const (
	// TruncationNone rejects oversized input.
	TruncationNone Truncation = "NONE"
	// TruncationStart removes content from the start.
	TruncationStart Truncation = "START"
	// TruncationEnd removes content from the end.
	TruncationEnd Truncation = "END"
)

// NovaPurpose controls how Nova optimizes an embedding.
type NovaPurpose string

const (
	// NovaPurposeGenericIndex identifies general indexed content.
	NovaPurposeGenericIndex NovaPurpose = "GENERIC_INDEX"
	// NovaPurposeGenericRetrieval identifies general retrieval queries.
	NovaPurposeGenericRetrieval NovaPurpose = "GENERIC_RETRIEVAL"
	// NovaPurposeTextRetrieval identifies text retrieval input.
	NovaPurposeTextRetrieval NovaPurpose = "TEXT_RETRIEVAL"
	// NovaPurposeClassification identifies classification input.
	NovaPurposeClassification NovaPurpose = "CLASSIFICATION"
	// NovaPurposeClustering identifies clustering input.
	NovaPurposeClustering NovaPurpose = "CLUSTERING"
)

// Settings builds Bedrock-specific values into portable embedding settings.
type Settings struct {
	Common           embeddings.Settings
	TitanNormalize   *bool
	CohereMaxTokens  *int
	CohereInputType  CohereInputType
	CohereTruncate   Truncation
	NovaTruncate     Truncation
	NovaPurpose      NovaPurpose
	InferenceProfile string
	MaxConcurrency   *int
}

// Build validates and returns detached portable settings.
func (settings Settings) Build() (embeddings.Settings, error) {
	built := settings.Common.Clone()
	if err := built.Validate(); err != nil {
		return embeddings.Settings{}, err
	}
	if settings.CohereMaxTokens != nil && *settings.CohereMaxTokens <= 0 {
		return embeddings.Settings{}, fmt.Errorf("bedrock embeddings: Cohere max tokens must be greater than zero")
	}
	if settings.MaxConcurrency != nil && *settings.MaxConcurrency < 1 {
		return embeddings.Settings{}, fmt.Errorf("bedrock embeddings: max concurrency must be at least one")
	}
	if settings.CohereInputType != "" && !validCohereInputType(settings.CohereInputType) {
		return embeddings.Settings{}, fmt.Errorf("bedrock embeddings: invalid Cohere input type %q", settings.CohereInputType)
	}
	if settings.CohereTruncate != "" && !validTruncation(settings.CohereTruncate) {
		return embeddings.Settings{}, fmt.Errorf("bedrock embeddings: invalid Cohere truncation %q", settings.CohereTruncate)
	}
	if settings.NovaTruncate != "" && !validTruncation(settings.NovaTruncate) {
		return embeddings.Settings{}, fmt.Errorf("bedrock embeddings: invalid Nova truncation %q", settings.NovaTruncate)
	}
	if settings.NovaPurpose != "" && !validNovaPurpose(settings.NovaPurpose) {
		return embeddings.Settings{}, fmt.Errorf("bedrock embeddings: invalid Nova purpose %q", settings.NovaPurpose)
	}
	if built.ExtraBody == nil {
		built.ExtraBody = map[string]any{}
	}
	values := []struct {
		key     string
		value   any
		include bool
	}{
		{titanNormalizeKey, pointerValue(settings.TitanNormalize), settings.TitanNormalize != nil},
		{cohereMaxTokensKey, pointerValue(settings.CohereMaxTokens), settings.CohereMaxTokens != nil},
		{cohereInputTypeKey, settings.CohereInputType, settings.CohereInputType != ""},
		{cohereTruncateKey, settings.CohereTruncate, settings.CohereTruncate != ""},
		{novaTruncateKey, settings.NovaTruncate, settings.NovaTruncate != ""},
		{novaPurposeKey, settings.NovaPurpose, settings.NovaPurpose != ""},
		{inferenceProfileKey, settings.InferenceProfile, settings.InferenceProfile != ""},
		{maxConcurrencyKey, pointerValue(settings.MaxConcurrency), settings.MaxConcurrency != nil},
	}
	for _, value := range values {
		if !value.include {
			continue
		}
		if _, exists := built.ExtraBody[value.key]; exists {
			return embeddings.Settings{}, fmt.Errorf(
				"bedrock embeddings: extra body field %q conflicts with typed settings", value.key,
			)
		}
		built.ExtraBody[value.key] = value.value
	}
	if len(built.ExtraBody) == 0 {
		built.ExtraBody = nil
	}
	return built, nil
}
