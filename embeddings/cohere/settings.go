package cohere

import (
	"fmt"

	"github.com/Kludex/pydantic-ai-go/embeddings"
)

const (
	inputTypeKey = "cohere_input_type"
	maxTokensKey = "cohere_max_tokens"
	truncateKey  = "cohere_truncate"
)

// InputType controls how Cohere optimizes an embedding.
type InputType string

const (
	// InputTypeSearchQuery identifies a search query.
	InputTypeSearchQuery InputType = "search_query"
	// InputTypeSearchDocument identifies a searchable document.
	InputTypeSearchDocument InputType = "search_document"
	// InputTypeClassification identifies classification input.
	InputTypeClassification InputType = "classification"
	// InputTypeClustering identifies clustering input.
	InputTypeClustering InputType = "clustering"
	// InputTypeImage identifies image input accepted by compatible Cohere models.
	InputTypeImage InputType = "image"
)

// Truncation controls which side Cohere removes when an input exceeds its limit.
type Truncation string

const (
	// TruncationNone rejects inputs that exceed the model limit.
	TruncationNone Truncation = "NONE"
	// TruncationEnd removes tokens from the end.
	TruncationEnd Truncation = "END"
	// TruncationStart removes tokens from the start.
	TruncationStart Truncation = "START"
)

// Settings builds Cohere-specific values into portable embedding settings.
type Settings struct {
	Common    embeddings.Settings
	InputType InputType
	MaxTokens *int
	Truncate  Truncation
}

// Build validates and returns detached portable settings.
func (settings Settings) Build() (embeddings.Settings, error) {
	built := settings.Common.Clone()
	if err := built.Validate(); err != nil {
		return embeddings.Settings{}, err
	}
	if settings.InputType != "" && !validInputType(settings.InputType) {
		return embeddings.Settings{}, fmt.Errorf("cohere embeddings: invalid input type %q", settings.InputType)
	}
	if settings.MaxTokens != nil && *settings.MaxTokens <= 0 {
		return embeddings.Settings{}, fmt.Errorf("cohere embeddings: max tokens must be greater than zero")
	}
	if settings.Truncate != "" && !validTruncation(settings.Truncate) {
		return embeddings.Settings{}, fmt.Errorf("cohere embeddings: invalid truncation %q", settings.Truncate)
	}
	if built.ExtraBody == nil {
		built.ExtraBody = map[string]any{}
	}
	values := []struct {
		key     string
		value   any
		include bool
	}{
		{key: inputTypeKey, value: settings.InputType, include: settings.InputType != ""},
		{key: maxTokensKey, value: pointerValue(settings.MaxTokens), include: settings.MaxTokens != nil},
		{key: truncateKey, value: settings.Truncate, include: settings.Truncate != ""},
	}
	for _, value := range values {
		if !value.include {
			continue
		}
		if _, exists := built.ExtraBody[value.key]; exists {
			return embeddings.Settings{}, fmt.Errorf(
				"cohere embeddings: extra body field %q conflicts with typed settings", value.key,
			)
		}
		built.ExtraBody[value.key] = value.value
	}
	if len(built.ExtraBody) == 0 {
		built.ExtraBody = nil
	}
	return built, nil
}

func validInputType(value InputType) bool {
	return value == InputTypeSearchQuery || value == InputTypeSearchDocument || value == InputTypeClassification ||
		value == InputTypeClustering || value == InputTypeImage
}

func validTruncation(value Truncation) bool {
	return value == TruncationNone || value == TruncationEnd || value == TruncationStart
}

func pointerValue[T any](value *T) any {
	if value == nil {
		return nil
	}
	return *value
}
