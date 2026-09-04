package voyageai

import (
	"fmt"

	"github.com/Kludex/pydantic-ai-go/embeddings"
)

const inputTypeKey = "voyageai_input_type"

// InputType controls VoyageAI's retrieval prefix.
type InputType string

const (
	// InputTypeQuery optimizes the input as a retrieval query.
	InputTypeQuery InputType = "query"
	// InputTypeDocument optimizes the input as a retrieval document.
	InputTypeDocument InputType = "document"
	// InputTypeNone disables VoyageAI's retrieval prefix.
	InputTypeNone InputType = "none"
)

// Settings builds VoyageAI-specific values into portable embedding settings.
type Settings struct {
	// Common contains portable embedding settings.
	Common embeddings.Settings
	// InputType selects query, document, or unprefixed embedding behavior.
	InputType InputType
}

// Build validates and returns detached portable settings.
func (settings Settings) Build() (embeddings.Settings, error) {
	built := settings.Common.Clone()
	if err := built.Validate(); err != nil {
		return embeddings.Settings{}, err
	}
	if settings.InputType != "" && !validInputType(settings.InputType) {
		return embeddings.Settings{}, fmt.Errorf("voyageai embeddings: invalid input type %q", settings.InputType)
	}
	if settings.InputType == "" {
		return built, nil
	}
	if built.ExtraBody == nil {
		built.ExtraBody = map[string]any{}
	}
	if _, exists := built.ExtraBody[inputTypeKey]; exists {
		return embeddings.Settings{}, fmt.Errorf(
			"voyageai embeddings: extra body field %q conflicts with typed settings", inputTypeKey,
		)
	}
	built.ExtraBody[inputTypeKey] = settings.InputType
	return built, nil
}

func extractSettings(settings embeddings.Settings) (embeddings.Settings, InputType, error) {
	if settings.ExtraBody == nil {
		return settings, "", nil
	}
	settings = settings.Clone()
	value, exists := settings.ExtraBody[inputTypeKey]
	if !exists {
		return settings, "", nil
	}
	delete(settings.ExtraBody, inputTypeKey)
	inputType, ok := value.(InputType)
	if !ok {
		text, textOK := value.(string)
		if !textOK {
			return embeddings.Settings{}, "", fmt.Errorf(
				"voyageai embeddings: %s must be an input type", inputTypeKey,
			)
		}
		inputType = InputType(text)
	}
	if !validInputType(inputType) {
		return embeddings.Settings{}, "", fmt.Errorf("voyageai embeddings: invalid input type %q", inputType)
	}
	return settings, inputType, nil
}

func mergeSettings(base, override embeddings.Settings) embeddings.Settings {
	merged := embeddings.MergeSettings(base, override)
	if override.ExtraBody == nil || base.ExtraBody == nil {
		return merged
	}
	if _, overridden := override.ExtraBody[inputTypeKey]; overridden {
		return merged
	}
	if value, exists := base.ExtraBody[inputTypeKey]; exists {
		merged.ExtraBody[inputTypeKey] = value
	}
	return merged
}

func validInputType(value InputType) bool {
	return value == InputTypeQuery || value == InputTypeDocument || value == InputTypeNone
}
