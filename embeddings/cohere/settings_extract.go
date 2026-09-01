package cohere

import (
	"fmt"

	"github.com/Kludex/pydantic-ai-go/embeddings"
)

type localSettings struct {
	inputType   InputType
	maxTokens   *int
	truncate    Truncation
	truncateSet bool
}

func extractSettings(settings embeddings.Settings) (embeddings.Settings, localSettings, error) {
	local := localSettings{}
	if settings.ExtraBody == nil {
		return settings, local, nil
	}
	settings = settings.Clone()
	body := settings.ExtraBody
	values := []struct {
		key string
		set func(any) error
	}{
		{key: inputTypeKey, set: func(value any) error {
			inputType, ok := value.(InputType)
			if !ok {
				text, textOK := value.(string)
				if !textOK {
					return fmt.Errorf("cohere embeddings: %s must be an input type", inputTypeKey)
				}
				inputType = InputType(text)
			}
			if !validInputType(inputType) {
				return fmt.Errorf("cohere embeddings: invalid input type %q", inputType)
			}
			local.inputType = inputType
			return nil
		}},
		{key: maxTokensKey, set: func(value any) error {
			maxTokens, ok := value.(int)
			if !ok || maxTokens <= 0 {
				return fmt.Errorf("cohere embeddings: %s must be a positive integer", maxTokensKey)
			}
			local.maxTokens = &maxTokens
			return nil
		}},
		{key: truncateKey, set: func(value any) error {
			truncate, ok := value.(Truncation)
			if !ok {
				text, textOK := value.(string)
				if !textOK {
					return fmt.Errorf("cohere embeddings: %s must be a truncation", truncateKey)
				}
				truncate = Truncation(text)
			}
			if !validTruncation(truncate) {
				return fmt.Errorf("cohere embeddings: invalid truncation %q", truncate)
			}
			local.truncate = truncate
			local.truncateSet = true
			return nil
		}},
	}
	for _, value := range values {
		if raw, exists := body[value.key]; exists {
			if err := value.set(raw); err != nil {
				return embeddings.Settings{}, localSettings{}, err
			}
			delete(body, value.key)
		}
	}
	settings.ExtraBody = body
	return settings, local, nil
}

func mergeSettings(base, override embeddings.Settings) embeddings.Settings {
	merged := embeddings.MergeSettings(base, override)
	if override.ExtraBody == nil || base.ExtraBody == nil {
		return merged
	}
	for _, key := range []string{inputTypeKey, maxTokensKey, truncateKey} {
		if _, overridden := override.ExtraBody[key]; overridden {
			continue
		}
		if value, exists := base.ExtraBody[key]; exists {
			merged.ExtraBody[key] = value
		}
	}
	return merged
}

func clonePointer[T any](value *T) *T {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}
