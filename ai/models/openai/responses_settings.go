package openai

import (
	"fmt"
	"slices"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

type responsesSettings struct {
	IncludeRawAnnotations bool
	Include               []string
	annotationsConfigured bool
}

func extractResponsesSettings(settings ai.ModelSettings) (ai.ModelSettings, responsesSettings, error) {
	settings, prediction, err := extractPredictionSettings(settings)
	if err != nil {
		return ai.ModelSettings{}, responsesSettings{}, err
	}
	if prediction != nil {
		return ai.ModelSettings{}, responsesSettings{}, fmt.Errorf(
			"openai: prediction is only supported by Chat Completions",
		)
	}
	settings, responseSettings, err := extractRawAnnotationSettings(settings)
	if err != nil {
		return ai.ModelSettings{}, responsesSettings{}, err
	}
	value, exists := settings.ExtraBody[responsesIncludeSetting]
	if !exists {
		return settings, responseSettings, nil
	}
	delete(settings.ExtraBody, responsesIncludeSetting)
	if len(settings.ExtraBody) == 0 {
		settings.ExtraBody = nil
	}
	included, ok := value.([]string)
	if !ok {
		return ai.ModelSettings{}, responsesSettings{}, fmt.Errorf("openai: Responses include values must be strings")
	}
	responseSettings.Include = slices.Clone(included)
	return settings, responseSettings, nil
}

func extractRawAnnotationSettings(settings ai.ModelSettings) (ai.ModelSettings, responsesSettings, error) {
	settings = settings.Clone()
	extra := settings.ExtraBody
	value, exists := extra[includeRawAnnotationsSetting]
	if !exists {
		return settings, responsesSettings{}, nil
	}
	delete(extra, includeRawAnnotationsSetting)
	if len(extra) == 0 {
		settings.ExtraBody = nil
	}
	include, ok := value.(bool)
	if !ok {
		return ai.ModelSettings{}, responsesSettings{}, fmt.Errorf(
			"openai: include raw annotations must be a boolean",
		)
	}
	return settings, responsesSettings{
		IncludeRawAnnotations: include,
		annotationsConfigured: true,
	}, nil
}
