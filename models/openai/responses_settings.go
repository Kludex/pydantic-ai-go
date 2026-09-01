package openai

import (
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go"
)

type responsesSettings struct {
	IncludeRawAnnotations bool
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
	return extractRawAnnotationSettings(settings)
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
