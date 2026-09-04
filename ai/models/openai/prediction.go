package openai

import (
	"encoding/json"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// Prediction supplies expected output so compatible Chat Completions models can
// generate unchanged content faster. Set either Content or ContentParts.
type Prediction struct {
	// Content is the complete expected text.
	Content string
	// ContentParts contains segmented expected text with optional cache boundaries.
	ContentParts []PredictionContentPart
}

// PredictionContentPart is one expected text segment.
type PredictionContentPart struct {
	// Text is one expected output segment.
	Text string
	// PromptCacheBreakpoint inserts an explicit cache boundary after this segment.
	PromptCacheBreakpoint bool
}

type chatPrediction struct {
	Type    string `json:"type"`
	Content any    `json:"content"`
}

type chatPredictionContentPart struct {
	Type                  string                       `json:"type"`
	Text                  string                       `json:"text"`
	PromptCacheBreakpoint *openAIPromptCacheBreakpoint `json:"prompt_cache_breakpoint,omitempty"`
}

func marshalPrediction(prediction *Prediction) (json.RawMessage, error) {
	if prediction == nil {
		return nil, nil
	}
	if prediction.Content != "" && len(prediction.ContentParts) > 0 {
		return nil, fmt.Errorf("openai: prediction must set either content or content parts, not both")
	}
	var content any
	switch {
	case prediction.Content != "":
		content = prediction.Content
	case len(prediction.ContentParts) > 0:
		parts := make([]chatPredictionContentPart, len(prediction.ContentParts))
		for index, part := range prediction.ContentParts {
			parts[index] = chatPredictionContentPart{Type: "text", Text: part.Text}
			if part.PromptCacheBreakpoint {
				parts[index].PromptCacheBreakpoint = &openAIPromptCacheBreakpoint{Mode: "explicit"}
			}
		}
		content = parts
	default:
		return nil, fmt.Errorf("openai: prediction content must not be empty")
	}
	encoded, _ := json.Marshal(chatPrediction{Type: "content", Content: content})
	return encoded, nil
}

func extractPredictionSettings(settings ai.ModelSettings) (ai.ModelSettings, *chatPrediction, error) {
	settings = settings.Clone()
	extra := settings.ExtraBody
	value, exists := extra[predictionSetting]
	if !exists {
		return settings, nil, nil
	}
	delete(extra, predictionSetting)
	if len(extra) == 0 {
		settings.ExtraBody = nil
	}
	encoded, ok := value.(json.RawMessage)
	if !ok {
		return ai.ModelSettings{}, nil, fmt.Errorf("openai: prediction must use Prediction")
	}
	var prediction chatPrediction
	if err := json.Unmarshal(encoded, &prediction); err != nil {
		return ai.ModelSettings{}, nil, fmt.Errorf("openai: decode prediction: %w", err)
	}
	if prediction.Type != "content" || prediction.Content == nil {
		return ai.ModelSettings{}, nil, fmt.Errorf("openai: invalid prediction payload")
	}
	return settings, &prediction, nil
}
