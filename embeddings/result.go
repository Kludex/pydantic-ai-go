package embeddings

import (
	"slices"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
)

// Result contains one vector for each original input.
type Result struct {
	Embeddings         [][]float64
	Inputs             []string
	InputType          InputType
	ModelName          string
	ProviderName       string
	ProviderURL        string
	Timestamp          time.Time
	Usage              ai.Usage
	ProviderDetails    map[string]any
	ProviderResponseID string
	Warnings           []string
}

// Clone returns a detached result.
func (result Result) Clone() Result {
	result.Embeddings = cloneEmbeddings(result.Embeddings)
	result.Inputs = slices.Clone(result.Inputs)
	result.Usage = result.Usage.Clone()
	result.ProviderDetails = cloneMap(result.ProviderDetails)
	result.Warnings = slices.Clone(result.Warnings)
	return result
}

// At returns a detached vector by position.
func (result Result) At(index int) ([]float64, bool) {
	if index < 0 || index >= len(result.Embeddings) {
		return nil, false
	}
	return slices.Clone(result.Embeddings[index]), true
}

// ForInput returns the first detached vector associated with input.
func (result Result) ForInput(input string) ([]float64, bool) {
	for index, candidate := range result.Inputs {
		if candidate == input && index < len(result.Embeddings) {
			return slices.Clone(result.Embeddings[index]), true
		}
	}
	return nil, false
}

// Price calculates the result's price using the bundled genai-prices snapshot.
func (result Result) Price() (ai.PriceCalculation, error) {
	response := ai.ModelResponse{
		ModelName: result.ModelName, ProviderName: result.ProviderName, ProviderURL: result.ProviderURL,
		Timestamp: result.Timestamp, Usage: result.Usage.Clone(),
	}
	return response.Price()
}

func cloneResult(result *Result) *Result {
	cloned := result.Clone()
	return &cloned
}

func cloneEmbeddings(embeddings [][]float64) [][]float64 {
	if embeddings == nil {
		return nil
	}
	cloned := make([][]float64, len(embeddings))
	for index, embedding := range embeddings {
		cloned[index] = slices.Clone(embedding)
	}
	return cloned
}
