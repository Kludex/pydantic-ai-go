package cohere

import (
	"encoding/json"
	"fmt"
	"slices"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
)

type embeddingResponse struct {
	ID         string `json:"id"`
	Embeddings struct {
		Float *[][]float64 `json:"float"`
	} `json:"embeddings"`
	Meta *struct {
		BilledUnits map[string]float64 `json:"billed_units"`
	} `json:"meta"`
}

func (model *Model) parseResponse(
	body []byte, inputs []string, inputType embeddings.InputType,
) (*embeddings.Result, error) {
	var response embeddingResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("cohere embeddings: decode response: %w", err)
	}
	if response.Embeddings.Float == nil {
		return nil, fmt.Errorf("cohere embeddings: response omitted float embeddings")
	}
	items := *response.Embeddings.Float
	if len(items) != len(inputs) {
		return nil, fmt.Errorf(
			"cohere embeddings: response returned %d vectors for %d inputs", len(items), len(inputs),
		)
	}
	vectors := make([][]float64, len(items))
	for index, vector := range items {
		if vector == nil {
			return nil, fmt.Errorf("cohere embeddings: response omitted vector at index %d", index)
		}
		vectors[index] = slices.Clone(vector)
	}
	usage := ai.Usage{Requests: 1}
	if response.Meta != nil {
		details := map[string]int{}
		for name, value := range response.Meta.BilledUnits {
			count := int(value)
			if count <= 0 {
				continue
			}
			if name == "input_tokens" {
				usage.InputTokens = count
			} else {
				details[name] = count
			}
		}
		if len(details) > 0 {
			usage.Details = details
		}
	}
	return &embeddings.Result{
		Embeddings: vectors, Inputs: slices.Clone(inputs), InputType: inputType,
		ModelName: model.name, ProviderName: model.providerName, ProviderURL: model.baseURL,
		Timestamp: time.Now().UTC(), Usage: usage, ProviderResponseID: response.ID,
	}, nil
}
