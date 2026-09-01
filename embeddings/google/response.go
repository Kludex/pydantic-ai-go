package google

import (
	"encoding/json"
	"fmt"
	"slices"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/embeddings"
	modelgoogle "github.com/Kludex/pydantic-ai-go/models/google"
)

type embeddingData struct {
	Values     *[]float64           `json:"values"`
	Statistics *embeddingStatistics `json:"statistics"`
}

type embeddingStatistics struct {
	TokenCount      float64 `json:"token_count"`
	TokenCountCamel float64 `json:"tokenCount"`
}

type embedResponse struct {
	Embeddings  []embeddingData `json:"embeddings"`
	Predictions []struct {
		Embeddings embeddingData `json:"embeddings"`
	} `json:"predictions"`
}

func (model *Model) parseResponse(
	body []byte, inputs []string, inputType embeddings.InputType,
) (*embeddings.Result, error) {
	var response embedResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("google embeddings: decode response: %w", err)
	}
	items := response.Embeddings
	if model.transport == modelgoogle.TransportVertexAI {
		items = make([]embeddingData, len(response.Predictions))
		for index, prediction := range response.Predictions {
			items[index] = prediction.Embeddings
		}
	}
	if len(items) != len(inputs) {
		return nil, fmt.Errorf(
			"google embeddings: response returned %d vectors for %d inputs", len(items), len(inputs),
		)
	}
	vectors := make([][]float64, len(items))
	usage := ai.Usage{Requests: 1}
	for index, item := range items {
		if item.Values == nil {
			return nil, fmt.Errorf("google embeddings: response omits vector at index %d", index)
		}
		vectors[index] = slices.Clone(*item.Values)
		if item.Statistics != nil {
			tokens := item.Statistics.TokenCount
			if tokens == 0 {
				tokens = item.Statistics.TokenCountCamel
			}
			usage.InputTokens += int(tokens)
		}
	}
	return &embeddings.Result{
		Embeddings: vectors, Inputs: slices.Clone(inputs), InputType: inputType,
		ModelName: model.name, ProviderName: model.providerName, ProviderURL: model.baseURL,
		Timestamp: time.Now().UTC(), Usage: usage,
	}, nil
}
