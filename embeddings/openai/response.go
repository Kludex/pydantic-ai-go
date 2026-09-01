package openai

import (
	"encoding/json"
	"fmt"
	"slices"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/embeddings"
)

type embeddingResponse struct {
	Data []struct {
		Embedding *[]float64 `json:"embedding"`
		Index     int        `json:"index"`
	} `json:"data"`
	Model string         `json:"model"`
	Usage map[string]int `json:"usage"`
}

func (model *Model) parseResponse(
	body []byte, inputs []string, inputType embeddings.InputType,
) (*embeddings.Result, error) {
	var response embeddingResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("openai embeddings: decode response: %w", err)
	}
	if len(response.Data) != len(inputs) {
		return nil, fmt.Errorf(
			"openai embeddings: response returned %d vectors for %d inputs", len(response.Data), len(inputs),
		)
	}
	vectors := make([][]float64, len(inputs))
	seen := make([]bool, len(inputs))
	for _, item := range response.Data {
		if item.Index < 0 || item.Index >= len(vectors) || seen[item.Index] {
			return nil, fmt.Errorf("openai embeddings: response contains invalid embedding index %d", item.Index)
		}
		if item.Embedding == nil {
			return nil, fmt.Errorf("openai embeddings: response omits vector at index %d", item.Index)
		}
		seen[item.Index] = true
		vectors[item.Index] = slices.Clone(*item.Embedding)
	}
	usage := ai.Usage{Requests: 1, InputTokens: response.Usage["prompt_tokens"]}
	details := make(map[string]int, len(response.Usage))
	for name, value := range response.Usage {
		if name != "prompt_tokens" && name != "total_tokens" {
			details[name] = value
		}
	}
	if len(details) > 0 {
		usage.Details = details
	}
	modelName := response.Model
	if modelName == "" {
		modelName = model.name
	}
	return &embeddings.Result{
		Embeddings: vectors, Inputs: slices.Clone(inputs), InputType: inputType,
		ModelName: modelName, ProviderName: model.providerName, ProviderURL: model.baseURL,
		Timestamp: time.Now().UTC(), Usage: usage,
	}, nil
}
