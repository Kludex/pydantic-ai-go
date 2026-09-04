package ollama

import (
	"encoding/json"
	"fmt"
	"slices"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/embeddings"
)

type embeddingResponse struct {
	Model           string       `json:"model"`
	Embeddings      *[][]float64 `json:"embeddings"`
	TotalDuration   *int64       `json:"total_duration"`
	LoadDuration    *int64       `json:"load_duration"`
	PromptEvalCount *int         `json:"prompt_eval_count"`
}

func (model *Model) parseResponse(
	body []byte, inputs []string, inputType embeddings.InputType,
) (*embeddings.Result, error) {
	var response embeddingResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("ollama embeddings: decode response: %w", err)
	}
	if response.Embeddings == nil {
		return nil, fmt.Errorf("ollama embeddings: response omits embeddings")
	}
	if len(*response.Embeddings) != len(inputs) {
		return nil, fmt.Errorf(
			"ollama embeddings: response returned %d vectors for %d inputs", len(*response.Embeddings), len(inputs),
		)
	}
	vectors := make([][]float64, len(*response.Embeddings))
	for index, vector := range *response.Embeddings {
		if vector == nil {
			return nil, fmt.Errorf("ollama embeddings: response omits vector at index %d", index)
		}
		vectors[index] = slices.Clone(vector)
	}
	usage := ai.Usage{Requests: 1}
	if response.PromptEvalCount != nil {
		usage.InputTokens = *response.PromptEvalCount
	}
	details := map[string]any{}
	if response.TotalDuration != nil {
		details["total_duration"] = *response.TotalDuration
	}
	if response.LoadDuration != nil {
		details["load_duration"] = *response.LoadDuration
	}
	if len(details) == 0 {
		details = nil
	}
	modelName := response.Model
	if modelName == "" {
		modelName = model.name
	}
	return &embeddings.Result{
		Embeddings: vectors, Inputs: slices.Clone(inputs), InputType: inputType,
		ModelName: modelName, ProviderName: model.ProviderName(), ProviderURL: model.baseURL,
		Timestamp: time.Now().UTC(), Usage: usage, ProviderDetails: details,
	}, nil
}
