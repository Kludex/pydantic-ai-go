package bedrock

import (
	"encoding/json"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func (model *Model) parseResponse(data []byte) ([][]float64, string, error) {
	switch model.family {
	case familyTitan:
		var response struct {
			Embedding []float64 `json:"embedding"`
		}
		if err := json.Unmarshal(data, &response); err != nil {
			return nil, "", responseError(err)
		}
		if response.Embedding == nil {
			return nil, "", fmt.Errorf("bedrock embeddings: Titan response is missing embedding")
		}
		return [][]float64{response.Embedding}, "", nil
	case familyCohere:
		var response struct {
			Embeddings json.RawMessage `json:"embeddings"`
			ID         string          `json:"id"`
		}
		if err := json.Unmarshal(data, &response); err != nil {
			return nil, "", responseError(err)
		}
		var vectors [][]float64
		if err := json.Unmarshal(response.Embeddings, &vectors); err != nil {
			var byType struct {
				Float [][]float64 `json:"float"`
			}
			if typedErr := json.Unmarshal(response.Embeddings, &byType); typedErr != nil {
				return nil, "", &ai.ModelTransportError{ProviderName: "bedrock embeddings", Operation: "decode Cohere response", Err: err}
			}
			vectors = byType.Float
		}
		if vectors == nil {
			return nil, "", fmt.Errorf("bedrock embeddings: Cohere response is missing float embeddings")
		}
		return vectors, response.ID, nil
	default:
		var response struct {
			Embeddings []struct {
				Embedding []float64 `json:"embedding"`
			} `json:"embeddings"`
		}
		if err := json.Unmarshal(data, &response); err != nil {
			return nil, "", responseError(err)
		}
		if len(response.Embeddings) == 0 || response.Embeddings[0].Embedding == nil {
			return nil, "", fmt.Errorf("bedrock embeddings: Nova response is missing embedding")
		}
		return [][]float64{response.Embeddings[0].Embedding}, "", nil
	}
}

func responseError(err error) error {
	return &ai.ModelTransportError{ProviderName: "bedrock embeddings", Operation: "decode response", Err: err}
}
