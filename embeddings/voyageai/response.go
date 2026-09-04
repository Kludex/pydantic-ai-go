package voyageai

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/embeddings"
)

type embeddingResponse struct {
	Data []struct {
		Embedding json.RawMessage `json:"embedding"`
		Index     int             `json:"index"`
	} `json:"data"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

func (model *Model) parseResponse(
	body []byte, inputs []string, inputType embeddings.InputType,
) (*embeddings.Result, error) {
	var response embeddingResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("voyageai embeddings: decode response: %w", err)
	}
	if len(response.Data) != len(inputs) {
		return nil, fmt.Errorf(
			"voyageai embeddings: response returned %d vectors for %d inputs", len(response.Data), len(inputs),
		)
	}
	vectors := make([][]float64, len(inputs))
	seen := make([]bool, len(inputs))
	for _, item := range response.Data {
		if item.Index < 0 || item.Index >= len(vectors) || seen[item.Index] {
			return nil, fmt.Errorf("voyageai embeddings: response contains invalid embedding index %d", item.Index)
		}
		vector, err := decodeEmbedding(item.Embedding)
		if err != nil {
			return nil, fmt.Errorf("voyageai embeddings: decode vector at index %d: %w", item.Index, err)
		}
		seen[item.Index] = true
		vectors[item.Index] = vector
	}
	return &embeddings.Result{
		Embeddings: vectors, Inputs: slices.Clone(inputs), InputType: inputType,
		ModelName: model.name, ProviderName: model.providerName, ProviderURL: model.baseURL,
		Timestamp: time.Now().UTC(), Usage: ai.Usage{Requests: 1, InputTokens: response.Usage.TotalTokens},
	}, nil
}

func decodeEmbedding(data json.RawMessage) ([]float64, error) {
	if len(data) == 0 || string(data) == "null" {
		return nil, fmt.Errorf("response omitted embedding")
	}
	if data[0] != '"' {
		var vector []float64
		if err := json.Unmarshal(data, &vector); err != nil {
			return nil, err
		}
		return vector, nil
	}
	encoded := string(data[1 : len(data)-1])
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, err
	}
	if len(decoded)%4 != 0 {
		return nil, fmt.Errorf("base64 embedding has %d bytes, want a multiple of 4", len(decoded))
	}
	vector := make([]float64, len(decoded)/4)
	for index := range vector {
		bits := binary.LittleEndian.Uint32(decoded[index*4 : index*4+4])
		vector[index] = float64(math.Float32frombits(bits))
	}
	return vector, nil
}
