package bedrock

import (
	"context"
	"fmt"
	"maps"

	"github.com/Kludex/pydantic-ai-go/ai/embeddings"
)

type indexedEmbedding struct {
	index  int
	vector []float64
	tokens int
	err    error
}

func (model *Model) embedConcurrent(
	ctx context.Context, client Client, inputs []string, inputType embeddings.InputType,
	settings requestSettings, extra map[string]any,
) ([][]float64, int, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	results := make(chan indexedEmbedding, len(inputs))
	jobs := make(chan struct {
		index int
		input string
	}, len(inputs))
	for index, input := range inputs {
		jobs <- struct {
			index int
			input string
		}{index: index, input: input}
	}
	close(jobs)
	for range min(settings.maxConcurrency, len(inputs)) {
		go func() {
			for job := range jobs {
				body, err := model.requestBody([]string{job.input}, inputType, settings, extra)
				if err != nil {
					results <- indexedEmbedding{index: job.index, err: err}
					continue
				}
				response, err := client.InvokeModel(ctx, Request{
					ModelID: modelID(model.modelName, settings), Body: body, Headers: maps.Clone(settings.headers),
				})
				if err != nil {
					results <- indexedEmbedding{index: job.index, err: fmt.Errorf(
						"bedrock embeddings: invoke %q: %w", model.modelName, err,
					)}
					continue
				}
				vectors, _, err := model.parseResponse(response.Body)
				if err != nil {
					results <- indexedEmbedding{index: job.index, err: err}
					continue
				}
				results <- indexedEmbedding{index: job.index, vector: vectors[0], tokens: response.InputTokens}
			}
		}()
	}
	vectors := make([][]float64, len(inputs))
	tokens := 0
	var firstErr error
	for range inputs {
		result := <-results
		if result.err != nil {
			if firstErr == nil {
				firstErr = result.err
				cancel()
			}
			continue
		}
		vectors[result.index] = result.vector
		tokens += result.tokens
	}
	if firstErr != nil {
		return nil, 0, firstErr
	}
	return vectors, tokens, nil
}
