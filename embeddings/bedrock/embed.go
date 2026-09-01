package bedrock

import (
	"context"
	"fmt"
	"maps"
	"time"

	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/embeddings"
)

// Embed generates one vector per input.
func (model *Model) Embed(
	ctx context.Context, inputs []string, inputType embeddings.InputType, settings embeddings.Settings,
) (*embeddings.Result, error) {
	if err := inputTypeValid(inputType); err != nil {
		return nil, err
	}
	merged := mergeSettings(model.settings, settings)
	if err := merged.Validate(); err != nil {
		return nil, err
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("bedrock embeddings: inputs must not be empty")
	}
	values, bodyFields, err := extractSettings(merged)
	if err != nil {
		return nil, err
	}
	client, err := model.resolveClient(ctx)
	if err != nil {
		return nil, err
	}
	var vectors [][]float64
	var inputTokens int
	var responseID string
	if model.family == familyCohere {
		body, err := model.requestBody(inputs, inputType, values, bodyFields)
		if err != nil {
			return nil, err
		}
		response, err := client.InvokeModel(ctx, Request{
			ModelID: modelID(model.modelName, values), Body: body, Headers: maps.Clone(values.headers),
		})
		if err != nil {
			return nil, fmt.Errorf("bedrock embeddings: invoke %q: %w", model.modelName, err)
		}
		vectors, responseID, err = model.parseResponse(response.Body)
		if err != nil {
			return nil, err
		}
		inputTokens = response.InputTokens
	} else {
		vectors, inputTokens, err = model.embedConcurrent(ctx, client, inputs, inputType, values, bodyFields)
		if err != nil {
			return nil, err
		}
	}
	if len(vectors) != len(inputs) {
		return nil, fmt.Errorf("bedrock embeddings: response returned %d vectors for %d inputs", len(vectors), len(inputs))
	}
	result := (&embeddings.Result{
		Embeddings: vectors, Inputs: append([]string(nil), inputs...), InputType: inputType,
		ModelName: model.modelName, ProviderName: "bedrock", ProviderURL: model.ProviderURL(),
		Timestamp: time.Now().UTC(), Usage: ai.Usage{Requests: requestCount(model.family, len(inputs)), InputTokens: inputTokens},
		ProviderResponseID: responseID,
	}).Clone()
	return &result, nil
}

func (model *Model) resolveClient(ctx context.Context) (Client, error) {
	model.loadOnce.Do(func() {
		if model.client != nil {
			return
		}
		config, err := awsconfig.LoadDefaultConfig(ctx, model.loadOptions...)
		if err != nil {
			model.loadErr = fmt.Errorf("bedrock embeddings: load AWS configuration: %w", err)
			return
		}
		model.client = &awsClient{client: bedrockruntime.NewFromConfig(config)}
		model.setProviderURL(awsProviderURL(config))
	})
	return model.client, model.loadErr
}

func inputTypeValid(inputType embeddings.InputType) error {
	if inputType != embeddings.InputTypeQuery && inputType != embeddings.InputTypeDocument {
		return fmt.Errorf("bedrock embeddings: invalid input type %q", inputType)
	}
	return nil
}

func modelID(name string, settings requestSettings) string {
	if settings.inferenceProfile != "" {
		return settings.inferenceProfile
	}
	return name
}

func requestCount(family family, inputs int) int {
	if family == familyCohere {
		return 1
	}
	return inputs
}
