package mcp

import (
	"context"
	"fmt"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

func samplingModelHandler(model ai.Model) func(
	context.Context, *mcpsdk.CreateMessageRequest,
) (*mcpsdk.CreateMessageResult, error) {
	return func(ctx context.Context, request *mcpsdk.CreateMessageRequest) (*mcpsdk.CreateMessageResult, error) {
		messages, err := samplingMessages(request.Params)
		if err != nil {
			return nil, err
		}
		settings, err := samplingSettings(request.Params)
		if err != nil {
			return nil, err
		}
		response, err := ai.RequestModel(ctx, model, messages, ai.ModelRequestParams{Settings: settings})
		if err != nil {
			return nil, fmt.Errorf("ai/mcp: sampling model request: %w", err)
		}
		content, err := samplingResponseContent(response)
		if err != nil {
			return nil, err
		}
		return &mcpsdk.CreateMessageResult{
			Role: mcpsdk.Role("assistant"), Content: content, Model: model.Name(),
		}, nil
	}
}

func samplingSettings(params *mcpsdk.CreateMessageParams) (ai.ModelSettings, error) {
	maxTokens := int(params.MaxTokens)
	if params.MaxTokens < 0 || int64(maxTokens) != params.MaxTokens {
		return ai.ModelSettings{}, fmt.Errorf("ai/mcp: sampling max tokens %d is outside the supported range", params.MaxTokens)
	}
	settings := ai.ModelSettings{MaxTokens: maxTokens}
	if params.Temperature != 0 {
		temperature := params.Temperature
		settings.Temperature = &temperature
	}
	settings.StopSequences = append([]string(nil), params.StopSequences...)
	return settings, nil
}

func samplingResponseContent(response *ai.ModelResponse) (mcpsdk.Content, error) {
	var text strings.Builder
	for _, part := range response.Parts {
		switch part := part.(type) {
		case ai.TextPart:
			text.WriteString(part.Content)
		case ai.ThinkingPart:
		default:
			return nil, fmt.Errorf("ai/mcp: sampling model returned unsupported part %T", part)
		}
	}
	return &mcpsdk.TextContent{Text: text.String()}, nil
}
