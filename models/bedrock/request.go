package bedrock

import (
	"context"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	ai "github.com/Kludex/pydantic-ai-go"
)

func buildConverseInput(
	ctx context.Context, modelName string, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*bedrockruntime.ConverseInput, error) {
	if params.OutputSchema != nil && params.OutputMode != ai.OutputModePrompted {
		return nil, fmt.Errorf("bedrock: native JSON output mode is not supported; use OutputModeTool")
	}
	if len(params.NativeTools) > 0 {
		return nil, fmt.Errorf("bedrock: provider-native tools are not supported")
	}
	input := &bedrockruntime.ConverseInput{ModelId: aws.String(modelName)}
	if params.Instructions != "" {
		input.System = append(input.System, &types.SystemContentBlockMemberText{Value: params.Instructions})
	}
	for _, message := range messages {
		switch value := message.(type) {
		case ai.ModelRequest:
			blocks, system, err := requestBlocks(ctx, value.Parts, input.Messages)
			if err != nil {
				return nil, err
			}
			input.System = append(input.System, system...)
			if len(blocks) > 0 {
				input.Messages = append(input.Messages, types.Message{Role: types.ConversationRoleUser, Content: blocks})
			}
		case ai.ModelResponse:
			blocks, err := responseBlocks(value)
			if err != nil {
				return nil, err
			}
			if len(blocks) > 0 {
				input.Messages = append(input.Messages, types.Message{Role: types.ConversationRoleAssistant, Content: blocks})
			}
		}
	}
	limitCachePoints(input.Messages, 4)
	input.ToolConfig = toolConfiguration(params)
	input.InferenceConfig = inferenceConfiguration(params.Settings)
	if len(params.Settings.ExtraBody) > 0 {
		input.AdditionalModelRequestFields = document.NewLazyDocument(params.Settings.ExtraBody)
	}
	input.ServiceTier = serviceTier(params.Settings.ServiceTier)
	return input, nil
}

func requestBlocks(
	ctx context.Context, parts []ai.RequestPart, priorMessages []types.Message,
) ([]types.ContentBlock, []types.SystemContentBlock, error) {
	var blocks []types.ContentBlock
	var system []types.SystemContentBlock
	for _, part := range parts {
		switch value := part.(type) {
		case ai.SystemPromptPart:
			if value.Content != "" {
				system = append(system, &types.SystemContentBlockMemberText{Value: value.Content})
			}
		case ai.UserPromptPart:
			userBlocks, err := userContentBlocks(ctx, value, priorMessages)
			if err != nil {
				return nil, nil, err
			}
			blocks = append(blocks, userBlocks...)
		case ai.ToolReturnPart:
			block, err := toolResultBlock(value.ToolCallID, value.Content, value.Outcome)
			if err != nil {
				return nil, nil, err
			}
			blocks = append(blocks, block)
		case ai.RetryPromptPart:
			if value.ToolCallID == "" {
				blocks = append(blocks, &types.ContentBlockMemberText{Value: value.ModelResponse()})
				continue
			}
			block, _ := toolResultBlock(value.ToolCallID, value.ModelResponse(), ai.ToolReturnOutcomeFailed)
			blocks = append(blocks, block)
		case ai.ToolAvailabilityDeltaPart:
			return nil, nil, fmt.Errorf("bedrock: tool availability history is not supported")
		case ai.SpeechPart:
			if value.Audio == nil {
				blocks = append(blocks, &types.ContentBlockMemberText{Value: value.Content()})
				continue
			}
			block, err := binaryContentBlock(*value.Audio, 0)
			if err != nil {
				return nil, nil, err
			}
			blocks = append(blocks, block)
		}
	}
	return blocks, system, nil
}
