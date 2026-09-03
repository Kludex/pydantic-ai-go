package bedrock

import (
	"encoding/json"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	ai "github.com/Kludex/pydantic-ai-go"
)

func toolResultBlock(
	toolCallID string, content any, outcome ai.ToolReturnOutcome,
) (types.ContentBlock, error) {
	var result types.ToolResultContentBlock
	if text, ok := content.(string); ok {
		result = &types.ToolResultContentBlockMemberText{Value: text}
	} else {
		if _, err := json.Marshal(content); err != nil {
			return nil, fmt.Errorf("bedrock: marshal tool result: %w", err)
		}
		result = &types.ToolResultContentBlockMemberJson{Value: document.NewLazyDocument(content)}
	}
	status := types.ToolResultStatusSuccess
	if outcome == ai.ToolReturnOutcomeFailed || outcome == ai.ToolReturnOutcomeInterrupted || outcome == ai.ToolReturnOutcomeDenied {
		status = types.ToolResultStatusError
	}
	return &types.ContentBlockMemberToolResult{Value: types.ToolResultBlock{
		ToolUseId: aws.String(toolCallID), Content: []types.ToolResultContentBlock{result}, Status: status,
	}}, nil
}

func responseBlocks(response ai.ModelResponse) ([]types.ContentBlock, error) {
	blocks := make([]types.ContentBlock, 0, len(response.Parts))
	for _, part := range response.Parts {
		switch value := part.(type) {
		case ai.TextPart:
			blocks = append(blocks, &types.ContentBlockMemberText{Value: value.Content})
		case ai.ToolCallPart:
			var args any
			if err := json.Unmarshal(value.Args, &args); err != nil {
				return nil, fmt.Errorf("bedrock: decode tool call %q arguments: %w", value.ToolName, err)
			}
			blocks = append(blocks, &types.ContentBlockMemberToolUse{Value: types.ToolUseBlock{
				Name: aws.String(value.ToolName), ToolUseId: aws.String(value.ToolCallID), Input: document.NewLazyDocument(args),
			}})
		case ai.ThinkingPart:
			if value.ProviderName != "" && value.ProviderName != "bedrock" {
				continue
			}
			reasoning := types.ReasoningTextBlock{Text: aws.String(value.Content)}
			if value.Signature != "" {
				reasoning.Signature = aws.String(value.Signature)
			}
			blocks = append(blocks, &types.ContentBlockMemberReasoningContent{Value: &types.ReasoningContentBlockMemberReasoningText{Value: reasoning}})
		default:
			return nil, fmt.Errorf("bedrock: unsupported response part type %T", part)
		}
	}
	return blocks, nil
}

func toolConfiguration(params ai.ModelRequestParams) *types.ToolConfiguration {
	definitions := append([]ai.ToolDefinition(nil), params.Tools...)
	if params.OutputTool != nil {
		definitions = append(definitions, *params.OutputTool)
	}
	if len(definitions) == 0 {
		return nil
	}
	tools := make([]types.Tool, len(definitions))
	for index, definition := range definitions {
		tools[index] = &types.ToolMemberToolSpec{Value: types.ToolSpecification{
			Name: aws.String(definition.Name), Description: aws.String(definition.Description),
			InputSchema: &types.ToolInputSchemaMemberJson{Value: document.NewLazyDocument(definition.Schema)},
			Strict:      definition.Strict,
		}}
	}
	choice := types.ToolChoice(&types.ToolChoiceMemberAuto{Value: types.AutoToolChoice{}})
	if params.OutputTool != nil && !params.AllowText {
		choice = &types.ToolChoiceMemberTool{Value: types.SpecificToolChoice{Name: aws.String(params.OutputTool.Name)}}
	}
	return &types.ToolConfiguration{Tools: tools, ToolChoice: choice}
}

func inferenceConfiguration(settings ai.ModelSettings) *types.InferenceConfiguration {
	if settings.MaxTokens == 0 && settings.Temperature == nil && settings.TopP == nil && len(settings.StopSequences) == 0 {
		return nil
	}
	configuration := &types.InferenceConfiguration{StopSequences: append([]string(nil), settings.StopSequences...)}
	if settings.MaxTokens != 0 {
		configuration.MaxTokens = aws.Int32(int32(settings.MaxTokens))
	}
	if settings.Temperature != nil {
		configuration.Temperature = aws.Float32(float32(*settings.Temperature))
	}
	if settings.TopP != nil {
		configuration.TopP = aws.Float32(float32(*settings.TopP))
	}
	return configuration
}

func serviceTier(tier ai.ServiceTier) *types.ServiceTier {
	switch tier {
	case ai.ServiceTierDefault:
		return &types.ServiceTier{Type: types.ServiceTierTypeDefault}
	case ai.ServiceTierFlex:
		return &types.ServiceTier{Type: types.ServiceTierTypeFlex}
	case ai.ServiceTierPriority:
		return &types.ServiceTier{Type: types.ServiceTierTypePriority}
	default:
		return nil
	}
}

func countTokensInput(input *bedrockruntime.ConverseInput) *bedrockruntime.CountTokensInput {
	return &bedrockruntime.CountTokensInput{
		ModelId: input.ModelId,
		Input: &types.CountTokensInputMemberConverse{Value: types.ConverseTokensRequest{
			Messages: input.Messages, System: input.System, ToolConfig: input.ToolConfig,
			AdditionalModelRequestFields: input.AdditionalModelRequestFields,
		}},
	}
}
