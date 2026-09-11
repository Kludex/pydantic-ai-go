package bedrock

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func buildConverseInput(
	ctx context.Context, modelName string, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*bedrockruntime.ConverseInput, error) {
	settings, cache, requestSettings, err := extractSettings(params.Settings)
	if err != nil {
		return nil, err
	}
	params.Settings = settings
	if err := ai.ValidateNativeTools(params.NativeTools); err != nil {
		return nil, err
	}
	for _, nativeTool := range params.NativeTools {
		codeExecution, ok := nativeTool.CloneNativeTool().(ai.CodeExecutionTool)
		if !ok {
			return nil, fmt.Errorf("bedrock: native tool %q is not supported", nativeTool.Kind())
		}
		if len(codeExecution.Files) > 0 {
			return nil, fmt.Errorf("bedrock: code execution file attachments are not supported")
		}
	}
	modelID := modelName
	if requestSettings.inferenceProfile != "" {
		modelID = requestSettings.inferenceProfile
	}
	input := &bedrockruntime.ConverseInput{
		ModelId:                           aws.String(modelID),
		AdditionalModelResponseFieldPaths: slices.Clone(requestSettings.additionalModelResponseFieldPaths),
		RequestMetadata:                   maps.Clone(requestSettings.requestMetadata),
	}
	if requestSettings.guardrail != nil {
		input.GuardrailConfig = &types.GuardrailConfiguration{
			GuardrailIdentifier: aws.String(requestSettings.guardrail.Identifier),
			GuardrailVersion:    aws.String(requestSettings.guardrail.Version), Trace: requestSettings.guardrail.Trace,
		}
	}
	if requestSettings.performanceLatency != "" {
		input.PerformanceConfig = &types.PerformanceConfiguration{Latency: requestSettings.performanceLatency}
	}
	if len(requestSettings.promptVariables) > 0 {
		input.PromptVariables = make(map[string]types.PromptVariableValues, len(requestSettings.promptVariables))
		for name, value := range requestSettings.promptVariables {
			input.PromptVariables[name] = &types.PromptVariableValuesMemberText{Value: value}
		}
	}
	if params.Instructions != "" {
		input.System = append(input.System, &types.SystemContentBlockMemberText{Value: params.Instructions})
	}
	if cache.instructions != "" {
		if len(input.System) == 0 {
			return nil, fmt.Errorf("bedrock: instruction caching requires instructions")
		}
		input.System = append(input.System, &types.SystemContentBlockMemberCachePoint{
			Value: providerCachePoint(cache.instructions),
		})
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
	reservedCachePoints := 0
	if cache.instructions != "" {
		reservedCachePoints++
	}
	if cache.messages != "" {
		point := &types.ContentBlockMemberCachePoint{Value: providerCachePoint(cache.messages)}
		if err := attachCachePoint(input.Messages, point); err != nil {
			return nil, fmt.Errorf("bedrock: message caching: %w", err)
		}
	}
	input.ToolConfig = toolConfiguration(params)
	if cache.toolDefinitions != "" {
		if input.ToolConfig == nil {
			return nil, fmt.Errorf("bedrock: tool-definition caching requires tools")
		}
		input.ToolConfig.Tools = append(input.ToolConfig.Tools, &types.ToolMemberCachePoint{
			Value: providerCachePoint(cache.toolDefinitions),
		})
		reservedCachePoints++
	}
	limitCachePoints(input.Messages, 4-reservedCachePoints)
	if bedrockAnthropicDisallowsSampling(modelName) {
		params.Settings = params.Settings.Clone()
		params.Settings.Temperature = nil
		params.Settings.TopP = nil
	}
	input.InferenceConfig = inferenceConfiguration(params.Settings)
	input.OutputConfig, err = outputConfiguration(params)
	if err != nil {
		return nil, err
	}
	if len(params.Settings.ExtraBody) > 0 {
		input.AdditionalModelRequestFields = document.NewLazyDocument(params.Settings.ExtraBody)
	}
	input.ServiceTier = serviceTier(params.Settings.ServiceTier)
	return input, nil
}

func bedrockAnthropicDisallowsSampling(modelName string) bool {
	name := strings.ToLower(modelName)
	for _, prefix := range []string{
		"claude-fable-5", "claude-mythos-5", "claude-opus-4-7", "claude-opus-4-8",
		"claude-opus-5", "claude-sonnet-5",
	} {
		if strings.Contains(name, prefix) {
			return true
		}
	}
	return false
}

func outputConfiguration(params ai.ModelRequestParams) (*types.OutputConfig, error) {
	if params.OutputSchema == nil || params.OutputMode == ai.OutputModePrompted {
		return nil, nil
	}
	schema, err := json.Marshal(params.OutputSchema)
	if err != nil {
		return nil, fmt.Errorf("bedrock: marshal output schema: %w", err)
	}
	name := "response"
	description := "The structured final response."
	if params.OutputTool != nil {
		if params.OutputTool.Name != "" {
			name = params.OutputTool.Name
		}
		if params.OutputTool.Description != "" {
			description = params.OutputTool.Description
		}
	}
	return &types.OutputConfig{TextFormat: &types.OutputFormat{
		Type: types.OutputFormatTypeJsonSchema,
		Structure: &types.OutputFormatStructureMemberJsonSchema{Value: types.JsonSchemaDefinition{
			Name: aws.String(name), Description: aws.String(description), Schema: aws.String(string(schema)),
		}},
	}}, nil
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
