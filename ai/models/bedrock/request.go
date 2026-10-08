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
	"github.com/Kludex/pydantic-ai-go/ai/internal/promptcache"
)

func buildConverseInput(
	ctx context.Context, modelName string, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (*bedrockruntime.ConverseInput, error) {
	unified := params.Settings.Cache != nil && !hasCacheSettings(params.Settings)
	translated, err := translateCache(modelName, params.Settings)
	if err != nil {
		return nil, err
	}
	settings, cache, requestSettings, err := extractSettings(translated)
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
	if cache.instructions != "" && len(params.InstructionParts) > 0 {
		static := 0
		for _, part := range params.InstructionParts {
			if part.Dynamic {
				break
			}
			static++
		}
		for index, part := range params.InstructionParts {
			input.System = append(input.System, &types.SystemContentBlockMemberText{Value: part.Content})
			if index+1 == static {
				input.System = append(input.System, &types.SystemContentBlockMemberCachePoint{
					Value: providerCachePoint(cache.instructions),
				})
			}
		}
		if static == 0 {
			cache.instructions = ""
		}
	} else {
		if params.Instructions != "" {
			input.System = append(input.System, &types.SystemContentBlockMemberText{Value: params.Instructions})
		}
		if cache.instructions != "" {
			if len(input.System) == 0 {
				if !unified {
					return nil, fmt.Errorf("bedrock: instruction caching requires instructions")
				}
				cache.instructions = ""
			} else {
				input.System = append(input.System, &types.SystemContentBlockMemberCachePoint{
					Value: providerCachePoint(cache.instructions),
				})
			}
		}
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
		roles := make([]string, len(input.Messages))
		counts := make([]int, len(input.Messages))
		for index, message := range input.Messages {
			roles[index], counts[index] = string(message.Role), len(message.Content)
		}
		if previous := promptcache.PreviousTail(roles, counts); previous >= 0 {
			point := &types.ContentBlockMemberCachePoint{Value: providerCachePoint(cache.messages)}
			err := attachCachePoint(input.Messages[:previous+1], point, true)
			if err != nil { // pragma: no cover - previous is a nonempty user message.
				return nil, fmt.Errorf("bedrock: previous message caching: %w", err)
			}
		}
		point := &types.ContentBlockMemberCachePoint{Value: providerCachePoint(cache.messages)}
		if err := attachCachePoint(input.Messages, point, true); err != nil {
			return nil, fmt.Errorf("bedrock: message caching: %w", err)
		}
	}
	input.ToolConfig = toolConfiguration(params)
	if unified && input.ToolConfig == nil {
		cache.toolDefinitions = ""
	}
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
	if bedrockAnthropicDisallowsSampling(modelName) || bedrockOpenAIDisallowsSampling(modelName) {
		params.Settings = params.Settings.Clone()
		params.Settings.Temperature = nil
		params.Settings.TopP = nil
	}
	if err := applyThinkingSettings(modelName, &params); err != nil {
		return nil, err
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

func applyThinkingSettings(modelName string, params *ai.ModelRequestParams) error {
	thinking := params.Settings.Thinking
	if thinking == nil || thinking.Level == "" {
		return nil
	}
	extra := maps.Clone(params.Settings.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	name := strings.ToLower(modelName)
	if strings.Contains(name, "anthropic") || strings.Contains(name, "claude") {
		if _, configured := extra["thinking"]; !configured {
			adaptive := bedrockSupportsAdaptiveThinking(name)
			switch {
			case thinking.Level == ai.ThinkingLevelDisabled:
				extra["thinking"] = map[string]any{"type": "disabled"}
			case adaptive:
				extra["thinking"] = map[string]any{"type": "adaptive"}
				if bedrockSupportsEffort(name) {
					effort := string(thinking.Level)
					if effort == string(ai.ThinkingLevelEnabled) || effort == string(ai.ThinkingLevelMinimal) {
						effort = "low"
					}
					if effort == string(ai.ThinkingLevelXHigh) && !bedrockSupportsXHigh(name) {
						effort = "max"
					}
					if _, configured := extra["output_config"]; !configured {
						extra["output_config"] = map[string]any{"effort": effort}
					}
				}
			default:
				budgets := map[ai.ThinkingLevel]int{
					ai.ThinkingLevelEnabled: 10_000, ai.ThinkingLevelMinimal: 1_024,
					ai.ThinkingLevelLow: 2_048, ai.ThinkingLevelMedium: 10_000,
					ai.ThinkingLevelHigh: 16_384, ai.ThinkingLevelXHigh: 32_768,
				}
				budget, ok := budgets[thinking.Level]
				if !ok {
					return fmt.Errorf("bedrock: invalid thinking level %q", thinking.Level)
				}
				if thinking.TokenBudget != nil {
					budget = *thinking.TokenBudget
				}
				extra["thinking"] = map[string]any{"type": "enabled", "budget_tokens": budget}
			}
		}
		if params.OutputTool != nil && !params.AllowText && !bedrockSupportsAdaptiveThinking(name) {
			return fmt.Errorf("bedrock: extended thinking cannot force an output tool; use native or prompted output")
		}
	} else if strings.Contains(name, "openai") && thinking.Level != ai.ThinkingLevelDisabled {
		if _, configured := extra["reasoning_effort"]; !configured {
			extra["reasoning_effort"] = string(thinking.Level)
		}
	}
	params.Settings.ExtraBody = extra
	return nil
}

func bedrockSupportsAdaptiveThinking(name string) bool {
	return strings.Contains(name, "claude-sonnet-4-6") || strings.Contains(name, "claude-sonnet-5") ||
		strings.Contains(name, "claude-opus-4-6") || strings.Contains(name, "claude-opus-4-7") ||
		strings.Contains(name, "claude-opus-4-8") || strings.Contains(name, "claude-opus-5") ||
		strings.Contains(name, "claude-fable-5") || strings.Contains(name, "claude-mythos-5")
}

func bedrockSupportsEffort(name string) bool {
	return strings.Contains(name, "claude-opus-4-5") || bedrockSupportsAdaptiveThinking(name)
}

func bedrockSupportsXHigh(name string) bool {
	return strings.Contains(name, "claude-opus-4-7") || strings.Contains(name, "claude-opus-4-8") ||
		strings.Contains(name, "claude-opus-5") || strings.Contains(name, "claude-sonnet-5") ||
		strings.Contains(name, "claude-fable-5") || strings.Contains(name, "claude-mythos-5")
}

func bedrockOpenAIDisallowsSampling(modelName string) bool {
	name := strings.ToLower(modelName)
	for _, model := range []string{
		"gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-6-sol", "gpt-6-luna", "gpt-6-astra",
	} {
		if strings.HasSuffix(name, "openai."+model) || name == model {
			return true
		}
	}
	return false
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
