package mistral

import (
	"cmp"
	"context"
	"fmt"
	"slices"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func (model *Model) buildRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams, stream bool,
) (chatRequest, error) {
	if len(params.NativeTools) > 0 {
		return chatRequest{}, fmt.Errorf("mistral: provider-native tools are not supported")
	}
	if params.OutputMode == ai.OutputModeNative && len(params.OutputSchema) > 0 {
		return chatRequest{}, fmt.Errorf("mistral: native structured output is not supported; use tool or prompted output")
	}
	settings, provider, err := extractSettings(params.Settings)
	if err != nil {
		return chatRequest{}, err
	}
	if len(settings.LogitBias) > 0 || settings.Logprobs != nil || settings.TopLogprobs != nil || settings.ServiceTier != "" {
		return chatRequest{}, fmt.Errorf("mistral: logit bias, log probabilities, and service tiers are not supported")
	}
	if settings.Thinking != nil &&
		(settings.Thinking.TokenBudget != nil || settings.Thinking.IncludeThoughts != nil) {
		return chatRequest{}, fmt.Errorf("mistral: thinking token budgets and thought visibility are not supported")
	}
	request := chatRequest{
		Model: model.name, Stream: stream, MaxTokens: settings.MaxTokens,
		Temperature: settings.Temperature, RandomSeed: settings.Seed,
		PresencePenalty: settings.PresencePenalty, FrequencyPenalty: settings.FrequencyPenalty,
		Stop: slices.Clone(settings.StopSequences), ParallelToolCalls: settings.ParallelToolCalls,
		PromptCacheKey: provider.promptCacheKey,
	}
	one := 1
	if !stream {
		request.N = &one
	}
	if settings.TopP != nil {
		request.TopP = settings.TopP
	} else if !stream {
		topP := 1.0
		request.TopP = &topP
	}
	for _, message := range messages {
		converted, err := model.convertMessage(ctx, message)
		if err != nil {
			return chatRequest{}, err
		}
		request.Messages = append(request.Messages, converted...)
	}
	instructions := params.InstructionParts
	if len(instructions) == 0 && params.Instructions != "" {
		instructions = []ai.InstructionPart{{Content: params.Instructions}}
	}
	if len(instructions) > 0 {
		index := 0
		for index < len(request.Messages) && request.Messages[index].Role == "system" {
			index++
		}
		values := make([]mistralMessage, 0, len(instructions))
		for _, instruction := range instructions {
			values = append(values, mistralMessage{Role: "system", Content: instruction.Content})
		}
		request.Messages = slices.Insert(request.Messages, index, values...)
	}
	declared := slices.Clone(params.Tools)
	callable := slices.Clone(params.Tools)
	if provider.toolChoice == ToolChoiceNone {
		callable = nil
	} else if provider.allowedTools != nil {
		callable = slices.DeleteFunc(callable, func(tool ai.ToolDefinition) bool {
			return !slices.Contains(provider.allowedTools, tool.Name)
		})
	}
	if params.OutputTool != nil {
		declared = append(declared, *params.OutputTool)
		callable = append(callable, *params.OutputTool)
	}
	choice := provider.toolChoice
	if params.OutputTool != nil && !params.AllowText {
		choice = ToolChoiceAny
	} else if choice == ToolChoiceNone {
		callable, choice = nil, ""
	}
	tools := callable
	if choice == ToolChoiceAny && len(callable) == 1 && len(declared) > 1 {
		tools = declared
		forced := namedToolChoice{Type: "function"}
		forced.Function.Name = callable[0].Name
		request.ToolChoice = forced
	}
	for _, tool := range tools {
		request.Tools = append(request.Tools, mistralTool{Function: mistralFunction{
			Name: tool.Name, Description: tool.Description, Parameters: tool.Schema,
		}})
	}
	if len(request.Tools) > 0 {
		if stream {
			request.N = &one
		}
		if request.ToolChoice == nil {
			request.ToolChoice = cmp.Or(choice, ToolChoiceAuto)
		}
		if stream && request.TopP == nil {
			topP := 1.0
			request.TopP = &topP
		}
	}
	request.ReasoningEffort = model.reasoningEffort(settings.Thinking)
	request.Messages = insertToolUserBarriers(request.Messages)
	return request, nil
}
