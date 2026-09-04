package mistral

import (
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
	tools := slices.Clone(params.Tools)
	if provider.allowedTools != nil {
		allowed := map[string]bool{}
		for _, name := range provider.allowedTools {
			allowed[name] = true
		}
		tools = slices.DeleteFunc(tools, func(tool ai.ToolDefinition) bool { return !allowed[tool.Name] })
	}
	if params.OutputTool != nil {
		tools = append(tools, *params.OutputTool)
	}
	request.ToolChoice = provider.toolChoice
	if params.OutputTool != nil && !params.AllowText {
		request.ToolChoice = ToolChoiceAny
	} else if request.ToolChoice == ToolChoiceNone {
		tools = nil
		request.ToolChoice = ""
	}
	for _, tool := range tools {
		request.Tools = append(request.Tools, mistralTool{Function: mistralFunction{
			Name: tool.Name, Description: tool.Description, Parameters: tool.Schema,
		}})
	}
	if len(request.Tools) == 0 {
		request.ToolChoice = ""
	} else {
		if stream {
			request.N = &one
		}
		if request.ToolChoice == "" {
			request.ToolChoice = ToolChoiceAuto
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
