package snowflake

import (
	"fmt"
	"iter"
	"maps"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func (model *Model) prepareParams(params ai.ModelRequestParams) (ai.ModelRequestParams, error) {
	if model.configErr != nil {
		return ai.ModelRequestParams{}, model.configErr
	}
	if len(params.NativeTools) > 0 {
		return ai.ModelRequestParams{}, fmt.Errorf("snowflake: provider-native tools are not supported")
	}
	if model.family == familyOther {
		if len(params.Tools) > 0 || params.OutputTool != nil {
			return ai.ModelRequestParams{}, fmt.Errorf(
				"snowflake: model %q does not support function tools; use prompted output", model.Name(),
			)
		}
		if params.OutputMode == ai.OutputModeNative && len(params.OutputSchema) > 0 {
			return ai.ModelRequestParams{}, fmt.Errorf(
				"snowflake: model %q does not support native structured output; use prompted output", model.Name(),
			)
		}
	}
	settings, reasoning, err := extractReasoning(params.Settings)
	if err != nil {
		return ai.ModelRequestParams{}, err
	}
	if reasoning != nil && model.family != familyClaude {
		return ai.ModelRequestParams{}, fmt.Errorf("snowflake: explicit reasoning is supported only by Claude models")
	}
	if settings.Thinking != nil && settings.Thinking.IncludeThoughts != nil {
		return ai.ModelRequestParams{}, fmt.Errorf("snowflake: thought visibility is not configurable")
	}
	if model.family == familyClaude {
		if reasoning == nil {
			reasoning, err = reasoningFromThinking(settings.Thinking)
			if err != nil {
				return ai.ModelRequestParams{}, err
			}
		}
		settings.Thinking = nil
		if reasoning != nil {
			extra := maps.Clone(settings.ExtraBody)
			if extra == nil {
				extra = map[string]any{}
			}
			if _, exists := extra["reasoning"]; exists {
				return ai.ModelRequestParams{}, fmt.Errorf(
					"snowflake: extra body field %q conflicts with typed or portable reasoning", "reasoning",
				)
			}
			extra["reasoning"] = *reasoning
			settings.ExtraBody = extra
			if settings.Temperature == nil {
				temperature := 1.0
				settings.Temperature = &temperature
			}
		}
	} else if settings.Thinking != nil && settings.Thinking.TokenBudget != nil {
		return ai.ModelRequestParams{}, fmt.Errorf("snowflake: thinking token budgets are supported only by Claude models")
	}
	params.Settings = settings
	return params, nil
}

func reasoningFromThinking(thinking *ai.ThinkingSettings) (*Reasoning, error) {
	if thinking == nil {
		return nil, nil
	}
	if thinking.TokenBudget != nil {
		reasoning := Reasoning{MaxTokens: *thinking.TokenBudget}
		if err := validateReasoning(reasoning); err != nil {
			return nil, err
		}
		return &reasoning, nil
	}
	if thinking.Level == "" || thinking.Level == ai.ThinkingLevelDisabled {
		return nil, nil
	}
	effort := ReasoningEffortMedium
	switch thinking.Level {
	case ai.ThinkingLevelMinimal, ai.ThinkingLevelLow:
		effort = ReasoningEffortLow
	case ai.ThinkingLevelHigh, ai.ThinkingLevelXHigh:
		effort = ReasoningEffortHigh
	}
	return &Reasoning{Effort: effort}, nil
}

func (model *Model) normalizeResponse(response *ai.ModelResponse) {
	if response.ModelName == "" {
		response.ModelName = model.Name()
	}
	if len(response.ToolCalls()) == 0 {
		return
	}
	if _, exists := response.ProviderDetails["finish_reason"]; exists {
		return
	}
	response.FinishReason = ai.FinishReasonToolCall
	response.ProviderDetails = maps.Clone(response.ProviderDetails)
	response.ProviderDetails["finish_reason"] = "tool_calls"
}

func (model *Model) normalizeStream(
	stream iter.Seq2[ai.ModelStreamEvent, error],
) iter.Seq2[ai.ModelStreamEvent, error] {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		sawToolCall := false
		for event, err := range stream {
			if err != nil {
				yield(nil, err)
				return
			}
			switch event := event.(type) {
			case ai.ToolCallStartEvent:
				sawToolCall = true
			case ai.FinishEvent:
				if _, exists := event.ProviderDetails["finish_reason"]; !exists && sawToolCall {
					event.ProviderDetails = maps.Clone(event.ProviderDetails)
					if event.ProviderDetails == nil {
						event.ProviderDetails = map[string]any{}
					}
					event.ProviderDetails["finish_reason"] = "tool_calls"
					event.FinishReason = ai.FinishReasonToolCall
				}
				if !yield(event, nil) {
					return
				}
				continue
			}
			if !yield(event, nil) {
				return
			}
		}
	}
}
