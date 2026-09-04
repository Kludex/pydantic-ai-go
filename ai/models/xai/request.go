package xai

import (
	"fmt"
	"maps"
	"slices"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

const explicitReasoningSetting = "xai_explicit_reasoning_effort"

func (model *Model) prepareParams(params ai.ModelRequestParams) (ai.ModelRequestParams, error) {
	prepared, err := ai.ResolveNativeToolPreferences(model, params)
	if err != nil {
		return ai.ModelRequestParams{}, err
	}
	nativeTools := make([]ai.NativeTool, 0, len(prepared.NativeTools))
	for _, tool := range prepared.NativeTools {
		if model.SupportsNativeTool(tool) {
			nativeTools = append(nativeTools, tool)
			continue
		}
		if !tool.IsOptional() {
			return ai.ModelRequestParams{}, fmt.Errorf("xai: native tool %q is not supported by model %q", tool.Kind(), model.Name())
		}
	}
	prepared.NativeTools = nativeTools
	settings := prepared.Settings.Clone()
	extra := maps.Clone(settings.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	explicitEffort := ""
	if value, exists := extra[explicitReasoningSetting]; exists {
		var ok bool
		explicitEffort, ok = value.(string)
		if !ok {
			return ai.ModelRequestParams{}, fmt.Errorf("xai: explicit reasoning effort must be a string")
		}
		delete(extra, explicitReasoningSetting)
	}
	thinking, err := prepareThinking(model.Name(), settings.Thinking, explicitEffort != "")
	if err != nil {
		return ai.ModelRequestParams{}, err
	}
	settings.Thinking = thinking
	set := func(name string, value any) error {
		if _, exists := extra[name]; exists {
			return fmt.Errorf("xai: extra body field %q conflicts with portable settings", name)
		}
		extra[name] = value
		return nil
	}
	fields := []struct {
		name  string
		value any
		set   bool
	}{
		{name: "seed", value: pointerValue(settings.Seed), set: settings.Seed != nil},
		{name: "presence_penalty", value: pointerValue(settings.PresencePenalty), set: settings.PresencePenalty != nil},
		{name: "frequency_penalty", value: pointerValue(settings.FrequencyPenalty), set: settings.FrequencyPenalty != nil},
		{name: "stop", value: slices.Clone(settings.StopSequences), set: len(settings.StopSequences) > 0},
		{name: "logprobs", value: pointerValue(settings.Logprobs), set: settings.Logprobs != nil},
		{name: "top_logprobs", value: pointerValue(settings.TopLogprobs), set: settings.TopLogprobs != nil},
	}
	for _, field := range fields {
		if field.set {
			if err := set(field.name, field.value); err != nil {
				return ai.ModelRequestParams{}, err
			}
		}
	}
	settings.Seed = nil
	settings.PresencePenalty = nil
	settings.FrequencyPenalty = nil
	settings.StopSequences = nil
	settings.Logprobs = nil
	settings.TopLogprobs = nil
	settings.LogitBias = nil
	settings.ServiceTier = ""
	if len(extra) == 0 {
		extra = nil
	}
	settings.ExtraBody = extra
	prepared.Settings = settings
	return prepared, nil
}

func prepareThinking(name string, thinking *ai.ThinkingSettings, explicit bool) (*ai.ThinkingSettings, error) {
	if thinking == nil {
		return nil, nil
	}
	if thinking.TokenBudget != nil {
		return nil, fmt.Errorf("xai: reasoning token budgets are not supported")
	}
	if thinking.IncludeThoughts != nil {
		return nil, fmt.Errorf("xai: reasoning visibility cannot be configured")
	}
	supported := reasoningEfforts(name)
	if len(supported) == 0 {
		if explicit {
			return nil, fmt.Errorf("xai: model %q does not support reasoning effort", name)
		}
		return nil, nil
	}
	effort := mapThinkingLevel(thinking.Level, supported)
	if effort == "" {
		if explicit {
			return nil, fmt.Errorf("xai: model %q does not support reasoning effort %q", name, thinking.Level)
		}
		return nil, nil
	}
	return &ai.ThinkingSettings{Level: thinkingLevelForEffort(effort)}, nil
}

func mapThinkingLevel(level ai.ThinkingLevel, supported map[ReasoningEffort]bool) ReasoningEffort {
	var effort ReasoningEffort
	switch level {
	case ai.ThinkingLevelDisabled:
		effort = ReasoningEffortNone
	case ai.ThinkingLevelEnabled:
		if supported[ReasoningEffortNone] {
			return ""
		}
		effort = ReasoningEffortMedium
	case ai.ThinkingLevelMinimal, ai.ThinkingLevelLow:
		effort = ReasoningEffortLow
	case ai.ThinkingLevelMedium:
		effort = ReasoningEffortMedium
	case ai.ThinkingLevelHigh, ai.ThinkingLevelXHigh:
		effort = ReasoningEffortHigh
	default:
		return ""
	}
	if supported[effort] {
		return effort
	}
	if effort == ReasoningEffortMedium && supported[ReasoningEffortHigh] {
		return ReasoningEffortHigh
	}
	return ""
}

func reasoningEfforts(name string) map[ReasoningEffort]bool {
	all := map[ReasoningEffort]bool{
		ReasoningEffortNone: true, ReasoningEffortLow: true,
		ReasoningEffortMedium: true, ReasoningEffortHigh: true,
	}
	switch name {
	case "grok-4.3", "grok-4.3-latest", "grok-latest", "grok-4-0709", "grok-4-1-fast-reasoning",
		"grok-4-1-fast-non-reasoning", "grok-4-fast-reasoning", "grok-4-fast-non-reasoning", "grok-3":
		return all
	case "grok-4.5", "grok-4.5-latest", "grok-4.6", "grok-build-latest":
		delete(all, ReasoningEffortNone)
		return all
	default:
		if len(name) >= len("grok-3-mini") && name[:len("grok-3-mini")] == "grok-3-mini" {
			return map[ReasoningEffort]bool{ReasoningEffortLow: true, ReasoningEffortHigh: true}
		}
		return nil
	}
}
