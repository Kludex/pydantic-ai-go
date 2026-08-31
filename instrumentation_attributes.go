package ai

import (
	"net/url"
	"strconv"

	"go.opentelemetry.io/otel/attribute"
)

func modelSettingAttributes(settings ModelSettings) []attribute.KeyValue {
	attributes := make([]attribute.KeyValue, 0, 8)
	if settings.MaxTokens != 0 {
		attributes = append(attributes, attribute.Int("gen_ai.request.max_tokens", settings.MaxTokens))
	}
	if settings.Temperature != nil {
		attributes = append(attributes, attribute.Float64("gen_ai.request.temperature", *settings.Temperature))
	}
	if settings.TopP != nil {
		attributes = append(attributes, attribute.Float64("gen_ai.request.top_p", *settings.TopP))
	}
	if settings.Seed != nil {
		attributes = append(attributes, attribute.Int("gen_ai.request.seed", *settings.Seed))
	}
	if settings.PresencePenalty != nil {
		attributes = append(attributes, attribute.Float64("gen_ai.request.presence_penalty", *settings.PresencePenalty))
	}
	if settings.FrequencyPenalty != nil {
		attributes = append(attributes, attribute.Float64("gen_ai.request.frequency_penalty", *settings.FrequencyPenalty))
	}
	if len(settings.StopSequences) > 0 {
		attributes = append(attributes, attribute.StringSlice("gen_ai.request.stop_sequences", settings.StopSequences))
	}
	return attributes
}

func responseTelemetryAttributes(response *ModelResponse) []attribute.KeyValue {
	attributes := usageTelemetryAttributes(response.Usage, "gen_ai.usage.")
	if response.ProviderName != "" {
		attributes = append(attributes,
			attribute.String("gen_ai.provider.name", response.ProviderName),
			attribute.String("gen_ai.system", response.ProviderName),
		)
	}
	if parsed, err := url.Parse(response.ProviderURL); err == nil && parsed.Hostname() != "" {
		attributes = append(attributes, attribute.String("server.address", parsed.Hostname()))
		if port, err := strconv.Atoi(parsed.Port()); err == nil && port != 0 {
			attributes = append(attributes, attribute.Int("server.port", port))
		}
	}
	if response.ModelName != "" {
		attributes = append(attributes, attribute.String("gen_ai.response.model", response.ModelName))
	}
	if response.ProviderResponseID != "" {
		attributes = append(attributes, attribute.String("gen_ai.response.id", response.ProviderResponseID))
	}
	if response.FinishReason != "" {
		attributes = append(attributes, attribute.StringSlice(
			"gen_ai.response.finish_reasons", []string{string(response.FinishReason)},
		))
	}
	if response.Usage.CostUSD != nil {
		attributes = append(attributes, attribute.Float64("operation.cost", *response.Usage.CostUSD))
	} else if calculation, err := response.Price(); err == nil {
		attributes = append(attributes, attribute.Float64("operation.cost", calculation.TotalPrice))
	}
	return attributes
}

func usageTelemetryAttributes(usage Usage, prefix string) []attribute.KeyValue {
	attributes := make([]attribute.KeyValue, 0, 12+len(usage.Details))
	if usage.InputTokens != 0 {
		attributes = append(attributes, attribute.Int(prefix+"input_tokens", usage.InputTokens))
	}
	if usage.OutputTokens != 0 {
		attributes = append(attributes, attribute.Int(prefix+"output_tokens", usage.OutputTokens))
	}
	if usage.CacheWriteTokens != 0 {
		attributes = append(attributes, attribute.Int(prefix+"cache_creation.input_tokens", usage.CacheWriteTokens))
	}
	if usage.CacheReadTokens != 0 {
		attributes = append(attributes, attribute.Int(prefix+"cache_read.input_tokens", usage.CacheReadTokens))
	}
	details := make(map[string]int, len(usage.Details)+7)
	for key, value := range usage.Details {
		details[key] = value
	}
	for key, value := range map[string]int{
		"cache_write_tokens": usage.CacheWriteTokens, "cache_read_tokens": usage.CacheReadTokens,
		"input_audio_tokens": usage.InputAudioTokens, "cache_audio_read_tokens": usage.CacheAudioReadTokens,
		"output_audio_tokens": usage.OutputAudioTokens, "reasoning_tokens": usage.ReasoningTokens,
		"accepted_prediction_tokens": usage.AcceptedPredictionTokens,
		"rejected_prediction_tokens": usage.RejectedPredictionTokens,
	} {
		if value != 0 {
			details[key] = value
		}
	}
	for key, value := range details {
		if key == "input_tokens" || key == "output_tokens" {
			continue
		}
		attributes = append(attributes, attribute.Int(prefix+"details."+key, value))
	}
	return attributes
}

func telemetryToolDefinitions(params ModelRequestParams) string {
	definitions := make([]map[string]any, 0, len(params.Tools)+1)
	for _, tool := range params.Tools {
		definitions = append(definitions, telemetryToolDefinition(tool))
	}
	if params.OutputTool != nil {
		definitions = append(definitions, telemetryToolDefinition(*params.OutputTool))
	}
	if len(definitions) == 0 {
		return ""
	}
	return telemetryJSON(definitions)
}

func telemetryToolDefinition(tool ToolDefinition) map[string]any {
	definition := map[string]any{"type": "function", "name": tool.Name}
	if tool.Description != "" {
		definition["description"] = tool.Description
	}
	if tool.Schema != nil {
		definition["parameters"] = tool.Schema
	}
	return definition
}

func telemetryRequestParameters(params ModelRequestParams) string {
	value := map[string]any{
		"function_tools": params.Tools, "deferred_tools": params.DeferredTools,
		"output_tool": params.OutputTool, "output_schema": params.OutputSchema,
		"output_mode": params.OutputMode, "allow_text_output": params.AllowText,
		"instruction_parts": params.InstructionParts,
	}
	return telemetryJSON(value)
}
