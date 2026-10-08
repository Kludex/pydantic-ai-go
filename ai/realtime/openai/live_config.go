package openai

import (
	"fmt"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
)

func (model *LiveModel) resolveLiveSettings(common realtime.Settings) (LiveSettings, error) {
	settings := model.config.liveSettings
	settings.Delegation = cloneLiveMap(settings.Delegation)
	settings.DataChannel = cloneLiveMap(settings.DataChannel)
	for key, value := range common.Provider {
		switch key {
		case "openai_voice":
			text, ok := value.(string)
			if !ok {
				return settings, fmt.Errorf("openai GPT-Live: %s must be a string", key)
			}
			settings.Voice = text
		case "openai_live_instructions":
			text, ok := value.(string)
			if !ok {
				return settings, fmt.Errorf("openai GPT-Live: %s must be a string", key)
			}
			settings.Instructions = text
		case "openai_live_delegation", "openai_live_data_channel":
			object, ok := value.(map[string]any)
			if !ok {
				return settings, fmt.Errorf("openai GPT-Live: %s must be an object", key)
			}
			if key == "openai_live_delegation" {
				settings.Delegation = cloneLiveMap(object)
			} else {
				settings.DataChannel = cloneLiveMap(object)
			}
		case "openai_live_store":
			stored, ok := value.(bool)
			if !ok {
				return settings, fmt.Errorf("openai GPT-Live: %s must be a boolean", key)
			}
			settings.Store = stored
		case "openai_live_turn_silence_ms":
			switch number := value.(type) {
			case int:
				settings.TurnSilence = time.Duration(number) * time.Millisecond
			case float64:
				settings.TurnSilence = time.Duration(number * float64(time.Millisecond))
			default:
				return settings, fmt.Errorf("openai GPT-Live: %s must be milliseconds", key)
			}
		case "openai_turn_detection", "openai_truncation", "openai_input_noise_reduction", "openai_output_speed":
			return settings, fmt.Errorf("openai GPT-Live: %s is unsupported", key)
		}
	}
	if settings.TurnSilence < 0 {
		return settings, fmt.Errorf("openai GPT-Live: turn silence must not be negative")
	}
	if settings.TurnSilence == 0 {
		settings.TurnSilence = 2 * time.Second
	}
	if settings.Instructions == "" {
		settings.Instructions = "You are a voice assistant. Keep replies short and conversational. When the user asks for something you cannot answer from this conversation alone, delegate the task and tell them you are looking it up."
	}
	return settings, nil
}

func (model *LiveModel) liveConfig(params realtime.ConnectParams, settings LiveSettings, media bool) (map[string]any, error) {
	common := params.Settings
	if common.TurnDetection != nil || common.MaxTokens != 0 || common.InputTranscriptionModel != nil || common.OutputModality != "" && common.OutputModality != realtime.OutputModalityAudio {
		return nil, fmt.Errorf("openai GPT-Live: turn detection, spoken token limits, transcription settings, and text output are unsupported")
	}
	if err := validateToolChoice(common.ToolChoice); err != nil {
		return nil, err
	}
	rate := model.config.profile.AudioInputSampleRate
	if !media && (rate != model.config.profile.AudioOutputSampleRate || rate != 16000 && rate != 24000) {
		return nil, fmt.Errorf("openai GPT-Live: input and output PCM rates must match at 16000 or 24000 Hz")
	}
	delegation := settings.Delegation
	backend := stringValue(delegation["model"])
	if backend == "" {
		backend = model.backend
	}
	if backend == "" || backend == "auto" {
		backend = "gpt-6-sol"
	}
	responses := map[string]any{"model": backend}
	instructions := params.Request.Instructions
	for _, message := range params.Messages {
		if request, ok := message.(ai.ModelRequest); ok {
			for _, part := range request.Parts {
				if system, ok := part.(ai.SystemPromptPart); ok && system.Content != "" {
					if instructions != "" {
						instructions += "\n\n"
					}
					instructions += system.Content
				}
			}
		}
	}
	if extra := stringValue(delegation["instructions"]); extra != "" {
		if instructions != "" {
			instructions += "\n\n"
		}
		instructions += extra
	}
	if instructions != "" {
		responses["instructions"] = instructions
	}
	for _, key := range []string{"max_output_tokens", "parallel_tool_calls", "service_tier"} {
		if value, ok := delegation[key]; ok {
			responses[key] = value
		}
	}
	if _, ok := responses["parallel_tool_calls"]; !ok && common.ParallelToolCalls != nil {
		responses["parallel_tool_calls"] = *common.ParallelToolCalls
	}
	effort := stringValue(delegation["reasoning_effort"])
	if effort == "" && common.Thinking != "" && (strings.HasPrefix(backend, "gpt-5") || strings.HasPrefix(backend, "gpt-6") || strings.HasPrefix(backend, "o")) {
		effort = string(common.Thinking)
		if common.Thinking == ai.ThinkingLevelEnabled {
			effort = "medium"
		}
		if common.Thinking == ai.ThinkingLevelDisabled {
			effort = "none"
		}
		if strings.HasPrefix(backend, "o") && effort == "none" {
			effort = ""
		}
		if (strings.HasPrefix(backend, "gpt-6") || strings.HasPrefix(backend, "o")) && effort == "minimal" {
			effort = "low"
		}
	}
	if effort != "" {
		responses["reasoning"] = map[string]any{"effort": effort}
	}
	if verbosity, ok := delegation["verbosity"]; ok {
		responses["text"] = map[string]any{"verbosity": verbosity}
	}
	var tools []any
	if common.ToolChoice != realtime.ToolChoiceNone {
		for _, tool := range params.Request.Tools {
			item := map[string]any{"type": "function", "name": tool.Name, "parameters": tool.Schema}
			if tool.Description != "" {
				item["description"] = tool.Description
			}
			if tool.Strict != nil {
				item["strict"] = *tool.Strict
			}
			tools = append(tools, item)
		}
	}
	for _, native := range params.Request.NativeTools {
		search, ok := native.(ai.WebSearchTool)
		if !ok {
			if native.IsOptional() {
				continue
			}
			return nil, fmt.Errorf("openai GPT-Live: native tool %q is unsupported", native.Kind())
		}
		size := search.SearchContextSize
		if size == "" {
			size = ai.WebSearchContextMedium
		}
		item := map[string]any{"type": "web_search", "search_context_size": size}
		if search.UserLocation != nil {
			location := map[string]any{"type": "approximate"}
			for key, value := range map[string]string{"city": search.UserLocation.City, "country": search.UserLocation.Country, "region": search.UserLocation.Region, "timezone": search.UserLocation.Timezone} {
				if value != "" {
					location[key] = value
				}
			}
			item["user_location"] = location
		}
		filters := map[string]any{}
		if len(search.AllowedDomains) > 0 {
			filters["allowed_domains"] = search.AllowedDomains
		}
		if len(search.BlockedDomains) > 0 {
			filters["blocked_domains"] = search.BlockedDomains
		}
		if len(filters) > 0 {
			item["filters"] = filters
		}
		if search.ExternalWebAccess != nil {
			item["external_web_access"] = *search.ExternalWebAccess
		}
		tools = append(tools, item)
	}
	if len(tools) > 0 {
		responses["tools"] = tools
	}
	if common.ToolChoice != "" {
		responses["tool_choice"] = common.ToolChoice
	}
	audio := map[string]any{}
	if !media {
		audio["format"] = map[string]any{"type": "audio/pcm", "rate": rate}
	}
	voice := settings.Voice
	if voice == "" {
		voice = model.config.settings.Voice
	}
	if voice != "" {
		audio["output"] = map[string]any{"voice": voice}
	}
	config := map[string]any{"model": model.Name(), "instructions": settings.Instructions,
		"delegation": map[string]any{"type": "responses", "responses": responses}}
	if len(audio) > 0 {
		config["audio"] = audio
	}
	if settings.Store {
		config["store"] = true
	}
	seed, err := seedLiveItems(params.Messages, false)
	if err != nil {
		return nil, err
	}
	if len(seed) > 0 {
		config["input"] = seed
	}
	if media {
		channel := settings.DataChannel
		if channel == nil {
			channel = map[string]any{"allowed_client_events": []string{}, "allowed_server_events": []string{}}
		}
		config["client"] = map[string]any{"data_channel": channel}
	}
	return config, nil
}
