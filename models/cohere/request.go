package cohere

import (
	"encoding/json"
	"fmt"
	"slices"

	ai "github.com/Kludex/pydantic-ai-go"
)

type chatRequest struct {
	Model            string          `json:"model"`
	Messages         []cohereMessage `json:"messages"`
	Stream           bool            `json:"stream"`
	Tools            []cohereTool    `json:"tools,omitempty"`
	ToolChoice       string          `json:"tool_choice,omitempty"`
	MaxTokens        int             `json:"max_tokens,omitempty"`
	StopSequences    []string        `json:"stop_sequences,omitempty"`
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"p,omitempty"`
	Seed             *int            `json:"seed,omitempty"`
	PresencePenalty  *float64        `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64        `json:"frequency_penalty,omitempty"`
}

type cohereMessage struct {
	Role       string           `json:"role"`
	Content    any              `json:"content,omitempty"`
	ToolCalls  []cohereToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type cohereContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	Thinking string `json:"thinking,omitempty"`
}

type cohereTool struct {
	Type     string         `json:"type"`
	Function cohereFunction `json:"function"`
}

type cohereToolCall struct {
	ID       string          `json:"id,omitempty"`
	Type     string          `json:"type,omitempty"`
	Function *cohereFunction `json:"function,omitempty"`
}

type cohereFunction struct {
	Name        string         `json:"name,omitempty"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
	Arguments   string         `json:"arguments,omitempty"`
}

func (model *Model) buildRequest(messages []ai.ModelMessage, params ai.ModelRequestParams) (chatRequest, error) {
	if len(params.NativeTools) > 0 {
		return chatRequest{}, fmt.Errorf("cohere: provider-native tools are not supported")
	}
	if params.OutputMode == ai.OutputModeNative && len(params.OutputSchema) > 0 {
		return chatRequest{}, fmt.Errorf("cohere: native structured output is not supported; use tool or prompted output")
	}
	request := chatRequest{
		Model: model.name, Messages: []cohereMessage{}, Stream: false,
		MaxTokens: params.Settings.MaxTokens, StopSequences: slices.Clone(params.Settings.StopSequences),
		Temperature: params.Settings.Temperature, TopP: params.Settings.TopP, Seed: params.Settings.Seed,
		PresencePenalty: params.Settings.PresencePenalty, FrequencyPenalty: params.Settings.FrequencyPenalty,
	}
	for _, message := range messages {
		converted, err := model.convertMessage(message)
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
		values := make([]cohereMessage, 0, len(instructions))
		for _, instruction := range instructions {
			values = append(values, cohereMessage{Role: "system", Content: instruction.Content})
		}
		request.Messages = slices.Insert(request.Messages, index, values...)
	}
	tools := slices.Clone(params.Tools)
	if params.OutputTool != nil {
		tools = append(tools, *params.OutputTool)
		if !params.AllowText {
			request.ToolChoice = "REQUIRED"
		}
	}
	for _, tool := range tools {
		request.Tools = append(request.Tools, cohereTool{
			Type: "function",
			Function: cohereFunction{
				Name: tool.Name, Description: tool.Description, Parameters: tool.Schema,
			},
		})
	}
	return request, nil
}

func (model *Model) convertMessage(message ai.ModelMessage) ([]cohereMessage, error) {
	if request, ok := message.(ai.ModelRequest); ok {
		var converted []cohereMessage
		for _, part := range request.Parts {
			switch part := part.(type) {
			case ai.SystemPromptPart:
				converted = append(converted, cohereMessage{Role: "system", Content: part.Content})
			case ai.UserPromptPart:
				content, err := cohereUserContent(part)
				if err != nil {
					return nil, err
				}
				converted = append(converted, cohereMessage{Role: "user", Content: content})
			case ai.ToolReturnPart:
				content, err := cohereToolResult(part.Content)
				if err != nil {
					return nil, err
				}
				if part.ToolCallID == "" {
					return nil, fmt.Errorf("cohere: tool result %q requires a tool call ID", part.ToolName)
				}
				converted = append(converted, cohereMessage{
					Role: "tool", ToolCallID: part.ToolCallID, Content: content,
				})
			case ai.RetryPromptPart:
				if part.ToolName == "" {
					converted = append(converted, cohereMessage{Role: "user", Content: part.ModelResponse()})
				} else {
					if part.ToolCallID == "" {
						return nil, fmt.Errorf("cohere: tool retry %q requires a tool call ID", part.ToolName)
					}
					converted = append(converted, cohereMessage{
						Role: "tool", ToolCallID: part.ToolCallID, Content: part.ModelResponse(),
					})
				}
			case ai.ToolAvailabilityDeltaPart:
				return nil, fmt.Errorf("cohere: tool availability deltas must be synthesized before transport")
			case ai.SpeechPart:
				return nil, ai.ErrUnpreparedSpeech
			}
		}
		return converted, nil
	}
	response := message.(ai.ModelResponse)
	var content []cohereContent
	var calls []cohereToolCall
	for _, part := range response.Parts {
		switch part := part.(type) {
		case ai.TextPart:
			content = append(content, cohereContent{Type: "text", Text: part.Content})
		case ai.ThinkingPart:
			content = append(content, cohereContent{Type: "thinking", Thinking: part.Content})
		case ai.ToolCallPart:
			if part.ToolCallID == "" {
				return nil, fmt.Errorf("cohere: tool call %q requires an ID", part.ToolName)
			}
			calls = append(calls, cohereToolCall{
				ID: part.ToolCallID, Type: "function",
				Function: &cohereFunction{Name: part.ToolName, Arguments: string(part.Args)},
			})
		case ai.SpeechPart:
			return nil, ai.ErrUnpreparedSpeech
		}
	}
	if len(content) == 0 && len(calls) == 0 {
		return nil, nil
	}
	return []cohereMessage{{Role: "assistant", Content: content, ToolCalls: calls}}, nil
}

func cohereUserContent(part ai.UserPromptPart) (any, error) {
	if len(part.Contents) == 0 {
		return part.Content, nil
	}
	content := make([]cohereContent, 0, len(part.Contents))
	for _, item := range part.Contents {
		switch item := item.(type) {
		case ai.TextContent:
			content = append(content, cohereContent{Type: "text", Text: item.Text})
		case ai.CachePoint:
			continue
		default:
			return nil, fmt.Errorf("cohere: multimodal input %T is not supported", item)
		}
	}
	return content, nil
}

func cohereToolResult(content any) (string, error) {
	if rich, ok := content.(ai.ToolReturn); ok {
		content = rich.ReturnValue
	}
	if rich, ok := content.(*ai.ToolReturn); ok && rich != nil {
		content = rich.ReturnValue
	}
	if text, ok := content.(string); ok {
		return text, nil
	}
	data, err := json.Marshal(content)
	if err != nil {
		return "", fmt.Errorf("cohere: marshal tool result: %w", err)
	}
	return string(data), nil
}
