package agui

import (
	"encoding/json"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go"
)

// PrepareInput converts and sanitizes untrusted AG-UI history. The latest user
// message becomes the new prompt and is removed from returned history.
func PrepareInput(
	input RunAgentInput, options ai.MessageSanitizationOptions,
) (ai.UserPromptPart, []ai.ModelMessage, ai.MessageSanitizationReport, error) {
	messages, err := convertMessages(input.Messages)
	if err != nil {
		return ai.UserPromptPart{}, nil, ai.MessageSanitizationReport{}, err
	}
	sanitized, report, err := ai.SanitizeMessages(messages, options)
	if err != nil {
		return ai.UserPromptPart{}, nil, report, err
	}
	for index := len(sanitized) - 1; index >= 0; index-- {
		request, ok := sanitized[index].(ai.ModelRequest)
		if !ok || len(request.Parts) != 1 {
			continue
		}
		prompt, ok := request.Parts[0].(ai.UserPromptPart)
		if !ok {
			continue
		}
		history := append([]ai.ModelMessage(nil), sanitized[:index]...)
		history = append(history, sanitized[index+1:]...)
		return prompt, history, report, nil
	}
	return ai.UserPromptPart{}, nil, report, fmt.Errorf("agui: input requires a user message")
}

func convertMessages(messages []Message) ([]ai.ModelMessage, error) {
	converted := make([]ai.ModelMessage, 0, len(messages))
	for _, message := range messages {
		if message.ID == "" {
			return nil, fmt.Errorf("agui: message ID must not be empty")
		}
		switch message.Role {
		case "system":
			converted = append(converted, ai.ModelRequest{Parts: []ai.RequestPart{
				ai.SystemPromptPart{Content: message.Content},
			}})
		case "user":
			converted = append(converted, ai.ModelRequest{Parts: []ai.RequestPart{
				ai.UserPromptPart{Content: message.Content},
			}})
		case "assistant":
			parts := make([]ai.ResponsePart, 0, len(message.ToolCalls)+1)
			if message.Content != "" {
				parts = append(parts, ai.TextPart{Content: message.Content})
			}
			for _, call := range message.ToolCalls {
				if call.ID == "" || call.Function.Name == "" {
					return nil, fmt.Errorf("agui: assistant tool call requires an ID and name")
				}
				args := json.RawMessage(call.Function.Arguments)
				if !json.Valid(args) {
					return nil, fmt.Errorf("agui: tool call %q arguments are not valid JSON", call.ID)
				}
				parts = append(parts, ai.ToolCallPart{
					ToolName: call.Function.Name, ToolCallID: call.ID, Args: append(json.RawMessage(nil), args...),
				})
			}
			converted = append(converted, ai.ModelResponse{Parts: parts})
		case "tool":
			if message.ToolCallID == "" {
				return nil, fmt.Errorf("agui: tool result requires a toolCallId")
			}
			converted = append(converted, ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{
				ToolName: message.Name, ToolCallID: message.ToolCallID, Content: message.Content,
			}}})
		default:
			return nil, fmt.Errorf("agui: unsupported message role %q", message.Role)
		}
	}
	return converted, nil
}
