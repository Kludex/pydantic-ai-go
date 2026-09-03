package agui

import (
	"encoding/json"
	"fmt"
	"strings"

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

func prepareRunInput(
	input RunAgentInput, options ai.MessageSanitizationOptions,
) (ai.UserPromptPart, []ai.ModelMessage, *ai.DeferredToolResults, error) {
	if len(input.Resume) == 0 {
		prompt, history, _, err := PrepareInput(input, options)
		return prompt, history, nil, err
	}
	messages, err := convertMessages(input.Messages)
	if err != nil {
		return ai.UserPromptPart{}, nil, nil, err
	}
	for _, entry := range input.Resume {
		if strings.HasPrefix(entry.InterruptID, "int-") {
			options.ResolvedToolCallIDs = append(options.ResolvedToolCallIDs, strings.TrimPrefix(entry.InterruptID, "int-"))
		}
	}
	history, _, err := ai.SanitizeMessages(messages, options)
	if err != nil {
		return ai.UserPromptPart{}, nil, nil, err
	}
	results := ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{}}
	for _, entry := range input.Resume {
		if !strings.HasPrefix(entry.InterruptID, "int-") || len(entry.InterruptID) == len("int-") {
			return ai.UserPromptPart{}, nil, nil, fmt.Errorf("agui: invalid interrupt ID %q", entry.InterruptID)
		}
		results.Approvals[strings.TrimPrefix(entry.InterruptID, "int-")] = resumeApproval(entry)
	}
	return ai.UserPromptPart{}, history, &results, nil
}

func resumeApproval(entry ResumeEntry) ai.ToolApproval {
	if entry.Status == "cancelled" {
		return ai.ToolDenied{Message: "Cancelled by user."}
	}
	var payload struct {
		Approved   *bool           `json:"approved"`
		EditedArgs json.RawMessage `json:"editedArgs"`
		Reason     string          `json:"reason"`
	}
	if json.Unmarshal(entry.Payload, &payload) != nil || payload.Approved == nil {
		return ai.ToolDenied{}
	}
	if *payload.Approved {
		if len(payload.EditedArgs) > 0 && json.Valid(payload.EditedArgs) && payload.EditedArgs[0] == '{' {
			return ai.ToolApproved{OverrideArgs: append(json.RawMessage(nil), payload.EditedArgs...)}
		}
		if len(payload.EditedArgs) > 0 {
			return ai.ToolDenied{}
		}
		return ai.ToolApproved{}
	}
	return ai.ToolDenied{Message: payload.Reason}
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
