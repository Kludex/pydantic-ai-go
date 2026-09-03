package vercel

import (
	"encoding/json"
	"fmt"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
)

// PrepareInput converts and sanitizes client-held Vercel AI history. The
// latest user message becomes the new prompt and is removed from history.
func PrepareInput(
	input RequestData, options ai.MessageSanitizationOptions,
) (ai.UserPromptPart, []ai.ModelMessage, ai.MessageSanitizationReport, error) {
	if input.Trigger != "submit-message" && input.Trigger != "regenerate-message" {
		return ai.UserPromptPart{}, nil, ai.MessageSanitizationReport{}, fmt.Errorf(
			"vercel: unsupported trigger %q", input.Trigger,
		)
	}
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
	return ai.UserPromptPart{}, nil, report, fmt.Errorf("vercel: input requires a user message")
}

func prepareRunInput(
	input RequestData, options ai.MessageSanitizationOptions,
) (ai.UserPromptPart, []ai.ModelMessage, *ai.DeferredToolResults, error) {
	decisions := map[string]ai.ToolApproval{}
	for _, message := range input.Messages {
		for _, part := range message.Parts {
			if !strings.HasPrefix(part.Type, "tool-") || part.State != "approval-responded" {
				continue
			}
			if part.ToolCallID == "" {
				return ai.UserPromptPart{}, nil, nil, fmt.Errorf("vercel: approval requires a toolCallId")
			}
			options.ResolvedToolCallIDs = append(options.ResolvedToolCallIDs, part.ToolCallID)
			if part.Approval != nil && part.Approval.Approved != nil && *part.Approval.Approved {
				decisions[part.ToolCallID] = ai.ToolApproved{}
			} else {
				reason := ""
				if part.Approval != nil {
					reason = part.Approval.Reason
				}
				decisions[part.ToolCallID] = ai.ToolDenied{Message: reason}
			}
		}
	}
	if len(decisions) == 0 {
		prompt, history, _, err := PrepareInput(input, options)
		return prompt, history, nil, err
	}
	messages, err := convertMessages(input.Messages)
	if err != nil {
		return ai.UserPromptPart{}, nil, nil, err
	}
	history, _, err := ai.SanitizeMessages(messages, options)
	if err != nil {
		return ai.UserPromptPart{}, nil, nil, err
	}
	return ai.UserPromptPart{}, history, &ai.DeferredToolResults{Approvals: decisions}, nil
}

func convertMessages(messages []UIMessage) ([]ai.ModelMessage, error) {
	var converted []ai.ModelMessage
	for _, message := range messages {
		if message.ID == "" {
			return nil, fmt.Errorf("vercel: message ID must not be empty")
		}
		switch message.Role {
		case "system", "user":
			text, err := messageText(message.Parts)
			if err != nil {
				return nil, err
			}
			var part ai.RequestPart = ai.UserPromptPart{Content: text}
			if message.Role == "system" {
				part = ai.SystemPromptPart{Content: text}
			}
			converted = append(converted, ai.ModelRequest{Parts: []ai.RequestPart{part}})
		case "assistant":
			response, results, err := assistantMessage(message.Parts)
			if err != nil {
				return nil, err
			}
			converted = append(converted, response)
			converted = append(converted, results...)
		default:
			return nil, fmt.Errorf("vercel: unsupported message role %q", message.Role)
		}
	}
	return converted, nil
}

func messageText(parts []UIMessagePart) (string, error) {
	var text []string
	for _, part := range parts {
		if part.Type != "text" {
			return "", fmt.Errorf("vercel: %q messages support only text parts", part.Type)
		}
		text = append(text, part.Text)
	}
	return strings.Join(text, ""), nil
}

func assistantMessage(parts []UIMessagePart) (ai.ModelResponse, []ai.ModelMessage, error) {
	response := ai.ModelResponse{}
	var results []ai.ModelMessage
	for _, part := range parts {
		switch {
		case part.Type == "text":
			response.Parts = append(response.Parts, ai.TextPart{Content: part.Text})
		case part.Type == "reasoning":
			response.Parts = append(response.Parts, ai.ThinkingPart{Content: part.Text})
		case strings.HasPrefix(part.Type, "tool-"):
			if part.ToolCallID == "" {
				return ai.ModelResponse{}, nil, fmt.Errorf("vercel: tool part requires a toolCallId")
			}
			name := strings.TrimPrefix(part.Type, "tool-")
			input := part.Input
			if len(input) == 0 {
				input = json.RawMessage(`{}`)
			}
			if !json.Valid(input) {
				return ai.ModelResponse{}, nil, fmt.Errorf("vercel: tool %q input is not valid JSON", name)
			}
			response.Parts = append(response.Parts, ai.ToolCallPart{
				ToolName: name, ToolCallID: part.ToolCallID, Args: append(json.RawMessage(nil), input...),
			})
			if part.State == "output-available" || part.State == "output-error" || part.State == "output-denied" {
				content := any(nil)
				if len(part.Output) > 0 {
					if err := json.Unmarshal(part.Output, &content); err != nil {
						return ai.ModelResponse{}, nil, fmt.Errorf("vercel: decode tool %q output: %w", name, err)
					}
				}
				outcome := ai.ToolReturnOutcomeSuccess
				switch part.State {
				case "output-error":
					outcome = ai.ToolReturnOutcomeFailed
					content = part.ErrorText
				case "output-denied":
					outcome = ai.ToolReturnOutcomeDenied
				}
				results = append(results, ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{
					ToolName: name, ToolCallID: part.ToolCallID, Content: content, Outcome: outcome,
				}}})
			}
		default:
			return ai.ModelResponse{}, nil, fmt.Errorf("vercel: unsupported assistant part %q", part.Type)
		}
	}
	return response, results, nil
}
