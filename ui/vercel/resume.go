package vercel

import (
	"encoding/json"
	"fmt"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
)

func prepareRunInput(
	input RequestData, options ai.MessageSanitizationOptions,
) (ai.UserPromptPart, []ai.ModelMessage, *ai.DeferredToolResults, error) {
	decisions := map[string]ai.ToolApproval{}
	callResults := map[string]any{}
	externalIDs := map[string]struct{}{}
	if len(input.Messages) > 0 {
		message := input.Messages[len(input.Messages)-1]
		if message.Role == "assistant" {
			callIDs, err := externalCallIDs(message.Metadata)
			if err != nil {
				return ai.UserPromptPart{}, nil, nil, err
			}
			for _, callID := range callIDs {
				externalIDs[callID] = struct{}{}
			}
			for _, part := range message.Parts {
				if !strings.HasPrefix(part.Type, "tool-") {
					continue
				}
				if part.State == "approval-responded" {
					if part.ToolCallID == "" {
						return ai.UserPromptPart{}, nil, nil, fmt.Errorf("vercel: approval requires a toolCallId")
					}
					options.ResolvedToolCallIDs = append(options.ResolvedToolCallIDs, part.ToolCallID)
					if part.Approval != nil && part.Approval.Approved != nil && *part.Approval.Approved {
						approval := ai.ToolApproved{}
						if len(part.Input) > 0 {
							approval.OverrideArgs = append(json.RawMessage(nil), part.Input...)
						}
						decisions[part.ToolCallID] = approval
					} else {
						reason := ""
						if part.Approval != nil {
							reason = part.Approval.Reason
						}
						decisions[part.ToolCallID] = ai.ToolDenied{Message: reason}
					}
				}
				if _, ok := externalIDs[part.ToolCallID]; !ok {
					continue
				}
				result, err := externalToolResult(part)
				if err != nil {
					return ai.UserPromptPart{}, nil, nil, err
				}
				callResults[part.ToolCallID] = result
				options.ResolvedToolCallIDs = append(options.ResolvedToolCallIDs, part.ToolCallID)
			}
		}
	}
	if len(callResults) != len(externalIDs) {
		return ai.UserPromptPart{}, nil, nil, fmt.Errorf("vercel: external tool results are incomplete")
	}
	if len(decisions) == 0 && len(callResults) == 0 {
		prompt, history, _, err := PrepareInput(input, options)
		return prompt, history, nil, err
	}
	messages, err := convertMessages(input.Messages, externalIDs)
	if err != nil {
		return ai.UserPromptPart{}, nil, nil, err
	}
	history, _, err := ai.SanitizeMessages(messages, options)
	if err != nil {
		return ai.UserPromptPart{}, nil, nil, err
	}
	return ai.UserPromptPart{}, history, &ai.DeferredToolResults{Calls: callResults, Approvals: decisions}, nil
}

func externalToolResult(part UIMessagePart) (any, error) {
	if part.State != "output-available" && part.State != "output-error" && part.State != "output-denied" {
		return nil, fmt.Errorf("vercel: external tool %q result is incomplete", part.ToolCallID)
	}
	name := strings.TrimPrefix(part.Type, "tool-")
	content, outcome, err := toolOutput(part, name)
	if err != nil {
		return nil, err
	}
	if outcome != ai.ToolReturnOutcomeSuccess {
		return ai.ToolFailedf("%v", content), nil
	}
	return content, nil
}
