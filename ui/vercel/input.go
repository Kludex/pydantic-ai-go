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
	for messageIndex := len(sanitized) - 1; messageIndex >= 0; messageIndex-- {
		request, ok := sanitized[messageIndex].(ai.ModelRequest)
		if !ok {
			continue
		}
		for partIndex := len(request.Parts) - 1; partIndex >= 0; partIndex-- {
			prompt, ok := request.Parts[partIndex].(ai.UserPromptPart)
			if !ok {
				continue
			}
			history := append([]ai.ModelMessage(nil), sanitized[:messageIndex]...)
			remaining := append([]ai.RequestPart(nil), request.Parts[:partIndex]...)
			remaining = append(remaining, request.Parts[partIndex+1:]...)
			if len(remaining) > 0 {
				request.Parts = remaining
				history = append(history, request)
			}
			history = append(history, sanitized[messageIndex+1:]...)
			return prompt, history, report, nil
		}
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
		metadata, timestamp := loadMessageMetadata(message.Metadata)
		switch message.Role {
		case "system":
			text, err := messageText(message.Parts)
			if err != nil {
				return nil, err
			}
			converted = append(converted, ai.ModelRequest{
				Parts: []ai.RequestPart{ai.SystemPromptPart{Content: text}}, Metadata: metadata, Timestamp: timestamp,
			})
		case "user":
			parts, err := userMessage(message.Parts)
			if err != nil {
				return nil, err
			}
			converted = append(converted, ai.ModelRequest{Parts: parts, Metadata: metadata, Timestamp: timestamp})
		case "assistant":
			response, results, err := assistantMessage(message.Parts)
			if err != nil {
				return nil, err
			}
			response.Metadata = metadata
			response.Timestamp = timestamp
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

func userMessage(parts []UIMessagePart) ([]ai.RequestPart, error) {
	if onlyText := len(parts) == 0 || allText(parts); onlyText {
		text, err := messageText(parts)
		return []ai.RequestPart{ai.UserPromptPart{Content: text}}, err
	}
	contents := make([]ai.UserContent, 0, len(parts))
	requestParts := make([]ai.RequestPart, 0, 2)
	for _, part := range parts {
		switch part.Type {
		case "text":
			contents = append(contents, ai.TextContent{Text: part.Text})
		case "file":
			file, err := userFile(part)
			if err != nil {
				return nil, err
			}
			contents = append(contents, file)
		case string(ChunkDataToolAvailability):
			requestParts = append(requestParts, toolAvailabilityPart(part.Data))
		default:
			if !strings.HasPrefix(part.Type, "data-") {
				return nil, fmt.Errorf("vercel: unsupported user part %q", part.Type)
			}
		}
	}
	if len(contents) > 0 {
		requestParts = append(requestParts, ai.UserPromptPart{Contents: contents})
	}
	return requestParts, nil
}

func allText(parts []UIMessagePart) bool {
	for _, part := range parts {
		if part.Type != "text" {
			return false
		}
	}
	return true
}

func assistantMessage(parts []UIMessagePart) (ai.ModelResponse, []ai.ModelMessage, error) {
	response := ai.ModelResponse{}
	var results []ai.ModelMessage
	for _, part := range parts {
		switch {
		case part.Type == "text":
			metadata := loadPartMetadata(part.ProviderMetadata)
			response.Parts = append(response.Parts, ai.TextPart{
				Content: part.Text, ID: metadata.id, ProviderName: metadata.providerName,
				ProviderDetails: cloneMap(metadata.providerDetails),
			})
		case part.Type == "reasoning":
			metadata := loadPartMetadata(part.ProviderMetadata)
			response.Parts = append(response.Parts, ai.ThinkingPart{
				Content: part.Text, ID: metadata.id, Signature: metadata.signature,
				ProviderName: metadata.providerName, ProviderDetails: cloneMap(metadata.providerDetails),
			})
		case part.Type == "file":
			file, err := binaryFile(part.URL)
			if err != nil {
				return ai.ModelResponse{}, nil, fmt.Errorf("vercel: decode assistant file: %w", err)
			}
			metadata := loadPartMetadata(part.ProviderMetadata)
			file.VendorMetadata = cloneMap(metadata.vendorMetadata)
			response.Parts = append(response.Parts, ai.FilePart{
				Content: file, ID: metadata.id, ProviderName: metadata.providerName,
				ProviderDetails: cloneMap(metadata.providerDetails),
			})
		case part.Type == string(ChunkDataCompaction):
			if compaction, ok := compactionPart(part.Data); ok {
				response.Parts = append(response.Parts, compaction)
			}
		case strings.HasPrefix(part.Type, "data-"):
		case strings.HasPrefix(part.Type, "tool-"):
			if part.ToolCallID == "" {
				return ai.ModelResponse{}, nil, fmt.Errorf("vercel: tool part requires a toolCallId")
			}
			name := strings.TrimPrefix(part.Type, "tool-")
			metadata := loadPartMetadata(part.CallProviderMetadata)
			input := part.Input
			if len(input) == 0 {
				input = json.RawMessage(`{}`)
			}
			if !json.Valid(input) {
				return ai.ModelResponse{}, nil, fmt.Errorf("vercel: tool %q input is not valid JSON", name)
			}
			providerExecuted := part.ProviderExecuted != nil && *part.ProviderExecuted
			if providerExecuted {
				response.Parts = append(response.Parts, ai.NativeToolCallPart{
					ToolName: name, ToolCallID: part.ToolCallID, Args: append(json.RawMessage(nil), input...),
					ToolKind: metadata.toolKind, ID: metadata.id, ProviderName: metadata.providerName,
					ProviderDetails: cloneMap(metadata.providerDetails),
				})
			} else {
				response.Parts = append(response.Parts, ai.ToolCallPart{
					ToolName: name, ToolCallID: part.ToolCallID, Args: append(json.RawMessage(nil), input...),
					ToolKind: metadata.toolKind, ID: metadata.id, ProviderName: metadata.providerName,
					ProviderDetails: cloneMap(metadata.providerDetails),
				})
			}
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
				if providerExecuted {
					response.Parts = append(response.Parts, ai.NativeToolReturnPart{
						ToolName: name, ToolCallID: part.ToolCallID, Content: content, Outcome: outcome,
						ToolKind: metadata.toolKind, ProviderName: metadata.providerName,
						ProviderDetails: cloneMap(metadata.providerDetails),
					})
				} else {
					results = append(results, ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{
						ToolName: name, ToolCallID: part.ToolCallID, Content: content, Outcome: outcome,
						ToolKind: metadata.toolKind,
					}}})
				}
			}
		default:
			return ai.ModelResponse{}, nil, fmt.Errorf("vercel: unsupported assistant part %q", part.Type)
		}
	}
	return response, results, nil
}
