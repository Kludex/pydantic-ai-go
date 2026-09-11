package vercel

import (
	"encoding/json"
	"fmt"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
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
	messages, err := convertMessages(input.Messages, nil)
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

func convertMessages(messages []UIMessage, externalIDs map[string]struct{}) ([]ai.ModelMessage, error) {
	var converted []ai.ModelMessage
	for messageIndex, message := range messages {
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
			var messageExternalIDs map[string]struct{}
			if messageIndex == len(messages)-1 {
				messageExternalIDs = externalIDs
			}
			response, results, err := assistantMessage(message.Parts, messageExternalIDs)
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
		if !validPartState(part.State) {
			return "", fmt.Errorf("vercel: unsupported text state %q", part.State)
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
			if !validPartState(part.State) {
				return nil, fmt.Errorf("vercel: unsupported text state %q", part.State)
			}
			contents = append(contents, ai.TextContent{Text: part.Text})
		case "file":
			file, err := userFile(part)
			if err != nil {
				return nil, err
			}
			contents = append(contents, file)
		case string(ChunkDataToolAvailability):
			if data, ok := part.Data.(map[string]any); ok {
				requestParts = append(requestParts, toolAvailabilityPart(data))
			}
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

func assistantMessage(
	parts []UIMessagePart, externalIDs map[string]struct{},
) (ai.ModelResponse, []ai.ModelMessage, error) {
	response := ai.ModelResponse{}
	var results []ai.ModelMessage
	for _, part := range parts {
		switch {
		case part.Type == "text":
			if !validPartState(part.State) {
				return ai.ModelResponse{}, nil, fmt.Errorf("vercel: unsupported text state %q", part.State)
			}
			metadata := loadPartMetadata(part.ProviderMetadata)
			response.Parts = append(response.Parts, ai.TextPart{
				Content: part.Text, ID: metadata.id, ProviderName: metadata.providerName,
				ProviderDetails: cloneMap(metadata.providerDetails),
			})
		case part.Type == "reasoning":
			if !validPartState(part.State) {
				return ai.ModelResponse{}, nil, fmt.Errorf("vercel: unsupported reasoning state %q", part.State)
			}
			metadata := loadPartMetadata(part.ProviderMetadata)
			signature := metadata.signature
			if part.State == "streaming" {
				signature = ""
			}
			response.Parts = append(response.Parts, ai.ThinkingPart{
				Content: part.Text, ID: metadata.id, Signature: signature,
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
			if data, ok := part.Data.(map[string]any); ok {
				if compaction, ok := compactionPart(data); ok {
					response.Parts = append(response.Parts, compaction)
				}
			}
		case part.Type == string(ChunkSourceURL):
			if part.SourceID == "" || part.URL == "" {
				return ai.ModelResponse{}, nil, fmt.Errorf("vercel: source-url requires sourceId and url")
			}
		case part.Type == string(ChunkSourceDocument):
			if part.SourceID == "" || part.MediaType == "" || part.Title == "" {
				return ai.ModelResponse{}, nil, fmt.Errorf(
					"vercel: source-document requires sourceId, mediaType, and title",
				)
			}
		case strings.HasPrefix(part.Type, "data-"), part.Type == "step-start":
		case isToolPart(part):
			if part.ToolCallID == "" {
				return ai.ModelResponse{}, nil, fmt.Errorf("vercel: tool part requires a toolCallId")
			}
			name, _ := toolPartName(part)
			if name == "" {
				return ai.ModelResponse{}, nil, fmt.Errorf("vercel: dynamic tool part requires a toolName")
			}
			switch part.State {
			case "input-streaming", "input-available", "output-available", "output-error", "approval-requested",
				"approval-responded", "output-denied":
			default:
				return ai.ModelResponse{}, nil, fmt.Errorf("vercel: unsupported tool state %q", part.State)
			}
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
				content, outcome, err := toolOutput(part, name)
				if err != nil {
					return ai.ModelResponse{}, nil, err
				}
				if providerExecuted {
					response.Parts = append(response.Parts, ai.NativeToolReturnPart{
						ToolName: name, ToolCallID: part.ToolCallID, Content: content, Outcome: outcome,
						ToolKind: metadata.toolKind, ProviderName: metadata.providerName,
						ProviderDetails: cloneMap(metadata.providerDetails),
					})
				} else if _, external := externalIDs[part.ToolCallID]; !external {
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

func validPartState(state string) bool {
	return state == "" || state == "streaming" || state == "done"
}

func isToolPart(part UIMessagePart) bool {
	_, ok := toolPartName(part)
	return ok
}

func toolPartName(part UIMessagePart) (string, bool) {
	if part.Type == "dynamic-tool" {
		return part.ToolName, true
	}
	if strings.HasPrefix(part.Type, "tool-") {
		return strings.TrimPrefix(part.Type, "tool-"), true
	}
	return "", false
}

func toolOutput(part UIMessagePart, name string) (any, ai.ToolReturnOutcome, error) {
	content := any(nil)
	if len(part.Output) > 0 {
		if err := json.Unmarshal(part.Output, &content); err != nil {
			return nil, "", fmt.Errorf("vercel: decode tool %q output: %w", name, err)
		}
		content = ai.NormalizeToolReturnContent(normalizeClientToolReturnContent(content))
	}
	outcome := ai.ToolReturnOutcomeSuccess
	switch part.State {
	case "output-error":
		outcome = ai.ToolReturnOutcomeFailed
		content = part.ErrorText
	case "output-denied":
		outcome = ai.ToolReturnOutcomeDenied
		if part.Approval != nil && part.Approval.Reason != "" {
			content = part.Approval.Reason
		}
	}
	return content, outcome, nil
}
