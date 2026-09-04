package agui

import (
	"encoding/base64"
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
			content, err := textMessageContent(message.Content, "system")
			if err != nil {
				return nil, err
			}
			converted = append(converted, ai.ModelRequest{Parts: []ai.RequestPart{
				ai.SystemPromptPart{Content: content},
			}})
		case "user":
			prompt, err := userPrompt(message.Content)
			if err != nil {
				return nil, err
			}
			converted = append(converted, ai.ModelRequest{Parts: []ai.RequestPart{prompt}})
		case "assistant":
			content, err := textMessageContent(message.Content, "assistant")
			if err != nil {
				return nil, err
			}
			parts := make([]ai.ResponsePart, 0, len(message.ToolCalls)+1)
			if content != "" {
				parts = append(parts, ai.TextPart{Content: content})
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
			encoded, err := json.Marshal(message.Content)
			if err != nil {
				return nil, fmt.Errorf("agui: encode tool result: %w", err)
			}
			var content any
			_ = json.Unmarshal(encoded, &content)
			converted = append(converted, ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{
				ToolName: message.Name, ToolCallID: message.ToolCallID, Content: content,
			}}})
		default:
			return nil, fmt.Errorf("agui: unsupported message role %q", message.Role)
		}
	}
	return converted, nil
}

func textMessageContent(content any, role string) (string, error) {
	if content == nil {
		return "", nil
	}
	text, ok := content.(string)
	if !ok {
		return "", fmt.Errorf("agui: %s message content must be text", role)
	}
	return text, nil
}

func userPrompt(content any) (ai.UserPromptPart, error) {
	if content == nil {
		return ai.UserPromptPart{}, nil
	}
	if text, ok := content.(string); ok {
		return ai.UserPromptPart{Content: text}, nil
	}
	var parts []InputContent
	switch value := content.(type) {
	case []InputContent:
		parts = append(parts, value...)
	default:
		encoded, err := json.Marshal(value)
		if err != nil {
			return ai.UserPromptPart{}, fmt.Errorf("agui: encode user content: %w", err)
		}
		if err := json.Unmarshal(encoded, &parts); err != nil {
			return ai.UserPromptPart{}, fmt.Errorf("agui: decode user content: %w", err)
		}
	}
	contents := make([]ai.UserContent, 0, len(parts))
	for _, part := range parts {
		converted, err := inputContent(part)
		if err != nil {
			return ai.UserPromptPart{}, err
		}
		contents = append(contents, converted)
	}
	return ai.UserPromptPart{Contents: contents}, nil
}

func inputContent(part InputContent) (ai.UserContent, error) {
	if part.Type == "text" {
		return ai.TextContent{Text: part.Text}, nil
	}
	metadata, _ := part.Metadata.(map[string]any)
	vendorMetadata, _ := metadata["vendor_metadata"].(map[string]any)
	forceDownload := ai.FileDownloadNever
	if value, ok := metadata["force_download"].(string); ok {
		forceDownload = ai.FileDownloadMode(value)
		if err := forceDownload.Validate(); err != nil {
			return nil, fmt.Errorf("agui: multimodal force download: %w", err)
		}
	}
	if part.Type == "binary" {
		if part.URL != "" {
			if strings.HasPrefix(part.URL, "data:") {
				return decodeDataURL(part.URL, vendorMetadata)
			}
			return inputURL(part.MimeType, part.URL, forceDownload, vendorMetadata), nil
		}
		if part.Data == "" {
			return nil, fmt.Errorf("agui: binary input requires url or data")
		}
		data, err := base64.StdEncoding.DecodeString(part.Data)
		if err != nil {
			return nil, fmt.Errorf("agui: decode binary input: %w", err)
		}
		return ai.BinaryContent{Data: data, MediaType: part.MimeType, VendorMetadata: cloneMap(vendorMetadata)}, nil
	}
	if part.Type != "image" && part.Type != "audio" && part.Type != "video" && part.Type != "document" {
		return nil, fmt.Errorf("agui: unsupported user content type %q", part.Type)
	}
	if part.Source == nil || (part.Source.Type != "url" && part.Source.Type != "data") || part.Source.Value == "" {
		return nil, fmt.Errorf("agui: %s input requires a url or data source", part.Type)
	}
	if part.Source.Type == "url" {
		return inputURL(part.Source.MimeType, part.Source.Value, forceDownload, vendorMetadata), nil
	}
	data, err := base64.StdEncoding.DecodeString(part.Source.Value)
	if err != nil {
		return nil, fmt.Errorf("agui: decode %s input: %w", part.Type, err)
	}
	return ai.BinaryContent{
		Data: data, MediaType: part.Source.MimeType, VendorMetadata: cloneMap(vendorMetadata),
	}, nil
}

func inputURL(
	mediaType string, url string, forceDownload ai.FileDownloadMode, vendorMetadata map[string]any,
) ai.UserContent {
	base := strings.SplitN(mediaType, "/", 2)[0]
	switch base {
	case "image":
		return ai.ImageURL{URL: url, MediaType: mediaType, ForceDownload: forceDownload, VendorMetadata: cloneMap(vendorMetadata)}
	case "audio":
		return ai.AudioURL{URL: url, MediaType: mediaType, ForceDownload: forceDownload, VendorMetadata: cloneMap(vendorMetadata)}
	case "video":
		return ai.VideoURL{URL: url, MediaType: mediaType, ForceDownload: forceDownload, VendorMetadata: cloneMap(vendorMetadata)}
	default:
		return ai.DocumentURL{URL: url, MediaType: mediaType, ForceDownload: forceDownload, VendorMetadata: cloneMap(vendorMetadata)}
	}
}

func decodeDataURL(value string, vendorMetadata map[string]any) (ai.UserContent, error) {
	header, payload, found := strings.Cut(strings.TrimPrefix(value, "data:"), ",")
	if !found || !strings.HasSuffix(header, ";base64") {
		return nil, fmt.Errorf("agui: binary data URL must contain base64 data")
	}
	data, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return nil, fmt.Errorf("agui: decode binary data URL: %w", err)
	}
	return ai.BinaryContent{
		Data: data, MediaType: strings.TrimSuffix(header, ";base64"), VendorMetadata: cloneMap(vendorMetadata),
	}, nil
}
