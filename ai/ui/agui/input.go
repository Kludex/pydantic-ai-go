package agui

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// PrepareInput converts and sanitizes untrusted AG-UI history. The latest user
// message becomes the new prompt and is removed from returned history.
func PrepareInput(
	input RunAgentInput, options ai.MessageSanitizationOptions,
) (ai.UserPromptPart, []ai.ModelMessage, ai.MessageSanitizationReport, error) {
	return prepareInput(input, options, false)
}

func prepareInput(
	input RunAgentInput, options ai.MessageSanitizationOptions, preserveFileData bool,
) (ai.UserPromptPart, []ai.ModelMessage, ai.MessageSanitizationReport, error) {
	messages, err := convertMessages(input.Messages, preserveFileData)
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
	input RunAgentInput, options ai.MessageSanitizationOptions, preserveFileData bool,
) (ai.UserPromptPart, []ai.ModelMessage, *ai.DeferredToolResults, error) {
	if len(input.Resume) == 0 {
		prompt, history, _, err := prepareInput(input, options, preserveFileData)
		return prompt, history, nil, err
	}
	messages, err := convertMessages(input.Messages, preserveFileData)
	if err != nil {
		return ai.UserPromptPart{}, nil, nil, err
	}
	kinds := make(map[string]string, len(input.Resume))
	for _, entry := range input.Resume {
		toolCallID, kind, ok := resumeToolCall(entry.InterruptID)
		if !ok {
			return ai.UserPromptPart{}, nil, nil, fmt.Errorf("agui: invalid interrupt ID %q", entry.InterruptID)
		}
		if _, exists := kinds[toolCallID]; exists {
			return ai.UserPromptPart{}, nil, nil, fmt.Errorf("agui: duplicate resume for tool call %q", toolCallID)
		}
		kinds[toolCallID] = kind
		options.ResolvedToolCallIDs = append(options.ResolvedToolCallIDs, toolCallID)
	}
	messages = annotateDeferredKinds(messages, kinds)
	history, _, err := ai.SanitizeMessages(messages, options)
	if err != nil {
		return ai.UserPromptPart{}, nil, nil, err
	}
	results := ai.DeferredToolResults{Approvals: map[string]ai.ToolApproval{}, Calls: map[string]any{}}
	for _, entry := range input.Resume {
		toolCallID, kind, _ := resumeToolCall(entry.InterruptID)
		if kind == "approval" {
			results.Approvals[toolCallID] = resumeApproval(entry)
			continue
		}
		result, err := resumeExternalResult(entry)
		if err != nil {
			return ai.UserPromptPart{}, nil, nil, err
		}
		results.Calls[toolCallID] = result
	}
	return ai.UserPromptPart{}, history, &results, nil
}

func resumeToolCall(interruptID string) (string, string, bool) {
	if strings.HasPrefix(interruptID, "int-") && len(interruptID) > len("int-") {
		return strings.TrimPrefix(interruptID, "int-"), "approval", true
	}
	if strings.HasPrefix(interruptID, "ext-") && len(interruptID) > len("ext-") {
		return strings.TrimPrefix(interruptID, "ext-"), "external", true
	}
	return "", "", false
}

func annotateDeferredKinds(messages []ai.ModelMessage, kinds map[string]string) []ai.ModelMessage {
	for index, message := range messages {
		response, ok := message.(ai.ModelResponse)
		if !ok {
			continue
		}
		stored := map[string]any{}
		for _, call := range response.ToolCalls() {
			if kind := kinds[call.ToolCallID]; kind != "" {
				stored[call.ToolCallID] = kind
			}
		}
		if len(stored) == 0 {
			continue
		}
		response.Metadata = map[string]any{ai.DeferredToolKindsMetadataKey: stored}
		messages[index] = response
	}
	return messages
}

func resumeExternalResult(entry ResumeEntry) (any, error) {
	if entry.Status == "cancelled" {
		return &ai.ToolFailedError{Message: "Cancelled by user."}, nil
	}
	var payload struct {
		Result json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(entry.Payload, &payload); err != nil || len(payload.Result) == 0 {
		return nil, fmt.Errorf("agui: external tool result for %q requires a result", entry.InterruptID)
	}
	var result any
	_ = json.Unmarshal(payload.Result, &result)
	return result, nil
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

func convertMessages(messages []Message, preserveFileData bool) ([]ai.ModelMessage, error) {
	converted := make([]ai.ModelMessage, 0, len(messages))
	toolNames := map[string]string{}
	toolKinds := map[string]ai.ToolPartKind{}
	for _, message := range messages {
		if message.ID == "" {
			return nil, fmt.Errorf("agui: message ID must not be empty")
		}
		switch message.Role {
		case "system", "developer":
			content, err := textMessageContent(message.Content, message.Role)
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
		case "activity":
			content, ok := message.Content.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("agui: activity content must be an object")
			}
			switch message.ActivityType {
			case "pydantic_ai_compaction":
				part, ok := compactionActivityPart(content)
				if ok {
					converted = appendResponse(converted, []ai.ResponsePart{part})
				}
			case "pydantic_ai_tool_availability_delta":
				converted = append(converted, ai.ModelRequest{Parts: []ai.RequestPart{
					toolAvailabilityActivityPart(content),
				}})
			case "pydantic_ai_file":
				if preserveFileData {
					part, err := fileActivityPart(content)
					if err != nil {
						return nil, err
					}
					converted = appendResponse(converted, []ai.ResponsePart{part})
				}
			case "pydantic_ai_uploaded_file":
				if preserveFileData {
					part, err := uploadedFileActivityPart(content)
					if err != nil {
						return nil, err
					}
					converted = append(converted, ai.ModelRequest{Parts: []ai.RequestPart{part}})
				}
			}
		case "reasoning":
			content, err := textMessageContent(message.Content, "reasoning")
			if err != nil {
				return nil, err
			}
			metadata := struct {
				ID              string         `json:"id"`
				Signature       string         `json:"signature"`
				ProviderName    string         `json:"provider_name"`
				ProviderDetails map[string]any `json:"provider_details"`
			}{}
			_ = json.Unmarshal([]byte(message.EncryptedValue), &metadata)
			converted = appendResponse(converted, []ai.ResponsePart{ai.ThinkingPart{
				Content: content, ID: metadata.ID, Signature: metadata.Signature, ProviderName: metadata.ProviderName,
				ProviderDetails: cloneMap(metadata.ProviderDetails),
			}})
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
				toolNames[call.ID] = call.Function.Name
				kind, _ := encryptedToolMetadata(call.EncryptedValue)
				if kind != "" {
					toolKinds[call.ID] = kind
				}
				if providerName, originalID, ok := nativeToolCallID(call.ID); ok {
					parts = append(parts, ai.NativeToolCallPart{
						ToolName: call.Function.Name, ToolCallID: originalID, Args: append(json.RawMessage(nil), args...),
						ProviderName: providerName, ToolKind: kind,
					})
				} else {
					parts = append(parts, ai.ToolCallPart{
						ToolName: call.Function.Name, ToolCallID: call.ID, Args: append(json.RawMessage(nil), args...),
						ToolKind: kind,
					})
				}
			}
			converted = appendResponse(converted, parts)
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
			content = ai.NormalizeToolReturnContent(content)
			toolName := toolNames[message.ToolCallID]
			if toolName == "" {
				toolName = message.Name
			}
			kind, outcome := encryptedToolMetadata(message.EncryptedValue)
			if outcome == "" {
				if kind == "" {
					kind = toolKinds[message.ToolCallID]
				}
				outcome = ai.ToolReturnOutcomeSuccess
			}
			if message.Error != "" && outcome == ai.ToolReturnOutcomeSuccess {
				outcome = ai.ToolReturnOutcomeFailed
				kind = ""
			}
			if providerName, originalID, ok := nativeToolCallID(message.ToolCallID); ok {
				converted = appendResponse(converted, []ai.ResponsePart{ai.NativeToolReturnPart{
					ToolName: toolName, ToolCallID: originalID, Content: content, ProviderName: providerName,
					ToolKind: kind, Outcome: outcome,
				}})
			} else {
				converted = append(converted, ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{
					ToolName: toolName, ToolCallID: message.ToolCallID, Content: content, ToolKind: kind, Outcome: outcome,
				}}})
			}
		default:
			return nil, fmt.Errorf("agui: unsupported message role %q", message.Role)
		}
	}
	return converted, nil
}

func encryptedToolMetadata(value string) (ai.ToolPartKind, ai.ToolReturnOutcome) {
	var metadata struct {
		PydanticAI struct {
			ToolKind string `json:"tool_kind"`
			Outcome  string `json:"outcome"`
		} `json:"pydantic_ai"`
	}
	if json.Unmarshal([]byte(value), &metadata) != nil {
		return "", ""
	}
	kind := ai.ToolPartKind(metadata.PydanticAI.ToolKind)
	switch kind {
	case ai.ToolPartKindToolSearch, ai.ToolPartKindCapabilityLoad, ai.ToolPartKindWebSearch,
		ai.ToolPartKindWebFetch, ai.ToolPartKindCodeExecution, ai.ToolPartKindImageGeneration,
		ai.ToolPartKindFileSearch, ai.ToolPartKindMCPServer, ai.ToolPartKindAdvisor:
	default:
		kind = ""
	}
	outcome := ai.ToolReturnOutcome(metadata.PydanticAI.Outcome)
	switch outcome {
	case ai.ToolReturnOutcomeFailed, ai.ToolReturnOutcomeDenied, ai.ToolReturnOutcomeInterrupted:
		kind = ""
	default:
		outcome = ""
	}
	return kind, outcome
}

func nativeToolCallID(value string) (string, string, bool) {
	parts := strings.SplitN(value, "|", 3)
	if len(parts) != 3 || parts[0] != "pyd_ai_builtin" || parts[2] == "" {
		return "", "", false
	}
	return parts[1], parts[2], true
}

func fileActivityPart(content map[string]any) (ai.FilePart, error) {
	value, ok := content["url"].(string)
	if !ok || value == "" {
		return ai.FilePart{}, fmt.Errorf("agui: file activity requires a non-empty data URL")
	}
	vendorMetadata, _ := content["vendor_metadata"].(map[string]any)
	decoded, err := decodeDataURL(value, vendorMetadata)
	if err != nil {
		return ai.FilePart{}, fmt.Errorf("agui: file activity: %w", err)
	}
	binary := decoded.(ai.BinaryContent)
	part := ai.FilePart{Content: binary}
	part.ID, _ = content["id"].(string)
	part.ProviderName, _ = content["provider_name"].(string)
	if details, ok := content["provider_details"].(map[string]any); ok {
		part.ProviderDetails = cloneMap(details)
	}
	return part, nil
}

func uploadedFileActivityPart(content map[string]any) (ai.UserPromptPart, error) {
	fileID, fileOK := content["file_id"].(string)
	providerName, providerOK := content["provider_name"].(string)
	if !fileOK || fileID == "" || !providerOK || providerName == "" {
		return ai.UserPromptPart{}, fmt.Errorf("agui: uploaded-file activity requires a file ID and provider name")
	}
	mediaType, _ := content["media_type"].(string)
	identifier, _ := content["identifier"].(string)
	vendorMetadata, _ := content["vendor_metadata"].(map[string]any)
	return ai.UserPromptPart{Contents: []ai.UserContent{ai.UploadedFile{
		FileID: fileID, ProviderName: providerName, MediaType: mediaType, Identifier: identifier,
		VendorMetadata: cloneMap(vendorMetadata),
	}}}, nil
}

func compactionActivityPart(content map[string]any) (ai.CompactionPart, bool) {
	part := ai.CompactionPart{}
	if value, ok := content["content"].(string); ok {
		part.Content = value
	} else if content["content"] != nil {
		return ai.CompactionPart{}, false
	}
	if value, ok := content["id"].(string); ok {
		part.ID = value
	} else if content["id"] != nil {
		return ai.CompactionPart{}, false
	}
	if value, ok := content["provider_name"].(string); ok {
		part.ProviderName = value
	} else if content["provider_name"] != nil {
		return ai.CompactionPart{}, false
	}
	if value, ok := content["provider_details"].(map[string]any); ok {
		part.ProviderDetails = cloneMap(value)
	} else if content["provider_details"] != nil {
		return ai.CompactionPart{}, false
	}
	if part.ProviderName == "" && (part.ID != "" || len(part.ProviderDetails) > 0) {
		return ai.CompactionPart{}, false
	}
	return part, part.Content != "" || part.ID != "" || part.ProviderName != "" || len(part.ProviderDetails) > 0
}

func toolAvailabilityActivityPart(content map[string]any) ai.ToolAvailabilityDeltaPart {
	part := ai.ToolAvailabilityDeltaPart{}
	part.ToolCallID, _ = content["tool_call_id"].(string)
	switch values := content["added"].(type) {
	case []string:
		part.ToolsAdded = append(part.ToolsAdded, values...)
	case []any:
		for _, value := range values {
			if name, ok := value.(string); ok && name != "" {
				part.ToolsAdded = append(part.ToolsAdded, name)
			}
		}
	}
	return part
}

func appendResponse(messages []ai.ModelMessage, parts []ai.ResponsePart) []ai.ModelMessage {
	if len(messages) > 0 {
		if response, ok := messages[len(messages)-1].(ai.ModelResponse); ok {
			response.Parts = append(response.Parts, parts...)
			messages[len(messages)-1] = response
			return messages
		}
	}
	return append(messages, ai.ModelResponse{Parts: parts})
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
