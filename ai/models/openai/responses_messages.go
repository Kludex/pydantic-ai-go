package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func trimOpenAICompactionMessages(messages []ai.ModelMessage, providerNames ...string) []ai.ModelMessage {
	providerName := "openai"
	if len(providerNames) > 0 {
		providerName = providerNames[0]
	}
	for messageIndex := len(messages) - 1; messageIndex >= 0; messageIndex-- {
		response, ok := messages[messageIndex].(ai.ModelResponse)
		if !ok {
			continue
		}
		for partIndex := len(response.Parts) - 1; partIndex >= 0; partIndex-- {
			part, ok := response.Parts[partIndex].(ai.CompactionPart)
			if !ok || part.ProviderName != providerName {
				continue
			}
			encryptedContent, _ := part.ProviderDetails["encrypted_content"].(string)
			if encryptedContent == "" {
				continue
			}
			response.Parts = slices.Clone(response.Parts[partIndex:])
			trimmed := make([]ai.ModelMessage, 0, len(messages)-messageIndex+1)
			standingPromptPlanted, _ := part.ProviderDetails[ai.StandingPromptPlantedKey].(bool)
			if !standingPromptPlanted {
				var standingParts []ai.RequestPart
				for _, earlier := range messages[:messageIndex] {
					request, ok := earlier.(ai.ModelRequest)
					if !ok {
						continue
					}
					for _, requestPart := range request.Parts {
						if system, ok := requestPart.(ai.SystemPromptPart); ok {
							standingParts = append(standingParts, system)
						}
					}
				}
				if len(standingParts) > 0 {
					trimmed = append(trimmed, ai.ModelRequest{Parts: standingParts})
				}
			}
			trimmed = append(trimmed, response)
			return append(trimmed, messages[messageIndex+1:]...)
		}
	}
	return messages
}

func prepareResponsesFunctionTool(definition ai.ToolDefinition, strictSupport bool) (responsesTool, error) {
	definition, err := ai.PrepareToolReturnSchema(definition, false)
	if err != nil {
		return responsesTool{}, err
	}
	schema, strict, err := prepareOpenAITool(definition, strictSupport)
	if err != nil {
		return responsesTool{}, err
	}
	return responsesTool{
		Type: "function", Name: definition.Name, Description: definition.Description,
		Parameters: schema, Strict: strict,
	}, nil
}

type responsesMessageConverter struct {
	ctx                       context.Context
	providerName              string
	clientToolSearch          bool
	serverToolSearch          bool
	deferred                  map[string]ai.ToolDefinition
	rendered                  map[string]struct{}
	strictSupport             bool
	phaseSupport              bool
	promptCacheBreakpoints    bool
	responsesReasoningContent bool
}

func (c *responsesMessageConverter) convert(msg ai.ModelMessage) ([]responsesInput, error) {
	switch message := msg.(type) {
	case ai.ModelRequest:
		return c.convertRequest(message)
	case ai.ModelResponse:
		return c.convertResponse(message)
	default:
		return nil, fmt.Errorf("openai: unknown message type %T", msg)
	}
}

func (c *responsesMessageConverter) convertRequest(message ai.ModelRequest) ([]responsesInput, error) {
	var out []responsesInput
	for _, requestPart := range message.Parts {
		switch part := requestPart.(type) {
		case ai.SpeechPart:
			return nil, ai.ErrUnpreparedSpeech
		case ai.SystemPromptPart:
			out = append(out, responsesInput{Role: "system", Content: part.Content})
		case ai.UserPromptPart:
			content, err := responsesUserContent(c.ctx, part, c.providerName, c.promptCacheBreakpoints)
			if err != nil {
				return nil, err
			}
			out = append(out, responsesInput{Role: "user", Content: content})
		case ai.ToolReturnPart:
			if c.clientToolSearch && part.ToolKind == ai.ToolPartKindToolSearch {
				tools, err := c.discoveredTools(part.Content)
				if err != nil {
					return nil, err
				}
				out = append(out, responsesInput{
					Type: "tool_search_output", CallID: part.ToolCallID, Execution: "client",
					Status: "completed", Tools: &tools,
				})
				continue
			}
			content, err := contentString(part.Content)
			if err != nil {
				return nil, err
			}
			out = append(out, responsesInput{Type: "function_call_output", CallID: part.ToolCallID, Output: content})
		case ai.ToolAvailabilityDeltaPart:
			tools, err := c.additionalTools(part.ToolsAdded)
			if err != nil {
				return nil, err
			}
			if len(tools) > 0 {
				out = append(out, responsesInput{Type: "additional_tools", Role: "developer", Tools: &tools})
			}
		case ai.RetryPromptPart:
			content := part.ModelResponse()
			if part.ToolCallID != "" {
				out = append(out, responsesInput{Type: "function_call_output", CallID: part.ToolCallID, Output: content})
			} else {
				out = append(out, responsesInput{Role: "user", Content: content})
			}
		default:
			return nil, fmt.Errorf("openai: unknown request part type %T", requestPart)
		}
	}
	return out, nil
}

func responsesUserContent(
	ctx context.Context, prompt ai.UserPromptPart, providerName string, promptCacheBreakpoints bool,
) (any, error) {
	if len(prompt.Contents) == 0 {
		return prompt.Content, nil
	}
	content := make([]responsesInputContent, 0, len(prompt.Contents))
	for _, item := range prompt.Contents {
		switch item := item.(type) {
		case ai.CachePoint:
			if _, err := item.ResolvedTTL(); err != nil {
				return nil, err
			}
			if !promptCacheBreakpoints {
				continue
			}
			if len(content) == 0 {
				return nil, fmt.Errorf("openai: cache point must follow user content")
			}
			content[len(content)-1].PromptCacheBreakpoint = &openAIPromptCacheBreakpoint{Mode: "explicit"}
		case ai.TextContent:
			content = append(content, responsesInputContent{Type: "input_text", Text: item.Text})
		case ai.ImageURL:
			if err := item.ForceDownload.Validate(); err != nil {
				return nil, err
			}
			imageURL := item.URL
			if item.ForceDownload != ai.FileDownloadNever {
				downloaded, err := downloadFileContent(
					ctx, item.URL, item.ResolvedMediaType, item.ForceDownload,
				)
				if err != nil {
					return nil, err
				}
				imageURL = downloaded.dataURI
			}
			content = append(content, responsesInputContent{
				Type: "input_image", ImageURL: imageURL, Detail: imageDetail(item.VendorMetadata),
			})
		case ai.AudioURL:
			file, err := responsesFileURL(ctx, item.URL, item.ResolvedMediaType, item.ForceDownload)
			if err != nil {
				return nil, err
			}
			content = append(content, file)
		case ai.DocumentURL:
			file, err := responsesFileURL(ctx, item.URL, item.ResolvedMediaType, item.ForceDownload)
			if err != nil {
				return nil, err
			}
			content = append(content, file)
		case ai.VideoURL:
			return nil, fmt.Errorf("openai: Responses does not support video URL input")
		case ai.BinaryContent:
			dataURI := "data:" + item.MediaType + ";base64," + base64.StdEncoding.EncodeToString(item.Data)
			if isImageMediaType(item.MediaType) {
				content = append(content, responsesInputContent{
					Type: "input_image", ImageURL: dataURI, Detail: imageDetail(item.VendorMetadata),
				})
				continue
			}
			if isAudioMediaType(item.MediaType) || isVideoMediaType(item.MediaType) {
				return nil, fmt.Errorf("openai: Responses does not support inline %s input", item.MediaType)
			}
			extension, err := fileExtension(item.MediaType)
			if err != nil {
				return nil, err
			}
			content = append(content, responsesInputContent{
				Type: "input_file", FileData: dataURI, Filename: "filename." + extension,
			})
		case ai.UploadedFile:
			if item.ProviderName != providerName {
				return nil, fmt.Errorf("openai: uploaded file %q belongs to provider %q", item.FileID, item.ProviderName)
			}
			if strings.HasPrefix(strings.ToLower(item.ResolvedMediaType()), "image/") {
				detail, _ := item.VendorMetadata["detail"].(string)
				if detail == "" {
					detail = "auto"
				}
				content = append(content, responsesInputContent{
					Type: "input_image", FileID: item.FileID, Detail: detail,
				})
			} else {
				content = append(content, responsesInputContent{Type: "input_file", FileID: item.FileID})
			}
		default:
			return nil, fmt.Errorf("openai: unsupported Responses user content type %T", item)
		}
	}
	return content, nil
}

func responsesFileURL(
	ctx context.Context,
	rawURL string,
	resolveMediaType func() (string, error),
	mode ai.FileDownloadMode,
) (responsesInputContent, error) {
	if err := mode.Validate(); err != nil {
		return responsesInputContent{}, err
	}
	if mode == ai.FileDownloadNever {
		return responsesInputContent{Type: "input_file", FileURL: rawURL}, nil
	}
	downloaded, err := downloadFileContent(ctx, rawURL, resolveMediaType, mode)
	if err != nil {
		return responsesInputContent{}, err
	}
	extension, err := fileExtension(downloaded.mediaType)
	if err != nil {
		return responsesInputContent{}, err
	}
	return responsesInputContent{
		Type: "input_file", FileData: downloaded.dataURI, Filename: "filename." + extension,
	}, nil
}

func (c *responsesMessageConverter) convertResponse(message ai.ModelResponse) ([]responsesInput, error) {
	var out []responsesInput
	for _, responsePart := range message.Parts {
		switch part := responsePart.(type) {
		case ai.SpeechPart:
			return nil, ai.ErrUnpreparedSpeech
		case ai.TextPart:
			id := ""
			phase := ""
			if part.ProviderName == "" || part.ProviderName == c.providerName {
				id = part.ID
				if c.phaseSupport {
					candidate, _ := part.ProviderDetails["phase"].(string)
					if candidate == "commentary" || candidate == "final_answer" {
						phase = candidate
					}
				}
			}
			out = append(out, responsesInput{Role: "assistant", Content: part.Content, ID: id, Phase: phase})
		case ai.CompactionPart:
			if part.ProviderName != c.providerName {
				continue
			}
			encryptedContent, _ := part.ProviderDetails["encrypted_content"].(string)
			if encryptedContent == "" {
				continue
			}
			out = append(out, responsesInput{
				Type: "compaction", ID: part.ID, EncryptedContent: encryptedContent,
			})
		case ai.ThinkingPart:
			if part.ProviderName != "" && part.ProviderName != c.providerName {
				continue
			}
			syntheticChatID := part.ID == "content" || part.ID == "reasoning" || part.ID == "reasoning_content" ||
				part.ID == "reasoning_text"
			if syntheticChatID && part.Signature == "" {
				if part.Content != "" {
					out = append(out, responsesInput{Role: "assistant", Content: "<think>\n" + part.Content + "\n</think>"})
				}
				continue
			}
			if part.ID != "" || part.Signature != "" {
				input := responsesInput{Type: "reasoning", ID: part.ID, EncryptedContent: part.Signature}
				if c.responsesReasoningContent && part.ID != "" && part.Content != "" {
					input.Content = []map[string]string{{"type": "reasoning_text", "text": part.Content}}
				}
				out = append(out, input)
			}
		case ai.ToolCallPart:
			id := ""
			if part.ProviderName == "" || part.ProviderName == c.providerName {
				id = part.ID
			}
			if c.clientToolSearch && part.ToolKind == ai.ToolPartKindToolSearch {
				var arguments any = map[string]any{}
				if len(part.Args) > 0 {
					if err := json.Unmarshal(part.Args, &arguments); err != nil {
						return nil, fmt.Errorf("openai: parse tool search arguments: %w", err)
					}
				}
				out = append(out, responsesInput{
					Type: "tool_search_call", ID: id, CallID: part.ToolCallID,
					Arguments: arguments, Execution: "client", Status: "completed",
				})
				continue
			}
			namespace := ""
			if part.ProviderName == "" || part.ProviderName == c.providerName {
				namespace, _ = part.ProviderDetails["namespace"].(string)
			}
			out = append(out, responsesInput{
				Type: "function_call", ID: id, CallID: part.ToolCallID,
				Name: part.ToolName, Arguments: string(part.Args), Namespace: namespace,
			})
		case ai.NativeToolCallPart:
			if part.ProviderName != c.providerName {
				continue
			}
			if part.ToolKind == ai.ToolPartKindCodeExecution {
				var arguments struct {
					ContainerID string `json:"container_id"`
					Code        string `json:"code"`
				}
				if err := json.Unmarshal(part.Args, &arguments); err != nil {
					return nil, fmt.Errorf("openai: parse code execution arguments: %w", err)
				}
				if part.ToolCallID != "" && arguments.ContainerID != "" {
					out = append(out, responsesInput{
						Type: "code_interpreter_call", ID: part.ToolCallID, ContainerID: arguments.ContainerID,
						Code: arguments.Code, Outputs: (*struct{})(nil), Status: "completed",
					})
				}
				continue
			}
			if part.ToolKind == ai.ToolPartKindWebSearch || part.ToolKind == ai.ToolPartKindXSearch {
				status, _ := part.ProviderDetails["status"].(string)
				itemType := "web_search_call"
				if part.ToolKind == ai.ToolPartKindXSearch {
					itemType = "x_search_call"
				}
				out = append(out, responsesInput{
					Type: itemType, ID: part.ID, Action: slices.Clone(part.Args), Status: status,
				})
				continue
			}
			if part.ToolKind == ai.ToolPartKindImageGeneration && part.ToolCallID != "" {
				out = append(out, responsesInput{Type: "image_generation_call", ID: part.ToolCallID})
				continue
			}
			if part.ToolKind == ai.ToolPartKindFileSearch && part.ToolCallID != "" {
				if part.ToolName == "attachment_search" {
					out = append(out, responsesInput{
						Type: "attachment_search_call", ID: part.ToolCallID,
						Action: slices.Clone(part.Args), Status: "completed",
					})
					continue
				}
				var arguments struct {
					Queries []string `json:"queries"`
				}
				if err := json.Unmarshal(part.Args, &arguments); err != nil {
					return nil, fmt.Errorf("openai: parse file search arguments: %w", err)
				}
				itemType := "file_search_call"
				if c.providerName == "xai" {
					itemType = "collections_search_call"
				}
				out = append(out, responsesInput{
					Type: itemType, ID: part.ToolCallID, Queries: arguments.Queries, Status: "completed",
				})
				continue
			}
			if part.ToolKind == ai.ToolPartKindMCPServer && part.ToolCallID != "" {
				serverLabel, ok := strings.CutPrefix(part.ToolName, "mcp_server:")
				if !ok || serverLabel == "" {
					return nil, fmt.Errorf("openai: invalid MCP server tool name %q", part.ToolName)
				}
				var arguments struct {
					Action   string          `json:"action"`
					ToolName string          `json:"tool_name"`
					ToolArgs json.RawMessage `json:"tool_args"`
				}
				if err := json.Unmarshal(part.Args, &arguments); err != nil {
					return nil, fmt.Errorf("openai: parse MCP server arguments: %w", err)
				}
				switch arguments.Action {
				case "list_tools":
					empty := []responsesTool{}
					out = append(out, responsesInput{
						Type: "mcp_list_tools", ID: part.ToolCallID, ServerLabel: serverLabel, Tools: &empty,
					})
				case "call_tool":
					if arguments.ToolName == "" {
						return nil, fmt.Errorf("openai: MCP call tool name must not be empty")
					}
					if len(arguments.ToolArgs) == 0 {
						arguments.ToolArgs = json.RawMessage(`{}`)
					}
					out = append(out, responsesInput{
						Type: "mcp_call", ID: part.ToolCallID, ServerLabel: serverLabel,
						Name: arguments.ToolName, Arguments: string(arguments.ToolArgs),
					})
				default:
					return nil, fmt.Errorf("openai: invalid MCP server action %q", arguments.Action)
				}
				continue
			}
			if !c.serverToolSearch || part.ToolKind != ai.ToolPartKindToolSearch {
				continue
			}
			var arguments any = map[string]any{}
			if len(part.Args) > 0 {
				if err := json.Unmarshal(part.Args, &arguments); err != nil {
					return nil, fmt.Errorf("openai: parse native tool search arguments: %w", err)
				}
			}
			callID, status := openAIToolSearchReplayDetails(part.ToolCallID, part.ProviderDetails)
			out = append(out, responsesInput{
				Type: "tool_search_call", ID: part.ID, CallID: callID,
				Arguments: arguments, Execution: "server", Status: status,
			})
		case ai.NativeToolReturnPart:
			if !c.serverToolSearch || part.ProviderName != c.providerName ||
				part.ToolKind != ai.ToolPartKindToolSearch {
				continue
			}
			outputID, _ := part.ProviderDetails["id"].(string)
			if outputID == "" {
				continue
			}
			tools, err := c.discoveredTools(part.Content)
			if err != nil {
				return nil, err
			}
			callID, status := openAIToolSearchReplayDetails(part.ToolCallID, part.ProviderDetails)
			out = append(out, responsesInput{
				Type: "tool_search_output", ID: outputID, CallID: callID,
				Execution: "server", Status: status, Tools: &tools,
			})
		}
	}
	return out, nil
}

func openAIToolSearchReplayDetails(fallback string, details map[string]any) (any, string) {
	var callID any = fallback
	if value, exists := details["call_id"]; exists {
		switch value := value.(type) {
		case nil:
			callID = (*string)(nil)
		case string:
			callID = value
		}
	}
	status, _ := details["status"].(string)
	switch status {
	case "in_progress", "completed", "incomplete":
	default:
		status = "completed"
	}
	return callID, status
}

func (c *responsesMessageConverter) discoveredTools(content any) ([]responsesTool, error) {
	encoded, err := json.Marshal(content)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal tool search return: %w", err)
	}
	var result ai.ToolSearchResult
	if err := json.Unmarshal(encoded, &result); err != nil {
		return nil, fmt.Errorf("openai: parse tool search return: %w", err)
	}
	names := make([]string, len(result.DiscoveredTools))
	for index, match := range result.DiscoveredTools {
		names[index] = match.Name
	}
	return c.additionalTools(names)
}

func (c *responsesMessageConverter) additionalTools(names []string) ([]responsesTool, error) {
	tools := make([]responsesTool, 0, len(names))
	for _, name := range names {
		definition, ok := c.deferred[name]
		if !ok {
			continue
		}
		if _, duplicate := c.rendered[name]; duplicate {
			continue
		}
		converted, err := prepareResponsesFunctionTool(definition, c.strictSupport)
		if err != nil {
			return nil, err
		}
		c.rendered[name] = struct{}{}
		tools = append(tools, converted)
	}
	return tools, nil
}
