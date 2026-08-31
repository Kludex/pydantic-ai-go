package openai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go"
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
	providerName     string
	clientToolSearch bool
	serverToolSearch bool
	deferred         map[string]ai.ToolDefinition
	rendered         map[string]struct{}
	strictSupport    bool
	phaseSupport     bool
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
		case ai.SystemPromptPart:
			out = append(out, responsesInput{Role: "system", Content: part.Content})
		case ai.UserPromptPart:
			content, err := responsesUserContent(part, c.providerName)
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
					Status: "completed", Tools: tools,
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
				out = append(out, responsesInput{Type: "additional_tools", Role: "developer", Tools: tools})
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

func responsesUserContent(prompt ai.UserPromptPart, providerName string) (any, error) {
	if len(prompt.Contents) == 0 {
		return prompt.Content, nil
	}
	content := make([]responsesInputContent, 0, len(prompt.Contents))
	for _, item := range prompt.Contents {
		switch item := item.(type) {
		case ai.TextContent:
			content = append(content, responsesInputContent{Type: "input_text", Text: item.Text})
		case ai.ImageURL:
			content = append(content, responsesInputContent{Type: "input_image", ImageURL: item.URL})
		case ai.BinaryContent:
			if !strings.HasPrefix(item.MediaType, "image/") {
				return nil, fmt.Errorf("openai: Responses binary input requires an image media type, got %q", item.MediaType)
			}
			imageURL := "data:" + item.MediaType + ";base64," + base64.StdEncoding.EncodeToString(item.Data)
			content = append(content, responsesInputContent{Type: "input_image", ImageURL: imageURL})
		case ai.UploadedFile:
			if item.ProviderName != providerName {
				return nil, fmt.Errorf("openai: uploaded file %q belongs to provider %q", item.FileID, item.ProviderName)
			}
			if strings.HasPrefix(item.MediaType, "image/") {
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

func (c *responsesMessageConverter) convertResponse(message ai.ModelResponse) ([]responsesInput, error) {
	var out []responsesInput
	for _, responsePart := range message.Parts {
		switch part := responsePart.(type) {
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
			if (part.ProviderName == "" || part.ProviderName == c.providerName) &&
				(part.ID != "" || part.Signature != "") {
				out = append(out, responsesInput{
					Type: "reasoning", ID: part.ID, EncryptedContent: part.Signature,
				})
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
			if part.ToolKind == ai.ToolPartKindWebSearch {
				status, _ := part.ProviderDetails["status"].(string)
				out = append(out, responsesInput{
					Type: "web_search_call", ID: part.ID, Action: slices.Clone(part.Args), Status: status,
				})
				continue
			}
			if part.ToolKind == ai.ToolPartKindImageGeneration && part.ToolCallID != "" {
				out = append(out, responsesInput{Type: "image_generation_call", ID: part.ToolCallID})
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
				Execution: "server", Status: status, Tools: tools,
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
