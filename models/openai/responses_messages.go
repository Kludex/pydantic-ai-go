package openai

import (
	"encoding/json"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go"
)

func prepareResponsesFunctionTool(definition ai.ToolDefinition, strictSupport bool) (responsesTool, error) {
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
	clientToolSearch bool
	deferred         map[string]ai.ToolDefinition
	rendered         map[string]struct{}
	strictSupport    bool
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
			out = append(out, responsesInput{Role: "user", Content: part.Content})
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

func (c *responsesMessageConverter) convertResponse(message ai.ModelResponse) ([]responsesInput, error) {
	var out []responsesInput
	for _, responsePart := range message.Parts {
		switch part := responsePart.(type) {
		case ai.TextPart:
			id := ""
			if part.ProviderName == "" || part.ProviderName == "openai" {
				id = part.ID
			}
			out = append(out, responsesInput{Role: "assistant", Content: part.Content, ID: id})
		case ai.ThinkingPart:
			if (part.ProviderName == "" || part.ProviderName == "openai") &&
				(part.ID != "" || part.Signature != "") {
				out = append(out, responsesInput{
					Type: "reasoning", ID: part.ID, EncryptedContent: part.Signature,
				})
			}
		case ai.ToolCallPart:
			id := ""
			if part.ProviderName == "" || part.ProviderName == "openai" {
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
			if part.ProviderName == "" || part.ProviderName == "openai" {
				namespace, _ = part.ProviderDetails["namespace"].(string)
			}
			out = append(out, responsesInput{
				Type: "function_call", ID: id, CallID: part.ToolCallID,
				Name: part.ToolName, Arguments: string(part.Args), Namespace: namespace,
			})
		}
	}
	return out, nil
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
