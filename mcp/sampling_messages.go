package mcp

import (
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

func samplingMessages(params *mcpsdk.CreateMessageParams) ([]ai.ModelMessage, error) {
	var messages []ai.ModelMessage
	var requestParts []ai.RequestPart
	if params.SystemPrompt != "" {
		requestParts = append(requestParts, ai.SystemPromptPart{Content: params.SystemPrompt})
	}
	var responseParts []ai.ResponsePart
	flushRequest := func() {
		if len(requestParts) > 0 {
			messages = append(messages, ai.ModelRequest{Parts: requestParts})
			requestParts = nil
		}
	}
	flushResponse := func() {
		if len(responseParts) > 0 {
			messages = append(messages, ai.ModelResponse{Parts: responseParts})
			responseParts = nil
		}
	}
	for _, message := range params.Messages {
		switch message.Role {
		case mcpsdk.Role("user"):
			flushResponse()
			part, err := samplingUserPrompt(message.Content)
			if err != nil {
				return nil, err
			}
			requestParts = append(requestParts, part)
		case mcpsdk.Role("assistant"):
			flushRequest()
			part, err := samplingAssistantPart(message.Content)
			if err != nil {
				return nil, err
			}
			responseParts = append(responseParts, part)
		default:
			return nil, fmt.Errorf("ai/mcp: unsupported sampling role %q", message.Role)
		}
	}
	flushResponse()
	flushRequest()
	return messages, nil
}

func samplingUserPrompt(content mcpsdk.Content) (ai.UserPromptPart, error) {
	switch content := content.(type) {
	case *mcpsdk.TextContent:
		return ai.UserPromptPart{Content: content.Text}, nil
	case *mcpsdk.ImageContent:
		return ai.UserPromptPart{Contents: []ai.UserContent{
			ai.BinaryContent{Data: append([]byte(nil), content.Data...), MediaType: content.MIMEType},
		}}, nil
	case *mcpsdk.AudioContent:
		return ai.UserPromptPart{Contents: []ai.UserContent{
			ai.BinaryContent{Data: append([]byte(nil), content.Data...), MediaType: content.MIMEType},
		}}, nil
	default:
		return ai.UserPromptPart{}, fmt.Errorf("ai/mcp: unsupported user sampling content %T", content)
	}
}

func samplingAssistantPart(content mcpsdk.Content) (ai.ResponsePart, error) {
	if text, ok := content.(*mcpsdk.TextContent); ok {
		return ai.TextPart{Content: text.Text}, nil
	}
	return nil, fmt.Errorf("ai/mcp: unsupported assistant sampling content %T", content)
}
