package mistral

import (
	"context"
	"fmt"
	"slices"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func (model *Model) convertMessage(ctx context.Context, message ai.ModelMessage) ([]mistralMessage, error) {
	switch message := message.(type) {
	case ai.ModelRequest:
		return model.convertRequest(ctx, message)
	case ai.ModelResponse:
		return model.convertResponse(message)
	default:
		return nil, fmt.Errorf("mistral: unknown message type %T", message)
	}
}

func (model *Model) convertRequest(ctx context.Context, request ai.ModelRequest) ([]mistralMessage, error) {
	var converted []mistralMessage
	var trailing []ai.UserContent
	for _, part := range request.Parts {
		switch part := part.(type) {
		case ai.SystemPromptPart:
			converted = append(converted, mistralMessage{Role: "system", Content: part.Content})
		case ai.UserPromptPart:
			content, err := model.userContent(ctx, part)
			if err != nil {
				return nil, err
			}
			converted = append(converted, mistralMessage{Role: "user", Content: content})
		case ai.ToolReturnPart:
			if part.ToolCallID == "" {
				return nil, fmt.Errorf("mistral: tool result %q requires a tool call ID", part.ToolName)
			}
			content, files, err := toolResultContent(part.Content)
			if err != nil {
				return nil, err
			}
			trailing = append(trailing, files...)
			converted = append(converted, mistralMessage{Role: "tool", ToolCallID: part.ToolCallID, Content: content})
		case ai.RetryPromptPart:
			if part.ToolName == "" {
				converted = append(converted, mistralMessage{Role: "user", Content: part.ModelResponse()})
			} else {
				if part.ToolCallID == "" {
					return nil, fmt.Errorf("mistral: tool retry %q requires a tool call ID", part.ToolName)
				}
				converted = append(converted, mistralMessage{
					Role: "tool", ToolCallID: part.ToolCallID, Content: part.ModelResponse(),
				})
			}
		case ai.ToolAvailabilityDeltaPart:
			return nil, fmt.Errorf("mistral: tool availability deltas must be synthesized before transport")
		case ai.SpeechPart:
			return nil, ai.ErrUnpreparedSpeech
		}
	}
	if len(trailing) > 0 {
		content, err := model.userContent(ctx, ai.UserPromptPart{Contents: trailing})
		if err != nil {
			return nil, err
		}
		converted = append(converted, mistralMessage{Role: "user", Content: content})
	}
	return converted, nil
}

func (model *Model) convertResponse(response ai.ModelResponse) ([]mistralMessage, error) {
	var content []contentChunk
	var thinking []thinkingContent
	var calls []mistralToolCall
	for _, part := range response.Parts {
		switch part := part.(type) {
		case ai.TextPart:
			content = append(content, contentChunk{Type: "text", Text: part.Content})
		case ai.ThinkingPart:
			thinking = append(thinking, thinkingContent{Type: "text", Text: part.Content})
		case ai.ToolCallPart:
			if part.ToolCallID == "" {
				return nil, fmt.Errorf("mistral: tool call %q requires an ID", part.ToolName)
			}
			index := len(calls)
			calls = append(calls, mistralToolCall{
				ID: part.ToolCallID, Index: &index, Type: "function",
				Function: mistralFunction{Name: part.ToolName, Arguments: slices.Clone(part.Args)},
			})
		case ai.SpeechPart:
			return nil, ai.ErrUnpreparedSpeech
		}
	}
	if len(thinking) > 0 {
		content = slices.Insert(content, 0, contentChunk{Type: "thinking", Thinking: thinking})
	}
	if len(content) == 0 && len(calls) == 0 {
		return nil, nil
	}
	prefix := false
	return []mistralMessage{{Role: "assistant", Content: content, ToolCalls: calls, Prefix: &prefix}}, nil
}

func insertToolUserBarriers(messages []mistralMessage) []mistralMessage {
	for index := 0; index+1 < len(messages); index++ {
		if messages[index].Role != "tool" || messages[index+1].Role != "user" {
			continue
		}
		prefix := false
		barrier := mistralMessage{
			Role: "assistant", Content: []contentChunk{{Type: "text", Text: "OK"}}, Prefix: &prefix,
		}
		messages = slices.Insert(messages, index+1, barrier)
		index++
	}
	return messages
}
