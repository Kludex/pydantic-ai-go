package bedrockmantle

import (
	"iter"
	"slices"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func restoreProviderToolCallIDs(messages []ai.ModelMessage) []ai.ModelMessage {
	restored := slices.Clone(messages)
	responseID := ""
	for messageIndex, message := range messages {
		switch message := message.(type) {
		case ai.ModelResponse:
			responseID = message.ProviderResponseID
			message.Parts = slices.Clone(message.Parts)
			for partIndex, part := range message.Parts {
				call, ok := part.(ai.ToolCallPart)
				if ok && responseID != "" {
					call.ToolCallID = strings.TrimPrefix(call.ToolCallID, responseID+":")
					message.Parts[partIndex] = call
				}
			}
			restored[messageIndex] = message
		case ai.ModelRequest:
			message.Parts = slices.Clone(message.Parts)
			for partIndex, part := range message.Parts {
				switch part := part.(type) {
				case ai.ToolReturnPart:
					part.ToolCallID = strings.TrimPrefix(part.ToolCallID, responseID+":")
					message.Parts[partIndex] = part
				case ai.RetryPromptPart:
					part.ToolCallID = strings.TrimPrefix(part.ToolCallID, responseID+":")
					message.Parts[partIndex] = part
				}
			}
			restored[messageIndex] = message
		}
	}
	return restored
}

func qualifyResponseToolCallIDs(response *ai.ModelResponse) {
	if response.ProviderResponseID == "" {
		return
	}
	for index, part := range response.Parts {
		call, ok := part.(ai.ToolCallPart)
		if !ok || call.ToolCallID == "" {
			continue
		}
		call.ToolCallID = response.ProviderResponseID + ":" + call.ToolCallID
		response.Parts[index] = call
	}
}

func qualifyStreamToolCallIDs(
	stream iter.Seq2[ai.ModelStreamEvent, error],
) iter.Seq2[ai.ModelStreamEvent, error] {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		responseID := ""
		for event, err := range stream {
			switch typed := event.(type) {
			case ai.ResponseMetadataEvent:
				if typed.ProviderResponseID != "" {
					responseID = typed.ProviderResponseID
				}
			case ai.ToolCallStartEvent:
				if !typed.Native && typed.ToolCallID != "" && responseID != "" {
					typed.ToolCallID = responseID + ":" + typed.ToolCallID
					event = typed
				}
			case ai.FinishEvent:
				if typed.ProviderResponseID != "" {
					responseID = typed.ProviderResponseID
				}
				response := ai.ModelResponse{Parts: typed.Parts, ProviderResponseID: responseID}
				qualifyResponseToolCallIDs(&response)
				typed.Parts = response.Parts
				event = typed
			}
			if !yield(event, err) {
				return
			}
		}
	}
}
