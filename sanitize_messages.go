package ai

import "slices"

func (state *messageSanitizer) request(message ModelRequest) (ModelRequest, bool, error) {
	parts := make([]RequestPart, 0, len(message.Parts))
	for _, requestPart := range message.Parts {
		switch part := requestPart.(type) {
		case SystemPromptPart:
			if !state.allowSystemPrompts {
				state.strippedSystems++
				continue
			}
			parts = append(parts, part)
		case UserPromptPart:
			contents, err := state.userContents(part.Contents)
			if err != nil {
				return ModelRequest{}, false, err
			}
			part.Contents = contents
			parts = append(parts, part)
		case ToolReturnPart:
			part.Metadata = cloneSchemaMap(part.Metadata)
			content, keep, err := state.toolReturnContent(part.Content)
			if err != nil {
				return ModelRequest{}, false, err
			}
			if keep {
				part.Content = content
			} else {
				part.Content = nil
			}
			parts = append(parts, part)
		case ToolAvailabilityDeltaPart:
			part.ToolsAdded = slices.Clone(part.ToolsAdded)
			parts = append(parts, part)
		case RetryPromptPart:
			part.Errors = cloneDeferredValidationErrors(part.Errors)
			parts = append(parts, part)
		default:
			parts = append(parts, requestPart)
		}
	}
	if len(parts) == 0 {
		return ModelRequest{}, false, nil
	}
	message.Parts = parts
	message.Metadata = cloneSchemaMap(message.Metadata)
	return message, true, nil
}

func (state *messageSanitizer) response(message ModelResponse) (ModelResponse, bool, error) {
	cloned := cloneModelResponse(&message)
	parts := make([]ResponsePart, 0, len(cloned.Parts))
	for _, responsePart := range cloned.Parts {
		switch part := responsePart.(type) {
		case CompactionPart:
			if state.stripCompactionParts {
				state.strippedCompactions++
				continue
			}
			delete(part.ProviderDetails, StandingPromptPlantedKey)
			parts = append(parts, part)
		case NativeToolReturnPart:
			content, keep, err := state.toolReturnContent(part.Content)
			if err != nil {
				return ModelResponse{}, false, err
			}
			if keep {
				part.Content = content
			} else {
				part.Content = nil
			}
			parts = append(parts, part)
		default:
			parts = append(parts, responsePart)
		}
	}
	if len(parts) == 0 {
		return ModelResponse{}, false, nil
	}
	cloned.Parts = parts
	return *cloned, true, nil
}

func (state *messageSanitizer) stripTrailingToolCalls(messages *[]ModelMessage) {
	for len(*messages) > 0 {
		response, ok := (*messages)[len(*messages)-1].(ModelResponse)
		if !ok {
			return
		}
		parts := make([]ResponsePart, 0, len(response.Parts))
		for _, responsePart := range response.Parts {
			call, local := responsePart.(ToolCallPart)
			if !local {
				parts = append(parts, responsePart)
				continue
			}
			if _, resolved := state.resolvedToolCallIDs[call.ToolCallID]; resolved {
				parts = append(parts, call)
				continue
			}
			state.strippedToolCalls = append(state.strippedToolCalls, MessageSanitizationToolCall{
				Name: call.ToolName, ID: call.ToolCallID,
			})
		}
		if len(parts) == len(response.Parts) {
			return
		}
		if len(parts) > 0 {
			response.Parts = parts
			(*messages)[len(*messages)-1] = response
			return
		}
		*messages = (*messages)[:len(*messages)-1]
	}
}
