package ai

import "slices"

func adaptNativeToolSearchHistory(messages []ModelMessage, targetProvider string) []ModelMessage {
	adapted := make([]ModelMessage, 0, len(messages))
	generatedRequest := false
	for _, message := range cloneModelMessages(messages) {
		switch message := message.(type) {
		case ModelRequest:
			if generatedRequest {
				previous := adapted[len(adapted)-1].(ModelRequest)
				message.Parts = append(previous.Parts, message.Parts...)
				adapted[len(adapted)-1] = message
				generatedRequest = false
				continue
			}
			adapted = append(adapted, message)
			generatedRequest = false
		case ModelResponse:
			parts := make([]ResponsePart, 0, len(message.Parts))
			flushResponse := func() {
				if len(parts) == 0 {
					return
				}
				response := message
				response.Parts = slices.Clone(parts)
				adapted = append(adapted, response)
				parts = parts[:0]
				generatedRequest = false
			}
			for _, responsePart := range message.Parts {
				switch part := responsePart.(type) {
				case NativeToolCallPart:
					if localizeNativeToolSearchPart(part.ToolKind, part.ProviderName, targetProvider) {
						parts = append(parts, ToolCallPart{
							ToolName: part.ToolName, Args: slices.Clone(part.Args), ToolCallID: part.ToolCallID,
							ToolKind: part.ToolKind, ID: part.ID, ProviderName: part.ProviderName,
							ProviderDetails: cloneSchemaMap(part.ProviderDetails),
						})
					} else {
						parts = append(parts, part)
					}
				case NativeToolReturnPart:
					if !localizeNativeToolSearchPart(part.ToolKind, part.ProviderName, targetProvider) {
						parts = append(parts, part)
						continue
					}
					flushResponse()
					toolReturn := ToolReturnPart{
						ToolName: part.ToolName, Content: cloneSchemaValue(part.Content), ToolCallID: part.ToolCallID,
						ToolKind: part.ToolKind, Metadata: cloneSchemaMap(part.Metadata),
						Timestamp: part.Timestamp, Outcome: part.Outcome,
					}
					if generatedRequest {
						request := adapted[len(adapted)-1].(ModelRequest)
						request.Parts = append(request.Parts, toolReturn)
						adapted[len(adapted)-1] = request
					} else {
						adapted = append(adapted, ModelRequest{Parts: []RequestPart{toolReturn}})
						generatedRequest = true
					}
				default:
					parts = append(parts, responsePart)
				}
			}
			flushResponse()
			if len(message.Parts) == 0 {
				adapted = append(adapted, message)
				generatedRequest = false
			}
		}
	}
	return adapted
}

func localizeNativeToolSearchPart(kind ToolPartKind, providerName, targetProvider string) bool {
	return kind == ToolPartKindToolSearch &&
		(targetProvider == "" || providerName == "" || providerName != targetProvider)
}
