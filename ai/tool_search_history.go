package ai

import "slices"

func adaptNativeToolSearchHistory(messages []ModelMessage, targetProvider string, pairReturns bool) []ModelMessage {
	adapted := make([]ModelMessage, 0, len(messages))
	generatedRequest := false
	translatedCalls := map[string]bool{}
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
			var unanswered []string
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
						translatedCalls[part.ToolCallID] = true
						unanswered = append(unanswered, part.ToolCallID)
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
					callID := part.ToolCallID
					if index := slices.Index(unanswered, callID); index >= 0 {
						unanswered = slices.Delete(unanswered, index, index+1)
					} else if pairReturns && !translatedCalls[callID] && len(unanswered) > 0 {
						callID, unanswered = unanswered[0], unanswered[1:]
					}
					toolReturn := ToolReturnPart{
						ToolName: part.ToolName, Content: cloneSchemaValue(part.Content), ToolCallID: callID,
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

// repairTranslatedToolSearchCalls answers each tool-search call translated from untranslated
// that has no result. A trailing response is left alone, as the frontier a resumed run continues.
func repairTranslatedToolSearchCalls(messages, untranslated []ModelMessage, targetProvider string) []ModelMessage {
	translated := map[string]bool{}
	for _, message := range untranslated {
		if response, ok := message.(ModelResponse); ok {
			for _, part := range response.Parts {
				if call, ok := part.(NativeToolCallPart); ok &&
					localizeNativeToolSearchPart(call.ToolKind, call.ProviderName, targetProvider) {
					translated[call.ToolCallID] = true
				}
			}
		}
	}
	if len(translated) == 0 {
		return messages
	}
	repaired, _ := repairDanglingToolCalls(messages, false, translated)
	return repaired
}
