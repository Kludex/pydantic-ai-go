package openrouter

import ai "github.com/Kludex/pydantic-ai-go/ai"

// prepareMessages also returns the texts of the user messages rendering the
// request-scoped prompts that end the last request, which the next request leaves out.
func prepareMessages(messages []ai.ModelMessage) ([]ai.ModelMessage, []string) {
	prepared := ai.ModelRequestContext{Messages: messages}.Clone().Messages
	firstRequest := -1
	for index, message := range prepared {
		if _, ok := message.(ai.ModelRequest); ok {
			firstRequest = index
			break
		}
	}
	if firstRequest == -1 {
		return prepared, nil
	}
	var scopedTail []string
	requestOffset := 0
	for messageIndex := firstRequest; messageIndex < len(prepared); messageIndex++ {
		request, ok := prepared[messageIndex].(ai.ModelRequest)
		if !ok {
			continue
		}
		leading := 0
		if requestOffset == 0 {
			for _, part := range request.Parts {
				if system, ok := part.(ai.SystemPromptPart); !ok || system.Scope == ai.SystemPromptScopeRequest {
					break
				}
				leading++
			}
		}
		scopedTail = nil
		for partIndex := leading; partIndex < len(request.Parts); partIndex++ {
			system, ok := request.Parts[partIndex].(ai.SystemPromptPart)
			if !ok {
				scopedTail = nil
				continue
			}
			content := "<system>" + system.Content + "</system>"
			request.Parts[partIndex] = ai.UserPromptPart{Content: content, Timestamp: system.Timestamp}
			if system.Scope == ai.SystemPromptScopeRequest {
				scopedTail = append(scopedTail, content)
			} else {
				scopedTail = nil
			}
		}
		prepared[messageIndex] = request
		requestOffset++
	}
	return prepared, scopedTail
}
