package openrouter

import ai "github.com/Kludex/pydantic-ai-go/ai"

func prepareMessages(messages []ai.ModelMessage) []ai.ModelMessage {
	prepared := ai.ModelRequestContext{Messages: messages}.Clone().Messages
	firstRequest := -1
	for index, message := range prepared {
		if _, ok := message.(ai.ModelRequest); ok {
			firstRequest = index
			break
		}
	}
	if firstRequest == -1 {
		return prepared
	}
	requestOffset := 0
	for messageIndex := firstRequest; messageIndex < len(prepared); messageIndex++ {
		request, ok := prepared[messageIndex].(ai.ModelRequest)
		if !ok {
			continue
		}
		leading := 0
		if requestOffset == 0 {
			for _, part := range request.Parts {
				if _, ok := part.(ai.SystemPromptPart); !ok {
					break
				}
				leading++
			}
		}
		for partIndex := leading; partIndex < len(request.Parts); partIndex++ {
			system, ok := request.Parts[partIndex].(ai.SystemPromptPart)
			if !ok {
				continue
			}
			request.Parts[partIndex] = ai.UserPromptPart{
				Content: "<system>" + system.Content + "</system>", Timestamp: system.Timestamp,
			}
		}
		prepared[messageIndex] = request
		requestOffset++
	}
	return prepared
}
