package openai

import ai "github.com/Kludex/pydantic-ai-go/ai"

func cacheableInstructions(messages []ai.ModelMessage, params ai.ModelRequestParams) ([]ai.InstructionPart, int, bool) {
	for _, message := range messages {
		if request, ok := message.(ai.ModelRequest); ok {
			for _, part := range request.Parts {
				if prompt, ok := part.(ai.SystemPromptPart); ok && prompt.DynamicRef != "" {
					return nil, 0, false
				}
			}
		}
	}
	parts := params.InstructionParts
	if len(parts) == 0 && params.Instructions != "" {
		parts = []ai.InstructionPart{{Content: params.Instructions}}
	}
	static := 0
	for _, part := range parts {
		if part.Dynamic {
			break
		}
		static++
	}
	return parts, static, true
}
