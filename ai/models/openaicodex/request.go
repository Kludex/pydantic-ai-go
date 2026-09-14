package openaicodex

import (
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func prepareRequest(
	messages []ai.ModelMessage, params ai.ModelRequestParams,
) ([]ai.ModelMessage, ai.ModelRequestParams) {
	params.Settings = params.Settings.Clone()
	params.Settings.MaxTokens = 0
	params.Settings.Temperature = nil
	params.Settings.TopP = nil
	if params.Settings.ExtraBody == nil {
		params.Settings.ExtraBody = map[string]any{}
	}
	params.Settings.ExtraBody["store"] = false
	conversationID := latestConversationID(messages)
	if conversationID == "" {
		return messages, params
	}
	if _, explicit := params.Settings.ExtraBody["openai_prompt_cache_key"]; !explicit {
		if _, wireValue := params.Settings.ExtraBody["prompt_cache_key"]; !wireValue {
			params.Settings.ExtraBody["openai_prompt_cache_key"] = conversationID
		}
	}
	if params.Settings.ExtraHeaders == nil {
		params.Settings.ExtraHeaders = map[string]string{}
	}
	for _, name := range []string{"session-id", "thread-id", "x-client-request-id"} {
		if !headerExists(params.Settings.ExtraHeaders, name) {
			params.Settings.ExtraHeaders[name] = conversationID
		}
	}
	return messages, params
}

func latestConversationID(messages []ai.ModelMessage) string {
	for index := len(messages) - 1; index >= 0; index-- {
		switch message := messages[index].(type) {
		case ai.ModelRequest:
			if message.ConversationID != "" {
				return message.ConversationID
			}
		case ai.ModelResponse:
			if message.ConversationID != "" {
				return message.ConversationID
			}
		}
	}
	return ""
}

func headerExists(headers map[string]string, name string) bool {
	for candidate := range headers {
		if strings.EqualFold(candidate, name) {
			return true
		}
	}
	return false
}

func historyEndsWithSuspendedCodexResponse(messages []ai.ModelMessage) bool {
	if len(messages) == 0 {
		return false
	}
	response, ok := messages[len(messages)-1].(ai.ModelResponse)
	return ok && response.ProviderName == "openai-codex" && response.State == ai.ModelResponseStateSuspended
}
