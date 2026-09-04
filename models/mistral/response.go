package mistral

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func (model *Model) parseResponse(data []byte) (*ai.ModelResponse, error) {
	var response chatResponse
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, fmt.Errorf("mistral: decode response: %w", err)
	}
	if len(response.Choices) == 0 {
		return nil, &ai.UnexpectedModelBehaviorError{Message: "Mistral returned no choices"}
	}
	parts, err := model.responseParts(response.Choices[0].Message)
	if err != nil {
		return nil, err
	}
	timestamp := time.Now().UTC()
	providerDetails := map[string]any{"finish_reason": response.Choices[0].FinishReason}
	if response.Created != 0 {
		timestamp = time.Unix(response.Created, 0).UTC()
		providerDetails["timestamp"] = timestamp
	}
	modelName := response.Model
	if modelName == "" {
		modelName = model.name
	}
	return &ai.ModelResponse{
		Parts: parts, Usage: response.Usage.normalized(), ModelName: modelName, Timestamp: timestamp,
		ProviderName: model.providerName, ProviderURL: model.baseURL, ProviderResponseID: response.ID,
		ProviderDetails: providerDetails, FinishReason: mistralFinishReason(response.Choices[0].FinishReason),
		State: ai.ModelResponseStateComplete,
	}, nil
}

func (model *Model) responseParts(message responseMessage) ([]ai.ResponsePart, error) {
	text, thinking, err := parseContent(message.Content)
	if err != nil {
		return nil, fmt.Errorf("mistral: decode response content: %w", err)
	}
	parts := make([]ai.ResponsePart, 0, len(thinking)+len(message.ToolCalls)+1)
	for _, content := range thinking {
		parts = append(parts, ai.ThinkingPart{Content: content, ProviderName: model.providerName})
	}
	if text != "" {
		parts = append(parts, ai.TextPart{Content: text, ProviderName: model.providerName})
	}
	for _, call := range message.ToolCalls {
		if call.Type != "" && call.Type != "function" {
			return nil, fmt.Errorf("mistral: unsupported tool call type %q", call.Type)
		}
		arguments := normalizeArguments(call.Function.Arguments)
		callID := call.ID
		if callID == "" {
			callID = generatedToolCallID(call.Function.Name, arguments)
		}
		parts = append(parts, ai.ToolCallPart{
			ToolName: call.Function.Name, Args: arguments, ToolCallID: callID, ProviderName: model.providerName,
		})
	}
	return parts, nil
}

func marshalRequest(payload chatRequest, extra map[string]any) ([]byte, error) {
	cleaned := maps.Clone(extra)
	delete(cleaned, promptCacheKeySetting)
	delete(cleaned, toolChoiceSetting)
	delete(cleaned, allowedToolsSetting)
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("mistral: marshal request: %w", err)
	}
	if len(cleaned) == 0 {
		return data, nil
	}
	var object map[string]any
	_ = json.Unmarshal(data, &object)
	for key, value := range cleaned {
		if _, exists := object[key]; exists {
			return nil, fmt.Errorf("mistral: extra body field %q conflicts with a typed field", key)
		}
		object[key] = value
	}
	return json.Marshal(object)
}

func (model *Model) reasoningEffort(thinking *ai.ThinkingSettings) string {
	if thinking == nil || strings.HasPrefix(strings.ToLower(model.name), "magistral") ||
		!adjustableReasoningModel(model.name) {
		return ""
	}
	if thinking.Level == ai.ThinkingLevelDisabled {
		return "none"
	}
	return "high"
}

func adjustableReasoningModel(name string) bool {
	name = strings.ToLower(name)
	return map[string]bool{
		"mistral-small-latest": true, "mistral-small-2603": true,
		"mistral-medium-latest": true, "mistral-medium": true,
		"mistral-medium-3": true, "mistral-medium-3-5": true,
		"mistral-medium-3.5": true, "mistral-medium-2604": true,
	}[name]
}

func mistralFinishReason(reason string) ai.FinishReason {
	return map[string]ai.FinishReason{
		"stop": ai.FinishReasonStop, "length": ai.FinishReasonLength,
		"model_length": ai.FinishReasonLength, "error": ai.FinishReasonError,
		"tool_calls": ai.FinishReasonToolCall,
	}[reason]
}

func parseContent(raw json.RawMessage) (string, []string, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return "", nil, nil
	}
	if raw[0] == '"' {
		var text string
		_ = json.Unmarshal(raw, &text)
		return text, nil, nil
	}
	var chunks []contentChunk
	if err := json.Unmarshal(raw, &chunks); err != nil {
		return "", nil, err
	}
	var text strings.Builder
	var thinking []string
	for _, chunk := range chunks {
		switch chunk.Type {
		case "text":
			text.WriteString(chunk.Text)
		case "thinking":
			for _, thought := range chunk.Thinking {
				if thought.Type == "text" {
					thinking = append(thinking, thought.Text)
				}
			}
		}
	}
	return text.String(), thinking, nil
}

func normalizeArguments(raw json.RawMessage) json.RawMessage {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return json.RawMessage("{}")
	}
	if raw[0] != '"' {
		return json.RawMessage(append([]byte(nil), raw...))
	}
	var arguments string
	_ = json.Unmarshal(raw, &arguments)
	return json.RawMessage(arguments)
}
