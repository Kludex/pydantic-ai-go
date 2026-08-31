package ai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
)

func telemetryMessagesJSON(messages []ModelMessage, includeBinary bool) string {
	output := make([]map[string]any, 0, len(messages))
	for _, message := range messages {
		switch message := message.(type) {
		case ModelRequest:
			output = append(output, telemetryRequestMessageGroups(message, includeBinary)...)
		case ModelResponse:
			entry := map[string]any{"role": "assistant", "parts": telemetryResponseParts(message.Parts)}
			if message.FinishReason != "" {
				entry["finish_reason"] = message.FinishReason
			}
			output = append(output, entry)
		}
	}
	return telemetryJSON(output)
}

func telemetryRequestMessageGroups(request ModelRequest, includeBinary bool) []map[string]any {
	groups := make([]map[string]any, 0, 2)
	var role string
	var parts []any
	flush := func() {
		if len(parts) != 0 {
			groups = append(groups, map[string]any{"role": role, "parts": parts})
		}
		parts = nil
	}
	for _, part := range request.Parts {
		nextRole := "user"
		var nextParts []any
		switch part := part.(type) {
		case SystemPromptPart:
			nextRole = "system"
			nextParts = []any{map[string]any{"type": "text", "content": part.Content}}
		case UserPromptPart:
			if len(part.Contents) == 0 {
				nextParts = []any{map[string]any{"type": "text", "content": part.Content}}
			} else {
				for _, content := range part.Contents {
					nextParts = append(nextParts, telemetryUserContent(content, includeBinary))
				}
			}
		case ToolReturnPart:
			nextRole = "tool"
			nextParts = []any{map[string]any{
				"type": "tool_call_response", "id": part.ToolCallID,
				"name": part.ToolName, "result": telemetryValue(part.Content),
			}}
		case RetryPromptPart:
			if part.ToolName != "" {
				nextRole = "tool"
				nextParts = []any{map[string]any{
					"type": "tool_call_response", "id": part.ToolCallID,
					"name": part.ToolName, "result": part.ModelResponse(),
				}}
			} else {
				nextParts = []any{map[string]any{"type": "text", "content": part.ModelResponse()}}
			}
		case ToolAvailabilityDeltaPart:
			nextRole = "system"
			nextParts = []any{map[string]any{
				"type": "text", "content": "Tools available: " + strings.Join(part.ToolsAdded, ", "),
			}}
		default:
			continue
		}
		if role != "" && role != nextRole {
			flush()
		}
		role = nextRole
		parts = append(parts, nextParts...)
	}
	flush()
	return groups
}

func telemetryUserContent(content UserContent, includeBinary bool) any {
	switch content := content.(type) {
	case TextContent:
		return map[string]any{"type": "text", "content": content.Text}
	case ImageURL:
		return map[string]any{"type": "uri", "uri": content.URL, "modality": "image"}
	case BinaryContent:
		value := map[string]any{"type": "blob", "mime_type": content.MediaType}
		if slash := strings.IndexByte(content.MediaType, '/'); slash > 0 {
			value["modality"] = content.MediaType[:slash]
		}
		if includeBinary {
			value["content"] = base64.StdEncoding.EncodeToString(content.Data)
		}
		return value
	default:
		return map[string]any{"type": "text", "content": fmt.Sprint(content)}
	}
}

func telemetryResponseParts(parts []ResponsePart) []any {
	output := make([]any, 0, len(parts))
	for _, part := range parts {
		switch part := part.(type) {
		case TextPart:
			output = append(output, map[string]any{"type": "text", "content": part.Content})
		case ThinkingPart:
			output = append(output, map[string]any{"type": "reasoning", "content": part.Content})
		case CompactionPart:
			output = append(output, map[string]any{"type": "text", "content": part.Content})
		case ToolCallPart:
			output = append(output, telemetryToolCall(part.ToolName, part.ToolCallID, part.Args))
		case NativeToolCallPart:
			output = append(output, telemetryToolCall(part.ToolName, part.ToolCallID, part.Args))
		case NativeToolReturnPart:
			output = append(output, map[string]any{
				"type": "tool_call_response", "id": part.ToolCallID,
				"name": part.ToolName, "result": telemetryValue(part.Content),
			})
		}
	}
	return output
}

func telemetryToolCall(name, id string, arguments json.RawMessage) map[string]any {
	return map[string]any{
		"type": "tool_call", "id": id, "name": name, "arguments": telemetryRawJSON(arguments),
	}
}

func telemetryRawJSON(raw json.RawMessage) any {
	var value any
	if len(raw) != 0 && json.Unmarshal(raw, &value) == nil {
		return value
	}
	return string(raw)
}

func telemetryValue(value any) any {
	if _, err := json.Marshal(value); err == nil {
		return value
	}
	return fmt.Sprint(value)
}

func telemetryJSON(value any) string {
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprintf("%q", fmt.Sprint(value))
	}
	return string(data)
}
