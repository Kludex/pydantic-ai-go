package ai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"mime"
	"net/url"
	"path"
	"strings"
)

func telemetryMessagesJSON(messages []ModelMessage, includeContent, includeBinary bool, version int) string {
	output := make([]map[string]any, 0, len(messages))
	for _, message := range messages {
		switch message := message.(type) {
		case ModelRequest:
			output = append(output, telemetryRequestMessageGroups(message, includeContent, includeBinary, version)...)
		case ModelResponse:
			entry := map[string]any{
				"role": "assistant", "parts": telemetryResponseParts(message.Parts, includeContent, version),
			}
			if message.FinishReason != "" {
				entry["finish_reason"] = message.FinishReason
			}
			output = append(output, entry)
		}
	}
	return telemetryJSON(output)
}

func telemetryRequestMessageGroups(
	request ModelRequest, includeContent, includeBinary bool, version int,
) []map[string]any {
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
			nextParts = []any{telemetryText(part.Content, includeContent)}
		case UserPromptPart:
			if len(part.Contents) == 0 {
				nextParts = []any{telemetryText(part.Content, includeContent)}
			} else {
				for _, content := range part.Contents {
					nextParts = append(nextParts, telemetryUserContent(content, includeContent, includeBinary, version))
				}
			}
		case ToolReturnPart:
			if version >= 6 {
				nextRole = "tool"
			}
			result := map[string]any{
				"type": "tool_call_response", "id": part.ToolCallID, "name": part.ToolName,
			}
			if includeContent {
				result["result"] = telemetryValue(part.Content)
			}
			nextParts = []any{result}
		case RetryPromptPart:
			if part.ToolName != "" {
				if version >= 6 {
					nextRole = "tool"
				}
				result := map[string]any{
					"type": "tool_call_response", "id": part.ToolCallID, "name": part.ToolName,
				}
				if includeContent {
					result["result"] = part.ModelResponse()
				}
				nextParts = []any{result}
			} else {
				nextParts = []any{telemetryText(part.ModelResponse(), includeContent)}
			}
		case ToolAvailabilityDeltaPart:
			nextRole = "system"
			nextParts = []any{telemetryText("Tools available: "+strings.Join(part.ToolsAdded, ", "), includeContent)}
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

func telemetryText(content string, includeContent bool) map[string]any {
	value := map[string]any{"type": "text"}
	if includeContent {
		value["content"] = content
	}
	return value
}

func telemetryUserContent(content UserContent, includeContent, includeBinary bool, version int) any {
	switch content := content.(type) {
	case TextContent:
		return telemetryText(content.Text, includeContent)
	case ImageURL:
		if version <= 3 {
			value := map[string]any{"type": "image-url"}
			if includeContent {
				value["url"] = content.URL
			}
			return value
		}
		value := map[string]any{"type": "uri", "modality": "image"}
		if includeContent {
			value["uri"] = content.URL
		}
		if mediaType := telemetryURLMediaType(content.URL); mediaType != "" {
			value["mime_type"] = mediaType
		}
		return value
	case BinaryContent:
		if version <= 3 {
			value := map[string]any{"type": "binary", "media_type": content.MediaType}
			if includeContent && includeBinary {
				value["content"] = base64.StdEncoding.EncodeToString(content.Data)
			}
			return value
		}
		value := map[string]any{"type": "blob", "mime_type": content.MediaType}
		if slash := strings.IndexByte(content.MediaType, '/'); slash > 0 {
			modality := content.MediaType[:slash]
			if modality == "image" || modality == "audio" || modality == "video" {
				value["modality"] = modality
			}
		}
		if includeContent && includeBinary {
			value["content"] = base64.StdEncoding.EncodeToString(content.Data)
		}
		return value
	default:
		if includeContent {
			return map[string]any{"type": "text", "content": fmt.Sprint(content)}
		}
		return map[string]any{"type": "text"}
	}
}

func telemetryURLMediaType(rawURL string) string {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return ""
	}
	return mime.TypeByExtension(path.Ext(parsed.Path))
}

func telemetryResponseParts(parts []ResponsePart, includeContent bool, version int) []any {
	output := make([]any, 0, len(parts))
	for _, part := range parts {
		switch part := part.(type) {
		case TextPart:
			output = append(output, telemetryText(part.Content, includeContent))
		case ThinkingPart:
			if version >= 3 {
				value := map[string]any{"type": "reasoning"}
				if includeContent {
					value["content"] = part.Content
				}
				output = append(output, value)
			}
		case CompactionPart:
			output = append(output, telemetryText(part.Content, includeContent))
		case ToolCallPart:
			output = append(output, telemetryToolCall(part.ToolName, part.ToolCallID, part.Args, includeContent))
		case NativeToolCallPart:
			output = append(output, telemetryToolCall(part.ToolName, part.ToolCallID, part.Args, includeContent))
		case NativeToolReturnPart:
			value := map[string]any{
				"type": "tool_call_response", "id": part.ToolCallID, "name": part.ToolName, "builtin": true,
			}
			if includeContent {
				value["result"] = telemetryValue(part.Content)
			}
			output = append(output, value)
		}
	}
	return output
}

func telemetryToolCall(name, id string, arguments json.RawMessage, includeContent bool) map[string]any {
	value := map[string]any{"type": "tool_call", "id": id, "name": name}
	if includeContent {
		value["arguments"] = telemetryRawJSON(arguments)
	}
	return value
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
