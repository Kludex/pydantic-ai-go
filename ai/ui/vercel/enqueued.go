package vercel

import (
	"encoding/base64"
	"encoding/json"
	"strconv"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func enqueuedChunk(event ai.EnqueuedMessagesEvent, allowSystemPrompts bool) (Chunk, bool) {
	var messages []UIMessage
	add := func(role string, parts []UIMessagePart, metadata map[string]any) {
		if len(parts) > 0 {
			id := event.EnqueueID + "-" + role + "-" + strconv.Itoa(len(messages))
			messages = append(messages, UIMessage{ID: id, Role: role, Parts: parts, Metadata: metadata})
		}
	}
	for _, message := range event.Messages {
		switch message := message.(type) {
		case ai.ModelRequest:
			var system, user []UIMessagePart
			for _, part := range message.Parts {
				switch part := part.(type) {
				case ai.SystemPromptPart:
					if allowSystemPrompts {
						system = append(system, UIMessagePart{Type: "text", Text: part.Content, State: "done"})
					}
				case ai.UserPromptPart:
					parts := userUIParts(part)
					if len(user) > 0 && len(parts) > 0 {
						parts[0].ProviderMetadata = withUserPromptStart(parts[0].ProviderMetadata)
					}
					user = append(user, parts...)
				}
			}
			metadata := dumpMessageMetadata(message.Metadata, message.Timestamp)
			if len(user) == 0 {
				add("system", system, metadata)
				continue
			}
			add("system", system, nil)
			add("user", user, metadata)
		case ai.ModelResponse:
			var parts []UIMessagePart
			for _, part := range message.Parts {
				// shortcut: only text, thinking, and files are encoded; add tool parts when enqueue needs them.
				switch part := part.(type) {
				case ai.TextPart:
					parts = append(parts, UIMessagePart{Type: "text", Text: part.Content, State: "done"})
				case ai.ThinkingPart:
					parts = append(parts, UIMessagePart{Type: "reasoning", Text: part.Content, State: "done"})
				case ai.FilePart:
					chunk := fileChunk(part)
					parts = append(parts, UIMessagePart{
						Type: "file", URL: chunk.URL, MediaType: chunk.MediaType, ProviderMetadata: chunk.ProviderMetadata,
					})
				}
			}
			add("assistant", parts, dumpMessageMetadata(message.Metadata, message.Timestamp))
		}
	}
	if len(messages) == 0 {
		return Chunk{}, false
	}
	return Chunk{Type: ChunkDataEnqueuedMessages, ID: event.EnqueueID, Data: map[string]any{"messages": messages}}, true
}

func userUIParts(part ai.UserPromptPart) []UIMessagePart {
	if len(part.Contents) == 0 {
		return []UIMessagePart{{Type: "text", Text: part.Content, State: "done"}}
	}
	var parts []UIMessagePart
	for _, content := range part.Contents {
		switch content := content.(type) {
		case ai.TextContent:
			parts = append(parts, UIMessagePart{Type: "text", Text: content.Text, State: "done"})
		case ai.BinaryContent:
			parts = append(parts, UIMessagePart{
				Type: "file", MediaType: content.MediaType,
				URL:              "data:" + content.MediaType + ";base64," + base64.StdEncoding.EncodeToString(content.Data),
				ProviderMetadata: dumpPartMetadata("", "", "", nil, "", content.VendorMetadata),
			})
		case ai.ImageURL:
			parts = append(parts, urlUIPart(content.URL, content.MediaType, content.VendorMetadata))
		case ai.AudioURL:
			parts = append(parts, urlUIPart(content.URL, content.MediaType, content.VendorMetadata))
		case ai.VideoURL:
			parts = append(parts, urlUIPart(content.URL, content.MediaType, content.VendorMetadata))
		case ai.DocumentURL:
			parts = append(parts, urlUIPart(content.URL, content.MediaType, content.VendorMetadata))
		}
	}
	return parts
}

// withUserPromptStart marks where a request's second or later prompt begins,
// since all of a request's prompts share one user UIMessage.
func withUserPromptStart(metadata map[string]any) map[string]any {
	metadata = cloneMap(metadata)
	if metadata == nil {
		metadata = map[string]any{}
	}
	wrapped, _ := metadata[providerMetadataKey].(map[string]any)
	if wrapped == nil {
		wrapped = map[string]any{}
	}
	wrapped[userPromptStartKey] = true
	metadata[providerMetadataKey] = wrapped
	return metadata
}

func urlUIPart(url string, mediaType string, vendorMetadata map[string]any) UIMessagePart {
	return UIMessagePart{
		Type: "file", URL: url, MediaType: mediaType,
		ProviderMetadata: dumpPartMetadata("", "", "", nil, "", vendorMetadata),
	}
}

// enqueuedMessages skips malformed data and nested payloads, so a client cannot recurse.
func enqueuedMessages(data any) []UIMessage {
	encoded, _ := json.Marshal(data)
	var payload struct {
		Messages []UIMessage `json:"messages"`
	}
	if json.Unmarshal(encoded, &payload) != nil {
		return nil
	}
	for index, message := range payload.Messages {
		var parts []UIMessagePart
		for _, part := range message.Parts {
			if part.Type != string(ChunkDataEnqueuedMessages) {
				parts = append(parts, part)
			}
		}
		payload.Messages[index].Parts = parts
	}
	return payload.Messages
}
