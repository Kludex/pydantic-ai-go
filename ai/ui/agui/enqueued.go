package agui

import (
	"encoding/base64"
	"encoding/json"
	"strconv"
	"strings"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

const enqueuedMessagesActivity = "pydantic_ai_enqueued_messages"

func enqueuedContent(event ai.EnqueuedMessagesEvent, allowSystemPrompts bool) (map[string]any, bool) {
	var messages []Message
	add := func(role string, content any) {
		id := event.EnqueueID + "-" + strconv.Itoa(len(messages))
		messages = append(messages, Message{ID: id, Role: role, Content: content})
	}
	for _, message := range event.Messages {
		switch message := message.(type) {
		case ai.ModelRequest:
			for _, part := range message.Parts {
				switch part := part.(type) {
				case ai.SystemPromptPart:
					if allowSystemPrompts {
						add("system", part.Content)
					}
				case ai.UserPromptPart:
					if len(part.Contents) == 0 {
						add("user", part.Content)
					} else {
						add("user", userInputContent(part.Contents))
					}
				}
			}
		case ai.ModelResponse:
			// shortcut: only text and thinking are encoded; add tool calls and files when enqueue needs them.
			var text []string
			for _, part := range message.Parts {
				switch part := part.(type) {
				case ai.TextPart:
					text = append(text, part.Content)
				case ai.ThinkingPart:
					add("reasoning", part.Content)
				}
			}
			if len(text) > 0 {
				add("assistant", strings.Join(text, ""))
			}
		}
	}
	if len(messages) == 0 {
		return nil, false
	}
	return map[string]any{"messages": messages}, true
}

func userInputContent(contents []ai.UserContent) []InputContent {
	var inputs []InputContent
	for _, content := range contents {
		switch content := content.(type) {
		case ai.TextContent:
			inputs = append(inputs, InputContent{Type: "text", Text: content.Text})
		case ai.BinaryContent:
			value := base64.StdEncoding.EncodeToString(content.Data)
			inputs = append(inputs, sourceInput("data", value, content.MediaType, content.VendorMetadata))
		case ai.ImageURL:
			inputs = append(inputs, sourceInput("url", content.URL, content.MediaType, content.VendorMetadata))
		case ai.AudioURL:
			inputs = append(inputs, sourceInput("url", content.URL, content.MediaType, content.VendorMetadata))
		case ai.VideoURL:
			inputs = append(inputs, sourceInput("url", content.URL, content.MediaType, content.VendorMetadata))
		case ai.DocumentURL:
			inputs = append(inputs, sourceInput("url", content.URL, content.MediaType, content.VendorMetadata))
		}
	}
	return inputs
}

func sourceInput(sourceType string, value string, mediaType string, vendorMetadata map[string]any) InputContent {
	kind := strings.SplitN(mediaType, "/", 2)[0]
	if kind != "image" && kind != "audio" && kind != "video" {
		kind = "document"
	}
	input := InputContent{Type: kind, Source: &InputContentSource{Type: sourceType, Value: value, MimeType: mediaType}}
	if len(vendorMetadata) > 0 {
		input.Metadata = map[string]any{"vendor_metadata": cloneMap(vendorMetadata)}
	}
	return input
}

// enqueuedMessages skips malformed content and nested activities, so a client cannot recurse.
func enqueuedMessages(content map[string]any) []Message {
	encoded, _ := json.Marshal(content["messages"])
	var messages []Message
	if json.Unmarshal(encoded, &messages) != nil {
		return nil
	}
	kept := messages[:0]
	for _, message := range messages {
		if message.Role != "activity" || message.ActivityType != enqueuedMessagesActivity {
			kept = append(kept, message)
		}
	}
	return kept
}
