package a2a

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	protocol "github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"

	ai "github.com/Kludex/pydantic-ai-go"
)

func prepareRequest(
	request *a2asrv.RequestContext, options ai.MessageSanitizationOptions,
) ([]ai.UserContent, []ai.ModelMessage, error) {
	if request.Message.Role != protocol.MessageRoleUser {
		return nil, nil, fmt.Errorf("ai/a2a: request message must have the user role")
	}
	prompt, err := userContents(request.Message.Parts)
	if err != nil {
		return nil, nil, err
	}
	var history []ai.ModelMessage
	seenTasks := map[protocol.TaskID]struct{}{}
	if request.TaskID != "" {
		seenTasks[protocol.TaskID(request.TaskID)] = struct{}{}
	}
	for _, task := range request.RelatedTasks {
		if task == nil {
			continue
		}
		if _, seen := seenTasks[task.ID]; seen {
			continue
		}
		seenTasks[task.ID] = struct{}{}
		messages, err := relatedTaskMessages(task)
		if err != nil {
			return nil, nil, err
		}
		history = append(history, messages...)
	}
	if request.StoredTask != nil {
		for _, message := range request.StoredTask.History {
			if message == nil || message.ID == request.Message.ID {
				continue
			}
			converted, err := modelMessage(message)
			if err != nil {
				return nil, nil, err
			}
			history = append(history, converted)
		}
	}
	history, _, err = ai.SanitizeMessages(history, options)
	if err != nil {
		return nil, nil, err
	}
	return prompt, history, nil
}

func relatedTaskMessages(task *protocol.Task) ([]ai.ModelMessage, error) {
	messages := make([]ai.ModelMessage, 0, 1+len(task.History)+len(task.Artifacts))
	messages = append(messages, ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{
		Content: fmt.Sprintf("Related A2A task %q:", task.ID),
	}}})
	for _, message := range task.History {
		if message == nil {
			continue
		}
		converted, err := modelMessage(message)
		if err != nil {
			return nil, fmt.Errorf("ai/a2a: related task %q: %w", task.ID, err)
		}
		messages = append(messages, converted)
	}
	for _, artifact := range task.Artifacts {
		if artifact == nil {
			continue
		}
		parts, err := modelResponseParts(artifact.Parts)
		if err != nil {
			return nil, fmt.Errorf("ai/a2a: related task %q artifact %q: %w", task.ID, artifact.ID, err)
		}
		if len(parts) > 0 {
			messages = append(messages, ai.ModelResponse{Parts: parts})
		}
	}
	return messages, nil
}

func modelMessage(message *protocol.Message) (ai.ModelMessage, error) {
	switch message.Role {
	case protocol.MessageRoleUser:
		content, err := userContents(message.Parts)
		if err != nil {
			return nil, err
		}
		return ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Contents: content}}}, nil
	case protocol.MessageRoleAgent:
		parts, err := modelResponseParts(message.Parts)
		if err != nil {
			return nil, err
		}
		return ai.ModelResponse{Parts: parts}, nil
	default:
		return nil, fmt.Errorf("ai/a2a: unsupported message role %q", message.Role)
	}
}

func modelResponseParts(parts protocol.ContentParts) ([]ai.ResponsePart, error) {
	converted := make([]ai.ResponsePart, 0, len(parts))
	for _, part := range parts {
		switch value := part.(type) {
		case protocol.TextPart:
			converted = append(converted, ai.TextPart{Content: value.Text})
		case protocol.DataPart:
			encoded, _ := json.Marshal(value.Data)
			converted = append(converted, ai.TextPart{Content: string(encoded)})
		case protocol.FilePart:
			binary, err := fileBinary(value)
			if err != nil {
				return nil, err
			}
			converted = append(converted, ai.FilePart{Content: binary})
		}
	}
	return converted, nil
}

func userContents(parts protocol.ContentParts) ([]ai.UserContent, error) {
	content := make([]ai.UserContent, 0, len(parts))
	for _, part := range parts {
		switch value := part.(type) {
		case protocol.TextPart:
			content = append(content, ai.TextContent{Text: value.Text})
		case protocol.DataPart:
			encoded, _ := json.Marshal(value.Data)
			content = append(content, ai.TextContent{Text: string(encoded)})
		case protocol.FilePart:
			item, err := userFile(value)
			if err != nil {
				return nil, err
			}
			content = append(content, item)
		default:
			return nil, fmt.Errorf("ai/a2a: unsupported message part %T", part)
		}
	}
	if len(content) == 0 {
		return nil, fmt.Errorf("ai/a2a: request message must contain content")
	}
	return content, nil
}

func userFile(part protocol.FilePart) (ai.UserContent, error) {
	switch file := part.File.(type) {
	case protocol.FileBytes:
		data, err := base64.StdEncoding.DecodeString(file.Bytes)
		if err != nil {
			return nil, fmt.Errorf("ai/a2a: decode file bytes: %w", err)
		}
		return ai.BinaryContent{Data: data, MediaType: file.MimeType, Identifier: file.Name}, nil
	case protocol.FileURI:
		switch {
		case strings.HasPrefix(file.MimeType, "image/"):
			return ai.ImageURL{URL: file.URI, MediaType: file.MimeType, Identifier: file.Name}, nil
		case strings.HasPrefix(file.MimeType, "audio/"):
			return ai.AudioURL{URL: file.URI, MediaType: file.MimeType, Identifier: file.Name}, nil
		case strings.HasPrefix(file.MimeType, "video/"):
			return ai.VideoURL{URL: file.URI, MediaType: file.MimeType, Identifier: file.Name}, nil
		default:
			return ai.DocumentURL{URL: file.URI, MediaType: file.MimeType, Identifier: file.Name}, nil
		}
	default:
		return nil, fmt.Errorf("ai/a2a: unsupported file content %T", part.File)
	}
}

func fileBinary(part protocol.FilePart) (ai.BinaryContent, error) {
	file, ok := part.File.(protocol.FileBytes)
	if !ok {
		return ai.BinaryContent{}, fmt.Errorf("ai/a2a: agent file history must contain inline bytes")
	}
	data, err := base64.StdEncoding.DecodeString(file.Bytes)
	if err != nil {
		return ai.BinaryContent{}, fmt.Errorf("ai/a2a: decode agent file bytes: %w", err)
	}
	return ai.BinaryContent{Data: data, MediaType: file.MimeType, Identifier: file.Name}, nil
}
