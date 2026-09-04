package a2a

import (
	"encoding/base64"
	"fmt"
	"reflect"

	protocol "github.com/a2aproject/a2a-go/a2a"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func (model *Model) request(messages []ai.ModelMessage, params ai.ModelRequestParams) (*protocol.MessageSendParams, error) {
	if err := validateRemoteParams(params); err != nil {
		return nil, err
	}
	parts, err := latestRequestParts(messages)
	if err != nil {
		return nil, err
	}
	if params.Instructions != "" {
		parts = append(protocol.ContentParts{protocol.TextPart{Text: params.Instructions}}, parts...)
	}
	message := protocol.NewMessage(protocol.MessageRoleUser, parts...)
	message.Extensions = append([]string(nil), model.config.Extensions...)
	message.ReferenceTasks = append([]protocol.TaskID(nil), model.config.ReferenceTasks...)
	message.Metadata = cloneModelMetadata(model.config.MessageMetadata)
	message.TaskID, message.ContextID = remoteTaskInfo(messages)
	sendConfig := cloneSendConfig(model.config.SendConfig)
	if sendConfig == nil {
		sendConfig = &protocol.MessageSendConfig{}
	}
	blocking := true
	sendConfig.Blocking = &blocking
	return &protocol.MessageSendParams{
		Config: sendConfig, Message: message, Metadata: cloneModelMetadata(model.config.RequestMetadata),
	}, nil
}

func validateRemoteParams(params ai.ModelRequestParams) error {
	if len(params.Tools) > 0 || len(params.DeferredTools) > 0 || len(params.NativeTools) > 0 || params.OutputTool != nil {
		return fmt.Errorf("ai/a2a: remote models do not support caller-provided tools")
	}
	settings := params.Settings.Clone()
	settings.RequestTimeout = 0
	if !reflect.DeepEqual(settings, ai.ModelSettings{}) {
		return fmt.Errorf("ai/a2a: remote models do not support generation settings")
	}
	return nil
}

func latestRequestParts(messages []ai.ModelMessage) (protocol.ContentParts, error) {
	for index := len(messages) - 1; index >= 0; index-- {
		request, ok := messages[index].(ai.ModelRequest)
		if !ok {
			continue
		}
		parts := make(protocol.ContentParts, 0, len(request.Parts))
		for _, part := range request.Parts {
			converted, err := requestPart(part)
			if err != nil {
				return nil, err
			}
			parts = append(parts, converted...)
		}
		if len(parts) == 0 {
			return nil, fmt.Errorf("ai/a2a: latest model request contains no sendable content")
		}
		return parts, nil
	}
	return nil, fmt.Errorf("ai/a2a: model history contains no request")
}

func requestPart(part ai.RequestPart) (protocol.ContentParts, error) {
	var parts protocol.ContentParts
	var err error
	switch value := part.(type) {
	case ai.SystemPromptPart:
		parts = protocol.ContentParts{protocol.TextPart{Text: value.Content}}
	case ai.UserPromptPart:
		if len(value.Contents) == 0 {
			parts = protocol.ContentParts{protocol.TextPart{Text: value.Content}}
			break
		}
		parts = make(protocol.ContentParts, 0, len(value.Contents))
		for _, content := range value.Contents {
			var converted protocol.ContentParts
			converted, err = userPart(content)
			if err != nil {
				break
			}
			parts = append(parts, converted...)
		}
	case ai.RetryPromptPart:
		parts = protocol.ContentParts{protocol.TextPart{Text: value.ModelResponse()}}
	case ai.SpeechPart:
		if value.Audio != nil {
			parts = protocol.ContentParts{binaryPart(*value.Audio)}
		} else {
			parts = protocol.ContentParts{protocol.TextPart{Text: value.Content()}}
		}
	case ai.ToolAvailabilityDeltaPart:
	case ai.ToolReturnPart:
		err = fmt.Errorf("ai/a2a: remote models do not support local tool results")
	}
	return parts, err
}

func userPart(content ai.UserContent) (protocol.ContentParts, error) {
	var parts protocol.ContentParts
	var err error
	switch value := content.(type) {
	case ai.TextContent:
		parts = protocol.ContentParts{protocol.TextPart{Text: value.Text}}
	case ai.BinaryContent:
		parts = protocol.ContentParts{binaryPart(value)}
	case ai.ImageURL:
		parts = protocol.ContentParts{uriPart(value.URL, value.MediaType, value.Identifier)}
	case ai.AudioURL:
		parts = protocol.ContentParts{uriPart(value.URL, value.MediaType, value.Identifier)}
	case ai.VideoURL:
		parts = protocol.ContentParts{uriPart(value.URL, value.MediaType, value.Identifier)}
	case ai.DocumentURL:
		parts = protocol.ContentParts{uriPart(value.URL, value.MediaType, value.Identifier)}
	case ai.CachePoint:
	case ai.UploadedFile:
		err = fmt.Errorf("ai/a2a: uploaded provider files cannot be sent to a remote agent")
	}
	return parts, err
}

func binaryPart(content ai.BinaryContent) protocol.FilePart {
	return protocol.FilePart{File: protocol.FileBytes{
		FileMeta: protocol.FileMeta{MimeType: content.MediaType, Name: content.Identifier},
		Bytes:    base64.StdEncoding.EncodeToString(content.Data),
	}}
}

func uriPart(uri string, mediaType string, name string) protocol.FilePart {
	return protocol.FilePart{File: protocol.FileURI{
		FileMeta: protocol.FileMeta{MimeType: mediaType, Name: name}, URI: uri,
	}}
}

func remoteTaskInfo(messages []ai.ModelMessage) (protocol.TaskID, string) {
	for index := len(messages) - 1; index >= 0; index-- {
		response, ok := messages[index].(ai.ModelResponse)
		if !ok {
			continue
		}
		if response.ProviderName != modelProviderName {
			return "", ""
		}
		taskID, _ := response.ProviderDetails["task_id"].(string)
		contextID, _ := response.ProviderDetails["context_id"].(string)
		status, hasStatus := response.ProviderDetails["status"].(string)
		if hasStatus && status != string(protocol.TaskStateInputRequired) && status != string(protocol.TaskStateAuthRequired) {
			taskID = ""
		}
		return protocol.TaskID(taskID), contextID
	}
	return "", ""
}
