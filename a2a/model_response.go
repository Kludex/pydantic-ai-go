package a2a

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"time"

	protocol "github.com/a2aproject/a2a-go/a2a"

	ai "github.com/Kludex/pydantic-ai-go"
)

// TaskError reports a remote A2A task that did not complete successfully.
type TaskError struct {
	// TaskID identifies the remote task.
	TaskID protocol.TaskID
	// ContextID identifies the remote conversation context.
	ContextID string
	// State is the remote task state.
	State protocol.TaskState
	// Message contains the remote status message text when available.
	Message string
}

// Error describes the unsuccessful remote task.
func (taskError *TaskError) Error() string {
	if taskError.Message == "" {
		return fmt.Sprintf("ai/a2a: task %q entered state %q", taskError.TaskID, taskError.State)
	}
	return fmt.Sprintf("ai/a2a: task %q entered state %q: %s", taskError.TaskID, taskError.State, taskError.Message)
}

func modelResponse(result protocol.SendMessageResult, modelName string, providerURL string) (*ai.ModelResponse, error) {
	switch value := result.(type) {
	case *protocol.Message:
		parts, err := remoteResponseParts(value.Parts, "", nil)
		if err != nil {
			return nil, err
		}
		return &ai.ModelResponse{
			Parts: parts, ModelName: modelName, Timestamp: time.Now().UTC(), ProviderName: modelProviderName,
			ProviderURL: providerURL, ProviderDetails: eventDetails(value), ProviderResponseID: value.ID,
			FinishReason: ai.FinishReasonStop,
		}, nil
	case *protocol.Task:
		return taskResponse(value, modelName, providerURL)
	default:
		return nil, fmt.Errorf("ai/a2a: unsupported send result %T", result)
	}
}

func taskResponse(task *protocol.Task, modelName string, providerURL string) (*ai.ModelResponse, error) {
	if task == nil {
		return nil, fmt.Errorf("ai/a2a: remote agent returned a nil task")
	}
	if task.Status.State != protocol.TaskStateCompleted {
		return nil, taskError(task.ID, task.ContextID, task.Status)
	}
	parts := make([]ai.ResponsePart, 0)
	for _, artifact := range task.Artifacts {
		if artifact == nil {
			continue
		}
		converted, err := remoteResponseParts(artifact.Parts, string(artifact.ID), artifactDetails(artifact))
		if err != nil {
			return nil, err
		}
		parts = append(parts, converted...)
	}
	if len(parts) == 0 && task.Status.Message != nil {
		converted, err := remoteResponseParts(task.Status.Message.Parts, task.Status.Message.ID, nil)
		if err != nil {
			return nil, err
		}
		parts = append(parts, converted...)
	}
	if len(parts) == 0 {
		return nil, fmt.Errorf("ai/a2a: completed task %q contains no output", task.ID)
	}
	timestamp := time.Now().UTC()
	if task.Status.Timestamp != nil {
		timestamp = task.Status.Timestamp.UTC()
	}
	return &ai.ModelResponse{
		Parts: parts, ModelName: modelName, Timestamp: timestamp, ProviderName: modelProviderName,
		ProviderURL: providerURL, ProviderDetails: eventDetails(task), ProviderResponseID: string(task.ID),
		FinishReason: ai.FinishReasonStop,
	}, nil
}

func remoteResponseParts(
	parts protocol.ContentParts, ownerID string, ownerDetails map[string]any,
) ([]ai.ResponsePart, error) {
	converted := make([]ai.ResponsePart, 0, len(parts))
	for index, part := range parts {
		partID := ownerID
		if len(parts) > 1 {
			partID = fmt.Sprintf("%s:%d", ownerID, index)
		}
		details := mergePartDetails(ownerDetails, part.Meta())
		switch value := part.(type) {
		case protocol.TextPart:
			converted = append(converted, ai.TextPart{
				Content: value.Text, ID: partID, ProviderName: modelProviderName, ProviderDetails: details,
			})
		case protocol.DataPart:
			encoded, err := json.Marshal(value.Data)
			if err != nil {
				return nil, fmt.Errorf("ai/a2a: encode response data: %w", err)
			}
			converted = append(converted, ai.TextPart{
				Content: string(encoded), ID: partID, ProviderName: modelProviderName, ProviderDetails: details,
			})
		case protocol.FilePart:
			file, ok := value.File.(protocol.FileBytes)
			if !ok {
				return nil, fmt.Errorf("ai/a2a: response file %q is not inline", partID)
			}
			data, err := base64.StdEncoding.DecodeString(file.Bytes)
			if err != nil {
				return nil, fmt.Errorf("ai/a2a: decode response file %q: %w", partID, err)
			}
			converted = append(converted, ai.FilePart{
				Content: ai.BinaryContent{Data: data, MediaType: file.MimeType, Identifier: file.Name},
				ID:      partID, ProviderName: modelProviderName, ProviderDetails: details,
			})
		}
	}
	return converted, nil
}

func eventDetails(event protocol.Event) map[string]any {
	info := event.TaskInfo()
	details := map[string]any{
		"task_id": string(info.TaskID), "context_id": info.ContextID,
	}
	if metadata := event.Meta(); len(metadata) > 0 {
		details["metadata"] = cloneModelMetadata(metadata)
	}
	switch value := event.(type) {
	case *protocol.Task:
		details["status"] = string(value.Status.State)
	case *protocol.TaskStatusUpdateEvent:
		details["status"] = string(value.Status.State)
	}
	return details
}

func artifactDetails(artifact *protocol.Artifact) map[string]any {
	details := map[string]any{
		"artifact_id": string(artifact.ID),
	}
	if artifact.Name != "" {
		details["artifact_name"] = artifact.Name
	}
	if artifact.Description != "" {
		details["artifact_description"] = artifact.Description
	}
	if len(artifact.Extensions) > 0 {
		details["artifact_extensions"] = append([]string(nil), artifact.Extensions...)
	}
	if len(artifact.Metadata) > 0 {
		details["artifact_metadata"] = cloneModelMetadata(artifact.Metadata)
	}
	return details
}

func mergePartDetails(owner map[string]any, metadata map[string]any) map[string]any {
	if len(owner) == 0 && len(metadata) == 0 {
		return nil
	}
	merged := cloneModelMetadata(owner)
	if merged == nil {
		merged = make(map[string]any)
	}
	if len(metadata) > 0 {
		merged["metadata"] = cloneModelMetadata(metadata)
	}
	return merged
}

func taskError(taskID protocol.TaskID, contextID string, status protocol.TaskStatus) *TaskError {
	message := ""
	if status.Message != nil {
		for _, part := range status.Message.Parts {
			if text, ok := part.(protocol.TextPart); ok {
				message += text.Text
			}
		}
	}
	return &TaskError{TaskID: taskID, ContextID: contextID, State: status.State, Message: message}
}
