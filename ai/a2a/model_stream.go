package a2a

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"reflect"
	"time"

	protocol "github.com/a2aproject/a2a-go/a2a"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

type remoteArtifacts struct {
	order     []protocol.ArtifactID
	artifacts map[protocol.ArtifactID]*protocol.Artifact
}

func (model *Model) stream(
	ctx context.Context, request *protocol.MessageSendParams,
) iter.Seq2[ai.ModelStreamEvent, error] {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		state := remoteArtifacts{artifacts: make(map[protocol.ArtifactID]*protocol.Artifact)}
		for event, err := range model.client.SendStreamingMessage(ctx, request) {
			if err != nil {
				yield(nil, ai.NewModelTransportError(ctx, model, "stream message", err))
				return
			}
			if nilProtocolEvent(event) {
				yield(nil, fmt.Errorf("ai/a2a: remote agent streamed a nil event"))
				return
			}
			if !yield(ai.ResponseMetadataEvent{
				ModelName: model.name, ProviderName: modelProviderName, ProviderURL: model.config.ProviderURL,
				ProviderDetails: eventDetails(event), ProviderResponseID: remoteResponseID(event),
			}, nil) {
				return
			}
			switch value := event.(type) {
			case *protocol.Message:
				parts, partErr := remoteResponseParts(value.Parts, value.ID, nil)
				if partErr != nil {
					yield(nil, partErr)
					return
				}
				if !yieldResponseParts(yield, parts, false) {
					return
				}
				yield(finishEvent(parts, model, event, time.Now().UTC()), nil)
				return
			case *protocol.Task:
				if stateErr := state.setTask(value); stateErr != nil {
					yield(nil, fmt.Errorf("ai/a2a: retain task artifacts: %w", stateErr))
					return
				}
				parts, partErr := state.parts()
				if partErr != nil {
					yield(nil, partErr)
					return
				}
				if !yieldResponseParts(yield, parts, false) {
					return
				}
				if value.Status.State == protocol.TaskStateCompleted {
					if len(parts) == 0 {
						yield(nil, fmt.Errorf("ai/a2a: completed task %q contains no output", value.ID))
						return
					}
					timestamp := statusTimestamp(value.Status)
					yield(finishEvent(parts, model, event, timestamp), nil)
					return
				}
				if value.Status.State != protocol.TaskStateSubmitted && value.Status.State != protocol.TaskStateWorking {
					yield(nil, taskError(value.ID, value.ContextID, value.Status))
					return
				}
			case *protocol.TaskArtifactUpdateEvent:
				if value.Artifact == nil {
					yield(nil, fmt.Errorf("ai/a2a: artifact update contains no artifact"))
					return
				}
				replace, stateErr := state.update(value)
				if stateErr != nil {
					yield(nil, fmt.Errorf("ai/a2a: retain artifact update: %w", stateErr))
					return
				}
				parts, partErr := remoteResponseParts(
					value.Artifact.Parts, string(value.Artifact.ID), artifactDetails(value.Artifact),
				)
				if partErr != nil {
					yield(nil, partErr)
					return
				}
				if !yieldResponseParts(yield, parts, replace) {
					return
				}
			case *protocol.TaskStatusUpdateEvent:
				if value.Status.State == protocol.TaskStateSubmitted || value.Status.State == protocol.TaskStateWorking {
					continue
				}
				if value.Status.State != protocol.TaskStateCompleted {
					yield(nil, taskError(value.TaskID, value.ContextID, value.Status))
					return
				}
				parts, _ := state.parts()
				if len(parts) == 0 && value.Status.Message != nil {
					var partErr error
					parts, partErr = remoteResponseParts(value.Status.Message.Parts, value.Status.Message.ID, nil)
					if partErr != nil {
						yield(nil, partErr)
						return
					}
					if !yieldResponseParts(yield, parts, false) {
						return
					}
				}
				if len(parts) == 0 {
					yield(nil, fmt.Errorf("ai/a2a: completed task %q contains no output", value.TaskID))
					return
				}
				yield(finishEvent(parts, model, event, statusTimestamp(value.Status)), nil)
				return
			}
		}
		yield(nil, fmt.Errorf("ai/a2a: remote agent stream ended without a final event"))
	}
}

func yieldResponseParts(
	yield func(ai.ModelStreamEvent, error) bool, parts []ai.ResponsePart, replace bool,
) bool {
	for _, part := range parts {
		var event ai.ModelStreamEvent
		switch value := part.(type) {
		case ai.TextPart:
			event = ai.TextDeltaEvent{
				PartID: value.ID, Delta: value.Content, ID: value.ID,
				ProviderName: value.ProviderName, ProviderDetails: value.ProviderDetails,
			}
		case ai.FilePart:
			event = ai.FileEvent{PartID: value.ID, Part: value, Replace: replace}
		}
		if event != nil && !yield(event, nil) {
			return false
		}
	}
	return true
}

func finishEvent(
	parts []ai.ResponsePart, model *Model, event protocol.Event, timestamp time.Time,
) ai.FinishEvent {
	return ai.FinishEvent{
		Parts: parts, ModelName: model.name, Timestamp: timestamp, ProviderName: modelProviderName,
		ProviderURL: model.config.ProviderURL, ProviderDetails: eventDetails(event),
		ProviderResponseID: remoteResponseID(event), FinishReason: ai.FinishReasonStop,
	}
}

func remoteResponseID(event protocol.Event) string {
	info := event.TaskInfo()
	if info.TaskID != "" {
		return string(info.TaskID)
	}
	if message, ok := event.(*protocol.Message); ok {
		return message.ID
	}
	return ""
}

func statusTimestamp(status protocol.TaskStatus) time.Time {
	if status.Timestamp != nil {
		return status.Timestamp.UTC()
	}
	return time.Now().UTC()
}

func (state *remoteArtifacts) setTask(task *protocol.Task) error {
	for _, artifact := range task.Artifacts {
		if artifact == nil {
			continue
		}
		if _, exists := state.artifacts[artifact.ID]; !exists {
			state.order = append(state.order, artifact.ID)
		}
		cloned, err := cloneArtifact(artifact)
		if err != nil {
			return err
		}
		state.artifacts[artifact.ID] = cloned
	}
	return nil
}

func (state *remoteArtifacts) update(event *protocol.TaskArtifactUpdateEvent) (bool, error) {
	artifact := event.Artifact
	current, exists := state.artifacts[artifact.ID]
	if !exists {
		state.order = append(state.order, artifact.ID)
	}
	if event.Append && exists {
		parts, err := cloneContentParts(artifact.Parts)
		if err != nil {
			return false, err
		}
		current.Parts = append(current.Parts, parts...)
		if artifact.Name != "" {
			current.Name = artifact.Name
		}
		if artifact.Description != "" {
			current.Description = artifact.Description
		}
		if len(artifact.Extensions) > 0 {
			current.Extensions = append([]string(nil), artifact.Extensions...)
		}
		if len(artifact.Metadata) > 0 {
			current.Metadata = cloneModelMetadata(artifact.Metadata)
		}
		return false, nil
	}
	cloned, err := cloneArtifact(artifact)
	if err != nil {
		return false, err
	}
	state.artifacts[artifact.ID] = cloned
	return exists, nil
}

func (state *remoteArtifacts) parts() ([]ai.ResponsePart, error) {
	parts := make([]ai.ResponsePart, 0)
	for _, id := range state.order {
		artifact := state.artifacts[id]
		converted, err := remoteResponseParts(artifact.Parts, string(id), artifactDetails(artifact))
		if err != nil {
			return nil, err
		}
		parts = append(parts, converted...)
	}
	return parts, nil
}

func nilProtocolEvent(event protocol.Event) bool {
	if event == nil {
		return true
	}
	value := reflect.ValueOf(event)
	return value.Kind() == reflect.Pointer && value.IsNil()
}

func cloneArtifact(artifact *protocol.Artifact) (*protocol.Artifact, error) {
	cloned := *artifact
	cloned.Extensions = append([]string(nil), artifact.Extensions...)
	cloned.Metadata = cloneModelMetadata(artifact.Metadata)
	var err error
	cloned.Parts, err = cloneContentParts(artifact.Parts)
	if err != nil {
		return nil, err
	}
	return &cloned, nil
}

func cloneContentParts(parts protocol.ContentParts) (protocol.ContentParts, error) {
	cloned := make(protocol.ContentParts, len(parts))
	for index, part := range parts {
		switch value := part.(type) {
		case protocol.TextPart:
			value.Metadata = cloneModelMetadata(value.Metadata)
			cloned[index] = value
		case protocol.DataPart:
			encoded, err := json.Marshal(value.Data)
			if err != nil {
				return nil, fmt.Errorf("ai/a2a: encode response data: %w", err)
			}
			value.Data = nil
			_ = json.Unmarshal(encoded, &value.Data)
			value.Metadata = cloneModelMetadata(value.Metadata)
			cloned[index] = value
		case protocol.FilePart:
			value.Metadata = cloneModelMetadata(value.Metadata)
			cloned[index] = value
		}
	}
	return cloned, nil
}
