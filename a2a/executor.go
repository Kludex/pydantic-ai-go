// Package a2a adapts typed agents to the official Agent2Agent Go SDK.
package a2a

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"

	protocol "github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"

	ai "github.com/Kludex/pydantic-ai-go"
)

// DepsFunc resolves typed dependencies for one A2A execution.
type DepsFunc[Deps any] func(ctx context.Context, request *a2asrv.RequestContext) (Deps, error)

// Config controls A2A execution and inbound trust.
type Config[Deps any] struct {
	// Deps is reused when ResolveDeps is nil.
	Deps Deps
	// ResolveDeps resolves request-specific dependencies and may run concurrently.
	ResolveDeps DepsFunc[Deps]
	// Sanitization controls untrusted stored task history. The zero value is secure.
	Sanitization ai.MessageSanitizationOptions
	// RunOptions are reused across executions and must be concurrency-safe.
	RunOptions []ai.RunOption
	// ArtifactName names the streamed output artifact.
	ArtifactName string
	// ArtifactDescription describes the streamed output artifact.
	ArtifactDescription string
	// ArtifactExtensions lists extension URIs associated with the output artifact.
	ArtifactExtensions []string
	// ArtifactMetadata contains detached extension metadata for the output artifact.
	ArtifactMetadata map[string]any
}

// Executor implements the official SDK's a2asrv.AgentExecutor.
type Executor[Deps, Output any] struct {
	agent  *ai.Agent[Deps, Output]
	config Config[Deps]
}

// NewExecutor creates an A2A executor for a typed agent.
func NewExecutor[Deps, Output any](agent *ai.Agent[Deps, Output], config Config[Deps]) *Executor[Deps, Output] {
	if agent == nil {
		panic("ai/a2a: agent must not be nil")
	}
	config.RunOptions = append([]ai.RunOption(nil), config.RunOptions...)
	config.ArtifactExtensions = append([]string(nil), config.ArtifactExtensions...)
	config.ArtifactMetadata = cloneJSONMap(config.ArtifactMetadata)
	return &Executor[Deps, Output]{agent: agent, config: config}
}

// Execute converts one A2A request, runs the agent, and publishes task updates.
func (executor *Executor[Deps, Output]) Execute(
	ctx context.Context, request *a2asrv.RequestContext, queue eventqueue.Queue,
) error {
	if request == nil || request.Message == nil {
		return fmt.Errorf("ai/a2a: request message must not be nil")
	}
	if queue == nil {
		return fmt.Errorf("ai/a2a: event queue must not be nil")
	}
	prompt, history, err := prepareRequest(request, executor.config.Sanitization)
	if err != nil {
		return executor.fail(ctx, request, queue, err)
	}
	deps := executor.config.Deps
	if executor.config.ResolveDeps != nil {
		deps, err = executor.config.ResolveDeps(ctx, request)
		if err != nil {
			return executor.fail(ctx, request, queue, fmt.Errorf("ai/a2a: resolve dependencies: %w", err))
		}
	}
	if request.StoredTask == nil {
		if err := queue.Write(ctx, protocol.NewStatusUpdateEvent(request, protocol.TaskStateSubmitted, nil)); err != nil {
			return fmt.Errorf("ai/a2a: publish submitted state: %w", err)
		}
	}
	if err := queue.Write(ctx, protocol.NewStatusUpdateEvent(request, protocol.TaskStateWorking, nil)); err != nil {
		return fmt.Errorf("ai/a2a: publish working state: %w", err)
	}
	options := append([]ai.RunOption(nil), executor.config.RunOptions...)
	options = append(options, ai.WithMessageHistory(history), ai.WithConversationID(request.ContextID))
	stream := executor.agent.RunStreamParts(ctx, prompt, deps, options...)
	var artifactID protocol.ArtifactID
	emitted := false
	deferred := false
	for event, eventErr := range stream.Events() {
		if eventErr != nil {
			if errors.Is(eventErr, context.Canceled) || errors.Is(eventErr, ai.ErrRunCancelled) {
				return executor.finish(ctx, request, queue, protocol.TaskStateCanceled, eventErr)
			}
			return executor.fail(ctx, request, queue, eventErr)
		}
		switch value := event.(type) {
		case ai.PartStartEvent:
			parts := responseParts(value.Part)
			if len(parts) > 0 {
				var err error
				artifactID, err = executor.publishArtifact(ctx, request, queue, artifactID, parts)
				if err != nil {
					return err
				}
				emitted = true
			}
		case ai.PartDeltaEvent:
			parts := deltaParts(value.Delta)
			if len(parts) > 0 {
				var err error
				artifactID, err = executor.publishArtifact(ctx, request, queue, artifactID, parts)
				if err != nil {
					return err
				}
				emitted = true
			}
		case ai.DeferredToolRequestsEvent:
			deferred = true
		}
	}
	if deferred {
		return executor.finish(ctx, request, queue, protocol.TaskStateInputRequired, nil)
	}
	if !emitted {
		result := stream.Result()
		if result != nil {
			encoded, err := json.Marshal(result.Output)
			if err != nil {
				return executor.fail(ctx, request, queue, fmt.Errorf("ai/a2a: encode output: %w", err))
			}
			if _, err := executor.publishArtifact(ctx, request, queue, artifactID, []protocol.Part{
				protocol.TextPart{Text: string(encoded)},
			}); err != nil {
				return err
			}
		}
	}
	return executor.finish(ctx, request, queue, protocol.TaskStateCompleted, nil)
}

// Cancel publishes a terminal canceled state. Active execution observes the SDK's canceled context.
func (*Executor[Deps, Output]) Cancel(
	ctx context.Context, request *a2asrv.RequestContext, queue eventqueue.Queue,
) error {
	if request == nil || queue == nil {
		return fmt.Errorf("ai/a2a: cancellation requires a request and event queue")
	}
	event := protocol.NewStatusUpdateEvent(request, protocol.TaskStateCanceled, nil)
	event.Final = true
	if err := queue.Write(ctx, event); err != nil {
		return fmt.Errorf("ai/a2a: publish canceled state: %w", err)
	}
	return nil
}

func (executor *Executor[Deps, Output]) fail(
	ctx context.Context, request *a2asrv.RequestContext, queue eventqueue.Queue, runErr error,
) error {
	return executor.finish(ctx, request, queue, protocol.TaskStateFailed, runErr)
}

func (*Executor[Deps, Output]) finish(
	ctx context.Context, request *a2asrv.RequestContext, queue eventqueue.Queue,
	state protocol.TaskState, runErr error,
) error {
	var message *protocol.Message
	if runErr != nil {
		message = protocol.NewMessageForTask(protocol.MessageRoleAgent, request, protocol.TextPart{Text: runErr.Error()})
	}
	event := protocol.NewStatusUpdateEvent(request, state, message)
	event.Final = true
	if err := queue.Write(ctx, event); err != nil {
		return fmt.Errorf("ai/a2a: publish %s state: %w", state, err)
	}
	return nil
}

func (executor *Executor[Deps, Output]) publishArtifact(
	ctx context.Context, request *a2asrv.RequestContext, queue eventqueue.Queue,
	artifactID protocol.ArtifactID, parts []protocol.Part,
) (protocol.ArtifactID, error) {
	var event *protocol.TaskArtifactUpdateEvent
	if artifactID == "" {
		event = protocol.NewArtifactEvent(request, parts...)
		event.Artifact.Name = executor.config.ArtifactName
		event.Artifact.Description = executor.config.ArtifactDescription
		event.Artifact.Extensions = append([]string(nil), executor.config.ArtifactExtensions...)
		event.Artifact.Metadata = cloneJSONMap(executor.config.ArtifactMetadata)
		artifactID = event.Artifact.ID
	} else {
		event = protocol.NewArtifactUpdateEvent(request, artifactID, parts...)
	}
	if err := queue.Write(ctx, event); err != nil {
		return artifactID, fmt.Errorf("ai/a2a: publish artifact: %w", err)
	}
	return artifactID, nil
}

func cloneJSONMap(value map[string]any) map[string]any {
	if value == nil {
		return nil
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		panic(fmt.Sprintf("ai/a2a: artifact metadata must be JSON-compatible: %v", err))
	}
	var cloned map[string]any
	// Marshal output is always valid JSON.
	_ = json.Unmarshal(encoded, &cloned)
	return cloned
}

func responseParts(part ai.ResponsePart) []protocol.Part {
	switch value := part.(type) {
	case ai.TextPart:
		if value.Content != "" {
			return []protocol.Part{protocol.TextPart{Text: value.Content}}
		}
	case ai.FilePart:
		return []protocol.Part{protocol.FilePart{File: protocol.FileBytes{
			FileMeta: protocol.FileMeta{MimeType: value.Content.MediaType},
			Bytes:    base64.StdEncoding.EncodeToString(value.Content.Data),
		}}}
	}
	return nil
}

func deltaParts(delta ai.ResponsePartDelta) []protocol.Part {
	switch value := delta.(type) {
	case ai.TextPartDelta:
		if value.ContentDelta != "" {
			return []protocol.Part{protocol.TextPart{Text: value.ContentDelta}}
		}
	case ai.FilePartDelta:
		return responseParts(value.Part)
	}
	return nil
}
