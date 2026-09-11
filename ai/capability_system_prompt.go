package ai

import (
	"context"
	"slices"
)

// ReinjectSystemPrompt restores the agent's configured legacy system prompts
// when request history omitted them. Instructions do not need reinjection.
type ReinjectSystemPrompt struct {
	// ReplaceExisting removes history-provided system prompts before injecting
	// the agent's configuration. Use it when history is untrusted.
	ReplaceExisting bool
}

// CapabilityID identifies the single system-prompt reinjection policy for a run.
func (ReinjectSystemPrompt) CapabilityID() string { return "reinject_system_prompt" }

// CombineCapabilities merges repeated reinjection declarations.
func (ReinjectSystemPrompt) CombineCapabilities(capabilities []Capability) (Capability, error) {
	return MergeCapabilities(capabilities...)
}

// Setup implements Capability.
func (ReinjectSystemPrompt) Setup(*CapabilityRegistry) error { return nil }

// BeforeModelRequest prepares a request-only history snapshot. Persisted
// history remains unchanged and is filtered again on each model request.
func (capability ReinjectSystemPrompt) BeforeModelRequest(
	ctx context.Context, runInfo *RunInfo, request ModelRequestContext,
) (ModelRequestContext, error) {
	if runInfo == nil || runInfo.systemPrompts == nil {
		return request, nil
	}
	if !capability.ReplaceExisting && hasSystemPrompt(request.Messages) {
		return request, nil
	}
	messages := request.Messages
	if capability.ReplaceExisting {
		messages = stripSystemPrompts(messages)
	}
	parts, err := runInfo.systemPrompts(ctx)
	if err != nil {
		return ModelRequestContext{}, err
	}
	if len(parts) == 0 {
		request.Messages = messages
		return request, nil
	}
	// An agent request always contains the current user ModelRequest.
	index := slices.IndexFunc(messages, func(message ModelMessage) bool {
		_, ok := message.(ModelRequest)
		return ok
	})
	modelRequest := messages[index].(ModelRequest)
	prepended := make([]RequestPart, 0, len(parts)+len(modelRequest.Parts))
	for _, part := range parts {
		prepended = append(prepended, part)
	}
	modelRequest.Parts = append(prepended, modelRequest.Parts...)
	messages[index] = modelRequest
	request.Messages = messages
	return request, nil
}

func hasSystemPrompt(messages []ModelMessage) bool {
	for _, message := range messages {
		request, ok := message.(ModelRequest)
		if !ok {
			continue
		}
		for _, part := range request.Parts {
			if _, ok := part.(SystemPromptPart); ok {
				return true
			}
		}
	}
	return false
}

func stripSystemPrompts(messages []ModelMessage) []ModelMessage {
	filteredMessages := make([]ModelMessage, 0, len(messages))
	for _, message := range messages {
		request, ok := message.(ModelRequest)
		if !ok {
			filteredMessages = append(filteredMessages, message)
			continue
		}
		parts := make([]RequestPart, 0, len(request.Parts))
		for _, part := range request.Parts {
			if _, ok := part.(SystemPromptPart); !ok {
				parts = append(parts, part)
			}
		}
		if len(parts) == 0 {
			continue
		}
		request.Parts = parts
		filteredMessages = append(filteredMessages, request)
	}
	return filteredMessages
}
