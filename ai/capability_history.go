package ai

import (
	"context"
	"fmt"
)

// HistoryProcessor transforms the message view sent to each model request.
// The callback receives a detached history. Its result does not replace durable
// run history unless another hook explicitly sets ModelRequestContext.ReplaceHistory.
type HistoryProcessor func(
	ctx context.Context, runInfo *RunInfo, messages []ModelMessage,
) ([]ModelMessage, error)

// Setup validates the processor.
func (processor HistoryProcessor) Setup(*CapabilityRegistry) error {
	if processor == nil {
		return fmt.Errorf("ai: history processor must not be nil")
	}
	return nil
}

// BeforeModelRequest applies the processor to a detached message snapshot.
func (processor HistoryProcessor) BeforeModelRequest(
	ctx context.Context, runInfo *RunInfo, request ModelRequestContext,
) (ModelRequestContext, error) {
	messages, err := processor(ctx, runInfo, cloneModelMessages(request.Messages))
	if err != nil {
		return request, err
	}
	request.Messages = cloneModelMessages(messages)
	return request, nil
}
