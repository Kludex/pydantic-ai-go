package openai

import (
	"context"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go"
)

// CompactionTrigger decides whether the current history should be compacted.
type CompactionTrigger func(messages []ai.ModelMessage) bool

// Compaction configures stateful context management or stateless Responses compaction.
type Compaction struct {
	stateless             *bool
	tokenThreshold        *int
	messageCountThreshold *int
	trigger               CompactionTrigger
	triggerConfigured     bool
}

// CompactionOption configures Compaction.
type CompactionOption func(*Compaction)

// NewCompaction creates stateful compaction unless a stateless trigger option is present.
func NewCompaction(options ...CompactionOption) *Compaction {
	compaction := &Compaction{}
	for _, option := range options {
		option(compaction)
	}
	return compaction
}

// WithCompactionTokenThreshold sets the input-token threshold for stateful compaction.
func WithCompactionTokenThreshold(threshold int) CompactionOption {
	return func(compaction *Compaction) { compaction.tokenThreshold = &threshold }
}

// WithCompactionMessageCountThreshold invokes stateless compaction when message count exceeds threshold.
func WithCompactionMessageCountThreshold(threshold int) CompactionOption {
	return func(compaction *Compaction) { compaction.messageCountThreshold = &threshold }
}

// WithCompactionTrigger invokes stateless compaction when trigger returns true.
func WithCompactionTrigger(trigger CompactionTrigger) CompactionOption {
	return func(compaction *Compaction) {
		compaction.trigger = trigger
		compaction.triggerConfigured = true
	}
}

// WithStatelessCompaction explicitly selects the /responses/compact endpoint.
func WithStatelessCompaction() CompactionOption {
	return func(compaction *Compaction) {
		stateless := true
		compaction.stateless = &stateless
	}
}

// WithStatefulCompaction explicitly selects server-managed context compaction.
func WithStatefulCompaction() CompactionOption {
	return func(compaction *Compaction) {
		stateless := false
		compaction.stateless = &stateless
	}
}

// Setup validates the compaction configuration.
func (compaction *Compaction) Setup(*ai.CapabilityRegistry) error {
	if compaction == nil {
		return fmt.Errorf("openai: compaction capability must not be nil")
	}
	if compaction.tokenThreshold != nil && *compaction.tokenThreshold < 0 {
		return fmt.Errorf("openai: compaction token threshold must be non-negative, got %d", *compaction.tokenThreshold)
	}
	if compaction.messageCountThreshold != nil && *compaction.messageCountThreshold < 0 {
		return fmt.Errorf(
			"openai: compaction message count threshold must be non-negative, got %d",
			*compaction.messageCountThreshold,
		)
	}
	if compaction.triggerConfigured && compaction.trigger == nil {
		return fmt.Errorf("openai: compaction trigger must not be nil")
	}
	stateless := compaction.isStateless()
	if stateless && compaction.tokenThreshold != nil {
		return fmt.Errorf("openai: token threshold is only valid for stateful compaction")
	}
	if stateless && compaction.messageCountThreshold == nil && compaction.trigger == nil {
		return fmt.Errorf("openai: stateless compaction requires a message count threshold or trigger")
	}
	if !stateless && (compaction.messageCountThreshold != nil || compaction.trigger != nil) {
		return fmt.Errorf("openai: message count threshold and trigger are only valid for stateless compaction")
	}
	return nil
}

// CapabilityOrdering keeps compaction inside model-changing request hooks.
func (*Compaction) CapabilityOrdering() ai.CapabilityOrdering {
	return ai.CapabilityOrdering{Position: ai.CapabilityInnermost}
}

// BeforeModelRequest validates the model and performs triggered stateless compaction.
func (compaction *Compaction) BeforeModelRequest(
	ctx context.Context, runInfo *ai.RunInfo, request ai.ModelRequestContext,
) (ai.ModelRequestContext, error) {
	_, ok := ai.UnwrapModel(request.Model).(*ResponsesModel)
	if !ok {
		return request, fmt.Errorf("openai: compaction requires ResponsesModel, got %T", request.Model)
	}
	if !compaction.isStateless() || !compaction.shouldCompact(request.Messages) || len(request.Messages) < 2 {
		return request, nil
	}
	compacted, err := ai.CompactModelMessages(ctx, request.Model, request.Messages[:len(request.Messages)-1], request.Params)
	if err != nil {
		return request, err
	}
	compacted.RunID = runInfo.RunID
	compacted.ConversationID = runInfo.ConversationID
	request.Messages = []ai.ModelMessage{*compacted, request.Messages[len(request.Messages)-1]}
	request.ReplaceHistory = true
	request.AdditionalUsage.Add(compacted.Usage)
	return request.Clone(), nil
}

// ModelSettings contributes stateful Responses context management without replacing an explicit setting.
func (compaction *Compaction) ModelSettings(
	_ context.Context, _ *ai.RunInfo, current ai.ModelSettings,
) (ai.ModelSettings, error) {
	if compaction.isStateless() {
		return ai.ModelSettings{}, nil
	}
	if _, configured := current.ExtraBody["context_management"]; configured {
		return ai.ModelSettings{}, nil
	}
	edit := map[string]any{"type": "compaction"}
	if compaction.tokenThreshold != nil {
		edit["compact_threshold"] = *compaction.tokenThreshold
	}
	return ai.ModelSettings{ExtraBody: map[string]any{
		"context_management": []any{edit},
	}}, nil
}

func (compaction *Compaction) isStateless() bool {
	if compaction.stateless != nil {
		return *compaction.stateless
	}
	return compaction.messageCountThreshold != nil || compaction.triggerConfigured
}

func (compaction *Compaction) shouldCompact(messages []ai.ModelMessage) bool {
	if compaction.trigger != nil {
		return compaction.trigger(ai.ModelRequestContext{Messages: messages}.Clone().Messages)
	}
	return compaction.messageCountThreshold != nil && len(messages) > *compaction.messageCountThreshold
}
