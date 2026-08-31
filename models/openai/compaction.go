package openai

import (
	"context"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go"
)

// Compaction configures stateful server-side compaction for Responses models.
type Compaction struct {
	tokenThreshold *int
}

// CompactionOption configures Compaction.
type CompactionOption func(*Compaction)

// NewCompaction creates stateful compaction with OpenAI's managed threshold.
func NewCompaction(options ...CompactionOption) *Compaction {
	compaction := &Compaction{}
	for _, option := range options {
		option(compaction)
	}
	return compaction
}

// WithCompactionTokenThreshold sets the input-token threshold for compaction.
func WithCompactionTokenThreshold(threshold int) CompactionOption {
	return func(compaction *Compaction) { compaction.tokenThreshold = &threshold }
}

// Setup validates the compaction configuration.
func (compaction *Compaction) Setup(*ai.CapabilityRegistry) error {
	if compaction == nil {
		return fmt.Errorf("openai: compaction capability must not be nil")
	}
	if compaction.tokenThreshold != nil && *compaction.tokenThreshold < 0 {
		return fmt.Errorf("openai: compaction token threshold must be non-negative, got %d", *compaction.tokenThreshold)
	}
	return nil
}

// BeforeModelRequest rejects compaction on models that do not use the Responses API.
func (compaction *Compaction) BeforeModelRequest(
	_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
) (ai.ModelRequestContext, error) {
	if _, ok := request.Model.(*ResponsesModel); !ok {
		return request, fmt.Errorf("openai: compaction requires ResponsesModel, got %T", request.Model)
	}
	return request, nil
}

// ModelSettings contributes Responses context management without replacing an explicit setting.
func (compaction *Compaction) ModelSettings(
	_ context.Context, _ *ai.RunInfo, current ai.ModelSettings,
) (ai.ModelSettings, error) {
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
