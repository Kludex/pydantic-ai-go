package anthropic

import (
	"context"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go"
)

const defaultCompactionTokenThreshold = 150_000

// Compaction configures Anthropic server-side context compaction.
type Compaction struct {
	tokenThreshold       int
	instructions         *string
	pauseAfterCompaction bool
}

// CompactionOption configures Compaction.
type CompactionOption func(*Compaction)

// NewCompaction creates compaction with a 150,000 input-token threshold.
func NewCompaction(options ...CompactionOption) *Compaction {
	compaction := &Compaction{tokenThreshold: defaultCompactionTokenThreshold}
	for _, option := range options {
		option(compaction)
	}
	return compaction
}

// WithCompactionTokenThreshold sets the input-token threshold. Anthropic requires at least 50,000.
func WithCompactionTokenThreshold(threshold int) CompactionOption {
	return func(compaction *Compaction) { compaction.tokenThreshold = threshold }
}

// WithCompactionInstructions sets instructions used to summarize the compacted window.
func WithCompactionInstructions(instructions string) CompactionOption {
	return func(compaction *Compaction) { compaction.instructions = &instructions }
}

// WithPauseAfterCompaction stops the response after Anthropic emits the compaction block.
func WithPauseAfterCompaction() CompactionOption {
	return func(compaction *Compaction) { compaction.pauseAfterCompaction = true }
}

// Setup validates the compaction configuration.
func (compaction *Compaction) Setup(*ai.CapabilityRegistry) error {
	if compaction == nil {
		return fmt.Errorf("anthropic: compaction capability must not be nil")
	}
	if compaction.tokenThreshold < 50_000 {
		return fmt.Errorf(
			"anthropic: compaction token threshold must be at least 50000, got %d", compaction.tokenThreshold,
		)
	}
	return nil
}

// BeforeModelRequest rejects compaction on non-Anthropic models.
func (compaction *Compaction) BeforeModelRequest(
	_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
) (ai.ModelRequestContext, error) {
	if _, ok := request.Model.(*Model); !ok {
		return request, fmt.Errorf("anthropic: compaction requires Model, got %T", request.Model)
	}
	return request, nil
}

// ModelSettings appends a compaction edit while preserving existing context management.
func (compaction *Compaction) ModelSettings(
	_ context.Context, _ *ai.RunInfo, current ai.ModelSettings,
) (ai.ModelSettings, error) {
	contextManagement := map[string]any{}
	if existing, configured := current.Clone().ExtraBody["context_management"]; configured {
		var ok bool
		contextManagement, ok = existing.(map[string]any)
		if !ok {
			return ai.ModelSettings{}, fmt.Errorf("anthropic: context_management must be an object")
		}
	}
	var edits []any
	if existing, configured := contextManagement["edits"]; configured {
		var ok bool
		edits, ok = existing.([]any)
		if !ok {
			return ai.ModelSettings{}, fmt.Errorf("anthropic: context_management.edits must be an array")
		}
	}
	edit := map[string]any{
		"type": "compact_20260112",
		"trigger": map[string]any{
			"type":  "input_tokens",
			"value": compaction.tokenThreshold,
		},
	}
	if compaction.instructions != nil {
		edit["instructions"] = *compaction.instructions
	}
	if compaction.pauseAfterCompaction {
		edit["pause_after_compaction"] = true
	}
	contextManagement["edits"] = append(edits, edit)
	return ai.ModelSettings{ExtraBody: map[string]any{"context_management": contextManagement}}, nil
}

func hasCompactionEdit(extraBody map[string]any) bool {
	contextManagement, ok := extraBody["context_management"].(map[string]any)
	if !ok {
		return false
	}
	edits, ok := contextManagement["edits"].([]any)
	if !ok {
		return false
	}
	for _, value := range edits {
		edit, ok := value.(map[string]any)
		if ok && edit["type"] == "compact_20260112" {
			return true
		}
	}
	return false
}
