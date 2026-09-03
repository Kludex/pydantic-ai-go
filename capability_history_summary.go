package ai

import (
	"context"
	"fmt"
	"time"
)

// HistorySummary is the provider-neutral result of summarizing old turns.
type HistorySummary struct {
	// Content is the readable replacement for summarized turns.
	Content string
	// Usage attributes summarizer model work to the outer run.
	Usage Usage
}

// HistorySummaryFunc summarizes a detached sequence of complete old turns.
// Implementations must be safe for concurrent agent runs.
type HistorySummaryFunc func(
	ctx context.Context, runInfo *RunInfo, messages []ModelMessage,
) (HistorySummary, error)

// HistorySummarizer replaces complete old turns with one assistant text
// response. The replacement is durable, so the callback runs at most once for
// the same old turns during a run.
//
// MinimumRecentTurns defaults to one. The current run's complete turn is always
// retained. Usage returned by Summarize is attributed to the outer run and
// checked before its primary model request.
type HistorySummarizer struct {
	// MinimumRecentTurns protects this many newest turns. Zero defaults to one.
	MinimumRecentTurns int
	// Summarize produces the replacement text and reports side-request usage.
	Summarize HistorySummaryFunc
}

// Setup validates the summarizer.
func (summarizer HistorySummarizer) Setup(*CapabilityRegistry) error {
	if summarizer.MinimumRecentTurns < 0 {
		return fmt.Errorf("ai: minimum recent summary turns must be non-negative, got %d", summarizer.MinimumRecentTurns)
	}
	if summarizer.Summarize == nil {
		return fmt.Errorf("ai: history summary function must not be nil")
	}
	return nil
}

// CapabilityOrdering places summarization in the innermost tier of its capability scope.
func (HistorySummarizer) CapabilityOrdering() CapabilityOrdering {
	return CapabilityOrdering{Position: CapabilityInnermost}
}

// BeforeModelRequest summarizes old turns in a detached request and explicitly
// replaces durable history with the compacted result.
func (summarizer HistorySummarizer) BeforeModelRequest(
	ctx context.Context, runInfo *RunInfo, request ModelRequestContext,
) (ModelRequestContext, error) {
	starts := historyUserTurnStarts(request.Messages)
	split := protectedHistoryStart(
		request.Messages, starts, runInfo.RunID, summarizer.MinimumRecentTurns,
	)
	if split == 0 {
		return request, nil
	}
	old := cloneModelMessages(request.Messages[:split])
	summary, err := summarizer.Summarize(ctx, runInfo, old)
	if err != nil {
		return request, fmt.Errorf("ai: summarize history: %w", err)
	}
	if summary.Content == "" {
		return request, fmt.Errorf("ai: history summary must not be empty")
	}
	compacted := make([]ModelMessage, 0, len(request.Messages)-split+1)
	compacted = append(compacted, ModelResponse{
		Parts:          []ResponsePart{TextPart{Content: summary.Content}},
		Timestamp:      time.Now().UTC(),
		FinishReason:   FinishReasonStop,
		RunID:          runInfo.RunID,
		ConversationID: runInfo.ConversationID,
		State:          ModelResponseStateComplete,
	})
	compacted = append(compacted, cloneModelMessages(request.Messages[split:])...)
	request.Messages = compacted
	request.ReplaceHistory = true
	request.AdditionalUsage.Add(summary.Usage)
	return request, nil
}
