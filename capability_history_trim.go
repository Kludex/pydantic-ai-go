package ai

import (
	"context"
	"errors"
	"fmt"
)

// ErrHistoryTokenLimitExceeded reports that the protected recent history is
// larger than a TokenHistoryTrimmer limit.
var ErrHistoryTokenLimitExceeded = errors.New("ai: protected history exceeds the input token limit")

// HistoryTokenLimitError describes a history that cannot be trimmed enough.
type HistoryTokenLimitError struct {
	Limit int
	Usage Usage
}

// Error implements error.
func (err *HistoryTokenLimitError) Error() string {
	return fmt.Sprintf(
		"%s: counted %d input tokens with a limit of %d",
		ErrHistoryTokenLimitExceeded, err.Usage.InputTokens, err.Limit,
	)
}

// Unwrap supports errors.Is with ErrHistoryTokenLimitExceeded.
func (*HistoryTokenLimitError) Unwrap() error { return ErrHistoryTokenLimitExceeded }

// TokenHistoryTrimmer removes the oldest complete user turns until a request
// fits MaxInputTokens. It never changes durable run history.
//
// MinimumRecentTurns defaults to one. The current run's complete turn is always
// protected. Token counts include instructions, tools, and output schemas.
type TokenHistoryTrimmer struct {
	// MaxInputTokens is the required upper bound for the prospective request.
	MaxInputTokens int
	// MinimumRecentTurns protects this many newest turns. Zero defaults to one.
	MinimumRecentTurns int
}

// Setup validates the trimmer.
func (trimmer TokenHistoryTrimmer) Setup(*CapabilityRegistry) error {
	if trimmer.MaxInputTokens <= 0 {
		return fmt.Errorf("ai: history input token limit must be positive, got %d", trimmer.MaxInputTokens)
	}
	if trimmer.MinimumRecentTurns < 0 {
		return fmt.Errorf("ai: minimum recent history turns must be non-negative, got %d", trimmer.MinimumRecentTurns)
	}
	return nil
}

// CapabilityOrdering places trimming in the innermost tier of its capability scope.
func (TokenHistoryTrimmer) CapabilityOrdering() CapabilityOrdering {
	return CapabilityOrdering{Position: CapabilityInnermost}
}

// BeforeModelRequest trims a detached request snapshot using the selected
// model's token-counting endpoint.
func (trimmer TokenHistoryTrimmer) BeforeModelRequest(
	ctx context.Context, runInfo *RunInfo, request ModelRequestContext,
) (ModelRequestContext, error) {
	counted, err := CountModelTokens(ctx, request.Model, request.Messages, request.Params)
	if err != nil {
		return request, fmt.Errorf("ai: count history tokens: %w", err)
	}
	if counted.InputTokens <= trimmer.MaxInputTokens {
		return request, nil
	}

	starts := historyUserTurnStarts(request.Messages)
	maximumStart := protectedHistoryStart(request.Messages, starts, runInfo.RunID, trimmer.MinimumRecentTurns)
	maximumIndex := 0
	for maximumIndex+1 < len(starts) && starts[maximumIndex+1] <= maximumStart {
		maximumIndex++
	}
	if maximumIndex == 0 {
		return request, newHistoryTokenLimitError(trimmer.MaxInputTokens, counted)
	}

	countAt := func(index int) (Usage, error) {
		usage, countErr := CountModelTokens(ctx, request.Model, request.Messages[starts[index]:], request.Params)
		if countErr != nil {
			return Usage{}, fmt.Errorf("ai: count trimmed history tokens: %w", countErr)
		}
		return usage, nil
	}

	minimum, err := countAt(maximumIndex)
	if err != nil {
		return request, err
	}
	if minimum.InputTokens > trimmer.MaxInputTokens {
		return request, newHistoryTokenLimitError(trimmer.MaxInputTokens, minimum)
	}

	low, high := 1, maximumIndex
	for low < high {
		middle := low + (high-low)/2
		usage, countErr := countAt(middle)
		if countErr != nil {
			return request, countErr
		}
		if usage.InputTokens <= trimmer.MaxInputTokens {
			high = middle
		} else {
			low = middle + 1
		}
	}
	request.Messages = cloneModelMessages(request.Messages[starts[low]:])
	return request, nil
}

func newHistoryTokenLimitError(limit int, usage Usage) error {
	return &HistoryTokenLimitError{Limit: limit, Usage: usage.Clone()}
}

func historyUserTurnStarts(messages []ModelMessage) []int {
	if len(messages) == 0 {
		return nil
	}
	starts := []int{0}
	for index := 1; index < len(messages); index++ {
		request, ok := messages[index].(ModelRequest)
		if !ok {
			continue
		}
		hasUserPrompt := false
		hasToolResult := false
		for _, part := range request.Parts {
			if _, ok := part.(UserPromptPart); ok {
				hasUserPrompt = true
			}
			if _, _, ok := toolResultIdentity(part); ok {
				hasToolResult = true
			}
		}
		if hasUserPrompt && !hasToolResult {
			starts = append(starts, index)
		}
	}
	return starts
}

func protectedHistoryStart(messages []ModelMessage, starts []int, runID string, minimumRecentTurns int) int {
	if len(starts) == 0 {
		return 0
	}
	if minimumRecentTurns == 0 {
		minimumRecentTurns = 1
	}
	recentIndex := len(starts) - minimumRecentTurns
	if recentIndex < 0 {
		recentIndex = 0
	}
	protected := starts[recentIndex]
	for index, message := range messages {
		if modelMessageRunID(message) != runID {
			continue
		}
		for turnIndex := len(starts) - 1; turnIndex >= 0; turnIndex-- {
			if starts[turnIndex] <= index {
				if starts[turnIndex] < protected {
					protected = starts[turnIndex]
				}
				break
			}
		}
		break
	}
	return protected
}

func modelMessageRunID(message ModelMessage) string {
	if request, ok := message.(ModelRequest); ok {
		return request.RunID
	}
	return message.(ModelResponse).RunID
}
