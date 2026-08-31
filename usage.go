package ai

import (
	"encoding/json"
	"fmt"
)

// Usage counts model requests and tokens across a run.
// The zero value is an empty count.
type Usage struct {
	Requests                 int            `json:"requests,omitempty"`
	ToolCalls                int            `json:"tool_calls,omitempty"`
	InputTokens              int            `json:"input_tokens,omitempty"`
	CacheWriteTokens         int            `json:"cache_write_tokens,omitempty"`
	CacheReadTokens          int            `json:"cache_read_tokens,omitempty"`
	InputAudioTokens         int            `json:"input_audio_tokens,omitempty"`
	CacheAudioReadTokens     int            `json:"cache_audio_read_tokens,omitempty"`
	OutputTokens             int            `json:"output_tokens,omitempty"`
	OutputAudioTokens        int            `json:"output_audio_tokens,omitempty"`
	ReasoningTokens          int            `json:"reasoning_tokens,omitempty"`
	AcceptedPredictionTokens int            `json:"accepted_prediction_tokens,omitempty"`
	RejectedPredictionTokens int            `json:"rejected_prediction_tokens,omitempty"`
	Details                  map[string]int `json:"details,omitempty"`
	CostUSD                  *float64       `json:"cost,omitempty"`
}

// UnmarshalJSON accepts the upstream cost field and the legacy Go cost_usd alias.
func (u *Usage) UnmarshalJSON(data []byte) error {
	type usageAlias Usage
	var decoded usageAlias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	if decoded.CostUSD == nil {
		var legacy struct {
			CostUSD *float64 `json:"cost_usd"`
		}
		if err := json.Unmarshal(data, &legacy); err != nil {
			return err
		}
		decoded.CostUSD = legacy.CostUSD
	}
	*u = Usage(decoded)
	return nil
}

// Clone returns a detached usage value.
func (u Usage) Clone() Usage {
	if u.Details != nil {
		details := make(map[string]int, len(u.Details))
		for key, value := range u.Details {
			details[key] = value
		}
		u.Details = details
	}
	if u.CostUSD != nil {
		cost := *u.CostUSD
		u.CostUSD = &cost
	}
	return u
}

// IsZero reports whether usage contains no counts, details, or known cost.
func (u Usage) IsZero() bool {
	return u.Requests == 0 && u.ToolCalls == 0 && u.InputTokens == 0 && u.CacheWriteTokens == 0 &&
		u.CacheReadTokens == 0 && u.InputAudioTokens == 0 && u.CacheAudioReadTokens == 0 &&
		u.OutputTokens == 0 && u.OutputAudioTokens == 0 && u.ReasoningTokens == 0 &&
		u.AcceptedPredictionTokens == 0 && u.RejectedPredictionTokens == 0 && len(u.Details) == 0 && u.CostUSD == nil
}

// TotalTokens returns input plus output tokens.
func (u Usage) TotalTokens() int { return u.InputTokens + u.OutputTokens }

// CacheHitRatio returns the fraction of input tokens read from a provider cache.
func (u Usage) CacheHitRatio() float64 {
	if u.InputTokens == 0 {
		return 0
	}
	return float64(u.CacheReadTokens) / float64(u.InputTokens)
}

// Add accumulates another usage count into u.
func (u *Usage) Add(other Usage) {
	u.Requests += other.Requests
	u.ToolCalls += other.ToolCalls
	u.InputTokens += other.InputTokens
	u.CacheWriteTokens += other.CacheWriteTokens
	u.CacheReadTokens += other.CacheReadTokens
	u.InputAudioTokens += other.InputAudioTokens
	u.CacheAudioReadTokens += other.CacheAudioReadTokens
	u.OutputTokens += other.OutputTokens
	u.OutputAudioTokens += other.OutputAudioTokens
	u.ReasoningTokens += other.ReasoningTokens
	u.AcceptedPredictionTokens += other.AcceptedPredictionTokens
	u.RejectedPredictionTokens += other.RejectedPredictionTokens
	if len(other.Details) > 0 {
		if u.Details == nil {
			u.Details = make(map[string]int, len(other.Details))
		}
		for key, value := range other.Details {
			u.Details[key] += value
		}
	}
	if other.CostUSD != nil {
		cost := *other.CostUSD
		if u.CostUSD != nil {
			cost += *u.CostUSD
		}
		u.CostUSD = &cost
	}
}

// UsageLimits bounds a run. The zero value means unlimited.
type UsageLimits struct {
	RequestLimit     int
	InputTokenLimit  int
	OutputTokenLimit int
	TotalTokenLimit  int
	ToolCallLimit    *int
	CostLimitUSD     *float64
}

func (l UsageLimits) check(u Usage) error {
	switch {
	case l.RequestLimit > 0 && u.Requests > l.RequestLimit:
		return usageLimitError("request", u.Requests, l.RequestLimit)
	case l.InputTokenLimit > 0 && u.InputTokens > l.InputTokenLimit:
		return usageLimitError("input token", u.InputTokens, l.InputTokenLimit)
	case l.OutputTokenLimit > 0 && u.OutputTokens > l.OutputTokenLimit:
		return usageLimitError("output token", u.OutputTokens, l.OutputTokenLimit)
	case l.TotalTokenLimit > 0 && u.TotalTokens() > l.TotalTokenLimit:
		return usageLimitError("total token", u.TotalTokens(), l.TotalTokenLimit)
	case l.CostLimitUSD != nil && u.CostUSD != nil && *u.CostUSD > *l.CostLimitUSD:
		return fmt.Errorf("%w: cost $%g exceeds limit $%g", ErrUsageLimitExceeded, *u.CostUSD, *l.CostLimitUSD)
	}
	return nil
}

func usageLimitError(kind string, used, limit int) error {
	return fmt.Errorf("%w: %s count %d exceeds limit %d", ErrUsageLimitExceeded, kind, used, limit)
}
