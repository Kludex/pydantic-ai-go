package ai

import "fmt"

// Usage counts model requests and tokens across a run.
// The zero value is an empty count.
type Usage struct {
	Requests     int `json:"requests,omitempty"`
	InputTokens  int `json:"input_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`
}

// TotalTokens returns input plus output tokens.
func (u Usage) TotalTokens() int { return u.InputTokens + u.OutputTokens }

// Add accumulates another usage count into u.
func (u *Usage) Add(other Usage) {
	u.Requests += other.Requests
	u.InputTokens += other.InputTokens
	u.OutputTokens += other.OutputTokens
}

// UsageLimits bounds a run. The zero value means unlimited.
type UsageLimits struct {
	RequestLimit     int
	InputTokenLimit  int
	OutputTokenLimit int
	TotalTokenLimit  int
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
	}
	return nil
}

func usageLimitError(kind string, used, limit int) error {
	return fmt.Errorf("%w: %s count %d exceeds limit %d", ErrUsageLimitExceeded, kind, used, limit)
}
