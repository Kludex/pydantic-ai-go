package snowflake

import (
	"fmt"
	"maps"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

const reasoningSetting = "snowflake_reasoning"

// ReasoningEffort controls Snowflake Cortex reasoning for Claude models.
type ReasoningEffort string

const (
	// ReasoningEffortLow requests a small reasoning budget.
	ReasoningEffortLow ReasoningEffort = "low"
	// ReasoningEffortMedium requests a medium reasoning budget.
	ReasoningEffortMedium ReasoningEffort = "medium"
	// ReasoningEffortHigh requests a large reasoning budget.
	ReasoningEffortHigh ReasoningEffort = "high"
)

// Reasoning configures Snowflake Cortex reasoning for Claude models.
type Reasoning struct {
	// Effort lets Cortex select a reasoning token budget.
	Effort ReasoningEffort `json:"effort,omitempty"`
	// MaxTokens sets the reasoning token budget directly.
	MaxTokens int `json:"max_tokens,omitempty"`
}

// Settings combines portable settings with Snowflake-specific settings.
type Settings struct {
	// Common contains portable model settings.
	Common ai.ModelSettings
	// Reasoning configures Claude reasoning. It overrides portable thinking.
	Reasoning *Reasoning
}

// Build returns detached portable settings accepted by agents and direct requests.
func (settings Settings) Build() (ai.ModelSettings, error) {
	common := settings.Common.Clone()
	if settings.Reasoning == nil {
		return common, nil
	}
	if err := validateReasoning(*settings.Reasoning); err != nil {
		return ai.ModelSettings{}, err
	}
	extra := maps.Clone(common.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	if _, exists := extra[reasoningSetting]; exists {
		return ai.ModelSettings{}, fmt.Errorf("snowflake: setting field %q is reserved", reasoningSetting)
	}
	if _, exists := extra["reasoning"]; exists {
		return ai.ModelSettings{}, fmt.Errorf("snowflake: extra body field %q conflicts with typed settings", "reasoning")
	}
	extra[reasoningSetting] = *settings.Reasoning
	common.ExtraBody = extra
	return common, nil
}

func extractReasoning(settings ai.ModelSettings) (ai.ModelSettings, *Reasoning, error) {
	settings = settings.Clone()
	extra := maps.Clone(settings.ExtraBody)
	value, exists := extra[reasoningSetting]
	if !exists {
		return settings, nil, nil
	}
	delete(extra, reasoningSetting)
	var reasoning Reasoning
	switch value := value.(type) {
	case Reasoning:
		reasoning = value
	case *Reasoning:
		if value == nil {
			return ai.ModelSettings{}, nil, fmt.Errorf("snowflake: reasoning must not be nil")
		}
		reasoning = *value
	default:
		return ai.ModelSettings{}, nil, fmt.Errorf("snowflake: reasoning must use Reasoning")
	}
	if err := validateReasoning(reasoning); err != nil {
		return ai.ModelSettings{}, nil, err
	}
	if len(extra) == 0 {
		extra = nil
	}
	settings.ExtraBody = extra
	return settings, &reasoning, nil
}

func validateReasoning(reasoning Reasoning) error {
	switch reasoning.Effort {
	case "", ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh:
	default:
		return fmt.Errorf("snowflake: invalid reasoning effort %q", reasoning.Effort)
	}
	if reasoning.MaxTokens < 0 {
		return fmt.Errorf("snowflake: reasoning max tokens must be non-negative")
	}
	if reasoning.Effort != "" && reasoning.MaxTokens != 0 {
		return fmt.Errorf("snowflake: reasoning effort and max tokens are mutually exclusive")
	}
	if reasoning.Effort == "" && reasoning.MaxTokens == 0 {
		return fmt.Errorf("snowflake: reasoning requires effort or max tokens")
	}
	return nil
}
