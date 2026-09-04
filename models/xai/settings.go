package xai

import (
	"fmt"
	"maps"
	"slices"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/openai"
)

// ReasoningEffort selects a Grok reasoning level.
type ReasoningEffort string

const (
	// ReasoningEffortNone disables reasoning on models that support it.
	ReasoningEffortNone ReasoningEffort = "none"
	// ReasoningEffortLow requests low reasoning effort.
	ReasoningEffortLow ReasoningEffort = "low"
	// ReasoningEffortMedium requests medium reasoning effort.
	ReasoningEffortMedium ReasoningEffort = "medium"
	// ReasoningEffortHigh requests high reasoning effort.
	ReasoningEffortHigh ReasoningEffort = "high"
)

// Settings combines portable settings with xAI request options.
type Settings struct {
	// Common contains portable model settings.
	Common ai.ModelSettings
	// Logprobs requests output-token log probabilities.
	Logprobs *bool
	// TopLogprobs controls the alternatives returned for each output token.
	TopLogprobs *int
	// User identifies an end user for abuse monitoring.
	User string
	// StoreMessages enables xAI server-side conversation storage.
	StoreMessages *bool
	// PreviousResponseID continues a stored xAI response.
	PreviousResponseID string
	// IncludeEncryptedContent preserves opaque reasoning state for replay.
	IncludeEncryptedContent bool
	// IncludeCodeExecutionOutput requests code execution results.
	IncludeCodeExecutionOutput bool
	// IncludeWebSearchOutput requests web search results.
	IncludeWebSearchOutput bool
	// IncludeInlineCitations requests inline citations.
	IncludeInlineCitations bool
	// IncludeMCPOutput requests hosted MCP results.
	IncludeMCPOutput bool
	// IncludeXSearchOutput requests X search results.
	IncludeXSearchOutput bool
	// IncludeCollectionsSearchOutput requests managed collection search results.
	IncludeCollectionsSearchOutput bool
	// IncludeAttachmentSearchOutput requests uploaded attachment search results.
	IncludeAttachmentSearchOutput bool
	// ReasoningEffort overrides portable thinking for supported Grok models.
	ReasoningEffort ReasoningEffort
	// MaxTurns bounds xAI's server-side native-tool loop.
	MaxTurns int
	// AgentCount controls xAI multi-agent models.
	AgentCount int
}

// Build returns detached portable settings accepted by agents and direct requests.
func (settings Settings) Build() (ai.ModelSettings, error) {
	common := settings.Common.Clone()
	if settings.Logprobs != nil {
		value := *settings.Logprobs
		common.Logprobs = &value
	}
	if settings.TopLogprobs != nil {
		value := *settings.TopLogprobs
		common.TopLogprobs = &value
	}
	if common.TopLogprobs != nil {
		if *common.TopLogprobs < 0 || *common.TopLogprobs > 20 {
			return ai.ModelSettings{}, fmt.Errorf("xai: top logprobs must be between 0 and 20")
		}
		if common.Logprobs == nil || !*common.Logprobs {
			return ai.ModelSettings{}, fmt.Errorf("xai: top logprobs requires logprobs")
		}
	}
	if err := validateReasoningEffort(settings.ReasoningEffort); err != nil {
		return ai.ModelSettings{}, err
	}
	if settings.MaxTurns < 0 {
		return ai.ModelSettings{}, fmt.Errorf("xai: max turns must be non-negative")
	}
	if settings.AgentCount < 0 {
		return ai.ModelSettings{}, fmt.Errorf("xai: agent count must be non-negative")
	}
	extra := maps.Clone(common.ExtraBody)
	if extra == nil {
		extra = map[string]any{}
	}
	set := func(name string, value any) error {
		if _, exists := extra[name]; exists {
			return fmt.Errorf("xai: extra body field %q conflicts with typed settings", name)
		}
		extra[name] = value
		return nil
	}
	values := []struct {
		name  string
		value any
		set   bool
	}{
		{name: "user", value: settings.User, set: settings.User != ""},
		{name: "store_messages", value: pointerValue(settings.StoreMessages), set: settings.StoreMessages != nil},
		{name: "previous_response_id", value: settings.PreviousResponseID, set: settings.PreviousResponseID != ""},
		{name: "use_encrypted_content", value: true, set: settings.IncludeEncryptedContent},
		{name: "max_turns", value: settings.MaxTurns, set: settings.MaxTurns != 0},
		{name: "agent_count", value: settings.AgentCount, set: settings.AgentCount != 0},
		{name: explicitReasoningSetting, value: string(settings.ReasoningEffort), set: settings.ReasoningEffort != ""},
	}
	for _, value := range values {
		if value.set {
			if err := set(value.name, value.value); err != nil {
				return ai.ModelSettings{}, err
			}
		}
	}
	common.ExtraBody = extra
	includes := settings.responseIncludes()
	built, err := (openai.Settings{Common: common, ResponsesInclude: includes}).Build()
	if err != nil {
		return ai.ModelSettings{}, err
	}
	if settings.ReasoningEffort != "" {
		built.Thinking = &ai.ThinkingSettings{Level: thinkingLevelForEffort(settings.ReasoningEffort)}
	}
	return built, nil
}

func (settings Settings) responseIncludes() []string {
	values := []struct {
		name    string
		include bool
	}{
		{name: "code_interpreter_call.outputs", include: settings.IncludeCodeExecutionOutput},
		{name: "web_search_call.action.sources", include: settings.IncludeWebSearchOutput},
		{name: "inline_citations", include: settings.IncludeInlineCitations},
		{name: "mcp_call.outputs", include: settings.IncludeMCPOutput},
		{name: "x_search_call.outputs", include: settings.IncludeXSearchOutput},
		{name: "collections_search_call.outputs", include: settings.IncludeCollectionsSearchOutput},
		{name: "attachment_search_call.outputs", include: settings.IncludeAttachmentSearchOutput},
	}
	included := make([]string, 0, len(values))
	for _, value := range values {
		if value.include {
			included = append(included, value.name)
		}
	}
	return slices.Clone(included)
}

func thinkingLevelForEffort(effort ReasoningEffort) ai.ThinkingLevel {
	if effort == ReasoningEffortNone {
		return ai.ThinkingLevelDisabled
	}
	return ai.ThinkingLevel(effort)
}

func validateReasoningEffort(effort ReasoningEffort) error {
	switch effort {
	case "", ReasoningEffortNone, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh:
		return nil
	default:
		return fmt.Errorf("xai: invalid reasoning effort %q", effort)
	}
}

func pointerValue[T any](value *T) T {
	if value == nil {
		var zero T
		return zero
	}
	return *value
}
