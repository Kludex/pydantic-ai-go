package ai

import (
	"context"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"time"
)

// Model is the provider contract. A provider package implements Model;
// the agent calls Request for each response segment. Suspended responses may
// require multiple segments within one logical loop iteration.
//
// Generics never cross this boundary: providers deal only in messages
// and schemas, which keeps adding a provider trivial.
type Model interface {
	Request(ctx context.Context, msgs []ModelMessage, params ModelRequestParams) (*ModelResponse, error)

	// Name identifies the model (e.g. "gpt-5") for tracing and results.
	Name() string
}

// ToolSearchStrategyModel is implemented by models that support required
// named provider-managed tool-search strategies.
type ToolSearchStrategyModel interface {
	SupportsToolSearchStrategy(strategy ToolSearchStrategy) bool
}

// NativeToolSearchHistoryModel is implemented by models that can replay
// provider-native tool-search parts from their own provider.
type NativeToolSearchHistoryModel interface {
	NativeToolSearchProvider() string
}

// ModelCloseFunc releases resources acquired for one agent run.
type ModelCloseFunc func(ctx context.Context) error

// ModelOpener is an optional per-run lifecycle for models that acquire
// resources. OpenModel runs once per distinct selected model. Close functions
// run in reverse selection order after toolset resources are no longer used.
type ModelOpener interface {
	OpenModel(ctx context.Context) (ModelCloseFunc, error)
}

// ModelDefaultSettings is implemented by models with request-setting defaults.
// Agent, capability, and run settings override these values field by field.
type ModelDefaultSettings interface {
	DefaultModelSettings() ModelSettings
}

func modelName(model Model) string {
	if modelIsNil(model) {
		return ""
	}
	return model.Name()
}

func sameModelInstance(first, second Model) bool {
	if modelIsNil(first) || modelIsNil(second) || reflect.TypeOf(first) != reflect.TypeOf(second) {
		return false
	}
	if reflect.TypeOf(first).Comparable() {
		return first == second
	}
	return false
}

func modelIsNil(model Model) bool {
	if model == nil {
		return true
	}
	value := reflect.ValueOf(model)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}

// ModelSelection selects either a concrete model or an ID resolved by a
// registered ModelIDResolverFunc. The zero value makes no selection.
type ModelSelection struct {
	Model Model
	ID    string
}

// ModelSelectionContext is a read-only snapshot passed to a model selector.
// Step starts at 1. Model is the lower-precedence model on the first step and
// the model used by the previous request thereafter.
type ModelSelectionContext[Deps any] struct {
	Deps     Deps
	Model    Model
	ModelID  string
	Step     int
	Messages []ModelMessage
	Usage    Usage
}

// ModelSelectorFunc selects a model before one logical model request.
type ModelSelectorFunc[Deps any] func(
	ctx context.Context, selection ModelSelectionContext[Deps],
) (ModelSelection, error)

// ModelResolutionContext is passed when resolving an application model ID.
type ModelResolutionContext[Deps any] struct {
	Deps Deps
}

// ModelIDResolverFunc resolves an application model ID. Return nil, nil to
// let the next resolver try. Resolved IDs are cached for one agent run.
type ModelIDResolverFunc[Deps any] func(
	ctx context.Context, resolution ModelResolutionContext[Deps], modelID string,
) (Model, error)

// ModelRequestParams carries everything a provider needs beyond the messages.
type ModelRequestParams struct {
	// Instructions is the joined instruction text for providers that do not
	// need origin metadata.
	Instructions string
	// InstructionParts preserves static and dynamic instruction boundaries.
	// Providers may use these boundaries for prompt caching.
	InstructionParts []InstructionPart
	Tools            []ToolDefinition
	// DeferredTools contains prepared definitions hidden by the agent. Providers
	// with native deferral may advertise them without making them executable.
	DeferredTools []ToolDefinition
	// OutputTool, when non-nil, is the tool the model must call to
	// produce the final structured output.
	OutputTool *ToolDefinition
	// OutputSchema validates structured text output. Providers enforce it
	// natively unless OutputMode is OutputModePrompted.
	OutputSchema map[string]any
	// OutputMode identifies how structured output is requested.
	OutputMode OutputMode
	// OutputPrompt contains provider-neutral prompted-output instructions.
	OutputPrompt string
	// AllowText reports whether plain text is an acceptable final output.
	AllowText bool
	Settings  ModelSettings
}

// InstructionPart is one model instruction block.
type InstructionPart struct {
	Content string
	Dynamic bool
}

// ThinkingLevel configures provider reasoning with a portable effort level.
type ThinkingLevel string

const (
	ThinkingLevelDisabled ThinkingLevel = "disabled"
	ThinkingLevelEnabled  ThinkingLevel = "enabled"
	ThinkingLevelMinimal  ThinkingLevel = "minimal"
	ThinkingLevelLow      ThinkingLevel = "low"
	ThinkingLevelMedium   ThinkingLevel = "medium"
	ThinkingLevelHigh     ThinkingLevel = "high"
	ThinkingLevelXHigh    ThinkingLevel = "xhigh"
)

// ThinkingSettings configures reasoning generation. Level maps to the closest
// provider effort. TokenBudget and IncludeThoughts override that provider's
// defaults when supported.
type ThinkingSettings struct {
	Level           ThinkingLevel
	TokenBudget     *int
	IncludeThoughts *bool
}

// ServiceTier selects a provider's latency and capacity class.
type ServiceTier string

const (
	ServiceTierAuto     ServiceTier = "auto"
	ServiceTierDefault  ServiceTier = "default"
	ServiceTierFlex     ServiceTier = "flex"
	ServiceTierPriority ServiceTier = "priority"
)

// ModelSettings tunes a model request. The zero value uses provider defaults.
type ModelSettings struct {
	MaxTokens         int
	RequestTimeout    time.Duration
	Temperature       *float64
	TopP              *float64
	Seed              *int
	PresencePenalty   *float64
	FrequencyPenalty  *float64
	LogitBias         map[string]int
	Logprobs          *bool
	TopLogprobs       *int
	ServiceTier       ServiceTier
	ExtraHeaders      map[string]string
	ExtraBody         map[string]any
	StopSequences     []string
	ParallelToolCalls *bool
	Thinking          *ThinkingSettings
}

// Clone returns settings detached from pointer and slice fields.
func (s ModelSettings) Clone() ModelSettings {
	s.Temperature = clonePointer(s.Temperature)
	s.TopP = clonePointer(s.TopP)
	s.Seed = clonePointer(s.Seed)
	s.PresencePenalty = clonePointer(s.PresencePenalty)
	s.FrequencyPenalty = clonePointer(s.FrequencyPenalty)
	s.LogitBias = maps.Clone(s.LogitBias)
	s.Logprobs = clonePointer(s.Logprobs)
	s.TopLogprobs = clonePointer(s.TopLogprobs)
	s.ExtraHeaders = maps.Clone(s.ExtraHeaders)
	s.ExtraBody = cloneSchemaMap(s.ExtraBody)
	s.StopSequences = slices.Clone(s.StopSequences)
	s.ParallelToolCalls = clonePointer(s.ParallelToolCalls)
	if s.Thinking != nil {
		thinking := *s.Thinking
		thinking.TokenBudget = clonePointer(thinking.TokenBudget)
		thinking.IncludeThoughts = clonePointer(thinking.IncludeThoughts)
		s.Thinking = &thinking
	}
	return s
}

func validateModelSettings(settings ModelSettings) error {
	if settings.RequestTimeout < 0 {
		return fmt.Errorf("ai: request timeout must be non-negative, got %s", settings.RequestTimeout)
	}
	if settings.TopLogprobs != nil {
		if *settings.TopLogprobs < 0 {
			return fmt.Errorf("ai: top logprobs must be non-negative, got %d", *settings.TopLogprobs)
		}
		if settings.Logprobs == nil || !*settings.Logprobs {
			return fmt.Errorf("ai: top logprobs requires logprobs to be enabled")
		}
	}
	switch settings.ServiceTier {
	case "", ServiceTierAuto, ServiceTierDefault, ServiceTierFlex, ServiceTierPriority:
	default:
		return fmt.Errorf("ai: invalid service tier %q", settings.ServiceTier)
	}
	return validateThinkingSettings(settings.Thinking)
}

func validateThinkingSettings(settings *ThinkingSettings) error {
	if settings == nil || settings.Level == "" {
		return nil
	}
	switch settings.Level {
	case ThinkingLevelDisabled, ThinkingLevelEnabled, ThinkingLevelMinimal, ThinkingLevelLow,
		ThinkingLevelMedium, ThinkingLevelHigh, ThinkingLevelXHigh:
		return nil
	default:
		return fmt.Errorf("ai: invalid thinking level %q", settings.Level)
	}
}

func clonePointer[T any](value *T) *T {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func mergeModelSettings(base ModelSettings, override *ModelSettings) ModelSettings {
	base = base.Clone()
	if override == nil {
		return base
	}
	if override.MaxTokens != 0 {
		base.MaxTokens = override.MaxTokens
	}
	if override.RequestTimeout != 0 {
		base.RequestTimeout = override.RequestTimeout
	}
	if override.Temperature != nil {
		base.Temperature = clonePointer(override.Temperature)
	}
	if override.TopP != nil {
		base.TopP = clonePointer(override.TopP)
	}
	if override.Seed != nil {
		base.Seed = clonePointer(override.Seed)
	}
	if override.PresencePenalty != nil {
		base.PresencePenalty = clonePointer(override.PresencePenalty)
	}
	if override.FrequencyPenalty != nil {
		base.FrequencyPenalty = clonePointer(override.FrequencyPenalty)
	}
	if override.LogitBias != nil {
		base.LogitBias = maps.Clone(override.LogitBias)
	}
	if override.Logprobs != nil {
		base.Logprobs = clonePointer(override.Logprobs)
	}
	if override.TopLogprobs != nil {
		base.TopLogprobs = clonePointer(override.TopLogprobs)
	}
	if override.ServiceTier != "" {
		base.ServiceTier = override.ServiceTier
	}
	if override.ExtraHeaders != nil {
		base.ExtraHeaders = maps.Clone(override.ExtraHeaders)
	}
	if override.ExtraBody != nil {
		base.ExtraBody = cloneSchemaMap(override.ExtraBody)
	}
	if override.StopSequences != nil {
		base.StopSequences = slices.Clone(override.StopSequences)
	}
	if override.ParallelToolCalls != nil {
		base.ParallelToolCalls = clonePointer(override.ParallelToolCalls)
	}
	if override.Thinking != nil {
		thinking := ModelSettings{Thinking: override.Thinking}.Clone()
		base.Thinking = thinking.Thinking
	}
	return base
}

// OutputToolConfig customizes tool-based structured output. Empty Name and
// Description fields use the defaults.
type OutputToolConfig struct {
	Name        string
	Description string
	Sequential  bool
	Strict      *bool
	// MaxRetries overrides the run's output retry budget for this output tool.
	MaxRetries *int
}

// ToolDefinition describes a tool to the model.
type ToolDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"parameters_json_schema"`
	// ReturnSchema describes the tool value for discovery and dynamic clients.
	// It is local metadata and is not sent to providers.
	ReturnSchema map[string]any `json:"-"`
	// Sequential makes this tool an execution barrier. Calls before it
	// finish first; the tool then runs alone; later calls start afterward.
	Sequential bool `json:"-"`
	// Strict asks the provider to constrain generated arguments to Schema.
	// Nil uses the provider default; true forces strict mode; false disables it.
	Strict *bool `json:"-"`
	// Metadata is available to preparation and filtering hooks but is not sent
	// to the model.
	Metadata map[string]any `json:"-"`
	// ToolsetID identifies the dynamic toolset that owns this definition. It is
	// local lifecycle metadata and is not sent to providers.
	ToolsetID string `json:"-"`
	// DeferLoading hides the tool until a rich tool return reveals its name.
	// Native providers may advertise its schema without making it executable.
	DeferLoading bool `json:"-"`
	// RequiresApproval pauses before local execution until a caller approves.
	RequiresApproval bool `json:"-"`
	// ApprovalMetadata is returned with a pending approval request.
	ApprovalMetadata map[string]any `json:"-"`
	// DynamicApproval allows the tool function to return ToolApprovalRequest.
	DynamicApproval bool `json:"-"`
	// ExternalExecution returns the call for execution outside the agent.
	ExternalExecution bool `json:"-"`
	// DynamicExternalExecution allows a function to return ExternalToolRequest.
	DynamicExternalExecution bool `json:"-"`
	// ToolKind identifies framework-managed typed tool calls and returns.
	ToolKind ToolPartKind `json:"-"`
	// ToolSearchStrategy controls provider adaptation for the tool-search surface.
	// It is local routing metadata and is not sent as a function-tool field.
	ToolSearchStrategy ToolSearchStrategy `json:"-"`
	maxRetries         *int
	timeout            time.Duration
}
