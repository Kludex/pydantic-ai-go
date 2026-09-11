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
	// Request generates one complete response without retaining or mutating its inputs.
	Request(ctx context.Context, msgs []ModelMessage, params ModelRequestParams) (*ModelResponse, error)

	// Name identifies the model (e.g. "gpt-5") for tracing and results.
	Name() string
}

// ModelContextWindow is implemented by models that expose their maximum
// combined input and output token count. A non-positive value means unknown.
type ModelContextWindow interface {
	ContextWindow() int
}

// TokenCountingModel is implemented by models that can count request tokens
// before generation. The returned usage describes the prospective request and
// is not added to run usage.
type TokenCountingModel interface {
	// CountTokens returns prospective usage without adding it to run totals.
	CountTokens(ctx context.Context, msgs []ModelMessage, params ModelRequestParams) (Usage, error)
}

// ModelCompactor is implemented by models with an explicit history-compaction endpoint.
// The returned response contains the provider's durable compaction boundary. Implementations
// may be called concurrently and must not retain or mutate request values.
type ModelCompactor interface {
	// CompactMessages returns a durable provider compaction boundary.
	CompactMessages(ctx context.Context, msgs []ModelMessage, params ModelRequestParams) (*ModelResponse, error)
}

// ModelProviderIdentity is implemented by models that expose the provider
// identity used for pricing and telemetry.
type ModelProviderIdentity interface {
	// ProviderName returns the stable provider identifier used for telemetry and pricing.
	ProviderName() string
	// ProviderURL returns the configured provider endpoint.
	ProviderURL() string
}

// CountModelTokens counts a detached prospective request.
func CountModelTokens(
	ctx context.Context, model Model, msgs []ModelMessage, params ModelRequestParams,
) (Usage, error) {
	if modelIsNil(model) {
		return Usage{}, ErrNoModel
	}
	counter, ok := model.(TokenCountingModel)
	if !ok {
		return Usage{}, fmt.Errorf("%w by model %q", ErrTokenCountingUnsupported, model.Name())
	}
	request := ModelRequestContext{Messages: msgs, Params: params}.Clone()
	if err := validateModelSettings(request.Params.Settings); err != nil {
		return Usage{}, err
	}
	preparedMessages, err := PrepareModelMessages(model, request.Messages)
	if err != nil {
		return Usage{}, err
	}
	request.Messages = preparedMessages
	if request.Params.Settings.RequestTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, request.Params.Settings.RequestTimeout)
		defer cancel()
	}
	usage, err := counter.CountTokens(ctx, request.Messages, request.Params)
	return usage.Clone(), err
}

// CompactModelMessages calls a model's explicit compaction endpoint with detached input.
// The returned response is detached from provider-owned state.
func CompactModelMessages(
	ctx context.Context, model Model, msgs []ModelMessage, params ModelRequestParams,
) (*ModelResponse, error) {
	if modelIsNil(model) {
		return nil, ErrNoModel
	}
	compactor, ok := model.(ModelCompactor)
	if !ok {
		return nil, fmt.Errorf("%w by model %q", ErrCompactionUnsupported, model.Name())
	}
	request := ModelRequestContext{Messages: msgs, Params: params}.Clone()
	if err := validateModelSettings(request.Params.Settings); err != nil {
		return nil, err
	}
	preparedMessages, err := PrepareModelMessages(model, request.Messages)
	if err != nil {
		return nil, err
	}
	request.Messages = preparedMessages
	if request.Params.Settings.RequestTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, request.Params.Settings.RequestTimeout)
		defer cancel()
	}
	if configured, ok := ctx.Value(instrumentationRuntimeContextKey{}).(*InstrumentedModel); ok &&
		!compactionSpanActive(ctx) {
		runtime := *configured
		runtime.ModelWrapper = WrapModel(model)
		response, err := runtime.compactMessages(ctx, compactor, request.Messages, request.Params)
		if response == nil {
			return nil, err
		}
		return cloneModelResponse(response), err
	}
	response, err := compactor.CompactMessages(ctx, request.Messages, request.Params)
	if response == nil {
		return nil, err
	}
	return cloneModelResponse(response), err
}

// ToolSearchStrategyModel is implemented by models that support required
// named provider-managed tool-search strategies.
type ToolSearchStrategyModel interface {
	// SupportsToolSearchStrategy reports whether a named strategy can run provider-side.
	SupportsToolSearchStrategy(strategy ToolSearchStrategy) bool
}

// NativeToolSupportModel is implemented by models that can decide whether a
// provider-native tool is available for the selected model and transport.
// Calls may run concurrently and receive detached tool definitions.
type NativeToolSupportModel interface {
	// SupportsNativeTool reports whether the selected model and transport support tool.
	SupportsNativeTool(tool NativeTool) bool
}

// NativeToolSearchHistoryModel is implemented by models that can replay
// provider-native tool-search parts from their own provider.
type NativeToolSearchHistoryModel interface {
	// NativeToolSearchProvider identifies native tool-search history this model can replay.
	NativeToolSearchProvider() string
}

// ModelCloseFunc releases resources acquired for one agent run.
type ModelCloseFunc func(ctx context.Context) error

// ModelOpener is an optional per-run lifecycle for models that acquire
// resources. OpenModel runs once per distinct selected model. Close functions
// run in reverse selection order after toolset resources are no longer used.
type ModelOpener interface {
	// OpenModel acquires run-scoped resources and returns their cleanup function.
	OpenModel(ctx context.Context) (ModelCloseFunc, error)
}

// ModelDefaultSettings is implemented by models with request-setting defaults.
// Agent, capability, and run settings override these values field by field.
type ModelDefaultSettings interface {
	// DefaultModelSettings returns detached provider request defaults.
	DefaultModelSettings() ModelSettings
}

// PromptCacheRetentionModel is implemented by models that can report how long
// provider prompt-cache entries requested by settings may remain reusable.
type PromptCacheRetentionModel interface {
	// PromptCacheRetention returns the requested retention and whether it is known.
	PromptCacheRetention(settings ModelSettings) (time.Duration, bool)
}

// ResolvePromptCacheRetention reports the longest requested provider cache lifetime.
// Model defaults are merged before the provider interprets its settings.
func ResolvePromptCacheRetention(model Model, settings *ModelSettings) (time.Duration, bool) {
	if modelIsNil(model) {
		return 0, false
	}
	resolver, ok := model.(PromptCacheRetentionModel)
	if !ok {
		return 0, false
	}
	merged := ModelSettings{}
	if defaults, ok := model.(ModelDefaultSettings); ok {
		merged = defaults.DefaultModelSettings()
	}
	merged = mergeModelSettings(merged, settings)
	return resolver.PromptCacheRetention(merged)
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
	// Model selects a concrete model directly.
	Model Model
	// ID selects an application model identifier resolved by registered resolvers.
	ID string
}

// ModelSelectionContext is a read-only snapshot passed to a model selector.
// Step starts at 1. Model is the lower-precedence model on the first step and
// the model used by the previous request thereafter.
type ModelSelectionContext[Deps any] struct {
	// Deps is the run's typed dependency value.
	Deps Deps
	// Model is the lower-precedence or previously selected model.
	Model Model
	// ModelID is the lower-precedence or previously selected application ID.
	ModelID string
	// Step is the one-based logical model request number.
	Step int
	// Messages is a detached snapshot before the request.
	Messages []ModelMessage
	// Usage is detached usage accumulated before the request.
	Usage Usage
}

// ModelSelectorFunc selects a model before one logical model request.
type ModelSelectorFunc[Deps any] func(
	ctx context.Context, selection ModelSelectionContext[Deps],
) (ModelSelection, error)

// ModelResolutionContext is passed when resolving an application model ID.
type ModelResolutionContext[Deps any] struct {
	// Deps is the run's typed dependency value.
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
	// Tools are locally executed function-tool definitions.
	Tools []ToolDefinition
	// DeferredTools contains prepared definitions hidden by the agent. Providers
	// with native deferral may advertise them without making them executable.
	DeferredTools []ToolDefinition
	// NativeTools are executed by a compatible model provider.
	NativeTools []NativeTool
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
	// AllowImageOutput reports whether an image file is an acceptable final output.
	AllowImageOutput bool
	// Settings contains detached generation settings for this request.
	Settings ModelSettings

	nativeToolPreferencesResolved bool
	nativeToolSupportRequired     bool
}

// InstructionPart is one independently addressable model instruction block.
// Declare Name relative to its source. The agent resolves ID before the model
// request; callers should not invent qualified IDs.
type InstructionPart struct {
	// Content is the instruction text sent to the provider.
	Content string
	// Dynamic reports whether the content was evaluated for this run.
	Dynamic bool
	// Name is the source-relative address assigned by its owner.
	Name string
	// ID is the resolved stable instruction address, when one was assigned.
	ID *InstructionID
}

// ThinkingLevel configures provider reasoning with a portable effort level.
type ThinkingLevel string

const (
	// ThinkingLevelDisabled requests no provider reasoning.
	ThinkingLevelDisabled ThinkingLevel = "disabled"
	// ThinkingLevelEnabled enables reasoning with provider defaults.
	ThinkingLevelEnabled ThinkingLevel = "enabled"
	// ThinkingLevelMinimal requests the smallest reasoning effort.
	ThinkingLevelMinimal ThinkingLevel = "minimal"
	// ThinkingLevelLow requests low reasoning effort.
	ThinkingLevelLow ThinkingLevel = "low"
	// ThinkingLevelMedium requests medium reasoning effort.
	ThinkingLevelMedium ThinkingLevel = "medium"
	// ThinkingLevelHigh requests high reasoning effort.
	ThinkingLevelHigh ThinkingLevel = "high"
	// ThinkingLevelXHigh requests the largest portable reasoning effort.
	ThinkingLevelXHigh ThinkingLevel = "xhigh"
)

// ThinkingSettings configures reasoning generation. Level maps to the closest
// provider effort. TokenBudget and IncludeThoughts override that provider's
// defaults when supported.
type ThinkingSettings struct {
	// Level is the portable provider reasoning effort.
	Level ThinkingLevel
	// TokenBudget overrides the provider's reasoning token budget when supported.
	TokenBudget *int
	// IncludeThoughts controls whether supported providers return reasoning content.
	IncludeThoughts *bool
}

// ServiceTier selects a provider's latency and capacity class.
type ServiceTier string

const (
	// ServiceTierAuto lets the provider choose capacity.
	ServiceTierAuto ServiceTier = "auto"
	// ServiceTierDefault requests standard capacity.
	ServiceTierDefault ServiceTier = "default"
	// ServiceTierFlex requests lower-cost flexible capacity.
	ServiceTierFlex ServiceTier = "flex"
	// ServiceTierPriority requests prioritized capacity.
	ServiceTierPriority ServiceTier = "priority"
)

// ModelSettings tunes a model request. The zero value uses provider defaults.
type ModelSettings struct {
	// MaxTokens bounds generated tokens. Zero uses the provider default.
	MaxTokens int
	// RequestTimeout bounds the request and full stream lifetime. Zero adds no timeout.
	RequestTimeout time.Duration
	// Temperature controls sampling randomness when supported.
	Temperature *float64
	// TopP controls nucleus sampling when supported.
	TopP *float64
	// Seed requests deterministic sampling when supported.
	Seed *int
	// PresencePenalty penalizes tokens already present in generated text.
	PresencePenalty *float64
	// FrequencyPenalty penalizes tokens in proportion to their generated frequency.
	FrequencyPenalty *float64
	// LogitBias adjusts token likelihood by provider token identifier.
	LogitBias map[string]int
	// Logprobs requests output-token log probabilities.
	Logprobs *bool
	// TopLogprobs requests alternative token log probabilities and requires Logprobs.
	TopLogprobs *int
	// ServiceTier selects portable provider capacity.
	ServiceTier ServiceTier
	// ExtraHeaders replaces default request headers by name.
	ExtraHeaders map[string]string
	// ExtraBody contains provider-specific fields that must not conflict with typed fields.
	ExtraBody map[string]any
	// StopSequences ends generation when one value is emitted.
	StopSequences []string
	// ParallelToolCalls controls whether the provider may request tools in parallel.
	ParallelToolCalls *bool
	// Thinking configures portable provider reasoning.
	Thinking *ThinkingSettings
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
	// Name overrides the generated output-tool name.
	Name string
	// Description overrides the generated output-tool description.
	Description string
	// Sequential makes the output tool an execution barrier.
	Sequential bool
	// Strict controls provider schema enforcement when supported.
	Strict *bool
	// MaxRetries overrides the run's output retry budget for this output tool.
	MaxRetries *int
}

// ToolDefinition describes a tool to the model.
type ToolDefinition struct {
	// Name is the model-facing function name.
	Name string `json:"name"`
	// Description explains when and how the model should call the tool.
	Description string `json:"description,omitempty"`
	// Schema is the JSON Schema for function arguments.
	Schema map[string]any `json:"parameters_json_schema"`
	// ReturnSchema describes the tool value for discovery and dynamic clients.
	ReturnSchema map[string]any `json:"-"`
	// IncludeReturnSchema controls whether ReturnSchema is sent to the model.
	// Nil defaults to false unless a capability or toolset enables it.
	IncludeReturnSchema *bool `json:"-"`
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
	// NativeFallbackFor removes this function tool when the identified native
	// tool is supported. It remains available when that native tool is unsupported.
	NativeFallbackFor string `json:"unless_native,omitempty"`
	// NativeCompanionFor identifies a function tool managed by the named native
	// tool. The marker is cleared when that native tool is unsupported.
	NativeCompanionFor string `json:"with_native,omitempty"`
	maxRetries         *int
	timeout            time.Duration
}
