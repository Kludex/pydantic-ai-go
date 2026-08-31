package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"sync/atomic"
)

// Agent is a typed LLM agent. Deps is the dependency type passed to tools
// and dynamic instructions; Output is the final result type. Use string
// for plain-text output; any other type produces structured output via
// an output tool whose schema is reflected from Output.
type Agent[Deps, Output any] struct {
	model              Model
	name               string
	description        string
	descriptionSet     bool
	descriptionFunc    AgentDescriptionFunc[Deps]
	instructions       string
	instructionParts   []InstructionPart
	instructionsFuncs  []instructionRunner[Deps]
	systemPrompts      []string
	systemPromptFuncs  []systemPromptRunner[Deps]
	modelSettingsFuncs []ModelSettingsFunc[Deps]
	metadataFuncs      []RunMetadataFunc[Deps]
	modelSelectors     []ModelSelectorFunc[Deps]
	modelIDResolvers   []ModelIDResolverFunc[Deps]
	toolsPrepareFuncs  []ToolsPrepareFunc[Deps]
	settings           ModelSettings
	metadata           map[string]any
	usageLimits        UsageLimits
	retryLimits        RetryLimits
	outputMode         OutputMode
	outputTool         OutputToolConfig
	outputSchema       map[string]any
	outputDecoder      func([]byte) (decodedOutput, error)
	outputProcessor    func(context.Context, *RunContext[Deps], any, any) (Output, error)
	outputHasFunction  bool
	outputFunctionName string
	outputInputType    reflect.Type
	outputAllowsText   bool
	outputOverrideErr  error
	promptedTemplate   string
	outputToolPrepare  []OutputToolPrepareFunc[Deps]
	endStrategy        EndStrategy
	sequentialTools    bool
	capabilities       []Capability
	capInstructions    []InstructionPart
	capSettings        []capabilitySettingsLayer
	capInstructionIDs  map[string]struct{}
	outputValidators   []func(ctx context.Context, rc *RunContext[Deps], out Output) error

	tools             []toolEntry[Deps]
	nativeToolEntries []nativeToolEntry[Deps]
	toolsets          []Toolset[Deps]
	started           atomic.Bool
}

type toolEntry[Deps any] struct {
	def      ToolDefinition
	validate toolValidateFunc[Deps]
	execute  toolExecuteFunc[Deps]
	prepare  ToolPrepareFunc[Deps]
}

type instructionRunner[Deps any] struct {
	name string
	fn   InstructionsFunc[Deps]
}

type systemPromptRunner[Deps any] struct {
	id      string
	fn      InstructionsFunc[Deps]
	dynamic bool
}

// NewAgent creates an agent backed by model. Model may be nil when an agent
// selector, capability selector, or per-run model always supplies one.
func NewAgent[Deps, Output any](model Model, opts ...Option) *Agent[Deps, Output] {
	a := &Agent[Deps, Output]{
		model: model, retryLimits: RetryLimits{Tools: 1, Output: 1}, outputMode: OutputModeAuto,
		endStrategy: EndStrategyGraceful, capInstructionIDs: make(map[string]struct{}),
	}
	var cfg config
	for _, opt := range opts {
		opt(&cfg)
	}
	a.name = cfg.name
	a.description = cfg.description
	a.descriptionSet = cfg.descriptionSet
	if cfg.descriptionFunc != nil {
		descriptionFunc, ok := cfg.descriptionFunc.(AgentDescriptionFunc[Deps])
		if !ok {
			panic("ai: agent description dependencies do not match agent")
		}
		a.descriptionFunc = descriptionFunc
	}
	a.instructions = cfg.instructions
	a.instructionParts = cloneInstructionParts(cfg.instructionParts)
	a.systemPrompts = slices.Clone(cfg.systemPrompts)
	a.settings = cfg.settings.Clone()
	a.metadata = cloneSchemaMap(cfg.metadata)
	a.usageLimits = cfg.limits
	if cfg.outputMode != nil {
		a.outputMode = *cfg.outputMode
	}
	a.outputTool = cloneOutputToolConfig(cfg.outputTool)
	a.promptedTemplate = cfg.promptedTemplate
	validateOutputToolConfig(a.outputTool)
	validateOutputMode(a.outputMode)
	if cfg.endStrategy != "" {
		switch cfg.endStrategy {
		case EndStrategyEarly, EndStrategyGraceful, EndStrategyExhaustive:
			a.endStrategy = cfg.endStrategy
		default:
			panic(fmt.Sprintf("ai: invalid end strategy %q", cfg.endStrategy))
		}
	}
	a.sequentialTools = cfg.sequentialTools
	for _, tool := range cfg.nativeTools {
		a.nativeToolEntries = append(a.nativeToolEntries, nativeToolEntry[Deps]{tool: cloneNativeTool(tool)})
	}
	if err := validateNativeTools(staticNativeTools(a.nativeToolEntries)); err != nil {
		panic(err.Error())
	}
	var err error
	a.capabilities, err = sortCapabilities(cfg.capabilities, cfg.capabilities)
	if err != nil {
		panic(fmt.Sprintf("ai: capability ordering: %v", err))
	}
	if cfg.retryLimits != nil {
		validateRetryLimits(*cfg.retryLimits)
		a.retryLimits = *cfg.retryLimits
	}
	for _, capability := range a.capabilities {
		reg := &CapabilityRegistry{}
		if err := capability.Setup(reg); err != nil {
			panic(fmt.Sprintf("ai: capability setup: %v", err))
		}
		source, err := capabilityInstructionSource(capability)
		if err != nil {
			panic(fmt.Sprintf("ai: capability instructions: %v", err))
		}
		instructions, err := qualifyInstructionParts(reg.instructions, source)
		if err != nil {
			panic(fmt.Sprintf("ai: capability instructions: %v", err))
		}
		if source != nil && capabilityContributesInstructions(capability, instructions) {
			if _, duplicate := a.capInstructionIDs[source.ID]; duplicate {
				panic(fmt.Sprintf(
					"ai: capability ID %q is used by multiple capabilities that contribute instructions", source.ID,
				))
			}
			a.capInstructionIDs[source.ID] = struct{}{}
		}
		a.capInstructions = append(a.capInstructions, instructions...)
		for _, tool := range reg.tools {
			fn := tool.call
			a.tools = append(a.tools, toolEntry[Deps]{
				def: cloneToolDefinition(tool.def),
				validate: func(_ context.Context, _ *RunContext[Deps], rawArgs json.RawMessage) (any, error) {
					return slices.Clone(rawArgs), nil
				},
				execute: func(ctx context.Context, _ *RunContext[Deps], validated any) (any, error) {
					rawArgs, ok := validated.(json.RawMessage)
					if !ok {
						return nil, fmt.Errorf(
							"validated arguments for tool %q have type %T, expected json.RawMessage", tool.def.Name, validated,
						)
					}
					return fn(ctx, rawArgs)
				},
			})
		}
		for _, tool := range reg.nativeTools {
			a.nativeToolEntries = append(a.nativeToolEntries, nativeToolEntry[Deps]{tool: cloneNativeTool(tool)})
		}
		a.capSettings = append(a.capSettings, capabilitySettingsLayer{
			static: reg.modelSettings, provider: capabilityModelSettingsProvider(capability),
		})
	}
	if err := validateNativeTools(staticNativeTools(a.nativeToolEntries)); err != nil {
		panic(err.Error())
	}
	return a
}

// AgentDescriptionFunc renders an agent description for one run.
type AgentDescriptionFunc[Deps any] func(ctx context.Context, deps Deps) (string, error)

// Name returns the application-defined agent name. Instrumentation uses
// "agent" when no name is configured.
func (a *Agent[Deps, Output]) Name() string { return a.name }

// Description returns the static agent description. It returns an empty
// string when the description is dynamic or absent.
func (a *Agent[Deps, Output]) Description() string {
	if !a.descriptionSet || a.descriptionFunc != nil {
		return ""
	}
	return a.description
}

// RenderDescription resolves the description for the supplied dependencies.
// Dynamic descriptions may be called concurrently by concurrent runs.
func (a *Agent[Deps, Output]) RenderDescription(ctx context.Context, deps Deps) (string, error) {
	if a.descriptionFunc != nil {
		description, err := a.descriptionFunc(ctx, deps)
		if err != nil {
			return "", fmt.Errorf("ai: agent description: %w", err)
		}
		return description, nil
	}
	if a.descriptionSet {
		return a.description, nil
	}
	return "", nil
}

// InstructionsFunc returns instructions for one model-request step.
type InstructionsFunc[Deps any] func(ctx context.Context, rc *RunContext[Deps]) (string, error)

// AddInstructionsFunc registers dynamic instructions evaluated before every
// model request and appended to the static instructions.
func (a *Agent[Deps, Output]) AddInstructionsFunc(fn InstructionsFunc[Deps]) {
	if fn == nil {
		panic("ai: instructions function must not be nil")
	}
	a.checkNotStarted()
	a.instructionsFuncs = append(a.instructionsFuncs, instructionRunner[Deps]{fn: fn})
}

// AddNamedInstructionsFunc registers addressable dynamic instructions. Name is
// qualified as agent:<name> and must remain stable across application versions.
func (a *Agent[Deps, Output]) AddNamedInstructionsFunc(name string, fn InstructionsFunc[Deps]) {
	if err := validateInstructionName(name, false); err != nil {
		panic(err.Error())
	}
	if fn == nil {
		panic("ai: instructions function must not be nil")
	}
	a.checkNotStarted()
	a.instructionsFuncs = append(a.instructionsFuncs, instructionRunner[Deps]{name: name, fn: fn})
}

// AddInstructionPart appends an independently addressable literal instruction.
// Its Name is qualified under the agent source.
func (a *Agent[Deps, Output]) AddInstructionPart(part InstructionPart) {
	qualified, err := qualifyInstructionParts(
		[]InstructionPart{part}, &InstructionSource{Kind: InstructionSourceAgent},
	)
	if err != nil {
		panic(err.Error())
	}
	a.checkNotStarted()
	a.instructionParts = append(a.instructionParts, qualified...)
}

// AddSystemPromptFunc registers a legacy system prompt evaluated when a new
// conversation starts. Prefer AddInstructionsFunc for new applications.
func (a *Agent[Deps, Output]) AddSystemPromptFunc(fn InstructionsFunc[Deps]) {
	a.checkNotStarted()
	a.systemPromptFuncs = append(a.systemPromptFuncs, systemPromptRunner[Deps]{fn: fn})
}

// AddDynamicSystemPromptFunc registers a legacy system prompt under a stable
// application ID. Matching parts in resumed history are reevaluated with the
// new run context. IDs must remain stable across application versions.
func (a *Agent[Deps, Output]) AddDynamicSystemPromptFunc(id string, fn InstructionsFunc[Deps]) {
	if id == "" {
		panic("ai: dynamic system prompt ID must not be empty")
	}
	a.checkNotStarted()
	for _, runner := range a.systemPromptFuncs {
		if runner.dynamic && runner.id == id {
			panic(fmt.Sprintf("ai: duplicate dynamic system prompt ID %q", id))
		}
	}
	a.systemPromptFuncs = append(a.systemPromptFuncs, systemPromptRunner[Deps]{id: id, fn: fn, dynamic: true})
}

// ModelSettingsFunc returns settings to merge over settings resolved by
// earlier layers for one model-request step.
type ModelSettingsFunc[Deps any] func(ctx context.Context, rc *RunContext[Deps]) (ModelSettings, error)

// RunMetadataFunc computes detached application metadata at run startup and
// again after successful completion. Concurrent runs may call it concurrently.
type RunMetadataFunc[Deps any] func(ctx context.Context, rc *RunContext[Deps]) (map[string]any, error)

// AddModelSettingsFunc registers dynamic model settings evaluated before
// every model request. Each callback sees prior layers in rc.ModelSettings.
func (a *Agent[Deps, Output]) AddModelSettingsFunc(fn ModelSettingsFunc[Deps]) {
	a.checkNotStarted()
	a.modelSettingsFuncs = append(a.modelSettingsFuncs, fn)
}

// AddMetadataFunc appends dynamic run metadata. Functions run in registration
// order and later values override earlier keys. Keep functions free of side
// effects because successful runs evaluate them at startup and completion.
// Synchronize shared mutable state because concurrent runs share these functions.
func (a *Agent[Deps, Output]) AddMetadataFunc(fn RunMetadataFunc[Deps]) {
	if fn == nil {
		panic("ai: metadata function must not be nil")
	}
	a.checkNotStarted()
	a.metadataFuncs = append(a.metadataFuncs, fn)
}

// AddModelSelector registers an adaptive model layer evaluated before every
// model request. Later selectors see earlier selections and take precedence.
func (a *Agent[Deps, Output]) AddModelSelector(fn ModelSelectorFunc[Deps]) {
	a.checkNotStarted()
	a.modelSelectors = append(a.modelSelectors, fn)
}

// AddModelIDResolver registers an application model-ID resolver. Resolvers
// are tried in registration order until one returns a model.
func (a *Agent[Deps, Output]) AddModelIDResolver(fn ModelIDResolverFunc[Deps]) {
	a.checkNotStarted()
	a.modelIDResolvers = append(a.modelIDResolvers, fn)
}

// ToolsPrepareFunc filters or modifies per-step copies of function tool
// definitions. Output tools are not included.
type ToolsPrepareFunc[Deps any] func(
	ctx context.Context, rc *RunContext[Deps], tools []ToolDefinition,
) ([]ToolDefinition, error)

// AddToolsPrepareFunc registers a tool preparation hook. Hooks run in
// registration order before every model request. Returning an empty or nil
// slice exposes no function tools for that step. Concurrent runs may invoke
// hooks concurrently, so hooks must synchronize mutable state.
func (a *Agent[Deps, Output]) AddToolsPrepareFunc(fn ToolsPrepareFunc[Deps]) {
	a.checkNotStarted()
	a.toolsPrepareFuncs = append(a.toolsPrepareFuncs, fn)
}

// OutputToolPrepareFunc customizes the output tool before one model request.
// Return nil to omit it for that step.
type OutputToolPrepareFunc[Deps any] func(
	ctx context.Context, rc *RunContext[Deps], tool ToolDefinition,
) (*ToolDefinition, error)

// AddOutputToolPrepareFunc registers per-step output-tool preparation. It is
// ignored for text and native output modes.
func (a *Agent[Deps, Output]) AddOutputToolPrepareFunc(fn OutputToolPrepareFunc[Deps]) {
	a.checkNotStarted()
	a.outputToolPrepare = append(a.outputToolPrepare, fn)
}

// AddOutputValidator registers a semantic check on the final output.
// Return an error from Retryf to send the failure back to the model. The
// exhaustive end strategy may invoke validators concurrently for multiple
// output calls, so validators must synchronize mutable state.
func (a *Agent[Deps, Output]) AddOutputValidator(fn func(ctx context.Context, rc *RunContext[Deps], out Output) error) {
	a.checkNotStarted()
	a.outputValidators = append(a.outputValidators, fn)
}

// AddToolset registers a composable toolset on the agent.
func (a *Agent[Deps, Output]) AddToolset(toolset Toolset[Deps]) {
	a.checkNotStarted()
	a.toolsets = append(a.toolsets, toolset)
}

// AddTool registers a reusable tool on the agent.
func (a *Agent[Deps, Output]) AddTool(tool Tool[Deps]) {
	a.checkNotStarted()
	tool.entry.def = cloneToolDefinition(tool.entry.def)
	a.tools = append(a.tools, tool.entry)
}

// AddNativeTool registers a provider-executed tool. Required tools fail on
// providers that do not support their kind; optional tools may be omitted.
func (a *Agent[Deps, Output]) AddNativeTool(tool NativeTool) {
	a.checkNotStarted()
	entries := append(cloneNativeToolEntries(a.nativeToolEntries), nativeToolEntry[Deps]{tool: cloneNativeTool(tool)})
	if err := validateNativeTools(staticNativeTools(entries)); err != nil {
		panic(err.Error())
	}
	a.nativeToolEntries = entries
}

// AddNativeToolFunc registers a dependency-aware native tool resolved before
// every model request.
func (a *Agent[Deps, Output]) AddNativeToolFunc(fn NativeToolFunc[Deps]) {
	a.checkNotStarted()
	if fn == nil {
		panic("ai: native tool function must not be nil")
	}
	a.nativeToolEntries = append(a.nativeToolEntries, nativeToolEntry[Deps]{fn: fn})
}

func (a *Agent[Deps, Output]) checkNotStarted() {
	if a.started.Load() {
		panic("ai: cannot modify an agent after its first run")
	}
}

// Option configures an Agent at construction.
type Option func(*config)

type config struct {
	name             string
	description      string
	descriptionSet   bool
	descriptionFunc  any
	instructions     string
	instructionParts []InstructionPart
	systemPrompts    []string
	settings         ModelSettings
	metadata         map[string]any
	limits           UsageLimits
	retryLimits      *RetryLimits
	outputMode       *OutputMode
	outputTool       OutputToolConfig
	promptedTemplate string
	endStrategy      EndStrategy
	sequentialTools  bool
	nativeTools      []NativeTool
	capabilities     []Capability
}

// WithNativeTools registers provider-executed tools on every run.
func WithNativeTools(tools ...NativeTool) Option {
	cloned := CloneNativeTools(tools)
	return func(c *config) { c.nativeTools = append(c.nativeTools, CloneNativeTools(cloned)...) }
}

// OutputMode selects how structured output is requested from the model.
type OutputMode int

const (
	// OutputModeTool asks for structured output via a final output tool
	// the model must call. It works with every provider.
	OutputModeTool OutputMode = iota
	// OutputModeNative uses the provider's native JSON mode: the model
	// responds with JSON text conforming to the output schema. The
	// provider must support it; validation retries still apply.
	OutputModeNative
	// OutputModePrompted asks for schema-compatible JSON through instructions.
	// It works with providers that do not implement native structured output.
	OutputModePrompted
	// OutputModeAuto uses the selected model's profile. The default profile uses
	// OutputModeTool. This is the default for reflected structured output.
	OutputModeAuto
)

// WithOutputMode selects how structured output is requested. It has no
// effect when Output is string.
func WithOutputMode(mode OutputMode) Option {
	return func(c *config) { c.outputMode = &mode }
}

// WithPromptedOutputTemplate replaces the default prompted-output instructions.
// If template omits {schema}, the schema is appended.
func WithPromptedOutputTemplate(template string) Option {
	if template == "" {
		panic("ai: prompted output template must not be empty")
	}
	return func(c *config) { c.promptedTemplate = template }
}

// WithOutputTool customizes tool-based structured output. It has no effect
// for string or native output.
func WithOutputTool(outputTool OutputToolConfig) Option {
	return func(c *config) { c.outputTool = outputTool }
}

// EndStrategy controls calls emitted alongside a successful output tool.
type EndStrategy string

const (
	// EndStrategyEarly skips function tools as soon as an output succeeds.
	EndStrategyEarly EndStrategy = "early"
	// EndStrategyGraceful runs function tools in emission order, skips later
	// output tools, and discards the output if a function tool asks to retry.
	EndStrategyGraceful EndStrategy = "graceful"
	// EndStrategyExhaustive runs every call and selects the first successful
	// output in emission order, unless a function tool asks to retry.
	EndStrategyExhaustive EndStrategy = "exhaustive"
)

// WithEndStrategy controls calls emitted alongside a successful output tool.
// The default is EndStrategyGraceful.
func WithEndStrategy(strategy EndStrategy) Option {
	return func(c *config) { c.endStrategy = strategy }
}

// WithSequentialToolExecution runs every tool call serially for every run
// of the agent. Without it, independent calls run concurrently and tools
// registered with WithSequential form barriers.
func WithSequentialToolExecution() Option {
	return func(c *config) { c.sequentialTools = true }
}

// WithAgentName sets the stable application name used by telemetry and
// integrations. An empty name uses "agent" in telemetry.
func WithAgentName(name string) Option {
	return func(config *config) { config.name = name }
}

// WithAgentDescription sets a human-readable description for telemetry and
// integrations. It is not sent to the model.
func WithAgentDescription(description string) Option {
	return func(config *config) {
		config.description = description
		config.descriptionSet = true
		config.descriptionFunc = nil
	}
}

// WithAgentDescriptionFunc renders a description from run dependencies.
func WithAgentDescriptionFunc[Deps any](function AgentDescriptionFunc[Deps]) Option {
	if function == nil {
		panic("ai: agent description function must not be nil")
	}
	return func(config *config) {
		config.description = ""
		config.descriptionSet = false
		config.descriptionFunc = function
	}
}

// WithInstructions sets the static instructions sent with every model request.
func WithInstructions(instructions string) Option {
	return func(c *config) { c.instructions = instructions }
}

// WithInstructionParts appends literal instruction blocks. Names are
// qualified under the agent source when the agent is constructed.
func WithInstructionParts(parts ...InstructionPart) Option {
	qualified, err := qualifyInstructionParts(parts, &InstructionSource{Kind: InstructionSourceAgent})
	if err != nil {
		panic(err.Error())
	}
	return func(config *config) {
		config.instructionParts = append(config.instructionParts, cloneInstructionParts(qualified)...)
	}
}

// WithSystemPrompt appends a legacy system prompt to the first request in a
// conversation. Prefer WithInstructions for new applications.
func WithSystemPrompt(prompt string) Option {
	return func(c *config) { c.systemPrompts = append(c.systemPrompts, prompt) }
}

// WithModelSettings sets default model settings for every request.
func WithModelSettings(settings ModelSettings) Option {
	return func(c *config) { c.settings = settings }
}

// WithUsageLimits bounds every run of the agent.
func WithUsageLimits(limits UsageLimits) Option {
	return func(c *config) { c.limits = limits }
}

// RetryLimits contains the independent function-tool and output retry
// budgets. Both default to 1. Zero disables retries for that side.
type RetryLimits struct {
	Tools  int
	Output int
}

// WithMetadata sets static application metadata for every run. Metadata is
// not sent to the model and is copied before callbacks or results expose it.
func WithMetadata(metadata map[string]any) Option {
	metadata = cloneSchemaMap(metadata)
	return func(config *config) { config.metadata = cloneSchemaMap(metadata) }
}

// WithRetryLimits sets independent agent-wide retry budgets.
func WithRetryLimits(limits RetryLimits) Option {
	return func(c *config) { c.retryLimits = &limits }
}

// WithMaxRetries sets both retry budgets to n. It is a convenience for
// WithRetryLimits(RetryLimits{Tools: n, Output: n}).
func WithMaxRetries(n int) Option {
	return WithRetryLimits(RetryLimits{Tools: n, Output: n})
}

func validateRetryLimits(limits RetryLimits) {
	if limits.Tools < 0 || limits.Output < 0 {
		panic(fmt.Sprintf("ai: retry limits must be non-negative, got %+v", limits))
	}
}

// RunOption configures a single run.
type RunOption func(*runConfig)

type erasedModelSettingsFunc func(context.Context, any) (ModelSettings, error)
type erasedInstructionsFunc func(context.Context, any) (string, error)
type erasedModelSelectorFunc func(context.Context, any) (ModelSelection, error)
type erasedRunMetadataFunc func(context.Context, any) (map[string]any, error)
type erasedNativeToolFunc func(context.Context, any) (NativeTool, error)

type erasedNativeToolEntry struct {
	tool NativeTool
	fn   erasedNativeToolFunc
}

type erasedTool struct {
	entry any
}

type runConfig struct {
	history           []ModelMessage
	model             Model
	settings          *ModelSettings
	instructions      string
	instructionParts  []InstructionPart
	usageLimits       *UsageLimits
	retryLimits       *RetryLimits
	outputMode        *OutputMode
	outputTool        *OutputToolConfig
	promptedTemplate  *string
	modelID           string
	runID             string
	conversationID    *string
	metadata          map[string]any
	metadataFuncs     []erasedRunMetadataFunc
	settingsFuncs     []erasedModelSettingsFunc
	instructionsFuncs []erasedInstructionsFunc
	modelSelectors    []erasedModelSelectorFunc
	tools             []erasedTool
	nativeToolEntries []erasedNativeToolEntry
	toolsets          []any
	capabilities      []Capability
	deferredResults   *DeferredToolResults
	resumeSuspended   bool
}

// WithRunNativeTools adds provider-executed tools for one run without modifying the agent.
func WithRunNativeTools(tools ...NativeTool) RunOption {
	cloned := CloneNativeTools(tools)
	return func(c *runConfig) {
		for _, tool := range cloned {
			c.nativeToolEntries = append(c.nativeToolEntries, erasedNativeToolEntry{tool: cloneNativeTool(tool)})
		}
	}
}

// WithRunNativeToolFunc adds a dependency-aware native tool for one run.
func WithRunNativeToolFunc[Deps any](fn NativeToolFunc[Deps]) RunOption {
	if fn == nil {
		panic("ai: run native tool function must not be nil")
	}
	return func(c *runConfig) {
		c.nativeToolEntries = append(c.nativeToolEntries, erasedNativeToolEntry{fn: func(
			ctx context.Context, rc any,
		) (NativeTool, error) {
			typed, ok := rc.(*RunContext[Deps])
			if !ok {
				return nil, fmt.Errorf("ai: run native tool function dependencies do not match agent")
			}
			return fn(ctx, typed)
		}})
	}
}

// WithRunToolsets adds composable toolsets for one run without modifying the agent.
func WithRunToolsets[Deps any](toolsets ...Toolset[Deps]) RunOption {
	erased := make([]any, len(toolsets))
	for index, toolset := range toolsets {
		erased[index] = toolset
	}
	return func(c *runConfig) { c.toolsets = append(c.toolsets, erased...) }
}

// WithRunCapabilities adds capabilities for one run without modifying the
// agent. Agent capabilities remain outermost in middleware order.
func WithRunCapabilities(capabilities ...Capability) RunOption {
	capabilities = flattenCapabilities(capabilities)
	return func(config *runConfig) {
		config.capabilities = append(config.capabilities, capabilities...)
	}
}

// WithRunTools adds reusable tools for one run without modifying the agent.
// A run tool name must not duplicate an agent or another run tool name.
func WithRunTools[Deps any](tools ...Tool[Deps]) RunOption {
	entries := make([]erasedTool, len(tools))
	for index, tool := range tools {
		tool.entry.def = cloneToolDefinition(tool.entry.def)
		entries[index] = erasedTool{entry: tool.entry}
	}
	return func(c *runConfig) { c.tools = append(c.tools, entries...) }
}

// WithMessageHistory prepends prior conversation messages to the run.
func WithMessageHistory(msgs []ModelMessage) RunOption {
	return func(c *runConfig) { c.history = msgs }
}

// WithDeferredToolResults resolves pending calls in message history before
// the run's first model request.
func WithDeferredToolResults(results DeferredToolResults) RunOption {
	results = cloneDeferredToolResults(results)
	return func(c *runConfig) { c.deferredResults = &results }
}

// WithRunRetryLimits overrides both retry budgets for one run. Explicit
// per-tool limits still take precedence.
func WithRunRetryLimits(limits RetryLimits) RunOption {
	return func(c *runConfig) { c.retryLimits = &limits }
}

// WithRunModel uses model for one run without changing the agent default.
func WithRunModel(model Model) RunOption {
	if modelIsNil(model) {
		panic("ai: run model must not be nil")
	}
	return func(c *runConfig) { c.model = model }
}

// WithRunModelID resolves modelID once for this run. It cannot be combined
// with WithRunModel or WithRunModelSelector.
func WithRunModelID(modelID string) RunOption {
	if modelID == "" {
		panic("ai: run model ID must not be empty")
	}
	return func(c *runConfig) { c.modelID = modelID }
}

// WithRunModelSelector selects the model before every request in one run.
// It replaces agent and capability selectors for that run.
func WithRunModelSelector[Deps any](fn ModelSelectorFunc[Deps]) RunOption {
	return func(c *runConfig) {
		c.modelSelectors = append(c.modelSelectors, func(ctx context.Context, value any) (ModelSelection, error) {
			selection, ok := value.(ModelSelectionContext[Deps])
			if !ok {
				return ModelSelection{}, fmt.Errorf("ai: run model selector dependencies do not match agent")
			}
			return fn(ctx, selection)
		})
	}
}

// WithRunID sets the unique ID recorded on messages created by one run.
func WithRunID(runID string) RunOption {
	if runID == "" {
		panic("ai: run ID must not be empty")
	}
	return func(c *runConfig) { c.runID = runID }
}

// WithConversationID sets the conversation shared by related runs. Use
// "new" to ignore an ID inherited from message history.
func WithConversationID(conversationID string) RunOption {
	if conversationID == "" {
		panic("ai: conversation ID must not be empty")
	}
	return func(c *runConfig) { c.conversationID = &conversationID }
}

// WithRunMetadata merges application metadata over agent metadata for one run.
func WithRunMetadata(metadata map[string]any) RunOption {
	metadata = cloneSchemaMap(metadata)
	return func(config *runConfig) { config.metadata = cloneSchemaMap(metadata) }
}

// WithRunMetadataFunc appends dynamic metadata for one run. Later metadata
// layers override earlier keys.
func WithRunMetadataFunc[Deps any](fn RunMetadataFunc[Deps]) RunOption {
	if fn == nil {
		panic("ai: run metadata function must not be nil")
	}
	return func(config *runConfig) {
		config.metadataFuncs = append(config.metadataFuncs, func(ctx context.Context, value any) (map[string]any, error) {
			runContext, ok := value.(*RunContext[Deps])
			if !ok {
				return nil, fmt.Errorf("ai: run metadata dependencies do not match agent")
			}
			return fn(ctx, runContext)
		})
	}
}

// WithRunModelSettings merges settings over the agent defaults for one run.
// Non-zero scalar values, non-nil pointers, and non-nil slices override the
// corresponding defaults.
func WithRunModelSettings(settings ModelSettings) RunOption {
	return func(c *runConfig) { c.settings = &settings }
}

// WithRunModelSettingsFunc appends dynamic settings for one run. The callback
// runs before every model request and sees agent and capability settings in
// rc.ModelSettings.
func WithRunModelSettingsFunc[Deps any](fn ModelSettingsFunc[Deps]) RunOption {
	return func(c *runConfig) {
		c.settingsFuncs = append(c.settingsFuncs, func(ctx context.Context, rc any) (ModelSettings, error) {
			typed, ok := rc.(*RunContext[Deps])
			if !ok {
				return ModelSettings{}, fmt.Errorf("ai: run model settings dependencies do not match agent")
			}
			return fn(ctx, typed)
		})
	}
}

// WithRunInstructions appends static instructions for one run.
func WithRunInstructions(instructions string) RunOption {
	return func(c *runConfig) { c.instructions = instructions }
}

// WithRunInstructionParts appends unaddressable literal blocks for one run.
// Declared names are preserved, but per-run instructions have no stable source.
func WithRunInstructionParts(parts ...InstructionPart) RunOption {
	parts = cloneInstructionParts(parts)
	for index := range parts {
		parts[index].ID = nil
	}
	qualified, err := qualifyInstructionParts(parts, nil)
	if err != nil {
		panic(err.Error())
	}
	return func(config *runConfig) {
		config.instructionParts = append(config.instructionParts, cloneInstructionParts(qualified)...)
	}
}

// WithRunInstructionsFunc appends dynamic instructions for one run. The
// callback runs before every model request.
func WithRunInstructionsFunc[Deps any](fn InstructionsFunc[Deps]) RunOption {
	return func(c *runConfig) {
		c.instructionsFuncs = append(c.instructionsFuncs, func(ctx context.Context, rc any) (string, error) {
			typed, ok := rc.(*RunContext[Deps])
			if !ok {
				return "", fmt.Errorf("ai: run instructions dependencies do not match agent")
			}
			return fn(ctx, typed)
		})
	}
}

// WithRunUsageLimits replaces the agent usage limits for one run. The zero
// value disables agent-level limits for that run.
func WithRunUsageLimits(limits UsageLimits) RunOption {
	return func(c *runConfig) { c.usageLimits = &limits }
}

// WithRunOutputMode selects the structured-output mode for one run.
func WithRunOutputMode(mode OutputMode) RunOption {
	return func(c *runConfig) { c.outputMode = &mode }
}

// WithRunPromptedOutputTemplate replaces prompted-output instructions for one run.
func WithRunPromptedOutputTemplate(template string) RunOption {
	if template == "" {
		panic("ai: prompted output template must not be empty")
	}
	return func(c *runConfig) { c.promptedTemplate = &template }
}

// WithRunOutputTool customizes the structured output tool for one run.
func WithRunOutputTool(outputTool OutputToolConfig) RunOption {
	outputTool = cloneOutputToolConfig(outputTool)
	validateOutputToolConfig(outputTool)
	return func(c *runConfig) { c.outputTool = &outputTool }
}

func cloneOutputToolConfig(config OutputToolConfig) OutputToolConfig {
	config.Strict = clonePointer(config.Strict)
	config.MaxRetries = clonePointer(config.MaxRetries)
	return config
}

func validateOutputToolConfig(config OutputToolConfig) {
	if config.MaxRetries != nil && *config.MaxRetries < 0 {
		panic(fmt.Sprintf("ai: output tool max retries must be non-negative, got %d", *config.MaxRetries))
	}
}

func validateOutputMode(mode OutputMode) {
	if mode != OutputModeTool && mode != OutputModeNative && mode != OutputModePrompted && mode != OutputModeAuto {
		panic(fmt.Sprintf("ai: invalid output mode %d", mode))
	}
}

// RunResult is the outcome of a successful run.
type RunResult[Output any] struct {
	Output Output

	usage       Usage
	messages    []ModelMessage
	newMessages int
	metadata    map[string]any
	deferred    *DeferredToolRequests
}

// Metadata returns detached application metadata resolved after the run.
func (r *RunResult[Output]) Metadata() map[string]any { return cloneSchemaMap(r.metadata) }

// Deferred returns pending external calls and approvals, or nil when Output
// contains the completed result.
func (r *RunResult[Output]) Deferred() *DeferredToolRequests {
	if r.deferred == nil {
		return nil
	}
	cloned := r.deferred.Clone()
	return &cloned
}

// Usage returns the tokens and requests consumed by the run.
func (r *RunResult[Output]) Usage() Usage { return r.usage.Clone() }

// Messages returns the full conversation, including any history passed in.
func (r *RunResult[Output]) Messages() []ModelMessage { return r.messages }

// NewMessages returns only the messages produced by this run.
func (r *RunResult[Output]) NewMessages() []ModelMessage {
	return r.messages[r.newMessages:]
}
