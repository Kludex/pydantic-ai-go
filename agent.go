package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"sync/atomic"
)

// Agent is a typed LLM agent. Deps is the dependency type passed to tools
// and dynamic instructions; Output is the final result type. Use string
// for plain-text output; any other type produces structured output via
// an output tool whose schema is reflected from Output.
type Agent[Deps, Output any] struct {
	model              Model
	instructions       string
	instructionsFuncs  []InstructionsFunc[Deps]
	systemPrompts      []string
	systemPromptFuncs  []systemPromptRunner[Deps]
	modelSettingsFuncs []ModelSettingsFunc[Deps]
	modelSelectors     []ModelSelectorFunc[Deps]
	modelIDResolvers   []ModelIDResolverFunc[Deps]
	toolsPrepareFuncs  []ToolsPrepareFunc[Deps]
	settings           ModelSettings
	usageLimits        UsageLimits
	retryLimits        RetryLimits
	outputMode         OutputMode
	outputTool         OutputToolConfig
	outputToolPrepare  []OutputToolPrepareFunc[Deps]
	endStrategy        EndStrategy
	sequentialTools    bool
	capabilities       []Capability
	capInstructions    []string
	capSettings        []capabilitySettingsLayer
	outputValidators   []func(ctx context.Context, rc *RunContext[Deps], out Output) error

	tools    []toolEntry[Deps]
	toolsets []Toolset[Deps]
	started  atomic.Bool
}

type toolEntry[Deps any] struct {
	def     ToolDefinition
	call    toolFunc[Deps]
	prepare ToolPrepareFunc[Deps]
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
		model: model, retryLimits: RetryLimits{Tools: 1, Output: 1}, endStrategy: EndStrategyGraceful,
	}
	var cfg config
	for _, opt := range opts {
		opt(&cfg)
	}
	a.instructions = cfg.instructions
	a.systemPrompts = slices.Clone(cfg.systemPrompts)
	a.settings = cfg.settings.Clone()
	a.usageLimits = cfg.limits
	a.outputMode = cfg.outputMode
	a.outputTool = cfg.outputTool
	if cfg.outputTool.Strict != nil {
		strict := *cfg.outputTool.Strict
		a.outputTool.Strict = &strict
	}
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
	a.capabilities = cfg.capabilities
	if cfg.retryLimits != nil {
		validateRetryLimits(*cfg.retryLimits)
		a.retryLimits = *cfg.retryLimits
	}
	for _, capability := range a.capabilities {
		reg := &CapabilityRegistry{}
		if err := capability.Setup(reg); err != nil {
			panic(fmt.Sprintf("ai: capability setup: %v", err))
		}
		a.capInstructions = append(a.capInstructions, reg.instructions...)
		for _, tool := range reg.tools {
			fn := tool.call
			a.addTool(tool.def, func(ctx context.Context, _ *RunContext[Deps], rawArgs json.RawMessage) (any, error) {
				return fn(ctx, rawArgs)
			})
		}
		a.capSettings = append(a.capSettings, capabilitySettingsLayer{
			static: reg.modelSettings, provider: capabilityModelSettingsProvider(capability),
		})
	}
	return a
}

// InstructionsFunc returns instructions for one model-request step.
type InstructionsFunc[Deps any] func(ctx context.Context, rc *RunContext[Deps]) (string, error)

// AddInstructionsFunc registers dynamic instructions evaluated before every
// model request and appended to the static instructions.
func (a *Agent[Deps, Output]) AddInstructionsFunc(fn InstructionsFunc[Deps]) {
	a.checkNotStarted()
	a.instructionsFuncs = append(a.instructionsFuncs, fn)
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

// AddModelSettingsFunc registers dynamic model settings evaluated before
// every model request. Each callback sees prior layers in rc.ModelSettings.
func (a *Agent[Deps, Output]) AddModelSettingsFunc(fn ModelSettingsFunc[Deps]) {
	a.checkNotStarted()
	a.modelSettingsFuncs = append(a.modelSettingsFuncs, fn)
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
	a.addPreparedTool(tool.entry.def, tool.entry.call, tool.entry.prepare)
}

func (a *Agent[Deps, Output]) addTool(def ToolDefinition, fn toolFunc[Deps]) {
	a.addPreparedTool(def, fn, nil)
}

func (a *Agent[Deps, Output]) addPreparedTool(
	def ToolDefinition, fn toolFunc[Deps], prepare ToolPrepareFunc[Deps],
) {
	a.checkNotStarted()
	a.tools = append(a.tools, toolEntry[Deps]{def: cloneToolDefinition(def), call: fn, prepare: prepare})
}

func (a *Agent[Deps, Output]) checkNotStarted() {
	if a.started.Load() {
		panic("ai: cannot modify an agent after its first run")
	}
}

// Option configures an Agent at construction.
type Option func(*config)

type config struct {
	instructions    string
	systemPrompts   []string
	settings        ModelSettings
	limits          UsageLimits
	retryLimits     *RetryLimits
	outputMode      OutputMode
	outputTool      OutputToolConfig
	endStrategy     EndStrategy
	sequentialTools bool
	capabilities    []Capability
}

// OutputMode selects how structured output is requested from the model.
type OutputMode int

const (
	// OutputModeTool asks for structured output via a final output tool
	// the model must call. It works with every provider and is the default.
	OutputModeTool OutputMode = iota
	// OutputModeNative uses the provider's native JSON mode: the model
	// responds with JSON text conforming to the output schema. The
	// provider must support it; validation retries still apply.
	OutputModeNative
)

// WithOutputMode selects how structured output is requested. It has no
// effect when Output is string.
func WithOutputMode(mode OutputMode) Option {
	return func(c *config) { c.outputMode = mode }
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

// WithInstructions sets the static instructions sent with every model request.
func WithInstructions(instructions string) Option {
	return func(c *config) { c.instructions = instructions }
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

type erasedTool struct {
	entry any
}

type runConfig struct {
	history           []ModelMessage
	model             Model
	settings          *ModelSettings
	instructions      string
	usageLimits       *UsageLimits
	retryLimits       *RetryLimits
	outputMode        *OutputMode
	modelID           string
	runID             string
	conversationID    *string
	settingsFuncs     []erasedModelSettingsFunc
	instructionsFuncs []erasedInstructionsFunc
	modelSelectors    []erasedModelSelectorFunc
	tools             []erasedTool
	toolsets          []any
	capabilities      []Capability
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
	return func(c *runConfig) { c.capabilities = append(c.capabilities, capabilities...) }
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

func validateOutputMode(mode OutputMode) {
	if mode != OutputModeTool && mode != OutputModeNative {
		panic(fmt.Sprintf("ai: invalid output mode %d", mode))
	}
}

// RunResult is the outcome of a successful run.
type RunResult[Output any] struct {
	Output Output

	usage       Usage
	messages    []ModelMessage
	newMessages int
}

// Usage returns the tokens and requests consumed by the run.
func (r *RunResult[Output]) Usage() Usage { return r.usage.Clone() }

// Messages returns the full conversation, including any history passed in.
func (r *RunResult[Output]) Messages() []ModelMessage { return r.messages }

// NewMessages returns only the messages produced by this run.
func (r *RunResult[Output]) NewMessages() []ModelMessage {
	return r.messages[r.newMessages:]
}
