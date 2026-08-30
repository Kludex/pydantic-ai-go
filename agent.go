package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"sync/atomic"
)

// Agent is a typed LLM agent. Deps is the dependency type passed to tools
// and dynamic instructions; Output is the final result type. Use string
// for plain-text output; any other type produces structured output via
// an output tool whose schema is reflected from Output.
type Agent[Deps, Output any] struct {
	model             Model
	instructions      string
	instructionsFuncs []func(ctx context.Context, rc *RunContext[Deps]) (string, error)
	toolsPrepareFuncs []ToolsPrepareFunc[Deps]
	settings          ModelSettings
	maxRetries        int
	outputMode        OutputMode
	endStrategy       EndStrategy
	sequentialTools   bool
	capabilities      []Capability
	capInstructions   []string
	outputValidators  []func(ctx context.Context, rc *RunContext[Deps], out Output) error

	tools   []toolEntry[Deps]
	started atomic.Bool
}

type toolEntry[Deps any] struct {
	def     ToolDefinition
	call    toolFunc[Deps]
	prepare ToolPrepareFunc[Deps]
}

// NewAgent creates an agent backed by model.
func NewAgent[Deps, Output any](model Model, opts ...Option) *Agent[Deps, Output] {
	a := &Agent[Deps, Output]{model: model, maxRetries: 1, endStrategy: EndStrategyGraceful}
	var cfg config
	for _, opt := range opts {
		opt(&cfg)
	}
	a.instructions = cfg.instructions
	a.settings = cfg.settings
	a.outputMode = cfg.outputMode
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
	if cfg.limits != (UsageLimits{}) {
		a.capabilities = append([]Capability{usageLimitsCapability{limits: cfg.limits}}, a.capabilities...)
	}
	if cfg.maxRetries > 0 {
		a.maxRetries = cfg.maxRetries
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
	}
	return a
}

// AddInstructionsFunc registers dynamic instructions evaluated at the start
// of every run and appended to the static instructions.
func (a *Agent[Deps, Output]) AddInstructionsFunc(fn func(ctx context.Context, rc *RunContext[Deps]) (string, error)) {
	a.checkNotStarted()
	a.instructionsFuncs = append(a.instructionsFuncs, fn)
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

// AddOutputValidator registers a semantic check on the final output.
// Return an error from Retryf to send the failure back to the model. The
// exhaustive end strategy may invoke validators concurrently for multiple
// output calls, so validators must synchronize mutable state.
func (a *Agent[Deps, Output]) AddOutputValidator(fn func(ctx context.Context, rc *RunContext[Deps], out Output) error) {
	a.checkNotStarted()
	a.outputValidators = append(a.outputValidators, fn)
}

func (a *Agent[Deps, Output]) addTool(def ToolDefinition, fn toolFunc[Deps]) {
	a.addPreparedTool(def, fn, nil)
}

func (a *Agent[Deps, Output]) addPreparedTool(
	def ToolDefinition, fn toolFunc[Deps], prepare ToolPrepareFunc[Deps],
) {
	a.checkNotStarted()
	a.tools = append(a.tools, toolEntry[Deps]{def: def, call: fn, prepare: prepare})
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
	settings        ModelSettings
	limits          UsageLimits
	maxRetries      int
	outputMode      OutputMode
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

// WithInstructions sets the static system instructions.
func WithInstructions(instructions string) Option {
	return func(c *config) { c.instructions = instructions }
}

// WithModelSettings sets default model settings for every request.
func WithModelSettings(settings ModelSettings) Option {
	return func(c *config) { c.settings = settings }
}

// WithUsageLimits bounds every run of the agent.
func WithUsageLimits(limits UsageLimits) Option {
	return func(c *config) { c.limits = limits }
}

// WithMaxRetries caps how many times a tool or the output validation may
// ask the model to retry. The default is 1.
func WithMaxRetries(n int) Option {
	return func(c *config) { c.maxRetries = n }
}

// RunOption configures a single run.
type RunOption func(*runConfig)

type runConfig struct {
	history []ModelMessage
}

// WithMessageHistory prepends prior conversation messages to the run.
func WithMessageHistory(msgs []ModelMessage) RunOption {
	return func(c *runConfig) { c.history = msgs }
}

// RunResult is the outcome of a successful run.
type RunResult[Output any] struct {
	Output Output

	usage       Usage
	messages    []ModelMessage
	newMessages int
}

// Usage returns the tokens and requests consumed by the run.
func (r *RunResult[Output]) Usage() Usage { return r.usage }

// Messages returns the full conversation, including any history passed in.
func (r *RunResult[Output]) Messages() []ModelMessage { return r.messages }

// NewMessages returns only the messages produced by this run.
func (r *RunResult[Output]) NewMessages() []ModelMessage {
	return r.messages[r.newMessages:]
}
