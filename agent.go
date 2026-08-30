package ai

import (
	"context"
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
	settings          ModelSettings
	limits            UsageLimits
	maxRetries        int
	outputValidators  []func(ctx context.Context, rc *RunContext[Deps], out Output) error

	tools   []toolEntry[Deps]
	started atomic.Bool
}

type toolEntry[Deps any] struct {
	def  ToolDefinition
	call toolFunc[Deps]
}

// NewAgent creates an agent backed by model.
func NewAgent[Deps, Output any](model Model, opts ...Option) *Agent[Deps, Output] {
	a := &Agent[Deps, Output]{model: model, maxRetries: 1}
	var cfg config
	for _, opt := range opts {
		opt(&cfg)
	}
	a.instructions = cfg.instructions
	a.settings = cfg.settings
	a.limits = cfg.limits
	if cfg.maxRetries > 0 {
		a.maxRetries = cfg.maxRetries
	}
	return a
}

// AddInstructionsFunc registers dynamic instructions evaluated at the start
// of every run and appended to the static instructions.
func (a *Agent[Deps, Output]) AddInstructionsFunc(fn func(ctx context.Context, rc *RunContext[Deps]) (string, error)) {
	a.checkNotStarted()
	a.instructionsFuncs = append(a.instructionsFuncs, fn)
}

// AddOutputValidator registers a semantic check on the final output.
// Return an error from Retryf to send the failure back to the model.
func (a *Agent[Deps, Output]) AddOutputValidator(fn func(ctx context.Context, rc *RunContext[Deps], out Output) error) {
	a.checkNotStarted()
	a.outputValidators = append(a.outputValidators, fn)
}

func (a *Agent[Deps, Output]) addTool(def ToolDefinition, fn toolFunc[Deps]) {
	a.checkNotStarted()
	a.tools = append(a.tools, toolEntry[Deps]{def: def, call: fn})
}

func (a *Agent[Deps, Output]) checkNotStarted() {
	if a.started.Load() {
		panic("ai: cannot modify an agent after its first run")
	}
}

// Option configures an Agent at construction.
type Option func(*config)

type config struct {
	instructions string
	settings     ModelSettings
	limits       UsageLimits
	maxRetries   int
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
