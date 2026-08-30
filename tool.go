package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Kludex/pydantic-ai-go/internal/schema"
)

// RunContext carries run-scoped data into tools and dynamic hooks. The
// context.Context argument remains the cancellation signal carrier.
type RunContext[Deps any] struct {
	Deps          Deps
	Retry         int
	MaxRetries    int
	RunID         string
	ToolCallID    string
	PartialOutput bool
	Model         Model
	ModelSettings ModelSettings
	UsageLimits   UsageLimits

	usage        *Usage
	toolCalls    *atomic.Int64
	messages     *[]ModelMessage
	cancellation *runCancellation
}

// Usage returns the usage accumulated so far in this run.
func (rc *RunContext[Deps]) Usage() Usage {
	usage := *rc.usage
	usage.ToolCalls = int(rc.toolCalls.Load())
	return usage
}

// Messages returns the conversation so far in this run.
func (rc *RunContext[Deps]) Messages() []ModelMessage { return *rc.messages }

// Cancel requests cancellation of this run. In-flight model and tool calls
// receive cancellation through their context.Context. Concurrent tool calls
// are drained before the run returns a RunCancelledError. Cancel is
// idempotent and becomes a no-op after the run ends.
func (rc *RunContext[Deps]) Cancel() {
	if rc.cancellation != nil {
		rc.cancellation.cancelRun()
	}
}

type runCancellation struct {
	mutex  sync.Mutex
	cancel context.CancelCauseFunc
	active bool
}

func (c *runCancellation) cancelRun() {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	if c.active {
		c.cancel(ErrRunCancelled)
	}
}

func (c *runCancellation) stopStream() {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	if c.active {
		c.cancel(context.Canceled)
	}
}

func (c *runCancellation) finish() {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.active = false
	c.cancel(nil)
}

// ToolPrepareFunc customizes one tool definition before each model request.
// Return nil to omit the tool for that step.
type ToolPrepareFunc[Deps any] func(
	ctx context.Context, rc *RunContext[Deps], tool ToolDefinition,
) (*ToolDefinition, error)

// ArgsValidator validates typed tool arguments before the tool runs. Return
// an error from Retryf to ask the model for corrected arguments.
type ArgsValidator[Deps, Args any] func(
	ctx context.Context, rc *RunContext[Deps], args Args,
) error

// AddTool registers a tool on the agent. The argument schema is reflected
// from the Args struct's `json` and `jsonschema` tags. Registration panics
// after the agent's first run.
//
// If the model sends arguments that fail to unmarshal, the error is sent
// back to the model as a retry prompt instead of failing the run. The same
// happens when fn returns an error created with Retryf; any other error
// aborts the run.
func AddTool[Deps, Output, Args, Result any](
	a *Agent[Deps, Output],
	name string,
	fn func(ctx context.Context, rc *RunContext[Deps], args Args) (Result, error),
	opts ...ToolOption,
) {
	addReflectedTool(a, name, fn, nil, nil, opts)
}

// AddToolWithArgsValidator registers a tool with a typed semantic validator.
// JSON decoding succeeds before validate runs. The tool function runs only
// after validate returns nil.
func AddToolWithArgsValidator[Deps, Output, Args, Result any](
	a *Agent[Deps, Output],
	name string,
	fn func(ctx context.Context, rc *RunContext[Deps], args Args) (Result, error),
	validate ArgsValidator[Deps, Args],
	opts ...ToolOption,
) {
	addReflectedTool(a, name, fn, validate, nil, opts)
}

// AddPreparedTool registers a tool with a per-step preparation callback.
// Preparation receives a fresh definition and runs before agent-wide tool
// preparation. Returning nil omits this tool for the current model request.
func AddPreparedTool[Deps, Output, Args, Result any](
	a *Agent[Deps, Output],
	name string,
	fn func(ctx context.Context, rc *RunContext[Deps], args Args) (Result, error),
	prepare ToolPrepareFunc[Deps],
	opts ...ToolOption,
) {
	addReflectedTool(a, name, fn, nil, prepare, opts)
}

// AddPreparedToolWithArgsValidator combines typed argument validation with
// per-step tool-definition preparation.
func AddPreparedToolWithArgsValidator[Deps, Output, Args, Result any](
	a *Agent[Deps, Output],
	name string,
	fn func(ctx context.Context, rc *RunContext[Deps], args Args) (Result, error),
	validate ArgsValidator[Deps, Args],
	prepare ToolPrepareFunc[Deps],
	opts ...ToolOption,
) {
	addReflectedTool(a, name, fn, validate, prepare, opts)
}

func addReflectedTool[Deps, Output, Args, Result any](
	a *Agent[Deps, Output],
	name string,
	fn func(ctx context.Context, rc *RunContext[Deps], args Args) (Result, error),
	validate ArgsValidator[Deps, Args],
	prepare ToolPrepareFunc[Deps],
	opts []ToolOption,
) {
	def := toolDefinition[Args](name, opts)
	a.addPreparedTool(def, func(ctx context.Context, rc *RunContext[Deps], rawArgs json.RawMessage) (any, error) {
		var args Args
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return nil, Retryf("invalid arguments for tool %q: %v", name, err)
		}
		if validate != nil {
			if err := validate(ctx, rc, args); err != nil {
				return nil, err
			}
		}
		return fn(ctx, rc, args)
	}, prepare)
}

// AddSimpleTool registers a tool that needs no run context or deps.
func AddSimpleTool[Deps, Output, Args, Result any](
	a *Agent[Deps, Output],
	name string,
	fn func(ctx context.Context, args Args) (Result, error),
	opts ...ToolOption,
) {
	AddTool(a, name, func(ctx context.Context, _ *RunContext[Deps], args Args) (Result, error) {
		return fn(ctx, args)
	}, opts...)
}

// AddSimpleToolWithArgsValidator registers a context-free tool with a typed
// argument validator that can inspect the run context.
func AddSimpleToolWithArgsValidator[Deps, Output, Args, Result any](
	a *Agent[Deps, Output],
	name string,
	fn func(ctx context.Context, args Args) (Result, error),
	validate ArgsValidator[Deps, Args],
	opts ...ToolOption,
) {
	AddToolWithArgsValidator(a, name, func(
		ctx context.Context, _ *RunContext[Deps], args Args,
	) (Result, error) {
		return fn(ctx, args)
	}, validate, opts...)
}

// AddRawTool registers a tool from an explicit definition, skipping schema
// reflection. It is the escape hatch for dynamic tools (MCP, config-driven).
func (a *Agent[Deps, Output]) AddRawTool(
	def ToolDefinition,
	fn func(ctx context.Context, rawArgs json.RawMessage) (any, error),
	opts ...ToolOption,
) {
	a.AddRawToolWithArgsValidator(def, fn, nil, opts...)
}

// AddRawToolWithArgsValidator registers a raw-schema tool with validation
// that can inspect the run context and unmodified JSON arguments.
func (a *Agent[Deps, Output]) AddRawToolWithArgsValidator(
	def ToolDefinition,
	fn func(ctx context.Context, rawArgs json.RawMessage) (any, error),
	validate ArgsValidator[Deps, json.RawMessage],
	opts ...ToolOption,
) {
	for _, opt := range opts {
		opt(&def)
	}
	a.addTool(def, func(ctx context.Context, rc *RunContext[Deps], rawArgs json.RawMessage) (any, error) {
		if validate != nil {
			if err := validate(ctx, rc, rawArgs); err != nil {
				return nil, err
			}
		}
		return fn(ctx, rawArgs)
	})
}

// ToolOption configures a tool at registration.
type ToolOption func(*ToolDefinition)

// WithDescription sets the tool description shown to the model.
func WithDescription(description string) ToolOption {
	return func(d *ToolDefinition) { d.Description = description }
}

// WithToolMetadata attaches local metadata for preparation and filtering
// hooks. Providers do not receive it.
func WithToolMetadata(metadata map[string]any) ToolOption {
	return func(d *ToolDefinition) { d.Metadata = cloneSchemaMap(metadata) }
}

// WithSequential makes a tool an execution barrier. Independent tools run
// concurrently by default. Calls before this tool finish first, this tool
// runs alone, and later calls start afterward.
func WithSequential() ToolOption {
	return func(d *ToolDefinition) { d.Sequential = true }
}

// WithStrict forces provider-native schema enforcement for tool arguments.
func WithStrict() ToolOption {
	strict := true
	return func(d *ToolDefinition) { d.Strict = &strict }
}

// WithoutStrict disables provider-native schema enforcement for this tool.
func WithoutStrict() ToolOption {
	strict := false
	return func(d *ToolDefinition) { d.Strict = &strict }
}

// WithToolMaxRetries overrides the function-tool retry budget for this tool.
func WithToolMaxRetries(n int) ToolOption {
	if n < 0 {
		panic(fmt.Sprintf("ai: tool max retries must be non-negative, got %d", n))
	}
	return func(d *ToolDefinition) { d.maxRetries = &n }
}

// WithToolTimeout sets the maximum duration of one tool call. The tool must
// honor context cancellation. A timeout becomes a retry prompt and consumes
// this tool's retry budget.
func WithToolTimeout(timeout time.Duration) ToolOption {
	if timeout <= 0 {
		panic(fmt.Sprintf("ai: tool timeout must be positive, got %s", timeout))
	}
	return func(d *ToolDefinition) { d.timeout = timeout }
}

func toolDefinition[Args any](name string, opts []ToolOption) ToolDefinition {
	s, err := schema.For(reflect.TypeFor[Args]())
	if err != nil {
		panic(fmt.Sprintf("ai: tool %q: %v", name, err))
	}
	def := ToolDefinition{Name: name, Schema: s}
	for _, opt := range opts {
		opt(&def)
	}
	return def
}
