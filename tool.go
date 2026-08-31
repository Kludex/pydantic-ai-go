package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Kludex/pydantic-ai-go/internal/schema"
)

// RunContext carries run-scoped data into tools and dynamic hooks. The
// context.Context argument remains the cancellation signal carrier.
type RunContext[Deps any] struct {
	Deps             Deps
	Retry            int
	MaxRetries       int
	RunID            string
	ConversationID   string
	ToolName         string
	ToolCallID       string
	ToolCallApproved bool
	ToolCallMetadata map[string]any
	PartialOutput    bool
	Model            Model
	ModelID          string
	RunStep          int
	ModelSettings    ModelSettings
	UsageLimits      UsageLimits

	usage         *Usage
	toolCalls     *atomic.Int64
	messages      *[]ModelMessage
	revealedTools *map[string]struct{}
	cancellation  *runCancellation
}

// Usage returns the usage accumulated so far in this run.
func (rc *RunContext[Deps]) Usage() Usage {
	usage := rc.usage.Clone()
	usage.ToolCalls = int(rc.toolCalls.Load())
	return usage
}

// Messages returns the conversation so far in this run.
func (rc *RunContext[Deps]) Messages() []ModelMessage { return *rc.messages }

// RevealedTools returns model-facing deferred tool names revealed in this run
// or its resumed history, sorted for deterministic inspection.
func (rc *RunContext[Deps]) RevealedTools() []string {
	if rc.revealedTools == nil {
		return nil
	}
	names := make([]string, 0, len(*rc.revealedTools))
	for name := range *rc.revealedTools {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}

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

// Tool is a reusable typed function tool. Create one with NewTool and add it
// to an agent or one run. Tool values are safe to reuse concurrently when
// their callbacks are safe to invoke concurrently.
type Tool[Deps any] struct {
	entry toolEntry[Deps]
}

// Definition returns a detached copy of the definition shown to models.
func (t Tool[Deps]) Definition() ToolDefinition { return cloneToolDefinition(t.entry.def) }

// NewTool creates a reusable function tool. The argument schema is reflected
// from Args and validated before fn runs.
func NewTool[Deps, Args, Result any](
	name string,
	fn func(ctx context.Context, rc *RunContext[Deps], args Args) (Result, error),
	opts ...ToolOption,
) Tool[Deps] {
	return newReflectedTool(name, fn, nil, nil, opts)
}

// NewToolWithArgsValidator creates a reusable function tool with a typed
// semantic validator that runs after decoding and before execution.
func NewToolWithArgsValidator[Deps, Args, Result any](
	name string,
	fn func(ctx context.Context, rc *RunContext[Deps], args Args) (Result, error),
	validate ArgsValidator[Deps, Args],
	opts ...ToolOption,
) Tool[Deps] {
	return newReflectedTool(name, fn, validate, nil, opts)
}

// NewPreparedTool creates a reusable function tool with per-step preparation.
func NewPreparedTool[Deps, Args, Result any](
	name string,
	fn func(ctx context.Context, rc *RunContext[Deps], args Args) (Result, error),
	prepare ToolPrepareFunc[Deps],
	opts ...ToolOption,
) Tool[Deps] {
	return newReflectedTool(name, fn, nil, prepare, opts)
}

// NewPreparedToolWithArgsValidator combines typed argument validation with
// per-step preparation on a reusable tool.
func NewPreparedToolWithArgsValidator[Deps, Args, Result any](
	name string,
	fn func(ctx context.Context, rc *RunContext[Deps], args Args) (Result, error),
	validate ArgsValidator[Deps, Args],
	prepare ToolPrepareFunc[Deps],
	opts ...ToolOption,
) Tool[Deps] {
	return newReflectedTool(name, fn, validate, prepare, opts)
}

// NewExternalTool creates a typed tool whose calls are returned to the caller
// instead of executed in the agent process.
func NewExternalTool[Deps, Args, Result any](name string, opts ...ToolOption) Tool[Deps] {
	def := toolDefinition[Args](name, opts)
	if def.ReturnSchema == nil {
		def.ReturnSchema = reflectedToolReturnSchema(reflect.TypeFor[Result]())
	}
	def.ExternalExecution = true
	return Tool[Deps]{entry: toolEntry[Deps]{def: cloneToolDefinition(def)}}
}

// NewSimpleTool creates a reusable tool that needs no run context or dependencies.
func NewSimpleTool[Deps, Args, Result any](
	name string,
	fn func(ctx context.Context, args Args) (Result, error),
	opts ...ToolOption,
) Tool[Deps] {
	return NewTool(name, func(ctx context.Context, _ *RunContext[Deps], args Args) (Result, error) {
		return fn(ctx, args)
	}, opts...)
}

// NewSimpleToolWithArgsValidator creates a reusable context-free tool with
// a typed validator that can inspect the run context.
func NewSimpleToolWithArgsValidator[Deps, Args, Result any](
	name string,
	fn func(ctx context.Context, args Args) (Result, error),
	validate ArgsValidator[Deps, Args],
	opts ...ToolOption,
) Tool[Deps] {
	return NewToolWithArgsValidator(name, func(
		ctx context.Context, _ *RunContext[Deps], args Args,
	) (Result, error) {
		return fn(ctx, args)
	}, validate, opts...)
}

// AddTool registers a tool on the agent. Registration panics after the
// agent's first run. Invalid arguments and Retryf errors become retry prompts.
func AddTool[Deps, Output, Args, Result any](
	a *Agent[Deps, Output],
	name string,
	fn func(ctx context.Context, rc *RunContext[Deps], args Args) (Result, error),
	opts ...ToolOption,
) {
	a.AddTool(NewTool(name, fn, opts...))
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
	a.AddTool(NewToolWithArgsValidator(name, fn, validate, opts...))
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
	a.AddTool(NewPreparedTool(name, fn, prepare, opts...))
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
	a.AddTool(NewPreparedToolWithArgsValidator(name, fn, validate, prepare, opts...))
}

func newReflectedTool[Deps, Args, Result any](
	name string,
	fn func(ctx context.Context, rc *RunContext[Deps], args Args) (Result, error),
	validate ArgsValidator[Deps, Args],
	prepare ToolPrepareFunc[Deps],
	opts []ToolOption,
) Tool[Deps] {
	def := toolDefinition[Args](name, opts)
	if def.ReturnSchema == nil {
		def.ReturnSchema = reflectedToolReturnSchema(reflect.TypeFor[Result]())
	}
	call := func(ctx context.Context, rc *RunContext[Deps], rawArgs json.RawMessage) (any, error) {
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
	}
	return Tool[Deps]{entry: toolEntry[Deps]{def: cloneToolDefinition(def), call: call, prepare: prepare}}
}

// AddExternalTool registers a typed externally executed tool.
func AddExternalTool[Deps, Output, Args, Result any](
	a *Agent[Deps, Output], name string, opts ...ToolOption,
) {
	a.AddTool(NewExternalTool[Deps, Args, Result](name, opts...))
}

// AddSimpleTool registers a tool that needs no run context or deps.
func AddSimpleTool[Deps, Output, Args, Result any](
	a *Agent[Deps, Output],
	name string,
	fn func(ctx context.Context, args Args) (Result, error),
	opts ...ToolOption,
) {
	a.AddTool(NewSimpleTool[Deps](name, fn, opts...))
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
	a.AddTool(NewSimpleToolWithArgsValidator[Deps](name, fn, validate, opts...))
}

// NewRawTool creates a reusable tool from an explicit definition, skipping
// schema reflection. It is the escape hatch for dynamic tools.
func NewRawTool[Deps any](
	def ToolDefinition,
	fn func(ctx context.Context, rawArgs json.RawMessage) (any, error),
	opts ...ToolOption,
) Tool[Deps] {
	return NewRawToolWithArgsValidator[Deps](def, fn, nil, opts...)
}

// NewRawToolWithArgsValidator creates a reusable raw-schema tool with a
// validator that can inspect the run context and unmodified JSON arguments.
func NewRawToolWithArgsValidator[Deps any](
	def ToolDefinition,
	fn func(ctx context.Context, rawArgs json.RawMessage) (any, error),
	validate ArgsValidator[Deps, json.RawMessage],
	opts ...ToolOption,
) Tool[Deps] {
	for _, opt := range opts {
		opt(&def)
	}
	call := func(ctx context.Context, rc *RunContext[Deps], rawArgs json.RawMessage) (any, error) {
		if validate != nil {
			if err := validate(ctx, rc, rawArgs); err != nil {
				return nil, err
			}
		}
		return fn(ctx, rawArgs)
	}
	return Tool[Deps]{entry: toolEntry[Deps]{def: cloneToolDefinition(def), call: call}}
}

// NewRawExternalTool creates an externally executed tool from an explicit definition.
func NewRawExternalTool[Deps any](def ToolDefinition, opts ...ToolOption) Tool[Deps] {
	for _, opt := range opts {
		opt(&def)
	}
	def.ExternalExecution = true
	return Tool[Deps]{entry: toolEntry[Deps]{def: cloneToolDefinition(def)}}
}

// AddRawTool registers a tool from an explicit definition, skipping schema reflection.
func (a *Agent[Deps, Output]) AddRawTool(
	def ToolDefinition,
	fn func(ctx context.Context, rawArgs json.RawMessage) (any, error),
	opts ...ToolOption,
) {
	a.AddTool(NewRawTool[Deps](def, fn, opts...))
}

// AddRawToolWithArgsValidator registers a raw-schema tool with validation
// that can inspect the run context and unmodified JSON arguments.
func (a *Agent[Deps, Output]) AddRawToolWithArgsValidator(
	def ToolDefinition,
	fn func(ctx context.Context, rawArgs json.RawMessage) (any, error),
	validate ArgsValidator[Deps, json.RawMessage],
	opts ...ToolOption,
) {
	a.AddTool(NewRawToolWithArgsValidator[Deps](def, fn, validate, opts...))
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

// WithReturnSchema overrides the reflected local schema for a tool's return
// value. Use it for rich ToolReturn values whose inner type is not reflected.
func WithReturnSchema(returnSchema map[string]any) ToolOption {
	return func(d *ToolDefinition) { d.ReturnSchema = cloneSchemaMap(returnSchema) }
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

// WithDeferredLoading hides a tool until another tool reveals its name through
// ToolReturn.Tools. Hidden calls are rejected as unavailable.
func WithDeferredLoading() ToolOption {
	return func(d *ToolDefinition) { d.DeferLoading = true }
}

// WithApprovalRequired pauses before local execution until a caller approves.
func WithApprovalRequired() ToolOption {
	return func(d *ToolDefinition) { d.RequiresApproval = true }
}

// WithApprovalMetadata adds detached context to pending approval requests.
// It is local and is not included in provider tool definitions.
func WithApprovalMetadata(metadata map[string]any) ToolOption {
	return func(d *ToolDefinition) { d.ApprovalMetadata = cloneSchemaMap(metadata) }
}

// WithDynamicApproval allows a function to return RequestToolApproval based
// on its validated arguments or run context.
func WithDynamicApproval() ToolOption {
	return func(d *ToolDefinition) { d.DynamicApproval = true }
}

// WithExternalExecution returns calls for execution outside the agent process.
// Prefer NewExternalTool or NewRawExternalTool when no local function exists.
func WithExternalExecution() ToolOption {
	return func(d *ToolDefinition) { d.ExternalExecution = true }
}

// WithDynamicExternalExecution allows a function to return
// RequestExternalToolExecution for selected calls.
func WithDynamicExternalExecution() ToolOption {
	return func(d *ToolDefinition) { d.DynamicExternalExecution = true }
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

func reflectedToolReturnSchema(resultType reflect.Type) map[string]any {
	for resultType.Kind() == reflect.Pointer {
		resultType = resultType.Elem()
	}
	if resultType == reflect.TypeFor[ToolReturn]() || resultType.Kind() == reflect.Interface {
		return nil
	}
	resultSchema, err := schema.ForType(resultType)
	if err != nil {
		return nil
	}
	return resultSchema
}

func containsNestedToolReturn(value any) bool {
	valueType := reflect.TypeOf(value)
	if valueType == nil || valueType.Kind() != reflect.Array && valueType.Kind() != reflect.Slice {
		return false
	}
	items := reflect.ValueOf(value)
	for index := range items.Len() {
		item := items.Index(index)
		if item.Kind() == reflect.Interface {
			item = item.Elem()
		}
		if item.IsValid() && (item.Type() == reflect.TypeFor[ToolReturn]() ||
			item.Type() == reflect.TypeFor[*ToolReturn]()) {
			return true
		}
	}
	return false
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
