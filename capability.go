package ai

import (
	"context"
	"encoding/json"
	"sync/atomic"
)

// A Capability is a reusable, composable unit of agent behavior. Setup runs
// once per agent at registration and contributes tools, instructions, and
// settings. A capability opts into loop interception by also implementing
// any of RunWrapper, ModelRequestWrapper, ToolCallWrapper,
// RunEventStreamWrapper, StreamEventProcessor, or InstructionsProvider - the
// agent discovers them by type assertion.
//
// Capabilities are untyped so one implementation works with every agent
// regardless of its Deps and Output types.
type Capability interface {
	Setup(reg *CapabilityRegistry) error
}

// CapabilityRegistry collects what a capability contributes at setup.
type CapabilityRegistry struct {
	tools        []capabilityTool
	instructions []string
}

type capabilityTool struct {
	def  ToolDefinition
	call func(ctx context.Context, rawArgs json.RawMessage) (any, error)
}

// AddTool registers a tool from an explicit definition, like Agent.AddRawTool.
func (r *CapabilityRegistry) AddTool(def ToolDefinition, fn func(ctx context.Context, rawArgs json.RawMessage) (any, error)) {
	r.tools = append(r.tools, capabilityTool{def: def, call: fn})
}

// AddInstructions appends static instructions to the agent's.
func (r *CapabilityRegistry) AddInstructions(instructions string) {
	r.instructions = append(r.instructions, instructions)
}

// RunInfo is the untyped view of a run that capabilities receive. It is the
// erased counterpart of RunContext.
type RunInfo struct {
	RunID string

	usage     *Usage
	toolCalls *atomic.Int64
	messages  *[]ModelMessage
}

// Usage returns the usage accumulated so far in this run.
func (ri *RunInfo) Usage() Usage {
	usage := *ri.usage
	usage.ToolCalls = int(ri.toolCalls.Load())
	return usage
}

// Messages returns the conversation so far in this run.
func (ri *RunInfo) Messages() []ModelMessage { return *ri.messages }

// ModelRequestFunc continues the model-request chain.
type ModelRequestFunc func(ctx context.Context, msgs []ModelMessage, params ModelRequestParams) (*ModelResponse, error)

// ModelRequestWrapper intercepts every model request. Implementations call
// next to continue; they may modify messages and params on the way in and
// the response on the way out.
type ModelRequestWrapper interface {
	WrapModelRequest(ctx context.Context, ri *RunInfo, msgs []ModelMessage, params ModelRequestParams, next ModelRequestFunc) (*ModelResponse, error)
}

// ToolCallFunc continues the tool-call chain.
type ToolCallFunc func(ctx context.Context, call ToolCallPart) (any, error)

// ToolCallWrapper intercepts every tool execution. Implementations call
// next to continue; returning an error created with Retryf sends a retry
// prompt to the model instead of failing the run. Independent tool calls
// invoke this method concurrently, so implementations must synchronize
// mutable state.
type ToolCallWrapper interface {
	WrapToolCall(ctx context.Context, ri *RunInfo, call ToolCallPart, next ToolCallFunc) (any, error)
}

// RunFunc continues the run chain.
type RunFunc func(ctx context.Context) error

// RunWrapper intercepts the whole run. Implementations call next to
// continue; an error aborts the run.
type RunWrapper interface {
	WrapRun(ctx context.Context, ri *RunInfo, next RunFunc) error
}

// InstructionsProvider contributes per-run instructions, evaluated at the
// start of every run and appended to the agent's.
type InstructionsProvider interface {
	Instructions(ctx context.Context, ri *RunInfo) (string, error)
}

// RunEventStreamWrapper transforms the consumer-facing event stream. Changes
// do not affect accumulated messages, tool execution, or final output. The
// wrapper must consume stream and stop when its downstream yield returns false.
type RunEventStreamWrapper interface {
	WrapRunEventStream(ctx context.Context, ri *RunInfo, stream EventStream) EventStream
}

// StreamEventProcessor is a per-event shorthand for RunEventStreamWrapper.
// Return nil to hide an event from the consumer. An error stops the stream.
// If a capability implements both interfaces, RunEventStreamWrapper wins.
type StreamEventProcessor interface {
	ProcessStreamEvent(ctx context.Context, ri *RunInfo, event StreamEvent) (StreamEvent, error)
}

// WithCapabilities registers capabilities on the agent. Slice order is
// middleware order: the first capability is outermost.
func WithCapabilities(caps ...Capability) Option {
	return func(c *config) { c.capabilities = append(c.capabilities, caps...) }
}
