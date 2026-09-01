package ai

import (
	"context"
	"encoding/json"
	"slices"
	"sync"
	"sync/atomic"
)

// A Capability is a reusable, composable unit of agent behavior. Setup runs
// once per agent at registration and contributes tools, instructions, and
// settings. A capability opts into loop interception by also implementing
// lifecycle hooks or wrappers for runs, model requests, tools, outputs, and
// event streams. The agent discovers optional interfaces by type assertion.
//
// Capabilities are untyped so one implementation works with every agent
// regardless of its Deps and Output types.
type Capability interface {
	Setup(reg *CapabilityRegistry) error
}

// CapabilityIDProvider optionally gives instruction contributions a stable
// application source ID.
type CapabilityIDProvider interface {
	CapabilityID() string
}

// CapabilityRegistry collects what a capability contributes at setup.
type CapabilityRegistry struct {
	tools         []capabilityTool
	nativeTools   []NativeTool
	nativeOrLocal []any
	instructions  []InstructionPart
	modelSettings []ModelSettings
}

type capabilityTool struct {
	def  ToolDefinition
	call func(ctx context.Context, rawArgs json.RawMessage) (any, error)
}

// AddTool registers a tool from an explicit definition, like Agent.AddRawTool.
func (r *CapabilityRegistry) AddTool(def ToolDefinition, fn func(ctx context.Context, rawArgs json.RawMessage) (any, error)) {
	r.tools = append(r.tools, capabilityTool{def: def, call: fn})
}

// AddNativeTool registers a provider-executed tool for this capability.
func (r *CapabilityRegistry) AddNativeTool(tool NativeTool) {
	r.nativeTools = append(r.nativeTools, cloneNativeTool(tool))
}

// AddInstructions appends static instructions to the agent's.
func (r *CapabilityRegistry) AddInstructions(instructions string) {
	r.instructions = append(r.instructions, InstructionPart{Content: instructions})
}

// AddInstructionPart appends a named or cache-aware static instruction block.
// A capability ID qualifies the part's name after Setup returns.
func (r *CapabilityRegistry) AddInstructionPart(part InstructionPart) {
	r.instructions = append(r.instructions, part)
}

// AddModelSettings appends static settings between agent and run settings.
func (r *CapabilityRegistry) AddModelSettings(settings ModelSettings) {
	r.modelSettings = append(r.modelSettings, settings)
}

type runMetadataState struct {
	mu    sync.RWMutex
	value map[string]any
}

func (state *runMetadataState) set(value map[string]any) {
	state.mu.Lock()
	defer state.mu.Unlock()
	state.value = cloneSchemaMap(value)
}

func (state *runMetadataState) snapshot() map[string]any {
	state.mu.RLock()
	defer state.mu.RUnlock()
	return cloneSchemaMap(state.value)
}

// RunInfo is the untyped view of a run that capabilities receive. It is the
// erased counterpart of RunContext.
type RunInfo struct {
	RunID          string
	ConversationID string

	agentName        string
	agentDescription string
	usage            *Usage
	toolCalls        *atomic.Int64
	messages         *[]ModelMessage
	newMessages      int
	prompt           UserPromptPart
	metadata         *runMetadataState
	model            func() Model
	systemPrompts    func(context.Context) ([]SystemPromptPart, error)
}

// AgentName returns the configured application agent name.
func (ri *RunInfo) AgentName() string { return ri.agentName }

// AgentDescription returns the description resolved for this run.
func (ri *RunInfo) AgentDescription() string { return ri.agentDescription }

// Usage returns the usage accumulated so far in this run.
func (ri *RunInfo) Usage() Usage {
	usage := ri.usage.Clone()
	usage.ToolCalls = int(ri.toolCalls.Load())
	return usage
}

// Messages returns a detached snapshot of the conversation so far.
func (ri *RunInfo) Messages() []ModelMessage { return cloneModelMessages(*ri.messages) }

// Prompt returns the text or multimodal prompt that started the run.
func (ri *RunInfo) Prompt() UserPromptPart {
	return cloneUserPromptPart(ri.prompt)
}

// Metadata returns detached application metadata resolved for the run.
func (ri *RunInfo) Metadata() map[string]any {
	if ri.metadata == nil {
		return nil
	}
	return ri.metadata.snapshot()
}

// Model returns the model selected for the current request, or nil before selection.
func (ri *RunInfo) Model() Model {
	if ri.model == nil {
		return nil
	}
	return ri.model()
}

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

// ToolCallWrapper intercepts every locally executed tool after initial argument
// validation. Implementations call next to continue. If a wrapper changes Args,
// next validates them again before execution. Returning Retryf sends a retry
// prompt instead of failing the run. Independent calls invoke wrappers
// concurrently, so implementations must synchronize mutable state.
type ToolCallWrapper interface {
	WrapToolCall(ctx context.Context, ri *RunInfo, call ToolCallPart, next ToolCallFunc) (any, error)
}

// InstructionsProvider contributes instructions before every model request.
type InstructionsProvider interface {
	Instructions(ctx context.Context, ri *RunInfo) (string, error)
}

// InstructionPartsProvider contributes named or cache-aware instruction blocks
// before every model request. It takes precedence over InstructionsProvider.
type InstructionPartsProvider interface {
	InstructionParts(ctx context.Context, ri *RunInfo) ([]InstructionPart, error)
}

// ModelSettingsProvider contributes settings before every model request.
// current contains model, agent, and earlier capability settings.
type ModelSettingsProvider interface {
	ModelSettings(ctx context.Context, ri *RunInfo, current ModelSettings) (ModelSettings, error)
}

// ModelSelectionInfo is the untyped capability view of a model-selection
// step. Step starts at 1. Messages is a detached snapshot.
type ModelSelectionInfo struct {
	Model    Model
	ModelID  string
	Step     int
	Messages []ModelMessage
	Usage    Usage
}

// ModelSelectionProvider adaptively contributes a model. Return the zero
// ModelSelection to leave the current selection unchanged. Later capability
// providers take precedence.
type ModelSelectionProvider interface {
	SelectModel(ctx context.Context, ri *RunInfo, selection ModelSelectionInfo) (ModelSelection, error)
}

// ModelIDResolver resolves application model IDs for a capability. Return
// nil, nil to defer to the next resolver.
type ModelIDResolver interface {
	ResolveModelID(ctx context.Context, ri *RunInfo, modelID string) (Model, error)
}

type capabilitySettingsLayer struct {
	static   []ModelSettings
	provider ModelSettingsProvider
}

func capabilityModelSettingsProvider(capability Capability) ModelSettingsProvider {
	provider, _ := capability.(ModelSettingsProvider)
	return provider
}

// DeferredToolCallHandler resolves any subset of pending external calls and
// approvals inline. Return nil to leave every request for the next handler or
// caller. Handlers run in capability order.
type DeferredToolCallHandler interface {
	HandleDeferredToolCalls(
		ctx context.Context, ri *RunInfo, requests DeferredToolRequests,
	) (*DeferredToolResults, error)
}

// DeferredToolHandlerFunc adapts a function into a capability that only
// handles deferred tool calls.
type DeferredToolHandlerFunc func(
	ctx context.Context, ri *RunInfo, requests DeferredToolRequests,
) (*DeferredToolResults, error)

// Setup implements Capability.
func (DeferredToolHandlerFunc) Setup(*CapabilityRegistry) error { return nil }

// HandleDeferredToolCalls calls the adapted function.
func (f DeferredToolHandlerFunc) HandleDeferredToolCalls(
	ctx context.Context, ri *RunInfo, requests DeferredToolRequests,
) (*DeferredToolResults, error) {
	return f(ctx, ri, requests)
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

type combinedCapability struct {
	capabilities []Capability
}

func (combinedCapability) Setup(*CapabilityRegistry) error { return nil }

// CombineCapabilities groups capabilities while preserving their middleware order.
// Nested groups are flattened when registered on an agent or run.
func CombineCapabilities(capabilities ...Capability) Capability {
	return combinedCapability{capabilities: slices.Clone(capabilities)}
}

func flattenCapabilities(capabilities []Capability) []Capability {
	var flattened []Capability
	for _, capability := range capabilities {
		if combined, ok := capability.(combinedCapability); ok {
			flattened = append(flattened, flattenCapabilities(combined.capabilities)...)
			continue
		}
		flattened = append(flattened, capability)
		if wrapper, ok := capability.(interface{ wrappedCapability() Capability }); ok {
			wrapped := wrapper.wrappedCapability()
			if !capabilityIsNil(wrapped) {
				flattened = append(flattened, flattenCapabilities([]Capability{wrapped})...)
			}
		}
	}
	return flattened
}

// WithCapabilities registers capabilities on the agent. Slice order is
// middleware order: the first capability is outermost.
func WithCapabilities(capabilities ...Capability) Option {
	capabilities = flattenCapabilities(capabilities)
	return func(config *config) {
		config.capabilities = append(config.capabilities, capabilities...)
	}
}
