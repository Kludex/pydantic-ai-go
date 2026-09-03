package ai

import "context"

// ModelRequestContext is the mutable request passed through model lifecycle hooks.
type ModelRequestContext struct {
	// Model is the concrete selected model.
	Model Model
	// ModelID is the selected application model ID.
	ModelID string
	// Messages is the detached prospective history.
	Messages []ModelMessage
	// Params contains detached prepared request parameters.
	Params ModelRequestParams
	// Streaming reports whether the request expects provider deltas.
	Streaming bool
	// ReplaceHistory commits Messages as the run history before the request.
	ReplaceHistory bool
	// AdditionalUsage counts side requests performed by a hook.
	AdditionalUsage Usage
}

// Clone returns a request context detached from mutable messages, settings, schemas, and tools.
func (request ModelRequestContext) Clone() ModelRequestContext {
	request.Messages = cloneModelMessages(request.Messages)
	request.Params.InstructionParts = cloneInstructionParts(request.Params.InstructionParts)
	request.Params.Tools = cloneToolDefinitions(request.Params.Tools)
	request.Params.DeferredTools = cloneToolDefinitions(request.Params.DeferredTools)
	request.Params.NativeTools = CloneNativeTools(request.Params.NativeTools)
	if request.Params.OutputTool != nil {
		outputTool := cloneToolDefinition(*request.Params.OutputTool)
		request.Params.OutputTool = &outputTool
	}
	request.Params.OutputSchema = cloneSchemaMap(request.Params.OutputSchema)
	request.Params.Settings = request.Params.Settings.Clone()
	request.AdditionalUsage = request.AdditionalUsage.Clone()
	return request
}

// BeforeModelRequestHook modifies a prepared request before model middleware runs.
// InstructionParts is the source of truth when a hook changes it. Changing only
// Instructions remains supported as an aggregate replacement for compatibility.
type BeforeModelRequestHook interface {
	// BeforeModelRequest returns the request passed to model middleware.
	BeforeModelRequest(ctx context.Context, ri *RunInfo, request ModelRequestContext) (ModelRequestContext, error)
}

// AfterModelRequestHook modifies a successful response. Hooks run in reverse capability order.
type AfterModelRequestHook interface {
	// AfterModelRequest returns the successful response passed to outer hooks.
	AfterModelRequest(
		ctx context.Context, ri *RunInfo, request ModelRequestContext, response *ModelResponse,
	) (*ModelResponse, error)
}

// ModelRequestErrorHook may replace a model or middleware error with a response. Hooks run in reverse capability order.
type ModelRequestErrorHook interface {
	// OnModelRequestError returns a recovered response or replacement error.
	OnModelRequestError(
		ctx context.Context, ri *RunInfo, request ModelRequestContext, requestErr error,
	) (*ModelResponse, error)
}

// BeforeModelRequestFunc adapts a function into a model-request hook capability.
type BeforeModelRequestFunc func(
	ctx context.Context, ri *RunInfo, request ModelRequestContext,
) (ModelRequestContext, error)

// Setup implements Capability.
func (BeforeModelRequestFunc) Setup(*CapabilityRegistry) error { return nil }

// BeforeModelRequest calls the adapted function.
func (hook BeforeModelRequestFunc) BeforeModelRequest(
	ctx context.Context, ri *RunInfo, request ModelRequestContext,
) (ModelRequestContext, error) {
	return hook(ctx, ri, request)
}

// AfterModelRequestFunc adapts a function into a model-response hook capability.
type AfterModelRequestFunc func(
	ctx context.Context, ri *RunInfo, request ModelRequestContext, response *ModelResponse,
) (*ModelResponse, error)

// Setup implements Capability.
func (AfterModelRequestFunc) Setup(*CapabilityRegistry) error { return nil }

// AfterModelRequest calls the adapted function.
func (hook AfterModelRequestFunc) AfterModelRequest(
	ctx context.Context, ri *RunInfo, request ModelRequestContext, response *ModelResponse,
) (*ModelResponse, error) {
	return hook(ctx, ri, request, response)
}

// ModelRequestErrorFunc adapts a recovery function into a model-request hook capability.
type ModelRequestErrorFunc func(
	ctx context.Context, ri *RunInfo, request ModelRequestContext, requestErr error,
) (*ModelResponse, error)

// Setup implements Capability.
func (ModelRequestErrorFunc) Setup(*CapabilityRegistry) error { return nil }

// OnModelRequestError calls the adapted function.
func (hook ModelRequestErrorFunc) OnModelRequestError(
	ctx context.Context, ri *RunInfo, request ModelRequestContext, requestErr error,
) (*ModelResponse, error) {
	return hook(ctx, ri, request, requestErr)
}
