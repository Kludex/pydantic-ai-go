package ai

import (
	"context"
	"slices"
)

// ModelRequestContext is the mutable request passed through model lifecycle hooks.
type ModelRequestContext struct {
	Model     Model
	ModelID   string
	Messages  []ModelMessage
	Params    ModelRequestParams
	Streaming bool
}

// Clone returns a request context detached from mutable messages, settings, schemas, and tools.
func (request ModelRequestContext) Clone() ModelRequestContext {
	request.Messages = cloneModelMessages(request.Messages)
	request.Params.InstructionParts = slices.Clone(request.Params.InstructionParts)
	request.Params.Tools = cloneToolDefinitions(request.Params.Tools)
	request.Params.DeferredTools = cloneToolDefinitions(request.Params.DeferredTools)
	if request.Params.OutputTool != nil {
		outputTool := cloneToolDefinition(*request.Params.OutputTool)
		request.Params.OutputTool = &outputTool
	}
	request.Params.OutputSchema = cloneSchemaMap(request.Params.OutputSchema)
	request.Params.Settings = request.Params.Settings.Clone()
	return request
}

// BeforeModelRequestHook modifies a prepared request before model middleware runs.
type BeforeModelRequestHook interface {
	BeforeModelRequest(ctx context.Context, ri *RunInfo, request ModelRequestContext) (ModelRequestContext, error)
}

// AfterModelRequestHook modifies a successful response. Hooks run in reverse capability order.
type AfterModelRequestHook interface {
	AfterModelRequest(
		ctx context.Context, ri *RunInfo, request ModelRequestContext, response *ModelResponse,
	) (*ModelResponse, error)
}

// ModelRequestErrorHook may replace a model or middleware error with a response. Hooks run in reverse capability order.
type ModelRequestErrorHook interface {
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
