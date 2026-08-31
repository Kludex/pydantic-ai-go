package mcp

import (
	"context"
	"errors"
	"reflect"

	ai "github.com/Kludex/pydantic-ai-go"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// SamplingRequest is an MCP server request to invoke a client-side model.
type SamplingRequest = mcpsdk.CreateMessageRequest

// SamplingResult is the client-side model response returned to the MCP server.
type SamplingResult = mcpsdk.CreateMessageResult

// SamplingHandler handles an MCP sampling request directly.
type SamplingHandler func(context.Context, *SamplingRequest) (*SamplingResult, error)

// ElicitationRequest asks the client for structured user input.
type ElicitationRequest = mcpsdk.ElicitRequest

// ElicitationResult records whether the user accepted, declined, or cancelled an elicitation.
type ElicitationResult = mcpsdk.ElicitResult

// ElicitationHandler handles structured input requested by an MCP server.
type ElicitationHandler func(context.Context, *ElicitationRequest) (*ElicitationResult, error)

// WithSamplingModel lets an MCP server make sampling requests through model.
// It is mutually exclusive with sampling handlers in WithClientOptions.
func WithSamplingModel(model ai.Model) Option {
	if modelIsNil(model) {
		panic("ai/mcp: sampling model must not be nil")
	}
	return func(config *config) { config.samplingModel = model }
}

// WithSamplingHandler handles MCP sampling without adapting an ai.Model.
// It is mutually exclusive with WithSamplingModel and sampling handlers in
// WithClientOptions.
func WithSamplingHandler(handler SamplingHandler) Option {
	if handler == nil {
		panic("ai/mcp: sampling handler must not be nil")
	}
	return func(config *config) { config.samplingHandler = handler }
}

// WithElicitationHandler handles structured input requested by the server.
// The handler must obtain informed user approval before returning sensitive data.
func WithElicitationHandler(handler ElicitationHandler) Option {
	if handler == nil {
		panic("ai/mcp: elicitation handler must not be nil")
	}
	return func(config *config) { config.elicitation = handler }
}

func prepareClientOptions(cfg config) *mcpsdk.ClientOptions {
	options := &mcpsdk.ClientOptions{}
	if cfg.clientOptions != nil {
		*options = *cfg.clientOptions
	}
	if options.CreateMessageHandler != nil && options.CreateMessageWithToolsHandler != nil {
		panic("ai/mcp: client options cannot set both sampling handlers")
	}
	configuredSampling := options.CreateMessageHandler != nil || options.CreateMessageWithToolsHandler != nil
	if cfg.samplingModel != nil && (cfg.samplingHandler != nil || configuredSampling) {
		panic("ai/mcp: sampling model and sampling handler cannot both be set")
	}
	if cfg.samplingHandler != nil && configuredSampling {
		panic("ai/mcp: sampling handler is already set in client options")
	}
	if cfg.elicitation != nil && options.ElicitationHandler != nil {
		panic("ai/mcp: elicitation handler is already set in client options")
	}
	if cfg.samplingModel != nil {
		options.CreateMessageHandler = samplingModelHandler(cfg.samplingModel)
	} else if cfg.samplingHandler != nil {
		options.CreateMessageHandler = detachedSamplingHandler(cfg.samplingHandler)
	}
	if cfg.elicitation != nil {
		options.ElicitationHandler = detachedElicitationHandler(cfg.elicitation)
	}
	return options
}

func detachedSamplingHandler(handler SamplingHandler) func(
	context.Context, *mcpsdk.CreateMessageRequest,
) (*mcpsdk.CreateMessageResult, error) {
	return func(ctx context.Context, request *mcpsdk.CreateMessageRequest) (*mcpsdk.CreateMessageResult, error) {
		cloned := *request
		cloned.Params = cloneProtocol(request.Params)
		result, err := handler(ctx, &cloned)
		if err != nil {
			return nil, err
		}
		if result == nil {
			return nil, errors.New("ai/mcp: sampling handler returned no result")
		}
		return cloneProtocol(result), nil
	}
}

func detachedElicitationHandler(handler ElicitationHandler) func(
	context.Context, *mcpsdk.ElicitRequest,
) (*mcpsdk.ElicitResult, error) {
	return func(ctx context.Context, request *mcpsdk.ElicitRequest) (*mcpsdk.ElicitResult, error) {
		cloned := *request
		cloned.Params = cloneProtocol(request.Params)
		result, err := handler(ctx, &cloned)
		if err != nil {
			return nil, err
		}
		if result == nil {
			return nil, errors.New("ai/mcp: elicitation handler returned no result")
		}
		return cloneProtocol(result), nil
	}
}

func modelIsNil(model ai.Model) bool {
	if model == nil {
		return true
	}
	value := reflect.ValueOf(model)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}
