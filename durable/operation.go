// Package durable defines engine-neutral durable operation contracts.
package durable

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// Role selects the coarse backend configuration bucket for an operation.
type Role string

const (
	// RoleModel configures provider request operations.
	RoleModel Role = "model"
	// RoleEvent configures event observer operations.
	RoleEvent Role = "event"
	// RoleTool configures toolset operations.
	RoleTool Role = "tool"
	// RoleCapability configures capability-contributed operations.
	RoleCapability Role = "capability"
)

// Kind identifies one semantic durable operation.
type Kind string

const (
	// KindModelRequest identifies a non-streaming model request.
	KindModelRequest Kind = "model.request"
	// KindModelStream identifies a streaming model request.
	KindModelStream Kind = "model.request_stream"
	// KindModelCancel identifies suspended-response cancellation.
	KindModelCancel Kind = "model.cancel_suspended_response"
	// KindModelCompact identifies explicit message compaction.
	KindModelCompact Kind = "model.compact_messages"
	// KindEventHandler identifies durable event observation.
	KindEventHandler Kind = "event_stream_handler"
	// KindToolsetGetTools identifies dynamic tool discovery.
	KindToolsetGetTools Kind = "toolset.get_tools"
	// KindToolsetGetInstructions identifies MCP instruction discovery.
	KindToolsetGetInstructions Kind = "toolset.get_instructions"
	// KindToolsetValidate identifies tool argument validation.
	KindToolsetValidate Kind = "toolset.validate_args"
	// KindToolsetCall identifies tool execution.
	KindToolsetCall Kind = "toolset.call_tool"
	// KindCapability identifies a capability-contributed operation.
	KindCapability Kind = "capability"
)

// ToolsetKind identifies a durable toolset category.
type ToolsetKind string

const (
	// ToolsetFunction identifies a static function toolset.
	ToolsetFunction ToolsetKind = "function"
	// ToolsetMCP identifies an MCP toolset.
	ToolsetMCP ToolsetKind = "mcp"
	// ToolsetDynamic identifies a runtime-discovered toolset.
	ToolsetDynamic ToolsetKind = "dynamic"
)

// OperationID is the stable semantic identity persisted by durable engines.
type OperationID struct {
	// Kind identifies the semantic operation.
	Kind Kind `json:"kind"`
	// ModelID is the persisted model registry identity.
	ModelID string `json:"model_id,omitempty"`
	// ModelName is diagnostic provider model identity.
	ModelName string `json:"model_name,omitempty"`
	// ToolsetKind identifies function, MCP, or dynamic tools.
	ToolsetKind ToolsetKind `json:"toolset_kind,omitempty"`
	// ToolsetID is the stable application toolset identity.
	ToolsetID string `json:"toolset_id,omitempty"`
	// CapabilityID is the stable contributing capability identity.
	CapabilityID string `json:"capability_id,omitempty"`
	// Operation is the capability-relative operation name.
	Operation string `json:"operation,omitempty"`
}

// Name returns the stable journal operation name for one agent.
func (id OperationID) Name(agentName string, defaultModelID string) (string, error) {
	if agentName == "" {
		return "", fmt.Errorf("durable: agent name must not be empty")
	}
	if err := id.Validate(); err != nil {
		return "", err
	}
	modelSuffix := ""
	if id.ModelID != "" && id.ModelID != defaultModelID {
		modelSuffix = "." + id.ModelID
	}
	if id.Kind == KindModelRequest || id.Kind == KindModelStream || id.Kind == KindModelCancel || id.Kind == KindModelCompact {
		return agentName + "__" + string(id.Kind) + modelSuffix, nil
	}
	if id.Kind == KindEventHandler {
		return agentName + "__event_stream_handler", nil
	}
	if id.Kind == KindCapability {
		return agentName + "__capability__" + id.CapabilityID + "." + id.Operation, nil
	}
	prefix := string(id.ToolsetKind) + "_toolset"
	if id.ToolsetKind == ToolsetMCP {
		prefix = "mcp_server"
	}
	operation := strings.TrimPrefix(string(id.Kind), "toolset.")
	return agentName + "__" + prefix + "__" + id.ToolsetID + "." + operation, nil
}

// Validate checks required identity fields without applying a backend policy.
func (id OperationID) Validate() error {
	switch id.Kind {
	case KindModelRequest, KindModelStream, KindModelCancel, KindModelCompact:
		if id.ModelName == "" {
			return fmt.Errorf("durable: model operation requires a model name")
		}
	case KindEventHandler:
	case KindCapability:
		if id.CapabilityID == "" || id.Operation == "" {
			return fmt.Errorf("durable: capability operation requires capability and operation IDs")
		}
	case KindToolsetGetTools, KindToolsetValidate, KindToolsetCall:
		if id.ToolsetID == "" || !validToolsetKind(id.ToolsetKind) {
			return fmt.Errorf("durable: toolset operation requires a toolset ID and kind")
		}
	case KindToolsetGetInstructions:
		if id.ToolsetID == "" || id.ToolsetKind != ToolsetMCP {
			return fmt.Errorf("durable: instruction discovery requires an MCP toolset ID")
		}
	default:
		return fmt.Errorf("durable: unsupported operation kind %q", id.Kind)
	}
	return nil
}

func validToolsetKind(kind ToolsetKind) bool {
	return kind == ToolsetFunction || kind == ToolsetMCP || kind == ToolsetDynamic
}

// Codec serializes one value across an engine boundary.
type Codec[T any] interface {
	// Encode serializes a detached value.
	Encode(value T) ([]byte, error)
	// Decode rebuilds a value worker-side.
	Decode(payload []byte) (T, error)
}

// JSONCodec serializes values with encoding/json.
type JSONCodec[T any] struct{}

func (JSONCodec[T]) Encode(value T) ([]byte, error) { return json.Marshal(value) }
func (JSONCodec[T]) Decode(payload []byte) (T, error) {
	var value T
	err := json.Unmarshal(payload, &value)
	return value, err
}

// CacheIdentity projects semantic parameters into backend cache-key input.
type CacheIdentity[Params any] interface {
	// Project returns semantic hash input independent of transport encoding.
	Project(params Params) (any, error)
}

// CacheIdentityFunc adapts a function into CacheIdentity.
type CacheIdentityFunc[Params any] func(params Params) (any, error)

func (function CacheIdentityFunc[Params]) Project(params Params) (any, error) {
	return function(params)
}

// Operation declares one typed durable unit.
type Operation[Params, Result any] struct {
	// ID is stable persisted semantic identity.
	ID OperationID
	// Role selects the engine configuration bucket.
	Role Role
	// Handler runs inside the durable unit from rebuilt parameters.
	Handler func(ctx context.Context, params Params) (Result, error)
	// Observer marks an operation that cannot execute model, tool, or external side effects.
	Observer bool
	// ParameterCodec overrides JSON parameter transport.
	ParameterCodec Codec[Params]
	// ResultCodec overrides JSON result transport.
	ResultCodec Codec[Result]
	// CacheIdentity overrides the default full-parameter projection.
	CacheIdentity CacheIdentity[Params]
	// InvocationLabel adds non-identity display context.
	InvocationLabel func(params Params) string
}
