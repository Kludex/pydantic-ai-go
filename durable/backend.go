package durable

import (
	"context"
	"encoding/json"
	"fmt"
)

// WireHandler executes one operation from serialized parameters.
type WireHandler func(ctx context.Context, params []byte) ([]byte, error)

// Registration is the immutable worker-side operation definition.
type Registration struct {
	// ID is the semantic operation identity.
	ID OperationID
	// Name is the stable persisted backend name.
	Name string
	// Role selects the backend configuration bucket.
	Role Role
	// Handler executes serialized parameters worker-side.
	Handler WireHandler
}

// Invocation is one serialized operation call.
type Invocation struct {
	// ID is the semantic operation identity.
	ID OperationID
	// Name is the stable persisted backend name.
	Name string
	// Label is optional per-invocation display context.
	Label string
	// Params is a detached serialized parameter payload.
	Params []byte
	// CacheKey is detached serialized semantic cache identity.
	CacheKey []byte
}

// Backend binds semantic operations to one durable engine.
type Backend interface {
	// Bind registers or wraps one immutable operation definition.
	Bind(registration Registration) (BackendOperation, error)
}

// BackendOperation invokes one bound durable engine operation.
type BackendOperation interface {
	// Invoke dispatches one serialized call through the durable engine.
	Invoke(ctx context.Context, invocation Invocation) ([]byte, error)
}

// BoundOperation is a typed operation connected to a backend.
type BoundOperation[Params, Result any] struct {
	operation      Operation[Params, Result]
	backend        BackendOperation
	name           string
	parameterCodec Codec[Params]
	resultCodec    Codec[Result]
	cacheIdentity  CacheIdentity[Params]
}

// Bind validates, serializes, and connects an operation to a third-party backend.
func Bind[Params, Result any](
	backend Backend, agentName string, defaultModelID string, operation Operation[Params, Result],
) (*BoundOperation[Params, Result], error) {
	if backend == nil {
		return nil, fmt.Errorf("durable: backend must not be nil")
	}
	if operation.Handler == nil {
		return nil, fmt.Errorf("durable: operation handler must not be nil")
	}
	if !validRole(operation.Role) {
		return nil, fmt.Errorf("durable: invalid operation role %q", operation.Role)
	}
	name, err := operation.ID.Name(agentName, defaultModelID)
	if err != nil {
		return nil, err
	}
	parameterCodec := operation.ParameterCodec
	if parameterCodec == nil {
		parameterCodec = JSONCodec[Params]{}
	}
	resultCodec := operation.ResultCodec
	if resultCodec == nil {
		resultCodec = JSONCodec[Result]{}
	}
	cacheIdentity := operation.CacheIdentity
	if cacheIdentity == nil {
		cacheIdentity = CacheIdentityFunc[Params](func(params Params) (any, error) { return params, nil })
	}
	registration := Registration{
		ID: operation.ID, Name: name, Role: operation.Role,
		Handler: func(ctx context.Context, payload []byte) ([]byte, error) {
			params, err := parameterCodec.Decode(payload)
			if err != nil {
				return nil, fmt.Errorf("durable: decode operation parameters: %w", err)
			}
			result, err := operation.Handler(ctx, params)
			if err != nil {
				return nil, err
			}
			encoded, err := resultCodec.Encode(result)
			if err != nil {
				return nil, fmt.Errorf("durable: encode operation result: %w", err)
			}
			return encoded, nil
		},
	}
	bound, err := backend.Bind(registration)
	if err != nil {
		return nil, err
	}
	if bound == nil {
		return nil, fmt.Errorf("durable: backend returned a nil bound operation")
	}
	return &BoundOperation[Params, Result]{
		operation: operation, backend: bound, name: name,
		parameterCodec: parameterCodec, resultCodec: resultCodec, cacheIdentity: cacheIdentity,
	}, nil
}

// Invoke serializes parameters, computes backend cache identity, and decodes the result.
func (operation *BoundOperation[Params, Result]) Invoke(ctx context.Context, params Params) (Result, error) {
	var zero Result
	payload, err := operation.parameterCodec.Encode(params)
	if err != nil {
		return zero, fmt.Errorf("durable: encode operation parameters: %w", err)
	}
	projection, err := operation.cacheIdentity.Project(params)
	if err != nil {
		return zero, fmt.Errorf("durable: project cache identity: %w", err)
	}
	cacheKey, err := json.Marshal(projection)
	if err != nil {
		return zero, fmt.Errorf("durable: encode cache identity: %w", err)
	}
	label := ""
	if operation.operation.InvocationLabel != nil {
		label = operation.operation.InvocationLabel(params)
	}
	result, err := operation.backend.Invoke(ctx, Invocation{
		ID: operation.operation.ID, Name: operation.name, Label: label,
		Params: append([]byte(nil), payload...), CacheKey: append([]byte(nil), cacheKey...),
	})
	if err != nil {
		return zero, err
	}
	decoded, err := operation.resultCodec.Decode(result)
	if err != nil {
		return zero, fmt.Errorf("durable: decode operation result: %w", err)
	}
	return decoded, nil
}

func validRole(role Role) bool {
	return role == RoleModel || role == RoleEvent || role == RoleTool || role == RoleCapability
}

// CallableBackend executes registered handlers through one engine callback.
type CallableBackend struct {
	// Execute runs handler inside one named durable engine unit.
	Execute func(ctx context.Context, invocation Invocation, handler WireHandler) ([]byte, error)
}

// Bind implements Backend.
func (backend CallableBackend) Bind(registration Registration) (BackendOperation, error) {
	if backend.Execute == nil {
		return nil, fmt.Errorf("durable: callable backend execute function must not be nil")
	}
	return callableOperation{registration: registration, execute: backend.Execute}, nil
}

type callableOperation struct {
	registration Registration
	execute      func(context.Context, Invocation, WireHandler) ([]byte, error)
}

func (operation callableOperation) Invoke(ctx context.Context, invocation Invocation) ([]byte, error) {
	return operation.execute(ctx, invocation, operation.registration.Handler)
}
