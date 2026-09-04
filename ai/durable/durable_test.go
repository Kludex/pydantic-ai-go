package durable_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/durable"
)

type backend struct {
	registration durable.Registration
	bindErr      error
	nilOperation bool
	invokeErr    error
	invocation   durable.Invocation
}

func (backend *backend) Bind(registration durable.Registration) (durable.BackendOperation, error) {
	backend.registration = registration
	if backend.bindErr != nil {
		return nil, backend.bindErr
	}
	if backend.nilOperation {
		return nil, nil
	}
	return backendOperation{backend: backend}, nil
}

type backendOperation struct{ backend *backend }

func (operation backendOperation) Invoke(ctx context.Context, invocation durable.Invocation) ([]byte, error) {
	operation.backend.invocation = invocation
	if operation.backend.invokeErr != nil {
		return nil, operation.backend.invokeErr
	}
	return operation.backend.registration.Handler(ctx, invocation.Params)
}

type codec[T any] struct {
	encode func(T) ([]byte, error)
	decode func([]byte) (T, error)
}

func (codec codec[T]) Encode(value T) ([]byte, error) { return codec.encode(value) }
func (codec codec[T]) Decode(value []byte) (T, error) { return codec.decode(value) }

func TestBindAndInvoke(t *testing.T) {
	backend := &backend{}
	operation, err := durable.Bind(backend, "weather", "default", durable.Operation[int, int]{
		ID:      durable.OperationID{Kind: durable.KindToolsetCall, ToolsetKind: durable.ToolsetFunction, ToolsetID: "tools"},
		Role:    durable.RoleTool,
		Handler: func(_ context.Context, value int) (int, error) { return value * 2, nil },
		CacheIdentity: durable.CacheIdentityFunc[int](func(value int) (any, error) {
			return map[string]int{"value": value}, nil
		}),
		InvocationLabel: func(int) string { return "double" },
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := operation.Invoke(context.Background(), 4)
	if err != nil || result != 8 {
		t.Fatalf("unexpected result: %d %v", result, err)
	}
	if backend.invocation.Name != "weather__function_toolset__tools.call_tool" ||
		backend.invocation.Label != "double" || string(backend.invocation.CacheKey) != `{"value":4}` ||
		backend.registration.Role != durable.RoleTool {
		t.Fatalf("unexpected invocation: %#v registration=%#v", backend.invocation, backend.registration)
	}
	backend.invocation.Params[0] = 'x'
	if string(backend.invocation.CacheKey) != `{"value":4}` {
		t.Fatal("invocation values shared storage")
	}
}

func TestCallableBackend(t *testing.T) {
	called := false
	backend := durable.CallableBackend{Execute: func(
		ctx context.Context, invocation durable.Invocation, handler durable.WireHandler,
	) ([]byte, error) {
		called = true
		return handler(ctx, invocation.Params)
	}}
	operation, err := durable.Bind(backend, "agent", "default", durable.Operation[string, string]{
		ID: durable.OperationID{Kind: durable.KindModelRequest, ModelName: "model"}, Role: durable.RoleModel,
		Handler: func(context.Context, string) (string, error) { return "done", nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := operation.Invoke(context.Background(), "input")
	if err != nil || result != "done" || !called {
		t.Fatalf("unexpected callable backend: %q %v %v", result, err, called)
	}
	if _, err := durable.Bind(durable.CallableBackend{}, "agent", "default", durable.Operation[string, string]{
		ID: durable.OperationID{Kind: durable.KindModelRequest, ModelName: "model"}, Role: durable.RoleModel,
		Handler: func(context.Context, string) (string, error) { return "", nil },
	}); err == nil {
		t.Fatal("expected nil execute error")
	}
}

func TestOperationNames(t *testing.T) {
	tests := []struct {
		id   durable.OperationID
		want string
	}{
		{durable.OperationID{Kind: durable.KindModelRequest, ModelName: "m"}, "agent__model.request"},
		{durable.OperationID{Kind: durable.KindModelStream, ModelID: "other", ModelName: "m"}, "agent__model.request_stream.other"},
		{durable.OperationID{Kind: durable.KindModelCancel, ModelID: "default", ModelName: "m"}, "agent__model.cancel_suspended_response"},
		{durable.OperationID{Kind: durable.KindModelCompact, ModelName: "m"}, "agent__model.compact_messages"},
		{durable.OperationID{Kind: durable.KindEventHandler}, "agent__event_stream_handler"},
		{durable.OperationID{Kind: durable.KindCapability, CapabilityID: "audit", Operation: "record"}, "agent__capability__audit.record"},
		{durable.OperationID{Kind: durable.KindToolsetGetTools, ToolsetKind: durable.ToolsetDynamic, ToolsetID: "tools"}, "agent__dynamic_toolset__tools.get_tools"},
		{durable.OperationID{Kind: durable.KindToolsetGetInstructions, ToolsetKind: durable.ToolsetMCP, ToolsetID: "server"}, "agent__mcp_server__server.get_instructions"},
		{durable.OperationID{Kind: durable.KindToolsetValidate, ToolsetKind: durable.ToolsetFunction, ToolsetID: "tools"}, "agent__function_toolset__tools.validate_args"},
		{durable.OperationID{Kind: durable.KindToolsetCall, ToolsetKind: durable.ToolsetMCP, ToolsetID: "server"}, "agent__mcp_server__server.call_tool"},
	}
	for _, test := range tests {
		name, err := test.id.Name("agent", "default")
		if err != nil || name != test.want {
			t.Fatalf("id=%#v: got %q want %q err=%v", test.id, name, test.want, err)
		}
	}
}

func TestOperationValidation(t *testing.T) {
	tests := []struct {
		id    durable.OperationID
		match string
	}{
		{durable.OperationID{Kind: durable.KindModelRequest}, "model name"},
		{durable.OperationID{Kind: durable.KindCapability}, "capability and operation"},
		{durable.OperationID{Kind: durable.KindToolsetCall}, "toolset ID and kind"},
		{durable.OperationID{Kind: durable.KindToolsetGetInstructions, ToolsetKind: durable.ToolsetFunction, ToolsetID: "x"}, "MCP toolset"},
		{durable.OperationID{Kind: "future"}, "unsupported operation"},
	}
	for _, test := range tests {
		if err := test.id.Validate(); err == nil || !strings.Contains(err.Error(), test.match) {
			t.Fatalf("id=%#v: unexpected error %v", test.id, err)
		}
	}
	if _, err := (durable.OperationID{Kind: durable.KindEventHandler}).Name("", ""); err == nil {
		t.Fatal("expected empty agent error")
	}
	if _, err := (durable.OperationID{Kind: durable.KindModelRequest}).Name("agent", ""); err == nil {
		t.Fatal("expected invalid operation name error")
	}
}

func TestBindErrors(t *testing.T) {
	valid := durable.Operation[int, int]{
		ID: durable.OperationID{Kind: durable.KindEventHandler}, Role: durable.RoleEvent,
		Handler: func(context.Context, int) (int, error) { return 1, nil },
	}
	if _, err := durable.Bind[int, int](nil, "agent", "", valid); err == nil {
		t.Fatal("expected nil backend error")
	}
	invalidID := valid
	invalidID.ID = durable.OperationID{Kind: durable.KindModelRequest}
	if _, err := durable.Bind(&backend{}, "agent", "", invalidID); err == nil {
		t.Fatal("expected invalid operation ID error")
	}
	operation := valid
	operation.Handler = nil
	if _, err := durable.Bind(&backend{}, "agent", "", operation); err == nil {
		t.Fatal("expected nil handler error")
	}
	operation = valid
	operation.Role = "future"
	if _, err := durable.Bind(&backend{}, "agent", "", operation); err == nil {
		t.Fatal("expected role error")
	}
	bindErr := errors.New("bind failed")
	if _, err := durable.Bind(&backend{bindErr: bindErr}, "agent", "", valid); !errors.Is(err, bindErr) {
		t.Fatalf("unexpected bind error: %v", err)
	}
	if _, err := durable.Bind(&backend{nilOperation: true}, "agent", "", valid); err == nil {
		t.Fatal("expected nil operation error")
	}
}
