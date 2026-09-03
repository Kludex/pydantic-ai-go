package durable_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/Kludex/pydantic-ai-go/durable"
)

func TestCodecAndInvocationErrors(t *testing.T) {
	failure := errors.New("failure")
	tests := []struct {
		name      string
		operation durable.Operation[int, int]
		backend   *backend
		match     string
	}{
		{name: "parameter encode", operation: durable.Operation[int, int]{
			ID: durable.OperationID{Kind: durable.KindEventHandler}, Role: durable.RoleEvent,
			Handler:        func(context.Context, int) (int, error) { return 0, nil },
			ParameterCodec: codec[int]{encode: func(int) ([]byte, error) { return nil, failure }, decode: decodeInt},
		}, backend: &backend{}, match: "encode operation parameters"},
		{name: "cache project", operation: durable.Operation[int, int]{
			ID: durable.OperationID{Kind: durable.KindEventHandler}, Role: durable.RoleEvent,
			Handler:       func(context.Context, int) (int, error) { return 0, nil },
			CacheIdentity: durable.CacheIdentityFunc[int](func(int) (any, error) { return nil, failure }),
		}, backend: &backend{}, match: "project cache identity"},
		{name: "cache encode", operation: durable.Operation[int, int]{
			ID: durable.OperationID{Kind: durable.KindEventHandler}, Role: durable.RoleEvent,
			Handler:       func(context.Context, int) (int, error) { return 0, nil },
			CacheIdentity: durable.CacheIdentityFunc[int](func(int) (any, error) { return make(chan int), nil }),
		}, backend: &backend{}, match: "encode cache identity"},
		{name: "invoke", operation: durable.Operation[int, int]{
			ID: durable.OperationID{Kind: durable.KindEventHandler}, Role: durable.RoleEvent,
			Handler: func(context.Context, int) (int, error) { return 0, nil },
		}, backend: &backend{invokeErr: failure}, match: "failure"},
		{name: "parameter decode", operation: durable.Operation[int, int]{
			ID: durable.OperationID{Kind: durable.KindEventHandler}, Role: durable.RoleEvent,
			Handler:        func(context.Context, int) (int, error) { return 0, nil },
			ParameterCodec: codec[int]{encode: encodeInt, decode: func([]byte) (int, error) { return 0, failure }},
		}, backend: &backend{}, match: "decode operation parameters"},
		{name: "handler", operation: durable.Operation[int, int]{
			ID: durable.OperationID{Kind: durable.KindEventHandler}, Role: durable.RoleEvent,
			Handler: func(context.Context, int) (int, error) { return 0, failure },
		}, backend: &backend{}, match: "failure"},
		{name: "result encode", operation: durable.Operation[int, int]{
			ID: durable.OperationID{Kind: durable.KindEventHandler}, Role: durable.RoleEvent,
			Handler:     func(context.Context, int) (int, error) { return 0, nil },
			ResultCodec: codec[int]{encode: func(int) ([]byte, error) { return nil, failure }, decode: decodeInt},
		}, backend: &backend{}, match: "encode operation result"},
		{name: "result decode", operation: durable.Operation[int, int]{
			ID: durable.OperationID{Kind: durable.KindEventHandler}, Role: durable.RoleEvent,
			Handler:     func(context.Context, int) (int, error) { return 0, nil },
			ResultCodec: codec[int]{encode: encodeInt, decode: func([]byte) (int, error) { return 0, failure }},
		}, backend: &backend{}, match: "decode operation result"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			operation, err := durable.Bind(test.backend, "agent", "", test.operation)
			if err != nil {
				t.Fatal(err)
			}
			_, err = operation.Invoke(context.Background(), 1)
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func encodeInt(int) ([]byte, error) { return []byte("1"), nil }
func decodeInt([]byte) (int, error) { return 1, nil }

func TestJSONCodec(t *testing.T) {
	codec := durable.JSONCodec[map[string]int]{}
	encoded, err := codec.Encode(map[string]int{"value": 1})
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := codec.Decode(encoded)
	if err != nil || decoded["value"] != 1 {
		t.Fatalf("unexpected JSON codec: %#v %v", decoded, err)
	}
	if _, err := codec.Decode([]byte("{")); err == nil {
		t.Fatal("expected JSON decode error")
	}
}
