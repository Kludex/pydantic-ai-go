package temporal_test

import (
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/durable"
	"github.com/Kludex/pydantic-ai-go/ai/durable/temporal"
)

func TestNewBackend(t *testing.T) {
	backend, err := temporal.NewBackend(func(durable.Registration) (durable.BackendOperation, error) { return nil, nil })
	if err != nil || backend == nil {
		t.Fatalf("unexpected backend: %#v %v", backend, err)
	}
}
