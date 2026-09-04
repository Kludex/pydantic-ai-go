package prefect_test

import (
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/durable"
	"github.com/Kludex/pydantic-ai-go/ai/durable/prefect"
)

func TestNewBackend(t *testing.T) {
	backend, err := prefect.NewBackend(func(durable.Registration) (durable.BackendOperation, error) { return nil, nil })
	if err != nil || backend == nil {
		t.Fatalf("unexpected backend: %#v %v", backend, err)
	}
}
