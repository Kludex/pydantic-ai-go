package dbos_test

import (
	"testing"

	"github.com/Kludex/pydantic-ai-go/durable"
	"github.com/Kludex/pydantic-ai-go/durable/dbos"
)

func TestNewBackend(t *testing.T) {
	backend, err := dbos.NewBackend(func(durable.Registration) (durable.BackendOperation, error) { return nil, nil })
	if err != nil || backend == nil {
		t.Fatalf("unexpected backend: %#v %v", backend, err)
	}
}
