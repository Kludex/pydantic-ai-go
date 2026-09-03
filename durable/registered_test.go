package durable_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/Kludex/pydantic-ai-go/durable"
	"github.com/Kludex/pydantic-ai-go/durable/dbos"
	"github.com/Kludex/pydantic-ai-go/durable/prefect"
	"github.com/Kludex/pydantic-ai-go/durable/temporal"
)

func registrar(registration durable.Registration) (durable.BackendOperation, error) {
	backend := &backend{registration: registration}
	return backendOperation{backend: backend}, nil
}

func TestRegisteredBackendPolicies(t *testing.T) {
	for _, constructor := range []func(durable.RegisterFunc) (*durable.RegisteredBackend, error){
		temporal.NewBackend, dbos.NewBackend,
	} {
		backend, err := constructor(registrar)
		if err != nil {
			t.Fatal(err)
		}
		operation := durable.Operation[int, int]{
			ID: durable.OperationID{Kind: durable.KindEventHandler}, Role: durable.RoleEvent,
			Handler: func(context.Context, int) (int, error) { return 1, nil },
		}
		if _, err := durable.Bind(backend, "agent", "", operation); err != nil {
			t.Fatal(err)
		}
		backend.Freeze()
		backend.Freeze()
		operation.ID = durable.OperationID{Kind: durable.KindCapability, CapabilityID: "late", Operation: "observe"}
		operation.Role = durable.RoleCapability
		operation.Observer = true
		if _, err := durable.Bind(backend, "agent", "", operation); err == nil ||
			!strings.Contains(err.Error(), "before worker startup") {
			t.Fatalf("unexpected startup policy error: %v", err)
		}
	}

	backend, err := prefect.NewBackend(registrar)
	if err != nil {
		t.Fatal(err)
	}
	backend.Freeze()
	observer := durable.Operation[int, int]{
		ID:   durable.OperationID{Kind: durable.KindCapability, CapabilityID: "runtime", Operation: "observe"},
		Role: durable.RoleCapability, Observer: true,
		Handler: func(context.Context, int) (int, error) { return 1, nil },
	}
	bound, err := durable.Bind(backend, "agent", "", observer)
	if err != nil {
		t.Fatal(err)
	}
	if result, err := bound.Invoke(context.Background(), 1); err != nil || result != 1 {
		t.Fatalf("unexpected observer result: %d %v", result, err)
	}
	tool := durable.Operation[int, int]{
		ID:      durable.OperationID{Kind: durable.KindToolsetCall, ToolsetKind: durable.ToolsetDynamic, ToolsetID: "late"},
		Role:    durable.RoleTool,
		Handler: func(context.Context, int) (int, error) { return 1, nil },
	}
	if _, err := durable.Bind(backend, "agent", "", tool); err == nil {
		t.Fatal("expected late Prefect tool rejection")
	}
	registrations := backend.Registrations()
	registrations[0].Name = "mutated"
	if backend.Registrations()[0].Name == "mutated" {
		t.Fatal("registrations were not detached")
	}
}

func TestRegisteredBackendErrors(t *testing.T) {
	if _, err := durable.NewRegisteredBackend(nil, durable.RegistrationStartupOnly); err == nil {
		t.Fatal("expected nil registrar error")
	}
	if _, err := durable.NewRegisteredBackend(registrar, "future"); err == nil {
		t.Fatal("expected policy error")
	}
	failure := errors.New("register failed")
	backend, err := durable.NewRegisteredBackend(func(durable.Registration) (durable.BackendOperation, error) {
		return nil, failure
	}, durable.RegistrationStartupOnly)
	if err != nil {
		t.Fatal(err)
	}
	registration := durable.Registration{Name: "one"}
	if _, err := backend.Bind(registration); !errors.Is(err, failure) {
		t.Fatalf("unexpected register error: %v", err)
	}
	backend, err = durable.NewRegisteredBackend(func(durable.Registration) (durable.BackendOperation, error) {
		return nil, nil
	}, durable.RegistrationStartupOnly)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := backend.Bind(registration); err == nil {
		t.Fatal("expected nil operation error")
	}
	backend, err = durable.NewRegisteredBackend(registrar, durable.RegistrationStartupOnly)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := backend.Bind(registration); err != nil {
		t.Fatal(err)
	}
	if _, err := backend.Bind(registration); err == nil {
		t.Fatal("expected duplicate registration error")
	}
}
