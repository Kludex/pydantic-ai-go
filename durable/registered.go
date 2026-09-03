package durable

import (
	"fmt"
	"slices"
	"sync"
)

// RegistrationPolicy controls whether operations may bind after worker startup.
type RegistrationPolicy string

const (
	// RegistrationStartupOnly rejects every late operation, as required by Temporal and DBOS workers.
	RegistrationStartupOnly RegistrationPolicy = "startup-only"
	// RegistrationRuntimeObservers permits only explicitly non-executing late observers.
	RegistrationRuntimeObservers RegistrationPolicy = "runtime-observers"
)

// RegisterFunc registers one named worker handler with an engine SDK.
type RegisterFunc func(registration Registration) (BackendOperation, error)

// RegisteredBackend enforces worker-startup registration policy around an engine registrar.
type RegisteredBackend struct {
	mutex         sync.Mutex
	register      RegisterFunc
	policy        RegistrationPolicy
	frozen        bool
	registrations []Registration
	names         map[string]struct{}
}

// NewRegisteredBackend creates a backend that binds named worker handlers.
func NewRegisteredBackend(register RegisterFunc, policy RegistrationPolicy) (*RegisteredBackend, error) {
	if register == nil {
		return nil, fmt.Errorf("durable: register function must not be nil")
	}
	if policy != RegistrationStartupOnly && policy != RegistrationRuntimeObservers {
		return nil, fmt.Errorf("durable: invalid registration policy %q", policy)
	}
	return &RegisteredBackend{register: register, policy: policy, names: map[string]struct{}{}}, nil
}

// Bind registers an operation unless startup policy rejects it.
func (backend *RegisteredBackend) Bind(registration Registration) (BackendOperation, error) {
	backend.mutex.Lock()
	defer backend.mutex.Unlock()
	if backend.frozen && (backend.policy == RegistrationStartupOnly || !registration.Observer) {
		return nil, fmt.Errorf("durable: operation %q was not registered before worker startup", registration.Name)
	}
	if _, exists := backend.names[registration.Name]; exists {
		return nil, fmt.Errorf("durable: operation %q is already registered", registration.Name)
	}
	operation, err := backend.register(registration)
	if err != nil {
		return nil, err
	}
	if operation == nil {
		return nil, fmt.Errorf("durable: registrar returned nil for operation %q", registration.Name)
	}
	backend.names[registration.Name] = struct{}{}
	backend.registrations = append(backend.registrations, registration)
	return operation, nil
}

// Freeze marks worker registration complete. Calls are idempotent.
func (backend *RegisteredBackend) Freeze() {
	backend.mutex.Lock()
	defer backend.mutex.Unlock()
	backend.frozen = true
}

// Registrations returns operation definitions in binding order.
func (backend *RegisteredBackend) Registrations() []Registration {
	backend.mutex.Lock()
	defer backend.mutex.Unlock()
	return slices.Clone(backend.registrations)
}
