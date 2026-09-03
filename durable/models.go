package durable

import (
	"context"
	"fmt"
	"reflect"
	"sync"

	ai "github.com/Kludex/pydantic-ai-go"
)

// ModelFactory rebuilds one model inside a durable operation.
type ModelFactory func(ctx context.Context) (ai.Model, error)

// ModelLease contains a model and operation-scoped cleanup.
type ModelLease struct {
	// Model is the registered or operation-owned model.
	Model ai.Model
	// Close releases only operation-owned lifecycle resources.
	Close ai.ModelCloseFunc
}

// ModelRegistry distinguishes caller-owned instances from operation-owned factories.
type ModelRegistry struct {
	mutex     sync.RWMutex
	models    map[string]ai.Model
	factories map[string]ModelFactory
}

// Register stores a caller-owned model. Acquire never opens or closes it.
func (registry *ModelRegistry) Register(id string, model ai.Model) error {
	if id == "" || modelIsNil(model) {
		return fmt.Errorf("durable: registered model requires a non-empty ID and model")
	}
	registry.mutex.Lock()
	defer registry.mutex.Unlock()
	if registry.models == nil {
		registry.models = map[string]ai.Model{}
	}
	if _, exists := registry.models[id]; exists {
		return fmt.Errorf("durable: model ID %q is already registered", id)
	}
	if _, exists := registry.factories[id]; exists {
		return fmt.Errorf("durable: model ID %q is already registered", id)
	}
	registry.models[id] = model
	return nil
}

// RegisterFactory stores a model rebuilt and lifecycle-managed inside each operation.
func (registry *ModelRegistry) RegisterFactory(id string, factory ModelFactory) error {
	if id == "" || factory == nil {
		return fmt.Errorf("durable: model factory requires a non-empty ID and function")
	}
	registry.mutex.Lock()
	defer registry.mutex.Unlock()
	if registry.factories == nil {
		registry.factories = map[string]ModelFactory{}
	}
	if _, exists := registry.models[id]; exists {
		return fmt.Errorf("durable: model ID %q is already registered", id)
	}
	if _, exists := registry.factories[id]; exists {
		return fmt.Errorf("durable: model ID %q is already registered", id)
	}
	registry.factories[id] = factory
	return nil
}

// Acquire resolves one model for a durable unit. Factory models are opened in
// the unit and Close releases them. Registered instances return a no-op Close.
func (registry *ModelRegistry) Acquire(ctx context.Context, id string) (ModelLease, error) {
	registry.mutex.RLock()
	model := registry.models[id]
	factory := registry.factories[id]
	registry.mutex.RUnlock()
	if model != nil {
		return ModelLease{Model: model, Close: func(context.Context) error { return nil }}, nil
	}
	if factory == nil {
		return ModelLease{}, fmt.Errorf("durable: unknown model ID %q", id)
	}
	model, err := factory(ctx)
	if err != nil {
		return ModelLease{}, fmt.Errorf("durable: rebuild model %q: %w", id, err)
	}
	if modelIsNil(model) {
		return ModelLease{}, fmt.Errorf("durable: model factory %q returned nil", id)
	}
	close := ai.ModelCloseFunc(func(context.Context) error { return nil })
	if opener, ok := model.(ai.ModelOpener); ok {
		close, err = opener.OpenModel(ctx)
		if err != nil {
			return ModelLease{}, fmt.Errorf("durable: open model %q: %w", id, err)
		}
		if close == nil {
			close = func(context.Context) error { return nil }
		}
	}
	return ModelLease{Model: model, Close: close}, nil
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
