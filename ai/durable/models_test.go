package durable_test

import (
	"context"
	"errors"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/durable"
)

type valueModel struct{}

func (valueModel) Name() string { return "value" }
func (valueModel) Request(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
	return &ai.ModelResponse{}, nil
}

type lifecycleModel struct {
	opened   int
	closed   int
	openErr  error
	nilClose bool
}

func (*lifecycleModel) Name() string { return "model" }
func (*lifecycleModel) Request(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
	return &ai.ModelResponse{}, nil
}
func (model *lifecycleModel) OpenModel(context.Context) (ai.ModelCloseFunc, error) {
	model.opened++
	if model.openErr != nil {
		return nil, model.openErr
	}
	if model.nilClose {
		return nil, nil
	}
	return func(context.Context) error {
		model.closed++
		return nil
	}, nil
}

func TestModelRegistryOwnership(t *testing.T) {
	registry := &durable.ModelRegistry{}
	owned := &lifecycleModel{}
	if err := registry.Register("owned", owned); err != nil {
		t.Fatal(err)
	}
	lease, err := registry.Acquire(context.Background(), "owned")
	if err != nil {
		t.Fatal(err)
	}
	if err := lease.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if lease.Model != owned || owned.opened != 0 || owned.closed != 0 {
		t.Fatalf("caller-owned model lifecycle changed: %#v", owned)
	}

	rebuilt := &lifecycleModel{}
	builds := 0
	if err := registry.RegisterFactory("rebuilt", func(context.Context) (ai.Model, error) {
		builds++
		return rebuilt, nil
	}); err != nil {
		t.Fatal(err)
	}
	lease, err = registry.Acquire(context.Background(), "rebuilt")
	if err != nil {
		t.Fatal(err)
	}
	if err := lease.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if builds != 1 || rebuilt.opened != 1 || rebuilt.closed != 1 {
		t.Fatalf("rebuilt lifecycle mismatch: builds=%d model=%#v", builds, rebuilt)
	}

	plain := valueModel{}
	if err := registry.RegisterFactory("plain", func(context.Context) (ai.Model, error) { return plain, nil }); err != nil {
		t.Fatal(err)
	}
	lease, err = registry.Acquire(context.Background(), "plain")
	if err != nil || lease.Close(context.Background()) != nil {
		t.Fatalf("unexpected plain lease: %#v %v", lease, err)
	}

	nilClose := &lifecycleModel{nilClose: true}
	if err := registry.RegisterFactory("nil-close", func(context.Context) (ai.Model, error) { return nilClose, nil }); err != nil {
		t.Fatal(err)
	}
	lease, err = registry.Acquire(context.Background(), "nil-close")
	if err != nil || lease.Close(context.Background()) != nil {
		t.Fatalf("unexpected nil-close lease: %#v %v", lease, err)
	}
}

func TestModelRegistryErrors(t *testing.T) {
	registry := &durable.ModelRegistry{}
	var typedNil *lifecycleModel
	for _, test := range []struct {
		id    string
		model ai.Model
	}{
		{"", valueModel{}}, {"nil", nil}, {"typed", typedNil},
	} {
		if err := registry.Register(test.id, test.model); err == nil {
			t.Fatalf("expected register error for %q", test.id)
		}
	}
	if err := registry.RegisterFactory("", func(context.Context) (ai.Model, error) { return valueModel{}, nil }); err == nil {
		t.Fatal("expected empty factory ID error")
	}
	if err := registry.RegisterFactory("nil", nil); err == nil {
		t.Fatal("expected nil factory error")
	}
	if err := registry.Register("duplicate", valueModel{}); err != nil {
		t.Fatal(err)
	}
	if err := registry.Register("duplicate", valueModel{}); err == nil {
		t.Fatal("expected duplicate model error")
	}
	if err := registry.RegisterFactory("duplicate", func(context.Context) (ai.Model, error) { return valueModel{}, nil }); err == nil {
		t.Fatal("expected model/factory collision")
	}
	if err := registry.RegisterFactory("factory", func(context.Context) (ai.Model, error) { return valueModel{}, nil }); err != nil {
		t.Fatal(err)
	}
	if err := registry.RegisterFactory("factory", func(context.Context) (ai.Model, error) { return valueModel{}, nil }); err == nil {
		t.Fatal("expected duplicate factory error")
	}
	if err := registry.Register("factory", valueModel{}); err == nil {
		t.Fatal("expected factory/model collision")
	}
	if _, err := registry.Acquire(context.Background(), "missing"); err == nil {
		t.Fatal("expected unknown model error")
	}

	failure := errors.New("failed")
	if err := registry.RegisterFactory("build-error", func(context.Context) (ai.Model, error) { return nil, failure }); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Acquire(context.Background(), "build-error"); !errors.Is(err, failure) {
		t.Fatalf("unexpected factory error: %v", err)
	}
	if err := registry.RegisterFactory("nil-model", func(context.Context) (ai.Model, error) { return nil, nil }); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Acquire(context.Background(), "nil-model"); err == nil {
		t.Fatal("expected nil factory model error")
	}
	if err := registry.RegisterFactory("typed-nil", func(context.Context) (ai.Model, error) {
		var model *lifecycleModel
		return model, nil
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Acquire(context.Background(), "typed-nil"); err == nil {
		t.Fatal("expected typed nil factory model error")
	}
	openFailure := &lifecycleModel{openErr: failure}
	if err := registry.RegisterFactory("open-error", func(context.Context) (ai.Model, error) { return openFailure, nil }); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Acquire(context.Background(), "open-error"); !errors.Is(err, failure) {
		t.Fatalf("unexpected open error: %v", err)
	}
}
