package ai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type modelTraceWrapper struct {
	ai.WrappedCapability
	name string
	log  *[]string
}

func newModelTraceWrapper(name string, log *[]string, wrapped ai.Capability) *modelTraceWrapper {
	return &modelTraceWrapper{WrappedCapability: ai.WrapCapability(wrapped), name: name, log: log}
}

func (wrapper *modelTraceWrapper) WrapModelRequest(
	ctx context.Context,
	runInfo *ai.RunInfo,
	messages []ai.ModelMessage,
	params ai.ModelRequestParams,
	next ai.ModelRequestFunc,
) (*ai.ModelResponse, error) {
	*wrapper.log = append(*wrapper.log, wrapper.name+":in")
	response, err := next(ctx, messages, params)
	*wrapper.log = append(*wrapper.log, wrapper.name+":out")
	return response, err
}

func TestWrappedCapabilityDelegatesUnchangedLifecycle(t *testing.T) {
	var log []string
	leaf := &traceCapability{
		name: "leaf",
		log:  &log,
		setup: func(*ai.CapabilityRegistry) error {
			log = append(log, "leaf:setup")
			return nil
		},
	}
	inner := newModelTraceWrapper("inner", &log, leaf)
	outer := newModelTraceWrapper("outer", &log, inner)
	if outer.Wrapped() != inner || inner.Wrapped() != leaf {
		t.Fatal("wrapped capability identity was not preserved")
	}
	result, err := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(outer)).Run(
		t.Context(), "go", deps{},
	)
	if err != nil || result.Output == "" {
		t.Fatalf("wrapped run failed: result=%+v err=%v", result, err)
	}
	want := []string{
		"leaf:setup", "leaf:run-in",
		"outer:in", "inner:in", "leaf:model", "inner:out", "outer:out",
		"leaf:run-out",
	}
	if !slices.Equal(log, want) {
		t.Fatalf("unexpected wrapper order:\n got %v\nwant %v", log, want)
	}
}

func TestBareWrappedCapabilityPreservesInnerOrdering(t *testing.T) {
	var log []string
	inner := &innerOrderingCapability{&orderingRecorder{name: "inner", log: &log}}
	wrapped := ai.WrapCapability(inner)
	defaultCapability := &defaultOrderingCapability{&orderingRecorder{name: "default", log: &log}}
	if _, err := ai.NewAgent[deps, string](
		fakes.NewTestModel(), ai.WithCapabilities(wrapped, defaultCapability),
	).Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	want := []string{
		"default:setup", "inner:setup",
		"default:in", "inner:in", "inner:out", "default:out",
	}
	if !slices.Equal(log, want) {
		t.Fatalf("wrapper did not adopt inner ordering: got %v want %v", log, want)
	}
}

func TestWrappedCapabilityCanAddSetupContributions(t *testing.T) {
	var calls int
	leaf := setupToolCapability{calls: &calls}
	wrapper := &setupDecorator{WrappedCapability: ai.WrapCapability(leaf), calls: &calls}
	result, err := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(wrapper)).Run(
		t.Context(), "use tools", deps{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 || result.Output == "" {
		t.Fatalf("wrapper or wrapped setup did not run once: calls=%d result=%+v", calls, result)
	}
}

type setupToolCapability struct{ calls *int }

func (capability setupToolCapability) Setup(registry *ai.CapabilityRegistry) error {
	*capability.calls++
	registry.AddTool(ai.ToolDefinition{Name: "leaf_tool", Description: "Leaf"}, func(
		context.Context, json.RawMessage,
	) (any, error) {
		return "leaf", nil
	})
	return nil
}

type setupDecorator struct {
	ai.WrappedCapability
	calls *int
}

func (wrapper *setupDecorator) Setup(registry *ai.CapabilityRegistry) error {
	*wrapper.calls++
	registry.AddInstructions("decorated")
	return nil
}

func TestWrappedCapabilityValidation(t *testing.T) {
	t.Run("zero value", func(t *testing.T) {
		deferred := func() (recovered any) {
			defer func() { recovered = recover() }()
			ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(ai.WrappedCapability{}))
			return nil
		}()
		if !strings.Contains(fmt.Sprint(deferred), "wrapped capability must not be nil") {
			t.Fatalf("unexpected zero wrapper panic: %v", deferred)
		}
	})
	t.Run("nil", func(t *testing.T) {
		defer func() {
			if recovered := recover(); recovered != "ai: wrapped capability must not be nil" {
				t.Fatalf("unexpected nil panic: %v", recovered)
			}
		}()
		var capability *traceCapability
		ai.WrapCapability(capability)
	})
	t.Run("combined", func(t *testing.T) {
		defer func() {
			if recovered := recover(); recovered != "ai: wrap individual capabilities before combining them" {
				t.Fatalf("unexpected combined panic: %v", recovered)
			}
		}()
		ai.WrapCapability(ai.CombineCapabilities(&traceCapability{}, &traceCapability{}))
	})
}
