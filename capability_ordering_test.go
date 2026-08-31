package ai_test

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type orderingRecorder struct {
	name string
	log  *[]string
}

func (capability *orderingRecorder) Setup(*ai.CapabilityRegistry) error {
	*capability.log = append(*capability.log, capability.name+":setup")
	return nil
}

func (capability *orderingRecorder) WrapRun(
	ctx context.Context,
	runInfo *ai.RunInfo,
	next ai.RunFunc,
) (ai.RunOutcome, error) {
	*capability.log = append(*capability.log, capability.name+":in")
	outcome, err := next(ctx)
	*capability.log = append(*capability.log, capability.name+":out")
	return outcome, err
}

type outerOrderingCapability struct{ *orderingRecorder }

func (*outerOrderingCapability) CapabilityOrdering() ai.CapabilityOrdering {
	return ai.CapabilityOrdering{Position: ai.CapabilityOutermost}
}

type innerOrderingCapability struct{ *orderingRecorder }

func (*innerOrderingCapability) CapabilityOrdering() ai.CapabilityOrdering {
	return ai.CapabilityOrdering{Position: ai.CapabilityInnermost}
}

type defaultOrderingCapability struct{ *orderingRecorder }

func TestCapabilityPositionTiersAreStable(t *testing.T) {
	var log []string
	capability := func(name string) *orderingRecorder { return &orderingRecorder{name: name, log: &log} }
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(
		&innerOrderingCapability{capability("inner-1")},
		&defaultOrderingCapability{capability("default-1")},
		&outerOrderingCapability{capability("outer-1")},
		&innerOrderingCapability{capability("inner-2")},
		ai.CombineCapabilities(
			&outerOrderingCapability{capability("outer-2")},
			&defaultOrderingCapability{capability("default-2")},
		),
	))
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	want := []string{
		"outer-1:setup", "outer-2:setup", "default-1:setup", "default-2:setup", "inner-1:setup", "inner-2:setup",
		"outer-1:in", "outer-2:in", "default-1:in", "default-2:in", "inner-1:in", "inner-2:in",
		"inner-2:out", "inner-1:out", "default-2:out", "default-1:out", "outer-2:out", "outer-1:out",
	}
	if !slices.Equal(log, want) {
		t.Fatalf("unexpected tier order:\n got %v\nwant %v", log, want)
	}
}

func TestRunCapabilityTiersRemainInsideAgentCapabilities(t *testing.T) {
	var log []string
	capability := func(name string) *orderingRecorder { return &orderingRecorder{name: name, log: &log} }
	agent := ai.NewAgent[deps, string](
		fakes.NewTestModel(), ai.WithCapabilities(&defaultOrderingCapability{capability("agent")}),
	)
	if _, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunCapabilities(
		&innerOrderingCapability{capability("run-inner")},
		&outerOrderingCapability{capability("run-outer")},
	)); err != nil {
		t.Fatal(err)
	}
	want := []string{
		"agent:setup", "run-outer:setup", "run-inner:setup",
		"agent:in", "run-outer:in", "run-inner:in",
		"run-inner:out", "run-outer:out", "agent:out",
	}
	if !slices.Equal(log, want) {
		t.Fatalf("unexpected scoped tier order: got %v want %v", log, want)
	}
}

type relativeTargetCapability struct{ *orderingRecorder }
type relativeOuterCapability struct{ *orderingRecorder }
type relativeInnerCapability struct{ *orderingRecorder }

func (*relativeOuterCapability) CapabilityOrdering() ai.CapabilityOrdering {
	return ai.CapabilityOrdering{Wraps: []ai.CapabilityReference{ai.CapabilityType[*relativeTargetCapability]()}}
}

func (*relativeInnerCapability) CapabilityOrdering() ai.CapabilityOrdering {
	return ai.CapabilityOrdering{WrappedBy: []ai.CapabilityReference{ai.CapabilityType[*relativeTargetCapability]()}}
}

func TestCapabilityRelativeOrderingAndInterfaceRequirements(t *testing.T) {
	var log []string
	capability := func(name string) *orderingRecorder { return &orderingRecorder{name: name, log: &log} }
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(
		&relativeInnerCapability{capability("inner")},
		&relativeTargetCapability{capability("target")},
		&relativeOuterCapability{capability("outer")},
		&requiresRunWrapperCapability{orderingRecorder: capability("requires")},
	))
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	wantSetup := []string{"outer:setup", "target:setup", "inner:setup", "requires:setup"}
	if !slices.Equal(log[:4], wantSetup) {
		t.Fatalf("unexpected relative setup order: got %v want %v", log[:4], wantSetup)
	}
}

type runWrapperCapability interface {
	ai.Capability
	ai.RunWrapper
}

type requiresRunWrapperCapability struct{ *orderingRecorder }

func (*requiresRunWrapperCapability) CapabilityOrdering() ai.CapabilityOrdering {
	return ai.CapabilityOrdering{Requires: []ai.CapabilityReference{ai.CapabilityType[runWrapperCapability]()}}
}

type instanceOrderingCapability struct {
	*orderingRecorder
	wrapped ai.CapabilityReference
}

func (capability *instanceOrderingCapability) CapabilityOrdering() ai.CapabilityOrdering {
	return ai.CapabilityOrdering{Wraps: []ai.CapabilityReference{capability.wrapped}}
}

func TestCapabilityInstanceReferenceMatchesOnlyTargetPointer(t *testing.T) {
	var log []string
	first := &defaultOrderingCapability{&orderingRecorder{name: "first", log: &log}}
	second := &defaultOrderingCapability{&orderingRecorder{name: "second", log: &log}}
	wrapper := &instanceOrderingCapability{
		orderingRecorder: &orderingRecorder{name: "wrapper", log: &log},
		wrapped:          ai.CapabilityInstance(second),
	}
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(first, second, wrapper))
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	wantSetup := []string{"first:setup", "wrapper:setup", "second:setup"}
	if !slices.Equal(log[:3], wantSetup) {
		t.Fatalf("instance reference matched wrong capability: got %v want %v", log[:3], wantSetup)
	}
}

type missingRequirementCapability struct{ *orderingRecorder }

func (*missingRequirementCapability) CapabilityOrdering() ai.CapabilityOrdering {
	return ai.CapabilityOrdering{Requires: []ai.CapabilityReference{ai.CapabilityType[*relativeTargetCapability]()}}
}

type wrapsInnerCapability struct{ *orderingRecorder }

func (*wrapsInnerCapability) CapabilityOrdering() ai.CapabilityOrdering {
	return ai.CapabilityOrdering{Wraps: []ai.CapabilityReference{ai.CapabilityType[*wrappedByOuterCapability]()}}
}

type wrappedByOuterCapability struct{ *orderingRecorder }

func (*wrappedByOuterCapability) CapabilityOrdering() ai.CapabilityOrdering {
	return ai.CapabilityOrdering{Wraps: []ai.CapabilityReference{ai.CapabilityType[*wrapsInnerCapability]()}}
}

type invalidPositionCapability struct{ *orderingRecorder }

func (*invalidPositionCapability) CapabilityOrdering() ai.CapabilityOrdering {
	return ai.CapabilityOrdering{Position: "sideways"}
}

type referenceRequirementCapability struct {
	*orderingRecorder
	reference ai.CapabilityReference
}

func (capability *referenceRequirementCapability) CapabilityOrdering() ai.CapabilityOrdering {
	return ai.CapabilityOrdering{Requires: []ai.CapabilityReference{capability.reference}}
}

func TestAgentCapabilityOrderingErrorsPanic(t *testing.T) {
	var log []string
	recorder := func(name string) *orderingRecorder { return &orderingRecorder{name: name, log: &log} }
	unregistered := &defaultOrderingCapability{recorder("unregistered")}
	tests := []struct {
		name         string
		capabilities []ai.Capability
		contains     string
	}{
		{
			name: "missing requirement",
			capabilities: []ai.Capability{
				&missingRequirementCapability{recorder("missing")},
			},
			contains: "requires *ai_test.relativeTargetCapability, but it is not registered",
		},
		{
			name: "missing instance requirement",
			capabilities: []ai.Capability{
				&referenceRequirementCapability{
					orderingRecorder: recorder("instance"), reference: ai.CapabilityInstance(unregistered),
				},
				&defaultOrderingCapability{recorder("other")},
			},
			contains: "requires instance of *ai_test.defaultOrderingCapability, but it is not registered",
		},
		{
			name: "invalid requirement reference",
			capabilities: []ai.Capability{
				&referenceRequirementCapability{orderingRecorder: recorder("invalid")},
			},
			contains: "requires <invalid capability reference>, but it is not registered",
		},
		{
			name: "cycle",
			capabilities: []ai.Capability{
				&wrapsInnerCapability{recorder("first")}, &wrappedByOuterCapability{recorder("second")},
			},
			contains: "circular capability ordering constraints",
		},
		{
			name: "invalid position",
			capabilities: []ai.Capability{
				&invalidPositionCapability{recorder("invalid")},
			},
			contains: `invalid position "sideways"`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			deferred := func() (recovered any) {
				defer func() { recovered = recover() }()
				ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(test.capabilities...))
				return nil
			}()
			if message := fmt.Sprint(deferred); !strings.Contains(message, test.contains) {
				t.Fatalf("unexpected panic: %v", deferred)
			}
		})
	}
}

func TestRunCapabilityRequirementCanUseAgentCapability(t *testing.T) {
	var log []string
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(
		&relativeTargetCapability{&orderingRecorder{name: "agent-target", log: &log}},
	))
	if _, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunCapabilities(
		&missingRequirementCapability{&orderingRecorder{name: "run-requirer", log: &log}},
	)); err != nil {
		t.Fatal(err)
	}
	wantPrefix := []string{"agent-target:setup", "run-requirer:setup"}
	if !slices.Equal(log[:2], wantPrefix) {
		t.Fatalf("unexpected requirement setup: got %v want %v", log[:2], wantPrefix)
	}
}

func TestRunCapabilityOrderingFailureIsReturned(t *testing.T) {
	var log []string
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	_, err := agent.Run(t.Context(), "go", deps{}, ai.WithRunCapabilities(
		&missingRequirementCapability{&orderingRecorder{name: "missing", log: &log}},
	))
	if err == nil || !strings.Contains(err.Error(), "ai: run capability ordering:") {
		t.Fatalf("unexpected run ordering error: %v", err)
	}
	if len(log) != 0 {
		t.Fatalf("capability was set up after ordering failure: %v", log)
	}
}

func TestCapabilityReferenceValidation(t *testing.T) {
	t.Run("value", func(t *testing.T) {
		defer func() {
			if recovered := recover(); recovered != "ai: capability instance reference requires a non-nil pointer" {
				t.Fatalf("unexpected panic: %v", recovered)
			}
		}()
		ai.CapabilityInstance(valueCapability{})
	})
	t.Run("nil pointer", func(t *testing.T) {
		defer func() {
			if recovered := recover(); recovered != "ai: capability instance reference requires a non-nil pointer" {
				t.Fatalf("unexpected panic: %v", recovered)
			}
		}()
		var capability *defaultOrderingCapability
		ai.CapabilityInstance(capability)
	})
}

type valueCapability struct{}

func (valueCapability) Setup(*ai.CapabilityRegistry) error { return errors.New("unused") }
