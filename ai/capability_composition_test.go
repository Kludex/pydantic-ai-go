package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type mergingCapability struct {
	Name   string
	Labels []string
	Values map[string]string
	Log    *[]string
}

func (capability mergingCapability) CapabilityID() string { return "merging" }

func (capability mergingCapability) CombineCapabilities(capabilities []ai.Capability) (ai.Capability, error) {
	values := make([]ai.Capability, len(capabilities))
	for index, candidate := range capabilities {
		typed := candidate.(mergingCapability)
		typed.Log = nil
		values[index] = typed
	}
	merged, err := ai.MergeCapabilities(values...)
	if err != nil {
		return nil, err
	}
	result := merged.(mergingCapability)
	result.Log = capability.Log
	return result, nil
}

func (capability mergingCapability) Setup(registry *ai.CapabilityRegistry) error {
	registry.AddInstructions(fmt.Sprintf("%s:%s:%s", capability.Name, strings.Join(capability.Labels, ","), capability.Values["key"]))
	return nil
}

func (capability mergingCapability) WrapModelRequest(
	ctx context.Context,
	_ *ai.RunInfo,
	messages []ai.ModelMessage,
	params ai.ModelRequestParams,
	next ai.ModelRequestFunc,
) (*ai.ModelResponse, error) {
	if capability.Log != nil {
		*capability.Log = append(*capability.Log, capability.Name)
	}
	return next(ctx, messages, params)
}

type pointerMergeCapability struct {
	Names  []string
	Labels map[string]string
	Limit  *int
	Value  any
}

func (pointerMergeCapability) Setup(*ai.CapabilityRegistry) error { return nil }

type functionCapability func()

func (functionCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func TestMergeCapabilitiesValidationAndFieldRules(t *testing.T) {
	if _, err := ai.MergeCapabilities(); err == nil || !strings.Contains(err.Error(), "empty") {
		t.Fatalf("unexpected empty merge error: %v", err)
	}
	var absent ai.Capability
	if _, err := ai.MergeCapabilities(absent); err == nil || !strings.Contains(err.Error(), "nil") {
		t.Fatalf("unexpected nil merge error: %v", err)
	}
	var typedNil *pointerMergeCapability
	if _, err := ai.MergeCapabilities(typedNil); err == nil || !strings.Contains(err.Error(), "nil") {
		t.Fatalf("unexpected typed nil merge error: %v", err)
	}
	if _, err := ai.MergeCapabilities(&pointerMergeCapability{}, collidingCapability{}); err == nil ||
		!strings.Contains(err.Error(), "different types") {
		t.Fatalf("unexpected mixed type error: %v", err)
	}
	if _, err := ai.MergeCapabilities(functionCapability(func() {})); err == nil ||
		!strings.Contains(err.Error(), "requires a struct") {
		t.Fatalf("unexpected non-struct error: %v", err)
	}

	limit := 3
	merged, err := ai.MergeCapabilities(
		&pointerMergeCapability{
			Names: []string{"a"}, Labels: map[string]string{"shared": "first"}, Limit: &limit,
			Value: []string{"one"},
		},
		&pointerMergeCapability{
			Names: []string{"b", "a"}, Labels: map[string]string{"shared": "second", "last": "yes"},
			Value: []string{"two"},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	got := merged.(*pointerMergeCapability)
	if !slices.Equal(got.Names, []string{"a", "b"}) || got.Labels["shared"] != "second" ||
		got.Labels["last"] != "yes" || got.Limit == nil || *got.Limit != 3 ||
		!slices.Equal(got.Value.([]string), []string{"one", "two"}) {
		t.Fatalf("unexpected default merge: %#v", got)
	}

	equal, err := ai.MergeCapabilities(
		pointerMergeCapability{Names: []string{"same"}, Labels: map[string]string{"same": "yes"}},
		pointerMergeCapability{Names: []string{"same"}, Labels: map[string]string{"same": "yes"}},
	)
	if err != nil || !slices.Equal(equal.(pointerMergeCapability).Names, []string{"same"}) {
		t.Fatalf("equal collections were not preserved: %#v, %v", equal, err)
	}
	later, err := ai.MergeCapabilities(
		pointerMergeCapability{Value: "first"}, pointerMergeCapability{Value: "second"},
	)
	if err != nil || later.(pointerMergeCapability).Value != "second" {
		t.Fatalf("later scalar did not win: %#v, %v", later, err)
	}
}

func TestRepeatedCapabilityIDsCombineWithinOneLayer(t *testing.T) {
	var instructions string
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		instructions = params.Instructions
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		mergingCapability{Name: "first", Labels: []string{"a"}, Values: map[string]string{"first": "1"}},
		mergingCapability{Name: "second", Labels: []string{"b", "a"}, Values: map[string]string{"key": "2"}},
	))
	if _, err := agent.Run(t.Context(), "go", struct{}{}); err != nil {
		t.Fatal(err)
	}
	if instructions != "second:a,b:2" {
		t.Fatalf("unexpected merged instructions %q", instructions)
	}
}

type failingCombineCapability struct {
	id     string
	result ai.Capability
	err    error
}

func (capability failingCombineCapability) CapabilityID() string    { return capability.id }
func (failingCombineCapability) Setup(*ai.CapabilityRegistry) error { return nil }
func (capability failingCombineCapability) CombineCapabilities([]ai.Capability) (ai.Capability, error) {
	return capability.result, capability.err
}

type shiftingIDCapability struct {
	calls int
}

func (capability *shiftingIDCapability) CapabilityID() string {
	capability.calls++
	if capability.calls < 4 {
		return "same"
	}
	return ""
}
func (*shiftingIDCapability) Setup(*ai.CapabilityRegistry) error { return nil }

type collidingCapability struct{ id string }

func (capability collidingCapability) CapabilityID() string    { return capability.id }
func (collidingCapability) Setup(*ai.CapabilityRegistry) error { return nil }

type otherCollidingCapability struct{ id string }

func (capability otherCollidingCapability) CapabilityID() string    { return capability.id }
func (otherCollidingCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func TestBuiltInCapabilityCombinationPolicies(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(
		ai.RaiseContentFilterError{}, ai.RaiseContentFilterError{},
		ai.ReinjectSystemPrompt{}, ai.ReinjectSystemPrompt{ReplaceExisting: true},
	))
	if _, err := agent.Run(t.Context(), "go", struct{}{}); err != nil {
		t.Fatal(err)
	}
	instrumentation := ai.NewInstrumentation()
	if _, err := instrumentation.CombineCapabilities(nil); err == nil {
		t.Fatal("empty instrumentation combination succeeded")
	}
	if _, err := instrumentation.CombineCapabilities([]ai.Capability{instrumentation, collidingCapability{}}); err == nil {
		t.Fatal("mixed instrumentation combination succeeded")
	}

	native := ai.NewImageGenerationCapability(ai.ImageGenerationCapabilityConfig[struct{}]{
		Native: ai.ImageGenerationTool{},
	})
	if _, err := native.CombineCapabilities(nil); err == nil {
		t.Fatal("empty native-or-local combination succeeded")
	}
	if _, err := native.CombineCapabilities([]ai.Capability{native, collidingCapability{}}); err == nil {
		t.Fatal("mixed native-or-local combination succeeded")
	}
	var absent *ai.NativeOrLocalTool[struct{}]
	if absent.CapabilityID() != "" {
		t.Fatal("nil native-or-local capability has an ID")
	}
}

func TestCapabilityCombinationFailuresAreRejected(t *testing.T) {
	tests := []struct {
		name         string
		capabilities []ai.Capability
		contains     string
	}{
		{
			name: "nil capability", capabilities: []ai.Capability{(*pointerMergeCapability)(nil)},
			contains: "capability must not be nil",
		},
		{
			name: "combiner error",
			capabilities: []ai.Capability{
				failingCombineCapability{id: "same"},
				failingCombineCapability{id: "same", err: errors.New("no merge")},
			},
			contains: "no merge",
		},
		{
			name: "nil result",
			capabilities: []ai.Capability{
				failingCombineCapability{id: "same"}, failingCombineCapability{id: "same"},
			},
			contains: "returned nil",
		},
		{
			name: "changed ID",
			capabilities: []ai.Capability{
				failingCombineCapability{id: "same"},
				failingCombineCapability{id: "same", result: failingCombineCapability{id: "other"}},
			},
			contains: `returned capability ID "other"`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recovered := fmt.Sprint(recover()); !strings.Contains(recovered, test.contains) {
					t.Fatalf("unexpected composition panic: %s", recovered)
				}
			}()
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(test.capabilities...))
		})
	}
}

func TestCapabilityIDCollisionsAreRejected(t *testing.T) {
	for _, capabilities := range [][]ai.Capability{
		{collidingCapability{id: "same"}, collidingCapability{id: "same"}},
		{collidingCapability{id: "same"}, otherCollidingCapability{id: "same"}},
	} {
		func() {
			defer func() {
				if recovered := fmt.Sprint(recover()); !strings.Contains(recovered, `capability ID "same"`) {
					t.Fatalf("unexpected collision panic: %s", recovered)
				}
			}()
			ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(capabilities...))
		}()
	}
}

type hiddenCollectionCapability struct {
	Names  []string
	cached []string
}

func (hiddenCollectionCapability) CapabilityID() string               { return "hidden" }
func (hiddenCollectionCapability) Setup(*ai.CapabilityRegistry) error { return nil }
func (hiddenCollectionCapability) CombineCapabilities(capabilities []ai.Capability) (ai.Capability, error) {
	return ai.MergeCapabilities(capabilities...)
}

func TestDefaultCapabilityMergeRejectsUndeclaredDerivedState(t *testing.T) {
	defer func() {
		message := fmt.Sprint(recover())
		if !strings.Contains(message, `unexported field "cached"`) || !strings.Contains(message, "preserve derived state") {
			t.Fatalf("unexpected merge panic: %s", message)
		}
	}()
	ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(
		hiddenCollectionCapability{Names: []string{"a"}, cached: []string{"a"}},
		hiddenCollectionCapability{Names: []string{"b"}, cached: []string{"b"}},
	))
}

type alternateNames []string

type unrebuildableCollectionCapability struct {
	Names any
}

func (unrebuildableCollectionCapability) CapabilityID() string               { return "collection" }
func (unrebuildableCollectionCapability) Setup(*ai.CapabilityRegistry) error { return nil }
func (unrebuildableCollectionCapability) CombineCapabilities(capabilities []ai.Capability) (ai.Capability, error) {
	return ai.MergeCapabilities(capabilities...)
}

func TestDefaultCapabilityMergeRejectsUnrebuildableCollections(t *testing.T) {
	defer func() {
		message := fmt.Sprint(recover())
		if !strings.Contains(message, "cannot be rebuilt") || !strings.Contains(message, `field "Names"`) {
			t.Fatalf("unexpected merge panic: %s", message)
		}
	}()
	ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(
		unrebuildableCollectionCapability{Names: []string{"a"}},
		unrebuildableCollectionCapability{Names: alternateNames{"b"}},
	))
}

type derivedCapability struct {
	Names []string
	count int
}

func (derivedCapability) CapabilityID() string { return "derived" }
func (capability derivedCapability) Setup(registry *ai.CapabilityRegistry) error {
	registry.AddInstructions(fmt.Sprintf("count:%d", capability.count))
	return nil
}
func (derivedCapability) CombineCapabilities(capabilities []ai.Capability) (ai.Capability, error) {
	var names []string
	for _, capability := range capabilities {
		for _, name := range capability.(derivedCapability).Names {
			if !slices.Contains(names, name) {
				names = append(names, name)
			}
		}
	}
	return derivedCapability{Names: names, count: len(names)}, nil
}

func TestCapabilityCombinerRebuildsDerivedState(t *testing.T) {
	var instructions string
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		instructions = params.Instructions
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		derivedCapability{Names: []string{"a"}, count: 1}, derivedCapability{Names: []string{"b"}, count: 1},
	))
	if _, err := agent.Run(t.Context(), "go", struct{}{}); err != nil {
		t.Fatal(err)
	}
	if instructions != "count:2" {
		t.Fatalf("derived state was not rebuilt: %q", instructions)
	}
}

type replacementCapability struct {
	ToolName string
}

func (replacementCapability) CapabilityID() string { return "replacement" }
func (replacementCapability) CombineCapabilities(capabilities []ai.Capability) (ai.Capability, error) {
	return ai.MergeCapabilities(capabilities...)
}
func (capability replacementCapability) Setup(registry *ai.CapabilityRegistry) error {
	registry.AddTool(ai.ToolDefinition{
		Name: capability.ToolName, Schema: map[string]any{"type": "object", "additionalProperties": false},
	}, func(context.Context, json.RawMessage) (any, error) {
		return capability.ToolName, nil
	})
	return nil
}

func TestRunCapabilityReplacesAgentSetupContributions(t *testing.T) {
	var tools []string
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		for _, tool := range params.Tools {
			tools = append(tools, tool.Name)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(replacementCapability{ToolName: "agent_tool"}))
	if _, err := agent.Run(
		t.Context(), "go", struct{}{}, ai.WithRunCapabilities(replacementCapability{ToolName: "run_tool"}),
	); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tools, []string{"run_tool"}) {
		t.Fatalf("agent capability contribution survived replacement: %v", tools)
	}
}

type compositionWrapper struct {
	ai.WrappedCapability
	id   string
	name string
	log  *[]string
}

func (wrapper compositionWrapper) CapabilityID() string { return wrapper.id }

func (wrapper compositionWrapper) WrapModelRequest(
	ctx context.Context,
	_ *ai.RunInfo,
	messages []ai.ModelMessage,
	params ai.ModelRequestParams,
	next ai.ModelRequestFunc,
) (*ai.ModelResponse, error) {
	*wrapper.log = append(*wrapper.log, wrapper.name+":in")
	response, err := next(ctx, messages, params)
	*wrapper.log = append(*wrapper.log, wrapper.name+":out")
	return response, err
}

func TestNestedCapabilityInstructionIDsRemainUnique(t *testing.T) {
	var log []string
	first := identifiedInstructionCapability{id: "nested", static: []ai.InstructionPart{{Content: "first"}}}
	second := identifiedInstructionCapability{id: "nested", static: []ai.InstructionPart{{Content: "second"}}}
	deferred := func() (recovered any) {
		defer func() { recovered = recover() }()
		ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(
			compositionWrapper{WrappedCapability: ai.WrapCapability(first), id: "first-root", log: &log},
			compositionWrapper{WrappedCapability: ai.WrapCapability(second), id: "second-root", log: &log},
		))
		return nil
	}()
	if !strings.Contains(fmt.Sprint(deferred), "multiple capabilities") {
		t.Fatalf("unexpected nested instruction collision: %v", deferred)
	}

	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(
		compositionWrapper{WrappedCapability: ai.WrapCapability(first), id: "agent-root", log: &log},
	))
	_, err := agent.Run(t.Context(), "go", struct{}{}, ai.WithRunCapabilities(
		compositionWrapper{WrappedCapability: ai.WrapCapability(second), id: "run-root", log: &log},
	))
	if err == nil || !strings.Contains(err.Error(), "multiple capabilities") {
		t.Fatalf("unexpected run nested instruction collision: %v", err)
	}
}

func TestRunCapabilityCompositionError(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	_, err := agent.Run(t.Context(), "go", struct{}{}, ai.WithRunCapabilities(
		collidingCapability{id: "same"}, collidingCapability{id: "same"},
	))
	if err == nil || !strings.Contains(err.Error(), "run capability composition") {
		t.Fatalf("unexpected run composition error: %v", err)
	}
}

func TestExplicitWrapperIdentityKeepsNestedMiddlewareActive(t *testing.T) {
	inner := ai.NewInstrumentation()
	wrapper := compositionWrapper{WrappedCapability: ai.WrapCapability(inner), id: "instrumented-wrapper", log: &[]string{}}
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(wrapper, ai.NewInstrumentation()))
	ai.AddSimpleTool(agent, "ping", func(context.Context, struct{}) (string, error) { return "pong", nil })
	if _, err := agent.Run(t.Context(), "go", struct{}{}); err != nil {
		t.Fatal(err)
	}
}

func TestRunCapabilityReplacesWholeWrapperSubtree(t *testing.T) {
	var log []string
	agentLeaf := mergingCapability{Name: "agent", Log: &log}
	runLeaf := mergingCapability{Name: "run", Log: &log}
	agentWrapper := compositionWrapper{WrappedCapability: ai.WrapCapability(agentLeaf), name: "agent-wrapper", log: &log}
	runWrapper := compositionWrapper{WrappedCapability: ai.WrapCapability(runLeaf), name: "run-wrapper", log: &log}
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(agentWrapper))
	if _, err := agent.Run(t.Context(), "go", struct{}{}, ai.WithRunCapabilities(runWrapper)); err != nil {
		t.Fatal(err)
	}
	want := []string{"run-wrapper:in", "run", "run-wrapper:out"}
	if !slices.Equal(log, want) {
		t.Fatalf("wrapper subtree replacement changed order: got %v want %v", log, want)
	}
}

func TestCapabilityIdentityTypeHandlesAnIdentityThatChanges(t *testing.T) {
	agentCapability := &shiftingIDCapability{}
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(agentCapability))
	runCapability := &shiftingIDCapability{}
	if _, err := agent.Run(t.Context(), "go", struct{}{}, ai.WithRunCapabilities(runCapability)); err != nil {
		t.Fatal(err)
	}
}

func TestRunCapabilityRejectsCrossTypeReplacement(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(collidingCapability{id: "same"}))
	_, err := agent.Run(
		t.Context(), "go", struct{}{}, ai.WithRunCapabilities(otherCollidingCapability{id: "same"}),
	)
	if err == nil || !strings.Contains(err.Error(), "different types") {
		t.Fatalf("unexpected cross-layer collision error: %v", err)
	}
}

func TestRunCapabilitiesMergeThenReplaceAgentCapability(t *testing.T) {
	var instructions string
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		instructions = params.Instructions
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		mergingCapability{Name: "agent", Labels: []string{"agent"}},
	))
	if _, err := agent.Run(t.Context(), "go", struct{}{}, ai.WithRunCapabilities(
		mergingCapability{Name: "run-one", Labels: []string{"one"}},
		mergingCapability{Name: "run-two", Labels: []string{"two"}},
	)); err != nil {
		t.Fatal(err)
	}
	if instructions != "run-two:one,two:" {
		t.Fatalf("run capability did not replace the agent declaration whole: %q", instructions)
	}
}
