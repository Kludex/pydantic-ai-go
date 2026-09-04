package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type identifiedInstructionCapability struct {
	id      string
	static  []ai.InstructionPart
	dynamic []ai.InstructionPart
}

func (capability identifiedInstructionCapability) Setup(registry *ai.CapabilityRegistry) error {
	for _, part := range capability.static {
		registry.AddInstructionPart(part)
	}
	return nil
}

func (capability identifiedInstructionCapability) CapabilityID() string { return capability.id }

func (capability identifiedInstructionCapability) InstructionParts(
	context.Context, *ai.RunInfo,
) ([]ai.InstructionPart, error) {
	return capability.dynamic, nil
}

type identifiedStringInstructionCapability struct {
	id           string
	instructions string
}

func (identifiedStringInstructionCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (capability identifiedStringInstructionCapability) CapabilityID() string { return capability.id }

func (capability identifiedStringInstructionCapability) Instructions(context.Context, *ai.RunInfo) (string, error) {
	return capability.instructions, nil
}

type mutableInstructionCapability struct {
	id string
}

func (*mutableInstructionCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (capability *mutableInstructionCapability) CapabilityID() string { return capability.id }

func (*mutableInstructionCapability) Instructions(context.Context, *ai.RunInfo) (string, error) {
	return "mutable", nil
}

type identifiedEmptyCapability struct{ id string }

func (identifiedEmptyCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (capability identifiedEmptyCapability) CapabilityID() string { return capability.id }

type badNameInstructionCapability struct{}

func (badNameInstructionCapability) Setup(registry *ai.CapabilityRegistry) error {
	registry.AddInstructionPart(ai.InstructionPart{Content: "bad", Name: "bad:name"})
	return nil
}

type identifiedInstructionToolset struct {
	id    string
	parts []ai.InstructionPart
}

func (toolset identifiedInstructionToolset) ToolsetID() string { return toolset.id }

func (identifiedInstructionToolset) Tools(context.Context, *ai.RunContext[struct{}]) ([]ai.Tool[struct{}], error) {
	return nil, nil
}

func (toolset identifiedInstructionToolset) ToolsetInstructions(
	context.Context, *ai.RunContext[struct{}],
) ([]ai.InstructionPart, error) {
	return toolset.parts, nil
}

func TestInstructionPartsReceiveStableSourceIDs(t *testing.T) {
	var got ai.ModelRequestParams
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		got = params
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	capability := identifiedInstructionCapability{
		id: "research",
		static: []ai.InstructionPart{
			{Content: "Cite sources.", Name: "citations"},
			{Content: "Check facts."},
		},
		dynamic: []ai.InstructionPart{{Content: "Current corpus.", Name: "corpus", Dynamic: true}},
	}
	runCapability := identifiedInstructionCapability{
		id: "tenant", static: []ai.InstructionPart{{Content: "Tenant policy."}},
	}
	agent := ai.NewAgent[struct{}, string](model,
		ai.WithInstructions("Be concise."),
		ai.WithInstructionParts(
			ai.InstructionPart{Content: "Friendly persona.", Name: "persona"},
			ai.InstructionPart{Content: "Base safety."},
		),
		ai.WithCapabilities(
			capability,
			identifiedStringInstructionCapability{id: "string-provider", instructions: "String provider."},
		),
	)
	agent.AddInstructionPart(ai.InstructionPart{Content: "Added later.", Name: "added"})
	agent.AddInstructionPart(ai.InstructionPart{Content: "   "})
	agent.AddNamedInstructionsFunc("local_time", func(context.Context, *ai.RunContext[struct{}]) (string, error) {
		return "The time is 10:00.", nil
	})
	agent.AddInstructionsFunc(func(context.Context, *ai.RunContext[struct{}]) (string, error) {
		return "The user is Frank.", nil
	})
	agent.AddToolset(ai.PrefixToolset[struct{}](identifiedInstructionToolset{
		id: "weather", parts: []ai.InstructionPart{{
			Content: "Use metric units.", Name: "units", Dynamic: true,
			ID: ai.ToolsetInstructionID("weather", "units"),
		}},
	}, "weather"))
	agent.AddToolset(identifiedInstructionToolset{
		parts: []ai.InstructionPart{{Content: "Cannot mint an ID.", ID: ai.ToolsetInstructionID("foreign")}},
	})
	result, err := agent.Run(t.Context(), "Capital?", struct{}{},
		ai.WithRunInstructions("This run only."),
		ai.WithRunInstructionParts(ai.InstructionPart{Content: "Temporary policy.", Name: "temporary"}),
		ai.WithRunCapabilities(runCapability),
	)
	if err != nil || result.Output != "done" {
		t.Fatalf("unexpected run result=%+v err=%v", result, err)
	}
	want := []ai.InstructionPart{
		{Content: "Be concise.", ID: ai.AgentInstructionID()},
		{Content: "Friendly persona.", Name: "persona", ID: ai.AgentInstructionID("persona")},
		{Content: "Base safety.", ID: ai.AgentInstructionID()},
		{Content: "Added later.", Name: "added", ID: ai.AgentInstructionID("added")},
		{Content: "Cite sources.", Name: "citations", ID: ai.CapabilityInstructionID("research", "citations")},
		{Content: "Check facts.", ID: ai.CapabilityInstructionID("research")},
		{Content: "Tenant policy.", ID: ai.CapabilityInstructionID("tenant")},
		{Content: "This run only."},
		{Content: "Temporary policy.", Name: "temporary"},
		{Content: "Cannot mint an ID."},
		{Content: "Use metric units.", Dynamic: true, Name: "units", ID: ai.ToolsetInstructionID("weather", "units")},
		{Content: "The time is 10:00.", Dynamic: true, Name: "local_time", ID: ai.AgentInstructionID("local_time")},
		{Content: "The user is Frank.", Dynamic: true},
		{Content: "Current corpus.", Dynamic: true, Name: "corpus", ID: ai.CapabilityInstructionID("research", "corpus")},
		{Content: "String provider.", Dynamic: true, ID: ai.CapabilityInstructionID("string-provider")},
	}
	if !reflect.DeepEqual(got.InstructionParts, want) {
		t.Fatalf("unexpected instruction parts:\n got: %+v\nwant: %+v", got.InstructionParts, want)
	}
	if got.Instructions != instructionContents(want) {
		t.Fatalf("joined instructions mismatch:\n%s", got.Instructions)
	}
	got.InstructionParts[0].ID.Name = "mutated"
	if ai.AgentInstructionID().Name != "" {
		t.Fatal("instruction ID constructor returned shared state")
	}
}

func TestBeforeModelRequestEditsInstructionsByID(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if params.Instructions != "Managed persona.\n\nKeep facts." {
			t.Fatalf("part rewrite did not reach model: %+v", params)
		}
		request := messages[len(messages)-1].(ai.ModelRequest)
		if request.Instructions != params.Instructions {
			t.Fatalf("history did not record rewritten instructions: %+v", request)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	rewrite := ai.BeforeModelRequestFunc(func(
		_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
	) (ai.ModelRequestContext, error) {
		for index, part := range request.Params.InstructionParts {
			if part.ID != nil && part.ID.String() == "agent:persona" {
				part.Content = "Managed persona."
				request.Params.InstructionParts[index] = part
			}
		}
		last := request.Messages[len(request.Messages)-1].(ai.ModelRequest)
		last.Instructions = "ignored message edit"
		request.Messages[len(request.Messages)-1] = last
		return request, nil
	})
	agent := ai.NewAgent[struct{}, string](model,
		ai.WithInstructionParts(
			ai.InstructionPart{Content: "Original persona.", Name: "persona"},
			ai.InstructionPart{Content: "Keep facts."},
		),
		ai.WithCapabilities(rewrite),
	)
	if _, err := agent.Run(t.Context(), "hello", struct{}{}); err != nil {
		t.Fatal(err)
	}
}

func TestBeforeModelRequestCanClearInstructionParts(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if params.Instructions != "" || params.InstructionParts != nil {
			t.Fatalf("cleared instruction parts still reached model: %+v", params)
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	clearParts := ai.BeforeModelRequestFunc(func(
		_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
	) (ai.ModelRequestContext, error) {
		request.Params.InstructionParts = nil
		return request, nil
	})
	agent := ai.NewAgent[struct{}, string](model,
		ai.WithInstructions("remove me"), ai.WithCapabilities(clearParts),
	)
	if _, err := agent.Run(t.Context(), "hello", struct{}{}); err != nil {
		t.Fatal(err)
	}
	clearJoined := ai.BeforeModelRequestFunc(func(
		_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext,
	) (ai.ModelRequestContext, error) {
		request.Params.Instructions = ""
		return request, nil
	})
	joinedAgent := ai.NewAgent[struct{}, string](model,
		ai.WithInstructions("remove joined"), ai.WithCapabilities(clearJoined),
	)
	if _, err := joinedAgent.Run(t.Context(), "hello", struct{}{}); err != nil {
		t.Fatal(err)
	}
}

func TestDirectRequestsDetachInstructionIDs(t *testing.T) {
	params := ai.ModelRequestParams{InstructionParts: []ai.InstructionPart{{
		Content: "Direct.", Name: "direct", ID: ai.AgentInstructionID("direct"),
	}}}
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, got ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if got.Instructions != "Direct." || got.InstructionParts[0].ID.String() != "agent:direct" {
			t.Fatalf("unexpected direct instructions: %+v", got)
		}
		got.InstructionParts[0].ID.Name = "mutated"
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	if _, err := ai.RequestModel(t.Context(), model, nil, params); err != nil {
		t.Fatal(err)
	}
	if params.InstructionParts[0].ID.String() != "agent:direct" {
		t.Fatal("direct model request aliased caller instruction ID")
	}
}

func TestInstructionPartJoiningAndStableCacheOrdering(t *testing.T) {
	parts := []ai.InstructionPart{
		{Content: " Dynamic. ", Dynamic: true, ID: ai.ToolsetInstructionID("weather")},
		{Content: "Static."},
		{Content: "Identified.", ID: ai.CapabilityInstructionID("memory")},
		{Content: "   ", Dynamic: true},
	}
	if got := ai.JoinInstructionParts(parts); got != "Dynamic.\n\nStatic.\n\nIdentified." {
		t.Fatalf("unexpected joined instructions: %q", got)
	}
	sorted := ai.SortInstructionParts(parts)
	if sorted[0].Content != "Static." || sorted[1].Content != "Identified." ||
		sorted[2].Content != " Dynamic. " || sorted[3].Content != "   " {
		t.Fatalf("unexpected stable instruction order: %+v", sorted)
	}
	sorted[2].ID.Name = "mutated"
	if parts[0].ID.Name != "" {
		t.Fatal("sorted instruction IDs alias input")
	}
}

func TestUpstreamInstructionPartFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/messages/upstream_instruction_parts.json")
	if err != nil {
		t.Fatal(err)
	}
	var parts []ai.InstructionPart
	if err := json.Unmarshal(data, &parts); err != nil {
		t.Fatal(err)
	}
	want := []ai.InstructionPart{
		{Content: "Agent.", ID: ai.AgentInstructionID()},
		{Content: "Persona.", Name: "persona", ID: ai.AgentInstructionID("persona")},
		{
			Content: "Weather.", Dynamic: true, Name: "limits",
			ID: ai.ToolsetInstructionID("weather", "limits"),
		},
		{Content: "Memory.", ID: ai.CapabilityInstructionID("memory")},
		{Content: "Anonymous.", Name: "local"},
	}
	if !reflect.DeepEqual(parts, want) {
		t.Fatalf("unexpected upstream instruction parts: %+v", parts)
	}
	roundTrip, err := json.Marshal(parts)
	if err != nil {
		t.Fatal(err)
	}
	var decoded []map[string]any
	var upstream []map[string]any
	if json.Unmarshal(roundTrip, &decoded) != nil || json.Unmarshal(data, &upstream) != nil || !reflect.DeepEqual(decoded, upstream) {
		t.Fatalf("instruction fixture changed on round trip: %s", roundTrip)
	}
}

func TestInstructionIDSerializationAndForwardCompatibility(t *testing.T) {
	tests := []struct {
		id   *ai.InstructionID
		want string
	}{
		{id: ai.AgentInstructionID(), want: "agent"},
		{id: ai.AgentInstructionID("persona"), want: "agent:persona"},
		{id: ai.ToolsetInstructionID("weather"), want: "toolset:weather"},
		{id: ai.ToolsetInstructionID("weather", "limits"), want: "toolset:weather:limits"},
		{id: ai.CapabilityInstructionID("memory"), want: "capability:memory"},
		{id: ai.CapabilityInstructionID("memory", "style"), want: "capability:memory:style"},
	}
	for _, test := range tests {
		if test.id.String() != test.want {
			t.Fatalf("instruction ID = %q, want %q", test.id, test.want)
		}
		data, err := json.Marshal(test.id)
		if err != nil {
			t.Fatal(err)
		}
		var decoded ai.InstructionID
		if err := json.Unmarshal(data, &decoded); err != nil || decoded != *test.id {
			t.Fatalf("instruction ID round trip failed: %s %+v %v", data, decoded, err)
		}
	}

	part := ai.InstructionPart{
		Content: "Weather.", Dynamic: true, Name: "limits", ID: ai.ToolsetInstructionID("weather", "limits"),
	}
	data, err := json.Marshal(part)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"content":"Weather.","dynamic":true,"name":"limits","id":"toolset:weather:limits","part_kind":"instruction"}` {
		t.Fatalf("unexpected instruction JSON: %s", data)
	}
	var decoded ai.InstructionPart
	if err := json.Unmarshal(data, &decoded); err != nil || !reflect.DeepEqual(decoded, part) {
		t.Fatalf("instruction part round trip failed: %+v %v", decoded, err)
	}
	if err := json.Unmarshal([]byte(`{"content":"Future.","id":"plugin:search:limits","part_kind":"instruction"}`), &decoded); err != nil || decoded.ID != nil {
		t.Fatalf("unknown instruction namespace did not become unaddressable: %+v %v", decoded, err)
	}
	if err := json.Unmarshal([]byte(`{"content":"Legacy.","dynamic":true,"part_kind":"instruction"}`), &decoded); err != nil ||
		decoded.Content != "Legacy." || !decoded.Dynamic || decoded.ID != nil {
		t.Fatalf("legacy instruction part did not decode: %+v %v", decoded, err)
	}
}

func TestInstructionIDValidation(t *testing.T) {
	for _, function := range []func(){
		func() { ai.AgentInstructionID("agent") },
		func() { ai.AgentInstructionID("a:b") },
		func() { ai.AgentInstructionID("one", "two") },
		func() { ai.ToolsetInstructionID("") },
		func() { ai.ToolsetInstructionID("bad:id") },
		func() { ai.CapabilityInstructionID("bad:id") },
		func() { ai.WithInstructionParts(ai.InstructionPart{Content: "x", Name: "agent"}) },
		func() { ai.WithRunInstructionParts(ai.InstructionPart{Content: "x", Name: "bad:name"}) },
		func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel()).AddInstructionPart(ai.InstructionPart{
				Content: "x", Name: "bad:name",
			})
		},
		func() { ai.NewAgent[struct{}, string](fakes.NewTestModel()).AddInstructionsFunc(nil) },
		func() { ai.NewAgent[struct{}, string](fakes.NewTestModel()).AddNamedInstructionsFunc("name", nil) },
		func() {
			ai.NewAgent[struct{}, string](fakes.NewTestModel()).AddNamedInstructionsFunc("bad:name", func(
				context.Context, *ai.RunContext[struct{}],
			) (string, error) {
				return "", nil
			})
		},
	} {
		assertOutputFunctionPanics(t, function)
	}
	if (ai.InstructionSource{Kind: ai.InstructionSourceToolset}).String() != "" {
		t.Fatal("invalid empty identified source produced a key")
	}
	for _, value := range []string{"plugin:search", "agent:agent", "agent:"} {
		if _, ok := ai.ParseInstructionID(value); ok {
			t.Fatalf("invalid instruction ID %q was accepted", value)
		}
	}
	var id ai.InstructionID
	if err := json.Unmarshal([]byte(`42`), &id); err == nil {
		t.Fatal("instruction ID accepted non-string JSON")
	}
	if err := json.Unmarshal([]byte(`"plugin:search"`), &id); err == nil {
		t.Fatal("direct instruction ID decode accepted unknown namespace")
	}
	var part ai.InstructionPart
	if err := json.Unmarshal([]byte(`{"content":"x","name":"bad:name"}`), &part); err == nil {
		t.Fatal("instruction part accepted invalid name")
	}
	if err := json.Unmarshal([]byte(`{"content":"x","part_kind":"other"}`), &part); err == nil {
		t.Fatal("instruction part accepted invalid kind")
	}
	for _, invalid := range []ai.InstructionID{
		{Source: ai.InstructionSource{Kind: "future"}},
		{Source: ai.InstructionSource{Kind: ai.InstructionSourceAgent, ID: "invalid"}},
	} {
		if _, err := json.Marshal(invalid); err == nil {
			t.Fatalf("invalid instruction ID marshaled: %+v", invalid)
		}
	}
	for _, invalidPart := range []ai.InstructionPart{
		{Content: "invalid", Name: "bad:name"},
		{Content: "invalid", ID: &ai.InstructionID{Source: ai.InstructionSource{Kind: "future"}}},
	} {
		if _, err := json.Marshal(invalidPart); err == nil {
			t.Fatalf("invalid instruction part marshaled: %+v", invalidPart)
		}
	}
	if err := json.Unmarshal([]byte(`{`), &part); err == nil {
		t.Fatal("instruction part accepted malformed JSON")
	}
}

func TestInstructionContributionErrors(t *testing.T) {
	if _, err := ai.NewAgent[struct{}, string](
		fakes.NewTestModel(), ai.WithCapabilities(identifiedEmptyCapability{id: "unused"}),
	).Run(t.Context(), "hello", struct{}{}); err != nil {
		t.Fatalf("identified capability without instructions failed: %v", err)
	}
	assertOutputFunctionPanics(t, func() {
		ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(badNameInstructionCapability{}))
	})

	mutable := &mutableInstructionCapability{id: "valid"}
	mutableAgent := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(mutable))
	mutable.id = "bad:id"
	if _, err := mutableAgent.Run(t.Context(), "hello", struct{}{}); err == nil ||
		!strings.Contains(err.Error(), "instruction ID delimiter") {
		t.Fatalf("mutated capability ID was accepted: %v", err)
	}

	duplicate := identifiedInstructionCapability{
		id: "duplicate", static: []ai.InstructionPart{{Content: "one"}},
	}
	assertOutputFunctionPanics(t, func() {
		ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(duplicate, duplicate))
	})
	base := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(duplicate))
	if _, err := base.Run(
		t.Context(), "hello", struct{}{}, ai.WithRunCapabilities(duplicate),
	); err == nil || !strings.Contains(err.Error(), "multiple capabilities") {
		t.Fatalf("duplicate run capability instruction ID was accepted: %v", err)
	}

	toolsetOne := identifiedInstructionToolset{
		id: "duplicate", parts: []ai.InstructionPart{{Content: "one"}, {Content: "two"}},
	}
	toolsetTwo := identifiedInstructionToolset{
		id: "duplicate", parts: []ai.InstructionPart{{Content: "other"}},
	}
	toolsetAgent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	toolsetAgent.AddToolset(toolsetOne)
	toolsetAgent.AddToolset(toolsetTwo)
	if _, err := toolsetAgent.Run(t.Context(), "hello", struct{}{}); err == nil ||
		!strings.Contains(err.Error(), "multiple toolsets") {
		t.Fatalf("duplicate toolset instruction ID was accepted: %v", err)
	}

	combinedAgent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	combinedAgent.AddToolset(ai.CombineToolsets[struct{}](toolsetOne, toolsetTwo))
	if _, err := combinedAgent.Run(t.Context(), "hello", struct{}{}); err == nil ||
		!strings.Contains(err.Error(), "multiple toolsets") {
		t.Fatalf("combined duplicate toolset instruction ID was accepted: %v", err)
	}

	badCapability := identifiedInstructionCapability{
		id: "bad:id", static: []ai.InstructionPart{{Content: "static"}},
	}
	assertOutputFunctionPanics(t, func() {
		ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(badCapability))
	})
	plainAgent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	if _, err := plainAgent.Run(
		t.Context(), "hello", struct{}{}, ai.WithRunCapabilities(badCapability),
	); err == nil || !strings.Contains(err.Error(), "instruction ID delimiter") {
		t.Fatalf("invalid run capability ID was accepted: %v", err)
	}
	if _, err := plainAgent.Run(
		t.Context(), "hello", struct{}{}, ai.WithRunCapabilities(badNameInstructionCapability{}),
	); err == nil || !strings.Contains(err.Error(), "instruction name") {
		t.Fatalf("invalid run capability instruction name was accepted: %v", err)
	}

	badToolset := identifiedInstructionToolset{
		id: "bad:id", parts: []ai.InstructionPart{{Content: "dynamic"}},
	}
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
	agent.AddToolset(badToolset)
	if _, err := agent.Run(t.Context(), "hello", struct{}{}); err == nil ||
		!strings.Contains(err.Error(), "instruction ID delimiter") {
		t.Fatalf("unexpected toolset instruction error: %v", err)
	}

	dynamic := identifiedInstructionCapability{
		id: "valid", dynamic: []ai.InstructionPart{{Content: "dynamic", Name: "bad:name"}},
	}
	if _, err := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(dynamic)).Run(
		t.Context(), "hello", struct{}{},
	); err == nil || !strings.Contains(err.Error(), "instruction name") {
		t.Fatalf("unexpected dynamic instruction error: %v", err)
	}

	errorCapability := instructionPartsErrorCapability{err: errors.New("instructions failed")}
	if _, err := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(errorCapability)).Run(
		t.Context(), "hello", struct{}{},
	); err == nil || !strings.Contains(err.Error(), "instructions failed") {
		t.Fatalf("unexpected provider error: %v", err)
	}
}

type instructionPartsErrorCapability struct{ err error }

func (instructionPartsErrorCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (capability instructionPartsErrorCapability) InstructionParts(
	context.Context, *ai.RunInfo,
) ([]ai.InstructionPart, error) {
	return nil, capability.err
}

func instructionContents(parts []ai.InstructionPart) string {
	contents := make([]string, len(parts))
	for index, part := range parts {
		contents[index] = part.Content
	}
	return strings.Join(contents, "\n\n")
}
