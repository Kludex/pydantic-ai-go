package ai_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

type identityDeps struct {
	Audience string
}

func TestAgentIdentityIsAvailableToRunsAndInstrumentation(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
	requests := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		requests++
		if requests == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "inspect_identity", ToolCallID: "identity", Args: []byte(`{}`),
			}}}, nil
		}
		for _, part := range messages[len(messages)-1].(ai.ModelRequest).Parts {
			if _, ok := part.(ai.ToolReturnPart); ok {
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "hello"}}}, nil
			}
		}
		return nil, errors.New("identity tool result missing")
	})
	before := ai.BeforeRunFunc(func(_ context.Context, info *ai.RunInfo) error {
		if info.AgentName() != "greeter" || info.AgentDescription() != "Greets Go developers" {
			t.Fatalf("capability received wrong identity: %q %q", info.AgentName(), info.AgentDescription())
		}
		return nil
	})
	descriptionCalls := 0
	agent := ai.NewAgent[identityDeps, string](model,
		ai.WithAgentName("greeter"),
		ai.WithAgentDescriptionFunc(func(_ context.Context, deps identityDeps) (string, error) {
			descriptionCalls++
			return "Greets " + deps.Audience, nil
		}),
		ai.WithCapabilities(before, ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(provider))),
	)
	if agent.Name() != "greeter" || agent.Description() != "" {
		t.Fatalf("unexpected unresolved identity: name=%q description=%q", agent.Name(), agent.Description())
	}
	ai.AddTool(agent, "inspect_identity", func(
		_ context.Context, rc *ai.RunContext[identityDeps], _ struct{},
	) (string, error) {
		if rc.AgentName != "greeter" || rc.AgentDescription != "Greets Go developers" {
			t.Fatalf("tool received wrong identity: %q %q", rc.AgentName, rc.AgentDescription)
		}
		return "ok", nil
	})
	run, err := agent.StartRun(t.Context(), "hello", identityDeps{Audience: "Go developers"})
	if err != nil {
		t.Fatal(err)
	}
	if run.AgentName() != "greeter" || run.AgentDescription() != "Greets Go developers" {
		t.Fatalf("manual run received wrong identity: %q %q", run.AgentName(), run.AgentDescription())
	}
	if _, err := drainAgentRun(run); err != nil {
		t.Fatal(err)
	}
	if descriptionCalls != 1 || run.Result().Output != "hello" {
		t.Fatalf("unexpected completed run: calls=%d result=%+v", descriptionCalls, run.Result())
	}
	var attributes map[string]any
	for _, span := range exporter.GetSpans() {
		if span.Name == "invoke_agent greeter" {
			attributes = instrumentationSpanAttributes(span.Attributes)
		}
	}
	if attributes["gen_ai.agent.name"] != "greeter" ||
		attributes["gen_ai.agent.description"] != "Greets Go developers" ||
		attributes["logfire.msg"] != "greeter run" {
		t.Fatalf("agent identity missing from telemetry: %+v", attributes)
	}
	if schema, _ := attributes["logfire.json_schema"].(string); !strings.Contains(schema, "final_result") ||
		!strings.Contains(schema, "pydantic_ai.all_messages") {
		t.Fatalf("run schema missing core fields: %s", schema)
	}
}

func TestAgentDescriptionConfigurationAndErrors(t *testing.T) {
	if (&ai.AgentRun[struct{}, string]{}).Metadata() != nil || (&ai.RunInfo{}).Metadata() != nil {
		t.Fatal("zero-value run metadata must be nil")
	}
	static := ai.NewAgent[struct{}, string](
		fakes.NewTestModel(),
		ai.WithAgentDescriptionFunc(func(context.Context, struct{}) (string, error) { return "dynamic", nil }),
		ai.WithAgentDescription("static"),
	)
	if static.Description() != "static" {
		t.Fatalf("static description did not replace dynamic description: %q", static.Description())
	}
	if rendered, err := static.RenderDescription(t.Context(), struct{}{}); err != nil || rendered != "static" {
		t.Fatalf("unexpected static description: %q %v", rendered, err)
	}
	if ai.NewAgent[struct{}, string](fakes.NewTestModel()).Description() != "" {
		t.Fatal("agent without a description did not return an empty description")
	}

	boom := errors.New("render failed")
	failing := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithAgentDescriptionFunc(
		func(context.Context, struct{}) (string, error) { return "", boom },
	))
	if _, err := failing.Run(t.Context(), "hello", struct{}{}); !errors.Is(err, boom) ||
		!strings.Contains(err.Error(), "ai: agent description") {
		t.Fatalf("unexpected description error: %v", err)
	}

	assertOutputFunctionPanics(t, func() { ai.WithAgentDescriptionFunc[struct{}](nil) })
	wrongDeps := ai.WithAgentDescriptionFunc(func(context.Context, string) (string, error) { return "", nil })
	assertOutputFunctionPanics(t, func() {
		ai.NewAgent[struct{}, string](fakes.NewTestModel(), wrongDeps)
	})
}

func TestInstrumentationUsesPersistedInstructionsBeforeRequestMiddleware(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
	boom := errors.New("request rejected")
	reject := ai.BeforeModelRequestFunc(func(
		context.Context, *ai.RunInfo, ai.ModelRequestContext,
	) (ai.ModelRequestContext, error) {
		return ai.ModelRequestContext{}, boom
	})
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel(),
		ai.WithInstructions("persisted instructions"),
		ai.WithCapabilities(
			ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(provider)),
			reject,
		),
	)
	if _, err := agent.Run(t.Context(), "hello", struct{}{}); !errors.Is(err, boom) {
		t.Fatalf("unexpected request error: %v", err)
	}
	var attributes map[string]any
	for _, span := range exporter.GetSpans() {
		if span.Name == "invoke_agent agent" {
			attributes = instrumentationSpanAttributes(span.Attributes)
		}
	}
	if instructions, _ := attributes["gen_ai.system_instructions"].(string); !strings.Contains(instructions, "persisted instructions") {
		t.Fatalf("persisted instructions missing from failed run: %+v", attributes)
	}

	exporter.Reset()
	withoutInstructions := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(
		ai.NewInstrumentation(ai.WithInstrumentationTracerProvider(provider)), reject,
	))
	if _, err := withoutInstructions.Run(t.Context(), "hello", struct{}{}); !errors.Is(err, boom) {
		t.Fatalf("unexpected request error without instructions: %v", err)
	}
	for _, span := range exporter.GetSpans() {
		if span.Name != "invoke_agent agent" {
			continue
		}
		attributes = instrumentationSpanAttributes(span.Attributes)
		if _, exists := attributes["gen_ai.system_instructions"]; exists {
			t.Fatalf("empty instructions were recorded: %+v", attributes)
		}
	}
}

type instructionState struct {
	mu    sync.Mutex
	calls int
}

func TestInstrumentationReportsVariableInstructions(t *testing.T) {
	tests := []struct {
		name           string
		changing       bool
		includeContent bool
		wantVariable   bool
	}{
		{name: "stable", includeContent: true},
		{name: "changing", changing: true, includeContent: true, wantVariable: true},
		{name: "changing without content", changing: true, wantVariable: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			exporter := tracetest.NewInMemoryExporter()
			provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
			t.Cleanup(func() { _ = provider.Shutdown(t.Context()) })
			requests := 0
			model := fakes.NewFunctionModel(func(
				_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				requests++
				if requests == 1 {
					return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
						ToolName: "continue_run", ToolCallID: "continue", Args: []byte(`{}`),
					}}}, nil
				}
				for _, part := range messages[len(messages)-1].(ai.ModelRequest).Parts {
					if _, ok := part.(ai.ToolReturnPart); ok {
						return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
					}
				}
				return nil, errors.New("tool result missing")
			})
			state := &instructionState{}
			agent := ai.NewAgent[*instructionState, string](model, ai.WithCapabilities(ai.NewInstrumentation(
				ai.WithInstrumentationTracerProvider(provider),
				ai.WithInstrumentationContent(test.includeContent),
			)))
			agent.AddInstructionsFunc(func(_ context.Context, deps *ai.RunContext[*instructionState]) (string, error) {
				deps.Deps.mu.Lock()
				defer deps.Deps.mu.Unlock()
				deps.Deps.calls++
				if test.changing && deps.Deps.calls > 1 {
					return "later instructions", nil
				}
				return "initial instructions", nil
			})
			ai.AddSimpleTool(agent, "continue_run", func(context.Context, struct{}) (string, error) {
				return "continue", nil
			})
			if _, err := agent.Run(t.Context(), "hello", state); err != nil {
				t.Fatal(err)
			}
			var attributes map[string]any
			for _, span := range exporter.GetSpans() {
				if span.Name == "invoke_agent agent" {
					attributes = instrumentationSpanAttributes(span.Attributes)
				}
			}
			_, variable := attributes["pydantic_ai.variable_instructions"]
			if variable != test.wantVariable {
				t.Fatalf("variable instruction attribute = %t, want %t: %+v", variable, test.wantVariable, attributes)
			}
			systemInstructions, hasInstructions := attributes["gen_ai.system_instructions"]
			if hasInstructions != test.includeContent {
				t.Fatalf("system instruction visibility = %t, want %t: %+v", hasInstructions, test.includeContent, attributes)
			}
			if test.includeContent {
				want := "initial instructions"
				if test.changing {
					want = "later instructions"
				}
				if !strings.Contains(systemInstructions.(string), want) {
					t.Fatalf("latest instructions missing from telemetry: %v", systemInstructions)
				}
			}
			schema := attributes["logfire.json_schema"].(string)
			if strings.Contains(schema, "pydantic_ai.variable_instructions") != test.wantVariable {
				t.Fatalf("run schema variable field mismatch: %s", schema)
			}
		})
	}
}
