package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type nilResponseCapability struct{}

func (nilResponseCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (nilResponseCapability) WrapModelRequest(
	context.Context, *ai.RunInfo, []ai.ModelMessage, ai.ModelRequestParams, ai.ModelRequestFunc,
) (*ai.ModelResponse, error) {
	return nil, nil
}

func TestCapabilityCannotCompleteWithNilModelResponse(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(nilResponseCapability{}))
	_, err := agent.Run(t.Context(), "go", deps{})
	if err == nil || err.Error() != "ai: unexpected model behavior: model request returned no response" {
		t.Fatalf("unexpected nil model response error: %v", err)
	}
}

// traceCapability records the order hooks fire in.
type traceCapability struct {
	name  string
	log   *[]string
	setup func(reg *ai.CapabilityRegistry) error
}

func (c *traceCapability) Setup(reg *ai.CapabilityRegistry) error {
	if c.setup != nil {
		return c.setup(reg)
	}
	return nil
}

func (c *traceCapability) WrapRun(
	ctx context.Context, _ *ai.RunInfo, next ai.RunFunc,
) (ai.RunOutcome, error) {
	*c.log = append(*c.log, c.name+":run-in")
	outcome, err := next(ctx)
	*c.log = append(*c.log, c.name+":run-out")
	return outcome, err
}

func (c *traceCapability) WrapModelRequest(ctx context.Context, _ *ai.RunInfo, msgs []ai.ModelMessage, params ai.ModelRequestParams, next ai.ModelRequestFunc) (*ai.ModelResponse, error) {
	*c.log = append(*c.log, c.name+":model")
	return next(ctx, msgs, params)
}

func (c *traceCapability) WrapToolCall(ctx context.Context, _ *ai.RunInfo, call ai.ToolCallPart, next ai.ToolCallFunc) (any, error) {
	*c.log = append(*c.log, c.name+":tool:"+call.ToolName)
	return next(ctx, call)
}

func TestCapabilityHookOrder(t *testing.T) {
	var log []string
	outer := &traceCapability{name: "outer", log: &log}
	inner := &traceCapability{name: "inner", log: &log}
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(),
		ai.WithCapabilities(outer, inner),
	)
	ai.AddSimpleTool(agent, "noop", func(context.Context, struct{}) (string, error) { return "", nil })
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	want := []string{
		"outer:run-in", "inner:run-in",
		"outer:model", "inner:model",
		"outer:tool:noop", "inner:tool:noop",
		"outer:model", "inner:model",
		"inner:run-out", "outer:run-out",
	}
	if len(log) != len(want) {
		t.Fatalf("unexpected log %v", log)
	}
	for i := range want {
		if log[i] != want[i] {
			t.Fatalf("step %d: expected %s, got %s (%v)", i, want[i], log[i], log)
		}
	}
}

// staticCapability contributes tools and instructions at setup.
type staticCapability struct{}

func (staticCapability) Setup(reg *ai.CapabilityRegistry) error {
	reg.AddInstructions("Always be brief.")
	reg.AddTool(ai.ToolDefinition{Name: "cap_tool", Schema: map[string]any{"type": "object"}},
		func(_ context.Context, _ json.RawMessage) (any, error) { return "from capability", nil })
	return nil
}

func TestCapabilityContributesToolsAndInstructions(t *testing.T) {
	var gotInstructions string
	var gotInstructionParts []ai.InstructionPart
	var toolReturn any
	model := fakes.NewFunctionModel(func(_ context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
		gotInstructions = params.Instructions
		gotInstructionParts = params.InstructionParts
		if len(msgs) == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "cap_tool", Args: json.RawMessage(`{}`), ToolCallID: "c1"},
			}}, nil
		}
		toolReturn = msgs[len(msgs)-1].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart).Content
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model,
		ai.WithInstructions("Base."),
		ai.WithCapabilities(staticCapability{}),
	)
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if gotInstructions != "Base.\n\nAlways be brief." {
		t.Fatalf("unexpected instructions %q", gotInstructions)
	}
	if !reflect.DeepEqual(gotInstructionParts, []ai.InstructionPart{
		{Content: "Base."}, {Content: "Always be brief."},
	}) {
		t.Fatalf("unexpected instruction parts %+v", gotInstructionParts)
	}
	if toolReturn != "from capability" {
		t.Fatalf("capability tool not executed: %v", toolReturn)
	}
}

// dynamicInstructions implements InstructionsProvider.
type dynamicInstructions struct{ err error }

func (dynamicInstructions) Setup(*ai.CapabilityRegistry) error { return nil }

func (c dynamicInstructions) Instructions(_ context.Context, ri *ai.RunInfo) (string, error) {
	if c.err != nil {
		return "", c.err
	}
	return "run " + ri.RunID[:2], nil
}

func TestCapabilityInstructionsProvider(t *testing.T) {
	var got string
	var gotParts []ai.InstructionPart
	model := fakes.NewFunctionModel(func(_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
		got = params.Instructions
		gotParts = params.InstructionParts
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "ok"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(dynamicInstructions{}))
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 || got[:4] != "run " {
		t.Fatalf("unexpected instructions %q", got)
	}
	if len(gotParts) != 1 || gotParts[0].Content != got || !gotParts[0].Dynamic {
		t.Fatalf("dynamic instruction boundary was lost: %+v", gotParts)
	}
}

func TestCapabilityInstructionsError(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(),
		ai.WithCapabilities(dynamicInstructions{err: errors.New("boom")}),
	)
	if _, err := agent.Run(t.Context(), "go", deps{}); err == nil {
		t.Fatal("expected error")
	}
}

func TestCapabilitySetupError(t *testing.T) {
	var log []string
	failing := &traceCapability{name: "f", log: &log, setup: func(*ai.CapabilityRegistry) error {
		return errors.New("cannot set up")
	}}
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(failing))
}

// guardCapability blocks tool calls by name - the guardrail pattern.
type guardCapability struct{ blocked string }

func (guardCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (c guardCapability) WrapToolCall(ctx context.Context, _ *ai.RunInfo, call ai.ToolCallPart, next ai.ToolCallFunc) (any, error) {
	if call.ToolName == c.blocked {
		return nil, ai.Retryf("tool %q is not allowed", call.ToolName)
	}
	return next(ctx, call)
}

func TestCapabilityGuardsToolCall(t *testing.T) {
	first := true
	model := fakes.NewFunctionModel(func(_ context.Context, msgs []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
		if first {
			first = false
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "dangerous", Args: json.RawMessage(`{}`), ToolCallID: "c1"},
			}}, nil
		}
		last := msgs[len(msgs)-1].(ai.ModelRequest)
		if _, ok := last.Parts[0].(ai.RetryPromptPart); !ok {
			return nil, fmt.Errorf("expected retry prompt, got %T", last.Parts[0])
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "blocked, giving up"}}}, nil
	})
	executed := false
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(guardCapability{blocked: "dangerous"}))
	ai.AddSimpleTool(agent, "dangerous", func(context.Context, struct{}) (string, error) {
		executed = true
		return "", nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if executed {
		t.Fatal("blocked tool must not execute")
	}
	if result.Output != "blocked, giving up" {
		t.Fatalf("unexpected output %q", result.Output)
	}
}

// rewriteCapability modifies the response - the redaction pattern.
type rewriteCapability struct{}

func (rewriteCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (rewriteCapability) WrapModelRequest(ctx context.Context, _ *ai.RunInfo, msgs []ai.ModelMessage, params ai.ModelRequestParams, next ai.ModelRequestFunc) (*ai.ModelResponse, error) {
	resp, err := next(ctx, msgs, params)
	if err != nil {
		return nil, err
	}
	for i, part := range resp.Parts {
		if text, ok := part.(ai.TextPart); ok {
			resp.Parts[i] = ai.TextPart{Content: text.Content + " [checked]"}
		}
	}
	return resp, nil
}

func TestCapabilityRewritesResponse(t *testing.T) {
	model := fakes.NewTestModel()
	model.CustomOutputText = "hello"
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(rewriteCapability{}))
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "hello [checked]" {
		t.Fatalf("unexpected output %q", result.Output)
	}
}

func TestCapabilityRunInfoExposesState(t *testing.T) {
	var sawUsage ai.Usage
	var sawMessages int
	capability := &inspectCapability{onRunOut: func(ri *ai.RunInfo) {
		sawUsage = ri.Usage()
		sawMessages = len(ri.Messages())
	}}
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(capability))
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatal(err)
	}
	if sawUsage.Requests != 1 || sawMessages != 2 {
		t.Fatalf("usage=%+v messages=%d", sawUsage, sawMessages)
	}
}

type inspectCapability struct{ onRunOut func(*ai.RunInfo) }

func (inspectCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (c *inspectCapability) WrapRun(
	ctx context.Context, ri *ai.RunInfo, next ai.RunFunc,
) (ai.RunOutcome, error) {
	outcome, err := next(ctx)
	c.onRunOut(ri)
	return outcome, err
}

func TestCapabilityWithStreaming(t *testing.T) {
	var log []string
	capability := &traceCapability{name: "s", log: &log}
	agent := ai.NewAgent[deps, string](fakes.NewTestModel(), ai.WithCapabilities(capability))
	stream := agent.RunStream(t.Context(), "go", deps{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	if stream.Result() == nil || len(log) == 0 {
		t.Fatalf("capability hooks did not fire during streaming: %v", log)
	}
}

func TestUsageLimitsPropagateModelErrors(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return nil, errors.New("down")
	})
	agent := ai.NewAgent[deps, string](model, ai.WithUsageLimits(ai.UsageLimits{RequestLimit: 5}))
	if _, err := agent.Run(t.Context(), "go", deps{}); err == nil {
		t.Fatal("expected error")
	}
}

// truncateHistory keeps only the last N messages - the history-processor
// pattern implemented as a ModelRequestWrapper.
type truncateHistory struct{ keep int }

func (truncateHistory) Setup(*ai.CapabilityRegistry) error { return nil }

func (c truncateHistory) WrapModelRequest(ctx context.Context, _ *ai.RunInfo, msgs []ai.ModelMessage, params ai.ModelRequestParams, next ai.ModelRequestFunc) (*ai.ModelResponse, error) {
	if len(msgs) > c.keep {
		msgs = msgs[len(msgs)-c.keep:]
	}
	return next(ctx, msgs, params)
}

func TestCapabilityProcessesHistory(t *testing.T) {
	var sawMessages int
	model := fakes.NewFunctionModel(func(_ context.Context, msgs []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
		sawMessages = len(msgs)
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "ok"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(truncateHistory{keep: 2}))
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "old 1"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "old answer"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "old 2"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "old answer 2"}}},
	}
	result, err := agent.Run(t.Context(), "new question", deps{}, ai.WithMessageHistory(history))
	if err != nil {
		t.Fatal(err)
	}
	if sawMessages != 2 {
		t.Fatalf("model saw %d messages, expected truncated 2", sawMessages)
	}
	if len(result.Messages()) != 6 {
		t.Fatalf("stored history must stay intact, got %d", len(result.Messages()))
	}
}
