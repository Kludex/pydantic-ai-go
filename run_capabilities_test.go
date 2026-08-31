package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type runOnlyCapability struct {
	setupCalls atomic.Int64
	runCalls   atomic.Int64
	events     atomic.Int64
}

func (c *runOnlyCapability) Setup(registry *ai.CapabilityRegistry) error {
	c.setupCalls.Add(1)
	registry.AddInstructions("Run capability static.")
	registry.AddModelSettings(ai.ModelSettings{MaxTokens: 321})
	registry.AddTool(ai.ToolDefinition{
		Name: "run_capability_tool",
		Schema: map[string]any{
			"type": "object", "additionalProperties": false,
		},
	}, func(context.Context, json.RawMessage) (any, error) {
		return "capability result", nil
	})
	return nil
}

func (c *runOnlyCapability) WrapRun(
	ctx context.Context, _ *ai.RunInfo, next ai.RunFunc,
) (ai.RunOutcome, error) {
	c.runCalls.Add(1)
	return next(ctx)
}

func (c *runOnlyCapability) Instructions(context.Context, *ai.RunInfo) (string, error) {
	return "Run capability dynamic.", nil
}

func (c *runOnlyCapability) ProcessStreamEvent(
	_ context.Context, _ *ai.RunInfo, event ai.StreamEvent,
) (ai.StreamEvent, error) {
	c.events.Add(1)
	return event, nil
}

func TestCapabilitiesCanBeScopedToOneRun(t *testing.T) {
	capability := &runOnlyCapability{}
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(params.Tools) == 0 {
			if params.Instructions != "Agent instructions." || params.Settings.MaxTokens != 0 {
				t.Fatalf("run capability leaked: params=%+v", params)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "without capability"}}}, nil
		}
		if len(params.Tools) != 1 || params.Tools[0].Name != "run_capability_tool" ||
			params.Settings.MaxTokens != 321 ||
			params.Instructions != "Agent instructions.\n\nRun capability static.\n\nRun instructions.\n\nRun capability dynamic." {
			t.Fatalf("run capability contributions missing: %+v", params)
		}
		if _, ok := messages[len(messages)-1].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart); ok {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "with capability"}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "run_capability_tool", ToolCallID: "call", Args: []byte(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithInstructions("Agent instructions."))
	withCapability, err := agent.Run(
		t.Context(), "go", deps{},
		ai.WithRunInstructions("Run instructions."),
		ai.WithRunCapabilities(capability),
	)
	if err != nil || withCapability.Output != "with capability" {
		t.Fatalf("run capability failed: result=%+v err=%v", withCapability, err)
	}
	if capability.setupCalls.Load() != 1 || capability.runCalls.Load() != 1 || capability.events.Load() == 0 {
		t.Fatalf("unexpected capability lifecycle: setup=%d run=%d events=%d",
			capability.setupCalls.Load(), capability.runCalls.Load(), capability.events.Load())
	}

	withoutCapability, err := agent.Run(t.Context(), "go", deps{})
	if err != nil || withoutCapability.Output != "without capability" {
		t.Fatalf("capability leaked into later run: result=%+v err=%v", withoutCapability, err)
	}
	if capability.setupCalls.Load() != 1 || capability.runCalls.Load() != 1 {
		t.Fatalf("run capability was reused unexpectedly: setup=%d run=%d",
			capability.setupCalls.Load(), capability.runCalls.Load())
	}
}

type failingRunCapability struct{ err error }

func (c failingRunCapability) Setup(*ai.CapabilityRegistry) error { return c.err }

func TestRunCapabilitySetupFailure(t *testing.T) {
	sentinel := errors.New("setup failed")
	_, err := ai.NewAgent[deps, string](fakes.NewTestModel()).Run(
		t.Context(), "go", deps{}, ai.WithRunCapabilities(failingRunCapability{err: sentinel}),
	)
	if !errors.Is(err, sentinel) || !strings.Contains(err.Error(), "ai: run capability setup") {
		t.Fatalf("unexpected setup error: %v", err)
	}
}

type duplicateRunCapability struct{}

func (duplicateRunCapability) Setup(registry *ai.CapabilityRegistry) error {
	registry.AddTool(ai.ToolDefinition{Name: "duplicate"}, func(context.Context, json.RawMessage) (any, error) {
		return nil, nil
	})
	return nil
}

func TestRunCapabilityRejectsDuplicateTool(t *testing.T) {
	tool := ai.NewTool("duplicate", func(
		context.Context, *ai.RunContext[deps], reusableToolArgs,
	) (string, error) {
		return "", nil
	})
	_, err := ai.NewAgent[deps, string](fakes.NewTestModel()).Run(
		t.Context(), "go", deps{}, ai.WithRunTools(tool), ai.WithRunCapabilities(duplicateRunCapability{}),
	)
	if err == nil || !strings.Contains(err.Error(), `duplicate run capability tool name "duplicate"`) {
		t.Fatalf("unexpected duplicate capability tool error: %v", err)
	}
}
