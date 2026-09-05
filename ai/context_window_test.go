package ai_test

import (
	"context"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type contextWindowDeps struct{}

type contextWindowResult struct {
	Value string `json:"value"`
}

// probeLoopModel returns a probe tool call until the probe has been called,
// then returns a final text response. usageOnProbe feeds the first response's
// token counts so the test can drive ContextWindowUsed with a known fraction.
func probeLoopModel(usageOnProbe ai.Usage, postProbeText string) *fakes.FunctionModel {
	return fakes.NewFunctionModel(
		func(_ context.Context, msgs []ai.ModelMessage, params ai.ModelRequestParams) (*ai.ModelResponse, error) {
			for _, msg := range msgs {
				resp, ok := msg.(ai.ModelResponse)
				if !ok {
					continue
				}
				for _, part := range resp.Parts {
					if call, ok := part.(ai.ToolCallPart); ok && call.ToolName == "probe" {
						return &ai.ModelResponse{
							Usage: ai.Usage{Requests: 1},
							Parts: []ai.ResponsePart{ai.TextPart{Content: postProbeText}},
						}, nil
					}
				}
			}
			for _, tool := range params.Tools {
				if tool.Name == "probe" {
					return &ai.ModelResponse{
						Usage: ai.Usage{Requests: 1, InputTokens: usageOnProbe.InputTokens, OutputTokens: usageOnProbe.OutputTokens},
						Parts: []ai.ResponsePart{ai.ToolCallPart{
							ToolName: "probe", ToolCallID: "call_probe", Args: []byte(`{}`),
						}},
					}, nil
				}
			}
			return &ai.ModelResponse{
				Usage: ai.Usage{Requests: 1},
				Parts: []ai.ResponsePart{ai.TextPart{Content: postProbeText}},
			}, nil
		},
	)
}

func TestRunContextContextWindowUsedReportsFractionFromTool(t *testing.T) {
	var observed *float64
	model := ai.NewProfiledModel(
		probeLoopModel(ai.Usage{InputTokens: 800, OutputTokens: 200}, "done"),
		ai.ModelProfile{ContextWindow: 1000},
	)
	agent := ai.NewAgent[contextWindowDeps, string](model)
	ai.AddTool(agent, "probe", func(_ context.Context, rc *ai.RunContext[contextWindowDeps], _ struct{}) (string, error) {
		observed = rc.ContextWindowUsed()
		return "ran", nil
	})
	if _, err := agent.Run(context.Background(), "go", contextWindowDeps{}); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if observed == nil {
		t.Fatal("expected ContextWindowUsed to report a fraction during tool execution")
	}
	if *observed != 1.0 {
		t.Fatalf("expected 1.0 (1000 / 1000), got %v", *observed)
	}
}

func TestRunContextContextWindowUsedNilWhenWindowUnknown(t *testing.T) {
	var observed *float64
	model := ai.NewProfiledModel(
		probeLoopModel(ai.Usage{InputTokens: 800, OutputTokens: 200}, "done"),
		ai.ModelProfile{},
	)
	agent := ai.NewAgent[contextWindowDeps, string](model)
	ai.AddTool(agent, "probe", func(_ context.Context, rc *ai.RunContext[contextWindowDeps], _ struct{}) (string, error) {
		observed = rc.ContextWindowUsed()
		return "ran", nil
	})
	if _, err := agent.Run(context.Background(), "go", contextWindowDeps{}); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if observed != nil {
		t.Fatalf("expected nil ContextWindowUsed when window unknown, got %v", *observed)
	}
}

func TestRunContextContextWindowUsedNilWhenTokensZero(t *testing.T) {
	var observed *float64
	model := ai.NewProfiledModel(
		probeLoopModel(ai.Usage{}, "done"),
		ai.ModelProfile{ContextWindow: 4096},
	)
	agent := ai.NewAgent[contextWindowDeps, string](model)
	ai.AddTool(agent, "probe", func(_ context.Context, rc *ai.RunContext[contextWindowDeps], _ struct{}) (string, error) {
		observed = rc.ContextWindowUsed()
		return "ran", nil
	})
	if _, err := agent.Run(context.Background(), "go", contextWindowDeps{}); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if observed != nil {
		t.Fatalf("expected nil ContextWindowUsed when accumulated tokens are zero, got %v", *observed)
	}
}

func TestRunContextContextWindowUsedNilWhenModelNil(t *testing.T) {
	var rc ai.RunContext[contextWindowDeps]
	if used := rc.ContextWindowUsed(); used != nil {
		t.Fatalf("expected nil ContextWindowUsed when model is nil, got %v", *used)
	}
}

func TestModelProfileContextWindowRoundTrips(t *testing.T) {
	base := fakes.NewFunctionModel(
		func(_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "ok"}}}, nil
		},
	)
	profiled := ai.NewProfiledModel(base, ai.ModelProfile{ContextWindow: 256_000, DefaultOutputMode: ai.OutputModeTool})
	if profiled.ModelProfile().ContextWindow != 256_000 {
		t.Fatalf("ContextWindow not preserved on profile: %+v", profiled.ModelProfile())
	}
}
