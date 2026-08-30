package ai_test

import (
	"context"
	"errors"
	"math"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

type usageCaptureCapability struct {
	seen []ai.Usage
}

func (*usageCaptureCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (c *usageCaptureCapability) WrapModelRequest(
	ctx context.Context,
	runInfo *ai.RunInfo,
	messages []ai.ModelMessage,
	params ai.ModelRequestParams,
	next ai.ModelRequestFunc,
) (*ai.ModelResponse, error) {
	c.seen = append(c.seen, runInfo.Usage())
	return next(ctx, messages, params)
}

func TestRichUsageAccumulatesAcrossRequestsAndTools(t *testing.T) {
	firstCost := 0.4
	secondCost := 0.6
	request := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		usage := ai.Usage{
			Requests: 1, InputTokens: 10, CacheWriteTokens: 2, CacheReadTokens: 4,
			InputAudioTokens: 3, CacheAudioReadTokens: 1, OutputTokens: 8,
			OutputAudioTokens: 2, ReasoningTokens: 5,
			AcceptedPredictionTokens: 2, RejectedPredictionTokens: 1,
			Details: map[string]int{"provider_units": request + 1},
		}
		if request == 1 {
			usage.CostUSD = &firstCost
			return &ai.ModelResponse{Usage: usage, Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "work-1", Args: []byte(`{}`),
			}}}, nil
		}
		usage.CostUSD = &secondCost
		return &ai.ModelResponse{Usage: usage, Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	capture := &usageCaptureCapability{}
	agent := ai.NewAgent[deps, string](model, ai.WithCapabilities(capture))
	ai.AddTool(agent, "work", func(
		_ context.Context, runContext *ai.RunContext[deps], _ struct{},
	) (string, error) {
		usage := runContext.Usage()
		if usage.InputTokens != 10 || usage.ToolCalls != 0 || usage.Details["provider_units"] != 2 {
			t.Fatalf("tool saw unexpected usage: %+v", usage)
		}
		usage.Details["provider_units"] = 0
		return "ok", nil
	})

	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	usage := result.Usage()
	if usage.Requests != 2 || usage.ToolCalls != 1 || usage.InputTokens != 20 ||
		usage.CacheWriteTokens != 4 || usage.CacheReadTokens != 8 || usage.InputAudioTokens != 6 ||
		usage.CacheAudioReadTokens != 2 || usage.OutputTokens != 16 || usage.OutputAudioTokens != 4 ||
		usage.ReasoningTokens != 10 || usage.AcceptedPredictionTokens != 4 ||
		usage.RejectedPredictionTokens != 2 || usage.Details["provider_units"] != 5 ||
		usage.CostUSD == nil || *usage.CostUSD != 1 {
		t.Fatalf("unexpected accumulated usage: %+v", usage)
	}
	if usage.IsZero() || !(ai.Usage{}).IsZero() ||
		math.Abs(usage.CacheHitRatio()-0.4) > 1e-9 || (ai.Usage{}).CacheHitRatio() != 0 {
		t.Fatalf("unexpected cache ratios: rich=%g empty=%g", usage.CacheHitRatio(), (ai.Usage{}).CacheHitRatio())
	}
	if len(capture.seen) != 2 || capture.seen[0].ToolCalls != 0 || capture.seen[1].ToolCalls != 1 {
		t.Fatalf("capability did not observe tool usage: %+v", capture.seen)
	}
	detached := result.Usage()
	detached.Details["provider_units"] = 0
	*detached.CostUSD = 99
	if result.Usage().Details["provider_units"] != 5 || *result.Usage().CostUSD != 1 {
		t.Fatalf("Usage returned mutable run state: %+v", result.Usage())
	}
}

func TestToolCallUsageLimit(t *testing.T) {
	for _, strategy := range []ai.EndStrategy{
		ai.EndStrategyEarly, ai.EndStrategyGraceful, ai.EndStrategyExhaustive,
	} {
		t.Run(string(strategy), func(t *testing.T) {
			zero := 0
			called := false
			model := fakes.NewFunctionModel(func(
				_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
					ToolName: params.Tools[0].Name, ToolCallID: "work-1", Args: []byte(`{}`),
				}}}, nil
			})
			agent := ai.NewAgent[deps, string](
				model, ai.WithEndStrategy(strategy),
				ai.WithUsageLimits(ai.UsageLimits{ToolCallLimit: &zero}),
			)
			ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) {
				called = true
				return "ok", nil
			})
			if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, ai.ErrUsageLimitExceeded) {
				t.Fatalf("expected tool call limit failure, got %v", err)
			}
			if called {
				t.Fatal("tool ran after projected usage exceeded the limit")
			}
		})
	}
}

func TestToolCallUsageLimitAllowsProjectedCalls(t *testing.T) {
	one := 1
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if len(messages) == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: params.Tools[0].Name, ToolCallID: "work-1", Args: []byte(`{}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithUsageLimits(ai.UsageLimits{ToolCallLimit: &one}))
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) { return "ok", nil })
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Usage().ToolCalls != 1 {
		t.Fatalf("unexpected tool usage: %+v", result.Usage())
	}
}

func TestStreamedToolCallUsageLimit(t *testing.T) {
	for _, strategy := range []ai.EndStrategy{ai.EndStrategyGraceful, ai.EndStrategyExhaustive} {
		t.Run(string(strategy), func(t *testing.T) {
			zero := 0
			model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
				return []ai.ModelStreamEvent{
					ai.TextDeltaEvent{PartID: "text", Delta: "done"},
					ai.ToolCallStartEvent{PartID: "tool", ToolName: "work", ToolCallID: "work-1"},
					ai.ToolCallDeltaEvent{PartID: "tool", ArgsDelta: `{}`}, ai.FinishEvent{},
				}
			})
			agent := ai.NewAgent[deps, string](
				model, ai.WithEndStrategy(strategy),
				ai.WithUsageLimits(ai.UsageLimits{ToolCallLimit: &zero}),
			)
			ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) {
				t.Fatal("tool ran after projected usage exceeded the limit")
				return "", nil
			})
			stream := agent.RunStream(t.Context(), "go", deps{})
			var got error
			for _, err := range stream.Events() {
				if err != nil {
					got = err
				}
			}
			if !errors.Is(got, ai.ErrUsageLimitExceeded) {
				t.Fatalf("expected streamed tool call limit failure, got %v", got)
			}
		})
	}
}

func TestEarlyOutputDoesNotReserveSkippedToolUsage(t *testing.T) {
	zero := 0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, params ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: params.OutputTool.Name, ToolCallID: "result", Args: []byte(`{"city":"Oslo","temp_c":3}`)},
			ai.ToolCallPart{ToolName: params.Tools[0].Name, ToolCallID: "work-1", Args: []byte(`{}`)},
		}}, nil
	})
	agent := ai.NewAgent[deps, weather](
		model, ai.WithEndStrategy(ai.EndStrategyEarly),
		ai.WithUsageLimits(ai.UsageLimits{ToolCallLimit: &zero}),
	)
	ai.AddSimpleTool(agent, "work", func(context.Context, struct{}) (string, error) {
		t.Fatal("skipped tool ran")
		return "", nil
	})
	result, err := agent.Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.City != "Oslo" || result.Usage().ToolCalls != 0 {
		t.Fatalf("unexpected early output or usage: result=%+v usage=%+v", result.Output, result.Usage())
	}
}

func TestCostUsageLimit(t *testing.T) {
	cost := 2.0
	limit := 1.0
	model := fakes.NewFunctionModel(func(
		_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{
			Usage: ai.Usage{Requests: 1, CostUSD: &cost},
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}},
		}, nil
	})
	agent := ai.NewAgent[deps, string](model, ai.WithUsageLimits(ai.UsageLimits{CostLimitUSD: &limit}))
	if _, err := agent.Run(t.Context(), "go", deps{}); !errors.Is(err, ai.ErrUsageLimitExceeded) {
		t.Fatalf("expected cost limit failure, got %v", err)
	}

	unknownCost := fakes.NewTestModel()
	agent = ai.NewAgent[deps, string](unknownCost, ai.WithUsageLimits(ai.UsageLimits{CostLimitUSD: &limit}))
	if _, err := agent.Run(t.Context(), "go", deps{}); err != nil {
		t.Fatalf("unknown cost should not fail the run: %v", err)
	}
}
