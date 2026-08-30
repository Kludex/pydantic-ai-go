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
		if usage.InputTokens != 10 || usage.ToolCalls != 0 {
			t.Fatalf("tool saw unexpected usage: %+v", usage)
		}
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
		usage.RejectedPredictionTokens != 2 || usage.CostUSD == nil || *usage.CostUSD != 1 {
		t.Fatalf("unexpected accumulated usage: %+v", usage)
	}
	if math.Abs(usage.CacheHitRatio()-0.4) > 1e-9 || (ai.Usage{}).CacheHitRatio() != 0 {
		t.Fatalf("unexpected cache ratios: rich=%g empty=%g", usage.CacheHitRatio(), (ai.Usage{}).CacheHitRatio())
	}
	if len(capture.seen) != 2 || capture.seen[0].ToolCalls != 0 || capture.seen[1].ToolCalls != 1 {
		t.Fatalf("capability did not observe tool usage: %+v", capture.seen)
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
