package ai_test

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
	genaiprices "github.com/pydantic/genai-prices/packages/go"
)

func TestModelResponsePrice(t *testing.T) {
	response := ai.ModelResponse{
		ModelName: "gpt-5", ProviderName: "openai", ProviderURL: "https://unknown.example",
		Timestamp: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
		Usage: ai.Usage{
			InputTokens: 1_000, OutputTokens: 100, CacheReadTokens: 100,
			ReasoningTokens: 10, Details: map[string]int{"reasoning_tokens": 10, "custom_tokens": 1},
		},
	}
	calculation, err := response.Price()
	if err != nil {
		t.Fatal(err)
	}
	if calculation.ProviderID != "openai" || calculation.ModelID != "gpt-5" || calculation.TotalPrice <= 0 {
		t.Fatalf("unexpected response price: %+v", calculation)
	}

	response.ProviderURL = "https://api.openai.com/v1"
	if _, err := response.Price(); err != nil {
		t.Fatal(err)
	}
	response.Usage.InputTokens = -1
	if _, err := response.Price(); !errors.Is(err, genaiprices.ErrInvalidUsage) {
		t.Fatalf("unexpected invalid usage error: %v", err)
	}
	response.ModelName = ""
	response.ProviderURL = ""
	if _, err := response.Price(); !errors.Is(err, genaiprices.ErrModelNotFound) {
		t.Fatalf("unexpected missing model error: %v", err)
	}
}

func TestRunCalculatesResponseCostAutomatically(t *testing.T) {
	cost := 0.123
	for name, response := range map[string]*ai.ModelResponse{
		"calculated": {
			ModelName: "gpt-5", ProviderName: "openai",
			Usage: ai.Usage{Requests: 1, InputTokens: 1_000, OutputTokens: 100},
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}},
		},
		"provider supplied": {
			ModelName: "unknown", ProviderName: "unknown",
			Usage: ai.Usage{Requests: 1, CostUSD: &cost},
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}},
		},
		"unknown": {
			ModelName: "unknown", ProviderName: "unknown", Usage: ai.Usage{Requests: 1},
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}},
		},
	} {
		t.Run(name, func(t *testing.T) {
			model := fakes.NewFunctionModel(func(
				context.Context, []ai.ModelMessage, ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				clone := *response
				clone.Usage = response.Usage.Clone()
				return &clone, nil
			})
			result, err := ai.NewAgent[deps, string](model).Run(t.Context(), "go", deps{})
			if err != nil {
				t.Fatal(err)
			}
			usage := result.Usage()
			switch name {
			case "calculated":
				if usage.CostUSD == nil || *usage.CostUSD <= 0 {
					t.Fatalf("cost was not calculated: %+v", usage)
				}
			case "provider supplied":
				if usage.CostUSD == nil || *usage.CostUSD != cost {
					t.Fatalf("provider cost was replaced: %+v", usage)
				}
			case "unknown":
				if usage.CostUSD != nil {
					t.Fatalf("unknown cost became known: %+v", usage)
				}
			}
		})
	}
}

func TestAutomaticCostParticipatesInUsageLimits(t *testing.T) {
	limit := 0.0
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{
			ModelName: "gpt-5", ProviderName: "openai",
			Usage: ai.Usage{Requests: 1, InputTokens: 1},
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}},
		}, nil
	})
	_, err := ai.NewAgent[deps, string](model, ai.WithUsageLimits(ai.UsageLimits{CostLimitUSD: &limit})).Run(
		t.Context(), "go", deps{},
	)
	if !errors.Is(err, ai.ErrUsageLimitExceeded) {
		t.Fatalf("automatic cost did not trigger the limit: %v", err)
	}
}

func TestContinuationPricesEachRequest(t *testing.T) {
	request := 0
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		request++
		state := ai.ModelResponseStateSuspended
		parts := []ai.ResponsePart(nil)
		if request == 2 {
			state = ai.ModelResponseStateComplete
			parts = []ai.ResponsePart{ai.TextPart{Content: "done"}}
		}
		return &ai.ModelResponse{
			ModelName: "gpt-5", ProviderName: "openai", State: state,
			Usage: ai.Usage{Requests: 1, InputTokens: 1_000, OutputTokens: 100}, Parts: parts,
		}, nil
	})
	result, err := ai.NewAgent[deps, string](model).Run(t.Context(), "go", deps{})
	if err != nil {
		t.Fatal(err)
	}
	single, err := (ai.ModelResponse{
		ModelName: "gpt-5", ProviderName: "openai",
		Usage: ai.Usage{InputTokens: 1_000, OutputTokens: 100},
	}).Price()
	if err != nil {
		t.Fatal(err)
	}
	usage := result.Usage()
	if request != 2 || usage.CostUSD == nil || math.Abs(*usage.CostUSD-single.TotalPrice*2) > 1e-12 {
		t.Fatalf("continuation cost was not summed per request: requests=%d usage=%+v single=%+v", request, usage, single)
	}
}
