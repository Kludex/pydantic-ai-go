package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type currentCacheModel struct{ ai.Model }

func (currentCacheModel) CacheRetention(ai.ModelSettings) (time.Duration, bool) {
	return time.Hour, true
}

func TestCacheRetentionContractsAndMixedHistory(t *testing.T) {
	current := currentCacheModel{Model: fakes.NewTestModel()}
	for _, model := range []ai.Model{current, ai.WrapModel(current), &fullyOptionalModel{}} {
		if duration, known := ai.ResolveCacheRetention(model, nil); !known || duration <= 0 {
			t.Fatalf("%T retention=%s known=%t", model, duration, known)
		}
	}
	profile := ai.ModelProfile{DefaultCacheRetention: time.Hour}
	history := []ai.ModelMessage{
		ai.ModelResponse{},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "standing"},
			ai.UserPromptPart{Contents: []ai.UserContent{ai.TextContent{Text: "prompt"}}},
		}},
		ai.ModelResponse{Timestamp: time.Now()},
	}
	if got := ai.PromptCacheOutlook(history, &profile, nil, time.Now()); got != ai.CacheOutlookWarm {
		t.Fatalf("mixed history outlook=%s", got)
	}
}

func TestConcurrencyLimitedCompactionAndEmptyStream(t *testing.T) {
	limiter := ai.NewConcurrencyLimiter(1, ai.WithMaxQueued(0))
	underlying := &explicitCompactionModel{compact: func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		if limiter.Running() != 1 {
			t.Fatal("compaction did not hold a slot")
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.CompactionPart{Content: "summary"}}}, nil
	}}
	model := ai.NewConcurrencyLimitedModel(underlying, limiter)
	if _, err := model.CompactMessages(t.Context(), nil, ai.ModelRequestParams{}); err != nil || limiter.Running() != 0 {
		t.Fatalf("compaction leaked slot: %v", err)
	}
	if err := limiter.Acquire(t.Context(), "holder"); err != nil {
		t.Fatal(err)
	}
	if _, err := model.CompactMessages(t.Context(), nil, ai.ModelRequestParams{}); !errors.Is(err, ai.ErrConcurrencyLimitExceeded) {
		t.Fatalf("compaction bypassed limit: %v", err)
	}
	limiter.Release()
	empty := streamingRequestModel{
		Model: underlying,
		stream: func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
			return nil, nil
		},
	}
	if _, err := ai.NewConcurrencyLimitedModel(empty, limiter).StreamRequest(t.Context(), nil, ai.ModelRequestParams{}); err == nil || limiter.Running() != 0 {
		t.Fatalf("empty stream leaked slot: %v", err)
	}
}

type validationDeferralWrapper struct{ early bool }

func (validationDeferralWrapper) Setup(*ai.CapabilityRegistry) error { return nil }

func (wrapper validationDeferralWrapper) WrapToolValidation(
	ctx context.Context, _ *ai.RunInfo, _ ai.ToolHookContext, raw json.RawMessage, next ai.ToolValidationFunc,
) (any, error) {
	if !wrapper.early {
		if _, err := next(ctx, raw); err != nil {
			return nil, err
		}
	}
	return ai.ToolApprovalRequest{}, nil
}

func TestToolValidationWrapperDefersOnlyAfterLifecycle(t *testing.T) {
	for _, early := range []bool{false, true} {
		model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "work", ToolCallID: "call", Args: json.RawMessage(`{"value":1}`),
			}}}, nil
		})
		agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(validationDeferralWrapper{early: early}))
		ai.AddSimpleTool(agent, "work", func(context.Context, hookedArgs) (string, error) {
			t.Fatal("deferred tool executed")
			return "", nil
		})
		result, err := agent.Run(t.Context(), "go", struct{}{})
		if early {
			if err == nil {
				t.Fatal("validation wrapper deferred before validating")
			}
		} else if err != nil || result.Deferred() == nil {
			t.Fatalf("validated wrapper deferral failed: result=%+v err=%v", result, err)
		}
	}
}

func TestOutputValidationAcceptsHookByteSlices(t *testing.T) {
	model := fakes.NewTestModel()
	hook := ai.BeforeOutputValidationFunc(func(
		context.Context, *ai.RunInfo, ai.OutputHookContext, any,
	) (any, error) {
		return []byte(`{"value":2}`), nil
	})
	result, err := ai.NewAgent[struct{}, hookedOutput](model, ai.WithCapabilities(hook)).Run(t.Context(), "go", struct{}{})
	if err != nil || result.Output.Value != 2 {
		t.Fatalf("byte-slice output=%+v err=%v", result, err)
	}
}

func TestWebSearchPricingUsesBilledCount(t *testing.T) {
	response := ai.ModelResponse{ModelName: "gpt-5", ProviderName: "openai", Usage: ai.Usage{
		Details: map[string]int{"web_search_requests": 10, "web_searches": 1},
	}}
	price, err := response.Price()
	if err != nil {
		t.Fatal(err)
	}
	response.Usage.Details = map[string]int{"web_search_requests": 1}
	comparison, err := response.Price()
	if err != nil || price.TotalPrice != comparison.TotalPrice {
		t.Fatalf("query count overbilled: price=%+v comparison=%+v err=%v", price, comparison, err)
	}
}

func TestHistoryRepairMergesRequestMetadata(t *testing.T) {
	messages := ai.RepairMessages([]ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "first"}}},
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "second"}}, Metadata: map[string]any{"source": "later"}},
	}, ai.MessageRepairOptions{})
	if len(messages) != 1 || messages[0].(ai.ModelRequest).Metadata["source"] != "later" {
		t.Fatalf("merged metadata=%+v", messages)
	}
}

func TestInstrumentationReportsInstructionsWhenRunIsShortCircuited(t *testing.T) {
	for _, instructions := range []string{"", "stored"} {
		calls := 0
		agent := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithCapabilities(
			ai.NewInstrumentation(), &shortCircuitRunCapability{output: "cached", calls: &calls},
		))
		result, err := agent.Run(t.Context(), "go", struct{}{}, ai.WithMessageHistory([]ai.ModelMessage{
			ai.ModelRequest{Instructions: instructions, Parts: []ai.RequestPart{ai.UserPromptPart{Content: "prior"}}},
			ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "prior"}}},
		}))
		if err != nil || result.Output != "cached" || calls != 1 {
			t.Fatalf("short-circuit instrumentation result=%+v err=%v", result, err)
		}
	}
}

type cachedModelRequest struct{}

func (cachedModelRequest) Setup(*ai.CapabilityRegistry) error { return nil }

func (cachedModelRequest) WrapModelRequest(
	context.Context, *ai.RunInfo, []ai.ModelMessage, ai.ModelRequestParams, ai.ModelRequestFunc,
) (*ai.ModelResponse, error) {
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "cached"}}}, nil
}

func TestInstrumentationReportsInstructionsForCachedRequests(t *testing.T) {
	agent := ai.NewAgent[struct{}, string](fakes.NewTestModel(), ai.WithInstructions("stable"),
		ai.WithCapabilities(cachedModelRequest{}, ai.NewInstrumentation()))
	if result, err := agent.Run(t.Context(), "go", struct{}{}); err != nil || result.Output != "cached" {
		t.Fatalf("cached request instrumentation=%+v err=%v", result, err)
	}
}

func TestChoicesDecoderHandlesUnconstrainedHookSchemas(t *testing.T) {
	attempt := 0
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		arguments := []json.RawMessage{json.RawMessage(`not-json`), json.RawMessage(`"missing"`), json.RawMessage(`"known"`)}
		response := &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "choices", ToolCallID: "result", Args: arguments[attempt]}}}
		attempt++
		return response, nil
	})
	hook := ai.BeforeModelRequestFunc(func(_ context.Context, _ *ai.RunInfo, request ai.ModelRequestContext) (ai.ModelRequestContext, error) {
		request.Params.OutputTool.Schema = nil
		return request, nil
	})
	agent := ai.NewChoicesAgent[struct{}](model, ai.NewStringChoices("", "", "known"), ai.WithCapabilities(hook),
		ai.WithRetryLimits(ai.RetryLimits{Output: 3}))
	result, err := agent.Run(t.Context(), "choose", struct{}{})
	if err != nil || result.Output != "known" || attempt != 3 {
		t.Fatalf("choice decoder result=%+v attempts=%d err=%v", result, attempt, err)
	}
}

func TestContentFilterPreservesStructuredOutput(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{FinishReason: ai.FinishReasonContentFilter, Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "final_result", ToolCallID: "final", Args: json.RawMessage(`{"value":1}`),
		}}}, nil
	})
	result, err := ai.NewAgent[struct{}, hookedOutput](model).Run(t.Context(), "go", struct{}{})
	if err != nil || result.Output.Value != 1 {
		t.Fatalf("partial structured output=%+v err=%v", result, err)
	}
}
