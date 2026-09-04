package ai_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

type countingModel struct {
	mu             sync.Mutex
	name           string
	providerName   string
	providerURL    string
	countUsage     ai.Usage
	countErr       error
	countCalls     int
	requestCalls   int
	countMessages  []ai.ModelMessage
	countParams    ai.ModelRequestParams
	requestHandler func(int) *ai.ModelResponse
}

func (model *countingModel) Name() string {
	if model.name != "" {
		return model.name
	}
	return "counting"
}

func (model *countingModel) ProviderName() string { return model.providerName }
func (model *countingModel) ProviderURL() string  { return model.providerURL }

func (model *countingModel) CountTokens(
	_ context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (ai.Usage, error) {
	model.mu.Lock()
	defer model.mu.Unlock()
	model.countCalls++
	model.countMessages = messages
	model.countParams = params
	return model.countUsage.Clone(), model.countErr
}

func (model *countingModel) Request(
	_ context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	model.mu.Lock()
	defer model.mu.Unlock()
	model.requestCalls++
	if model.requestHandler != nil {
		return model.requestHandler(model.requestCalls), nil
	}
	return &ai.ModelResponse{
		Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}},
		Usage: ai.Usage{Requests: 1, InputTokens: model.countUsage.InputTokens},
	}, nil
}

type tokenCountRewriteCapability struct{}

func (tokenCountRewriteCapability) Setup(*ai.CapabilityRegistry) error { return nil }

func (tokenCountRewriteCapability) WrapModelRequest(
	ctx context.Context,
	_ *ai.RunInfo,
	_ []ai.ModelMessage,
	params ai.ModelRequestParams,
	next ai.ModelRequestFunc,
) (*ai.ModelResponse, error) {
	messages := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.UserPromptPart{Content: "rewritten"},
	}}}
	params.Settings.ExtraHeaders = map[string]string{"X-Rewritten": "true"}
	return next(ctx, messages, params)
}

func TestCountModelTokensUsesDetachedRequest(t *testing.T) {
	model := &countingModel{countUsage: ai.Usage{InputTokens: 7, Details: map[string]int{"cached": 3}}}
	messages := []ai.ModelMessage{ai.ModelRequest{
		Parts: []ai.RequestPart{ai.UserPromptPart{Content: "original"}},
	}}
	params := ai.ModelRequestParams{
		Settings: ai.ModelSettings{ExtraHeaders: map[string]string{"X-Test": "original"}},
		Tools:    []ai.ToolDefinition{{Name: "tool", Schema: map[string]any{"type": "object"}}},
	}
	usage, err := ai.CountModelTokens(t.Context(), model, messages, params)
	if err != nil {
		t.Fatal(err)
	}
	model.countMessages[0].(ai.ModelRequest).Parts[0] = ai.UserPromptPart{Content: "changed"}
	model.countParams.Settings.ExtraHeaders["X-Test"] = "changed"
	model.countParams.Tools[0].Schema["type"] = "changed"
	usage.Details["cached"] = 99
	if messages[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content != "original" ||
		params.Settings.ExtraHeaders["X-Test"] != "original" || params.Tools[0].Schema["type"] != "object" ||
		model.countUsage.Details["cached"] != 3 {
		t.Fatal("token count request or result was not detached")
	}
	if usage.InputTokens != 7 {
		t.Fatalf("unexpected count usage: %+v", usage)
	}
}

func TestTokenCountingModelWrappers(t *testing.T) {
	model := &countingModel{
		providerName: "openai", providerURL: "https://api.openai.com/v1", countUsage: ai.Usage{InputTokens: 8},
	}
	wrapped := ai.WrapModel(model)
	if wrapped.ProviderName() != "openai" || wrapped.ProviderURL() != "https://api.openai.com/v1" {
		t.Fatalf("unexpected wrapped provider identity: %q %q", wrapped.ProviderName(), wrapped.ProviderURL())
	}
	usage, err := wrapped.CountTokens(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil || usage.InputTokens != 8 {
		t.Fatalf("unexpected wrapped count: %+v err=%v", usage, err)
	}
	limiter := ai.NewConcurrencyLimiter(1)
	limited := ai.NewConcurrencyLimitedModel(model, limiter)
	usage, err = limited.CountTokens(t.Context(), nil, ai.ModelRequestParams{})
	if err != nil || usage.InputTokens != 8 || limiter.Running() != 0 {
		t.Fatalf("unexpected limited count: %+v running=%d err=%v", usage, limiter.Running(), err)
	}
	if model.countCalls != 2 {
		t.Fatalf("unexpected wrapped count calls: %d", model.countCalls)
	}
	full := ai.NewConcurrencyLimiter(1, ai.WithMaxQueued(0))
	if err := full.Acquire(t.Context(), "test"); err != nil {
		t.Fatal(err)
	}
	limited = ai.NewConcurrencyLimitedModel(model, full)
	if _, err := limited.CountTokens(
		t.Context(), nil, ai.ModelRequestParams{},
	); !errors.Is(err, ai.ErrConcurrencyLimitExceeded) {
		t.Fatalf("unexpected count admission error: %v", err)
	}
	full.Release()
	withoutIdentity := ai.WrapModel(&requestOnlyModel{})
	if withoutIdentity.ProviderName() != "" || withoutIdentity.ProviderURL() != "" {
		t.Fatalf("unexpected absent provider identity: %q %q", withoutIdentity.ProviderName(), withoutIdentity.ProviderURL())
	}
}

func TestCountModelTokensErrors(t *testing.T) {
	if _, err := ai.CountModelTokens(t.Context(), nil, nil, ai.ModelRequestParams{}); !errors.Is(err, ai.ErrNoModel) {
		t.Fatalf("unexpected nil model error: %v", err)
	}
	var typedNil *countingModel
	if _, err := ai.CountModelTokens(
		t.Context(), typedNil, nil, ai.ModelRequestParams{},
	); !errors.Is(err, ai.ErrNoModel) {
		t.Fatalf("unexpected typed nil model error: %v", err)
	}
	unsupported := &requestOnlyModel{}
	if _, err := ai.CountModelTokens(
		t.Context(), unsupported, nil, ai.ModelRequestParams{},
	); !errors.Is(err, ai.ErrTokenCountingUnsupported) {
		t.Fatalf("unexpected unsupported error: %v", err)
	}
	invalid := &countingModel{}
	if _, err := ai.CountModelTokens(t.Context(), invalid, nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		RequestTimeout: -1,
	}}); err == nil || invalid.countCalls != 0 {
		t.Fatalf("unexpected invalid settings result: calls=%d err=%v", invalid.countCalls, err)
	}
	countErr := errors.New("count failed")
	model := &countingModel{countErr: countErr}
	if _, err := ai.CountModelTokens(
		t.Context(), model, nil, ai.ModelRequestParams{},
	); !errors.Is(err, countErr) {
		t.Fatalf("unexpected count error: %v", err)
	}
}

type blockingTokenModel struct{}

func (*blockingTokenModel) Name() string { return "blocking" }

func (*blockingTokenModel) Request(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return nil, nil
}

func (*blockingTokenModel) CountTokens(
	ctx context.Context, _ []ai.ModelMessage, _ ai.ModelRequestParams,
) (ai.Usage, error) {
	<-ctx.Done()
	return ai.Usage{}, context.Cause(ctx)
}

func TestCountModelTokensRequestTimeout(t *testing.T) {
	_, err := ai.CountModelTokens(t.Context(), &blockingTokenModel{}, nil, ai.ModelRequestParams{
		Settings: ai.ModelSettings{RequestTimeout: time.Millisecond},
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("unexpected token count timeout: %v", err)
	}
}

type requestOnlyModel struct{}

func (*requestOnlyModel) Name() string { return "request-only" }

func (*requestOnlyModel) Request(
	context.Context, []ai.ModelMessage, ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
}

func TestPreRequestTokenCountingUsesFinalMiddlewareRequest(t *testing.T) {
	model := &countingModel{countUsage: ai.Usage{InputTokens: 5, Requests: 99}}
	agent := ai.NewAgent[struct{}, string](
		model,
		ai.WithCapabilities(tokenCountRewriteCapability{}),
		ai.WithUsageLimits(ai.UsageLimits{CountTokensBeforeRequest: true, InputTokenLimit: 5}),
	)
	result, err := agent.Run(t.Context(), "original", struct{}{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "done" || model.countCalls != 1 || model.requestCalls != 1 {
		t.Fatalf("unexpected run result=%+v count=%d request=%d", result, model.countCalls, model.requestCalls)
	}
	request := model.countMessages[0].(ai.ModelRequest)
	if request.Parts[0].(ai.UserPromptPart).Content != "rewritten" ||
		model.countParams.Settings.ExtraHeaders["X-Rewritten"] != "true" {
		t.Fatalf("token counter did not receive final middleware request: %#v %+v", request, model.countParams)
	}
	if result.Usage().Requests != 1 || result.Usage().InputTokens != 5 {
		t.Fatalf("count usage leaked into run usage: %+v", result.Usage())
	}
}

func TestPreRequestTokenCountErrorStopsGeneration(t *testing.T) {
	countErr := errors.New("count failed")
	model := &countingModel{countErr: countErr}
	agent := ai.NewAgent[struct{}, string](model, ai.WithUsageLimits(ai.UsageLimits{
		CountTokensBeforeRequest: true,
	}))
	if _, err := agent.Run(t.Context(), "hello", struct{}{}); !errors.Is(err, countErr) {
		t.Fatalf("unexpected count error: %v", err)
	}
	if model.countCalls != 1 || model.requestCalls != 0 {
		t.Fatalf("unexpected calls after count error: count=%d request=%d", model.countCalls, model.requestCalls)
	}
}

func TestPreRequestTokenLimitsStopGeneration(t *testing.T) {
	tests := []struct {
		name   string
		limits ai.UsageLimits
	}{
		{name: "input", limits: ai.UsageLimits{CountTokensBeforeRequest: true, InputTokenLimit: 4}},
		{name: "total", limits: ai.UsageLimits{CountTokensBeforeRequest: true, TotalTokenLimit: 4}},
		{name: "per request", limits: ai.UsageLimits{
			CountTokensBeforeRequest: true, PerRequestInputTokenLimit: 4,
		}},
		{name: "cost", limits: func() ai.UsageLimits {
			limit := 0.000001
			return ai.UsageLimits{CountTokensBeforeRequest: true, CostLimitUSD: &limit}
		}()},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := &countingModel{
				name: "gpt-5-mini", providerName: "openai", countUsage: ai.Usage{InputTokens: 1_000_000},
			}
			agent := ai.NewAgent[struct{}, string](model, ai.WithUsageLimits(test.limits))
			if _, err := agent.Run(t.Context(), "hello", struct{}{}); !errors.Is(err, ai.ErrUsageLimitExceeded) {
				t.Fatalf("unexpected limit error: %v", err)
			}
			if model.countCalls != 1 || model.requestCalls != 0 {
				t.Fatalf("generation was not prevented: count=%d request=%d", model.countCalls, model.requestCalls)
			}
		})
	}
}

func TestPreRequestTokenLimitUsesCumulativeUsage(t *testing.T) {
	model := &countingModel{countUsage: ai.Usage{InputTokens: 6}}
	model.requestHandler = func(call int) *ai.ModelResponse {
		if call == 1 {
			return &ai.ModelResponse{
				Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "again", ToolCallID: "call", Args: []byte(`{}`)}},
				Usage: ai.Usage{Requests: 1, InputTokens: 6},
			}
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "unexpected"}}}
	}
	agent := ai.NewAgent[struct{}, string](model, ai.WithUsageLimits(ai.UsageLimits{
		CountTokensBeforeRequest: true, InputTokenLimit: 10,
	}))
	agent.AddTool(ai.NewSimpleTool[struct{}](
		"again", func(context.Context, struct{}) (string, error) { return "again", nil },
	))
	if _, err := agent.Run(t.Context(), "hello", struct{}{}); !errors.Is(err, ai.ErrUsageLimitExceeded) {
		t.Fatalf("unexpected cumulative limit error: %v", err)
	}
	if model.countCalls != 2 || model.requestCalls != 1 {
		t.Fatalf("unexpected cumulative calls: count=%d request=%d", model.countCalls, model.requestCalls)
	}
}

func TestPostRequestInputLimitAppliesToRunsAndStreams(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "run", true: "stream"}[stream], func(t *testing.T) {
			model := &countingModel{countUsage: ai.Usage{InputTokens: 6}}
			agent := ai.NewAgent[struct{}, string](model, ai.WithUsageLimits(ai.UsageLimits{
				PerRequestInputTokenLimit: 5,
			}))
			if !stream {
				if _, err := agent.Run(t.Context(), "hello", struct{}{}); !errors.Is(err, ai.ErrUsageLimitExceeded) {
					t.Fatalf("unexpected run limit error: %v", err)
				}
			} else {
				var got error
				for _, err := range agent.RunStream(t.Context(), "hello", struct{}{}).Events() {
					if err != nil {
						got = err
					}
				}
				if !errors.Is(got, ai.ErrUsageLimitExceeded) {
					t.Fatalf("unexpected stream limit error: %v", got)
				}
			}
			if model.requestCalls != 1 || model.countCalls != 0 {
				t.Fatalf("unexpected post-request calls: count=%d request=%d", model.countCalls, model.requestCalls)
			}
		})
	}
}

func TestRequestLimitRejectsCombinedRequestUsage(t *testing.T) {
	model := &countingModel{}
	model.requestHandler = func(int) *ai.ModelResponse {
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}, Usage: ai.Usage{Requests: 2},
		}
	}
	agent := ai.NewAgent[struct{}, string](model, ai.WithUsageLimits(ai.UsageLimits{RequestLimit: 1}))
	if _, err := agent.Run(t.Context(), "hello", struct{}{}); !errors.Is(err, ai.ErrUsageLimitExceeded) {
		t.Fatalf("unexpected combined request limit error: %v", err)
	}
	if model.requestCalls != 1 {
		t.Fatalf("unexpected generation calls: %d", model.requestCalls)
	}
}

func TestRequestLimitStopsNextGeneration(t *testing.T) {
	model := &countingModel{}
	model.requestHandler = func(call int) *ai.ModelResponse {
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: "again", ToolCallID: "call", Args: []byte(`{}`)}},
			Usage: ai.Usage{Requests: 1},
		}
	}
	agent := ai.NewAgent[struct{}, string](model, ai.WithUsageLimits(ai.UsageLimits{RequestLimit: 1}))
	agent.AddTool(ai.NewSimpleTool[struct{}](
		"again", func(context.Context, struct{}) (string, error) { return "again", nil },
	))
	if _, err := agent.Run(t.Context(), "hello", struct{}{}); !errors.Is(err, ai.ErrUsageLimitExceeded) {
		t.Fatalf("unexpected request limit error: %v", err)
	}
	if model.requestCalls != 1 {
		t.Fatalf("request limit allowed %d generations", model.requestCalls)
	}
}
