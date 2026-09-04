package ai_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

type historyCountingModel struct {
	mu       sync.Mutex
	calls    int
	failCall int
	requests [][]ai.ModelMessage
}

func (*historyCountingModel) Name() string { return "history-counter" }

func (model *historyCountingModel) CountTokens(
	_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
) (ai.Usage, error) {
	model.mu.Lock()
	defer model.mu.Unlock()
	model.calls++
	if model.calls == model.failCall {
		return ai.Usage{}, errors.New("counter unavailable")
	}
	return ai.Usage{InputTokens: len(messages) * 10, Details: map[string]int{"messages": len(messages)}}, nil
}

func (model *historyCountingModel) Request(
	_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
) (*ai.ModelResponse, error) {
	model.mu.Lock()
	defer model.mu.Unlock()
	model.requests = append(model.requests, messages)
	return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
}

func historyTurn(question, answer string) []ai.ModelMessage {
	return []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: question}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: answer}}},
	}
}

func TestTokenHistoryTrimmerKeepsLargestSuffixThatFits(t *testing.T) {
	model := &historyCountingModel{}
	history := append(historyTurn("one", "first"), historyTurn("two", "second")...)
	history = append(history, historyTurn("three", "third")...)
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(ai.TokenHistoryTrimmer{MaxInputTokens: 30}))
	result, err := agent.Run(t.Context(), "current", struct{}{}, ai.WithMessageHistory(history))
	if err != nil {
		t.Fatal(err)
	}
	if len(model.requests) != 1 || len(model.requests[0]) != 3 {
		t.Fatalf("unexpected trimmed request: %#v", model.requests)
	}
	first := model.requests[0][0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart)
	if first.Content != "three" {
		t.Fatalf("oldest fitting turn was not retained: %#v", model.requests[0])
	}
	if messages := result.Messages(); len(messages) != 8 ||
		messages[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content != "one" {
		t.Fatalf("durable history was trimmed: %#v", messages)
	}
	if model.calls != 4 {
		t.Fatalf("expected binary token counts, got %d", model.calls)
	}
}

func TestTokenHistoryTrimmerRunsAfterOrdinaryHistoryProcessors(t *testing.T) {
	model := &historyCountingModel{}
	keepCurrent := ai.HistoryProcessor(func(
		_ context.Context, _ *ai.RunInfo, messages []ai.ModelMessage,
	) ([]ai.ModelMessage, error) {
		return messages[len(messages)-1:], nil
	})
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(
		ai.TokenHistoryTrimmer{MaxInputTokens: 10}, keepCurrent,
	))
	if _, err := agent.Run(
		t.Context(), "current", struct{}{}, ai.WithMessageHistory(historyTurn("old", "answer")),
	); err != nil {
		t.Fatal(err)
	}
	if model.calls != 1 || len(model.requests[0]) != 1 {
		t.Fatalf("trimmer ran before history processing: counts=%d request=%#v", model.calls, model.requests[0])
	}
}

func TestTokenHistoryTrimmerDoesNotCountAgainWhenRequestFits(t *testing.T) {
	model := &historyCountingModel{}
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(ai.TokenHistoryTrimmer{MaxInputTokens: 10}))
	if _, err := agent.Run(t.Context(), "current", struct{}{}); err != nil {
		t.Fatal(err)
	}
	if model.calls != 1 || len(model.requests[0]) != 1 {
		t.Fatalf("unexpected fit behavior: counts=%d requests=%#v", model.calls, model.requests)
	}
}

func TestTokenHistoryTrimmerProtectsRecentTurns(t *testing.T) {
	model := &historyCountingModel{}
	history := append(historyTurn("one", "first"), historyTurn("two", "second")...)
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(ai.TokenHistoryTrimmer{
		MaxInputTokens: 20, MinimumRecentTurns: 2,
	}))
	_, err := agent.Run(t.Context(), "current", struct{}{}, ai.WithMessageHistory(history))
	var limitErr *ai.HistoryTokenLimitError
	if !errors.Is(err, ai.ErrHistoryTokenLimitExceeded) || !errors.As(err, &limitErr) ||
		limitErr.Limit != 20 || limitErr.Usage.InputTokens != 30 || limitErr.Usage.Details["messages"] != 3 {
		t.Fatalf("unexpected protected-history error: %v details=%+v", err, limitErr)
	}
	limitErr.Usage.Details["messages"] = 99
	if model.requests != nil {
		t.Fatalf("generation ran after trimming failure: %#v", model.requests)
	}
}

func TestTokenHistoryTrimmerProtectsCompleteCurrentToolTurn(t *testing.T) {
	model := &historyCountingModel{}
	modelRequest := func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		model.mu.Lock()
		defer model.mu.Unlock()
		model.requests = append(model.requests, messages)
		if len(model.requests) == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
				ToolName: "lookup", ToolCallID: "call", Args: []byte(`{}`),
			}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	}
	wrapped := fakes.NewFunctionModel(modelRequest)
	counting := &countingRequestModel{Model: wrapped, count: model.CountTokens}
	history := append(historyTurn("old one", "answer"), historyTurn("old two", "answer")...)
	agent := ai.NewAgent[struct{}, string](counting, ai.WithCapabilities(ai.TokenHistoryTrimmer{MaxInputTokens: 30}))
	agent.AddTool(ai.NewSimpleTool[struct{}](
		"lookup", func(context.Context, struct{}) (string, error) { return "result", nil },
	))
	if _, err := agent.Run(t.Context(), "current", struct{}{}, ai.WithMessageHistory(history)); err != nil {
		t.Fatal(err)
	}
	if len(model.requests) != 2 || len(model.requests[0]) != 3 || len(model.requests[1]) != 3 {
		t.Fatalf("current tool turn was split: %#v", model.requests)
	}
	call := model.requests[1][1].(ai.ModelResponse).Parts[0].(ai.ToolCallPart)
	result := model.requests[1][2].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart)
	if call.ToolCallID != result.ToolCallID {
		t.Fatalf("tool call and result were not retained together: %+v %+v", call, result)
	}
}

type countingRequestModel struct {
	ai.Model
	count func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (ai.Usage, error)
}

func (model *countingRequestModel) CountTokens(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (ai.Usage, error) {
	return model.count(ctx, messages, params)
}

func TestTokenHistoryTrimmerDoesNotSplitInterleavedUserPromptAndToolResult(t *testing.T) {
	model := &historyCountingModel{}
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "start"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "lookup", ToolCallID: "call", Args: []byte(`{}`),
		}}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "lookup", ToolCallID: "call", Content: "result"},
			ai.UserPromptPart{Content: "also consider this"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "answer"}}},
	}
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(ai.TokenHistoryTrimmer{MaxInputTokens: 30}))
	if _, err := agent.Run(t.Context(), "current", struct{}{}, ai.WithMessageHistory(history)); err != nil {
		t.Fatal(err)
	}
	if len(model.requests[0]) != 1 ||
		model.requests[0][0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content != "current" {
		t.Fatalf("interleaved tool turn was split: %#v", model.requests[0])
	}
}

func TestTokenHistoryTrimmerCannotTrimEmptyOrUngroupedHistory(t *testing.T) {
	trimmer := ai.TokenHistoryTrimmer{MaxInputTokens: 1}
	fixed := &countingModel{countUsage: ai.Usage{InputTokens: 2}}
	_, err := trimmer.BeforeModelRequest(t.Context(), &ai.RunInfo{RunID: "run"}, ai.ModelRequestContext{Model: fixed})
	if !errors.Is(err, ai.ErrHistoryTokenLimitExceeded) {
		t.Fatalf("unexpected empty-history error: %v", err)
	}

	model := &historyCountingModel{}
	request := ai.ModelRequestContext{Model: model, Messages: []ai.ModelMessage{
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "answer"}}},
	}}
	_, err = (ai.TokenHistoryTrimmer{MaxInputTokens: 5}).BeforeModelRequest(
		t.Context(), &ai.RunInfo{RunID: "run"}, request,
	)
	if !errors.Is(err, ai.ErrHistoryTokenLimitExceeded) {
		t.Fatalf("unexpected ungrouped-history error: %v", err)
	}
}

func TestTokenHistoryTrimmerProtectsEarlierCurrentRunAndLargeRecentWindow(t *testing.T) {
	model := &historyCountingModel{}
	messages := append(historyTurn("one", "first"), historyTurn("two", "second")...)
	messages = append(messages, historyTurn("three", "third")...)
	second := messages[2].(ai.ModelRequest)
	second.RunID = "run"
	messages[2] = second
	request := ai.ModelRequestContext{Model: model, Messages: messages}
	_, err := (ai.TokenHistoryTrimmer{MaxInputTokens: 20}).BeforeModelRequest(
		t.Context(), &ai.RunInfo{RunID: "run"}, request,
	)
	var limitErr *ai.HistoryTokenLimitError
	if !errors.As(err, &limitErr) || limitErr.Usage.InputTokens != 40 {
		t.Fatalf("earlier current-run turn was not protected: %v", err)
	}

	_, err = (ai.TokenHistoryTrimmer{MaxInputTokens: 20, MinimumRecentTurns: 10}).BeforeModelRequest(
		t.Context(), &ai.RunInfo{RunID: "other"}, request,
	)
	if !errors.As(err, &limitErr) || limitErr.Usage.InputTokens != 60 {
		t.Fatalf("large recent window was not protected: %v", err)
	}
}

func TestTokenHistoryTrimmerErrors(t *testing.T) {
	tests := []struct {
		name    string
		trimmer ai.TokenHistoryTrimmer
		want    string
	}{
		{name: "missing limit", trimmer: ai.TokenHistoryTrimmer{}, want: "must be positive"},
		{name: "negative limit", trimmer: ai.TokenHistoryTrimmer{MaxInputTokens: -1}, want: "got -1"},
		{name: "negative recent", trimmer: ai.TokenHistoryTrimmer{
			MaxInputTokens: 1, MinimumRecentTurns: -1,
		}, want: "must be non-negative"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.trimmer.Setup(&ai.CapabilityRegistry{})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected setup error: %v", err)
			}
		})
	}

	unsupported := ai.NewAgent[struct{}, string](
		fakes.NewTestModel(), ai.WithCapabilities(ai.TokenHistoryTrimmer{MaxInputTokens: 1}),
	)
	if _, err := unsupported.Run(t.Context(), "current", struct{}{}); !errors.Is(err, ai.ErrTokenCountingUnsupported) {
		t.Fatalf("unexpected unsupported-model error: %v", err)
	}

	history := append(historyTurn("one", "first"), historyTurn("two", "second")...)
	history = append(history, historyTurn("three", "third")...)
	for _, failCall := range []int{2, 3} {
		model := &historyCountingModel{failCall: failCall}
		_, err := ai.NewAgent[struct{}, string](
			model, ai.WithCapabilities(ai.TokenHistoryTrimmer{MaxInputTokens: 30}),
		).Run(t.Context(), "current", struct{}{}, ai.WithMessageHistory(history))
		if err == nil || !strings.Contains(err.Error(), "count trimmed history tokens: counter unavailable") {
			t.Fatalf("unexpected count %d error: %v", failCall, err)
		}
	}
}
