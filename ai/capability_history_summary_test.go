package ai_test

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestHistorySummarizerReplacesOldTurnsAndAttributesUsage(t *testing.T) {
	history := append(historyTurn("one", "first"), historyTurn("two", "second")...)
	summaryUsage := ai.Usage{Requests: 1, InputTokens: 7, Details: map[string]int{"summary": 7}}
	var summarized []ai.ModelMessage
	summarizer := ai.HistorySummarizer{Summarize: func(
		_ context.Context, runInfo *ai.RunInfo, messages []ai.ModelMessage,
	) (ai.HistorySummary, error) {
		if runInfo.RunID == "" || runInfo.ConversationID == "" {
			t.Fatal("summary callback did not receive run identity")
		}
		summarized = messages
		messages[0].(ai.ModelRequest).Parts[0] = ai.UserPromptPart{Content: "changed"}
		return ai.HistorySummary{Content: "The user discussed one and two.", Usage: summaryUsage}, nil
	}}
	var received []ai.ModelMessage
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		received = messages
		return &ai.ModelResponse{
			Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}},
			Usage: ai.Usage{Requests: 1, InputTokens: 2},
		}, nil
	})
	result, err := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(summarizer)).Run(
		t.Context(), "current", struct{}{}, ai.WithMessageHistory(history),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(summarized) != 4 || history[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content != "one" {
		t.Fatalf("summary input was not detached: summarized=%#v history=%#v", summarized, history)
	}
	if len(received) != 2 || received[0].(ai.ModelResponse).Text() != "The user discussed one and two." ||
		received[1].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content != "current" {
		t.Fatalf("unexpected compacted request: %#v", received)
	}
	messages := result.Messages()
	if len(messages) != 3 || messages[0].(ai.ModelResponse).Text() != "The user discussed one and two." ||
		messages[0].(ai.ModelResponse).RunID == "" || messages[0].(ai.ModelResponse).ConversationID == "" ||
		messages[0].(ai.ModelResponse).Timestamp.IsZero() {
		t.Fatalf("durable summary was not retained: %#v", messages)
	}
	usage := result.Usage()
	if usage.Requests != 2 || usage.InputTokens != 9 || usage.Details["summary"] != 7 {
		t.Fatalf("summary usage was not attributed: %+v", usage)
	}
	summaryUsage.Details["summary"] = 99
	if result.Usage().Details["summary"] != 7 {
		t.Fatal("summary usage was not detached")
	}
}

func TestHistorySummarizerRunsOnceAcrossToolLoop(t *testing.T) {
	var summaries atomic.Int32
	var requests atomic.Int32
	summarizer := ai.HistorySummarizer{Summarize: func(
		context.Context, *ai.RunInfo, []ai.ModelMessage,
	) (ai.HistorySummary, error) {
		summaries.Add(1)
		return ai.HistorySummary{Content: "Old discussion."}, nil
	}}
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		call := requests.Add(1)
		if call == 2 {
			if len(messages) != 4 || messages[0].(ai.ModelResponse).Text() != "Old discussion." {
				t.Fatalf("summary was not reused during tool loop: %#v", messages)
			}
			return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
		}
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{
			ToolName: "lookup", ToolCallID: "call", Args: []byte(`{}`),
		}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(summarizer))
	agent.AddTool(ai.NewSimpleTool[struct{}](
		"lookup", func(context.Context, struct{}) (string, error) { return "result", nil },
	))
	if _, err := agent.Run(
		t.Context(), "current", struct{}{}, ai.WithMessageHistory(historyTurn("old", "answer")),
	); err != nil {
		t.Fatal(err)
	}
	if summaries.Load() != 1 || requests.Load() != 2 {
		t.Fatalf("unexpected summary/model calls: summaries=%d requests=%d", summaries.Load(), requests.Load())
	}
}

func TestHistorySummarizerKeepsRecentTurns(t *testing.T) {
	history := append(historyTurn("one", "first"), historyTurn("two", "second")...)
	var summarized int
	summarizer := ai.HistorySummarizer{
		MinimumRecentTurns: 2,
		Summarize: func(
			_ context.Context, _ *ai.RunInfo, messages []ai.ModelMessage,
		) (ai.HistorySummary, error) {
			summarized = len(messages)
			return ai.HistorySummary{Content: "first summary"}, nil
		},
	}
	var received []ai.ModelMessage
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		received = messages
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	if _, err := ai.NewAgent[struct{}, string](model, ai.WithCapabilities(summarizer)).Run(
		t.Context(), "current", struct{}{}, ai.WithMessageHistory(history),
	); err != nil {
		t.Fatal(err)
	}
	if summarized != 2 || len(received) != 4 ||
		received[1].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content != "two" {
		t.Fatalf("recent turns were not preserved: summarized=%d request=%#v", summarized, received)
	}
}

func TestHistorySummarizerSkipsWhenThereAreNoOldTurns(t *testing.T) {
	called := false
	summarizer := ai.HistorySummarizer{Summarize: func(
		context.Context, *ai.RunInfo, []ai.ModelMessage,
	) (ai.HistorySummary, error) {
		called = true
		return ai.HistorySummary{Content: "unexpected"}, nil
	}}
	if _, err := ai.NewAgent[struct{}, string](
		fakes.NewTestModel(), ai.WithCapabilities(summarizer),
	).Run(t.Context(), "current", struct{}{}); err != nil {
		t.Fatal(err)
	}
	if called {
		t.Fatal("summary callback ran without old turns")
	}
}

func TestHistorySummarizerErrors(t *testing.T) {
	if err := (ai.HistorySummarizer{MinimumRecentTurns: -1, Summarize: func(
		context.Context, *ai.RunInfo, []ai.ModelMessage,
	) (ai.HistorySummary, error) {
		return ai.HistorySummary{}, nil
	}}).Setup(
		&ai.CapabilityRegistry{},
	); err == nil || !strings.Contains(err.Error(), "must be non-negative") {
		t.Fatalf("unexpected negative recent-turn error: %v", err)
	}
	if err := (ai.HistorySummarizer{}).Setup(&ai.CapabilityRegistry{}); err == nil ||
		err.Error() != "ai: history summary function must not be nil" {
		t.Fatalf("unexpected nil summary error: %v", err)
	}

	summaryErr := errors.New("summary unavailable")
	for name, summarize := range map[string]ai.HistorySummaryFunc{
		"callback": func(context.Context, *ai.RunInfo, []ai.ModelMessage) (ai.HistorySummary, error) {
			return ai.HistorySummary{}, summaryErr
		},
		"empty": func(context.Context, *ai.RunInfo, []ai.ModelMessage) (ai.HistorySummary, error) {
			return ai.HistorySummary{}, nil
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := ai.NewAgent[struct{}, string](
				fakes.NewTestModel(), ai.WithCapabilities(ai.HistorySummarizer{Summarize: summarize}),
			).Run(t.Context(), "current", struct{}{}, ai.WithMessageHistory(historyTurn("old", "answer")))
			if name == "callback" && (!errors.Is(err, summaryErr) || !strings.Contains(err.Error(), "summarize history")) {
				t.Fatalf("unexpected callback error: %v", err)
			}
			if name == "empty" && (err == nil || err.Error() != "ai: history summary must not be empty") {
				t.Fatalf("unexpected empty summary error: %v", err)
			}
		})
	}
}

func TestHistorySummarizerUsageLimitsStopPrimaryRequest(t *testing.T) {
	var primaryCalls atomic.Int32
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		primaryCalls.Add(1)
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "unexpected"}}}, nil
	})
	summarizer := ai.HistorySummarizer{Summarize: func(
		context.Context, *ai.RunInfo, []ai.ModelMessage,
	) (ai.HistorySummary, error) {
		return ai.HistorySummary{Content: "summary", Usage: ai.Usage{InputTokens: 6}}, nil
	}}
	_, err := ai.NewAgent[struct{}, string](
		model, ai.WithCapabilities(summarizer), ai.WithUsageLimits(ai.UsageLimits{InputTokenLimit: 5}),
	).Run(t.Context(), "current", struct{}{}, ai.WithMessageHistory(historyTurn("old", "answer")))
	if !errors.Is(err, ai.ErrUsageLimitExceeded) || primaryCalls.Load() != 0 {
		t.Fatalf("unexpected side-usage limit result: calls=%d err=%v", primaryCalls.Load(), err)
	}
}
