package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

// streamingModel wraps a FunctionModel and streams scripted events.
type streamingModel struct {
	ai.Model
	script func(msgs []ai.ModelMessage) []ai.StreamEvent
	fail   error
}

func (m *streamingModel) StreamRequest(_ context.Context, msgs []ai.ModelMessage, _ ai.ModelRequestParams) (iter.Seq2[ai.StreamEvent, error], error) {
	if m.fail != nil {
		return nil, m.fail
	}
	events := m.script(msgs)
	return func(yield func(ai.StreamEvent, error) bool) {
		for _, e := range events {
			if !yield(e, nil) {
				return
			}
		}
	}, nil
}

func newStreamingModel(script func(msgs []ai.ModelMessage) []ai.StreamEvent) *streamingModel {
	return &streamingModel{Model: fakes.NewTestModel(), script: script}
}

func TestRunStreamTextDeltas(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.StreamEvent {
		return []ai.StreamEvent{
			ai.TextDeltaEvent{Delta: "Hel"},
			ai.TextDeltaEvent{Delta: "lo!"},
			ai.FinishEvent{Usage: ai.Usage{Requests: 1, OutputTokens: 2}, ModelName: "scripted"},
		}
	})
	agent := ai.NewAgent[deps, string](model)
	stream := agent.RunStream(t.Context(), "hi", deps{})

	var text string
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if delta, ok := event.(ai.TextDeltaEvent); ok {
			text += delta.Delta
		}
	}
	if text != "Hello!" {
		t.Fatalf("unexpected streamed text %q", text)
	}
	result := stream.Result()
	if result == nil || result.Output != "Hello!" {
		t.Fatalf("unexpected result %+v", result)
	}
	if result.Usage().OutputTokens != 2 {
		t.Fatalf("unexpected usage %+v", result.Usage())
	}
}

func TestRunStreamWithToolCalls(t *testing.T) {
	model := newStreamingModel(func(msgs []ai.ModelMessage) []ai.StreamEvent {
		if len(msgs) == 1 {
			return []ai.StreamEvent{
				ai.ThinkingDeltaEvent{Delta: "let me check"},
				ai.ToolCallStartEvent{ToolName: "get_weather", ToolCallID: "c1"},
				ai.ToolCallDeltaEvent{ArgsDelta: `{"city":`},
				ai.ToolCallDeltaEvent{ArgsDelta: `"SF"}`},
				ai.FinishEvent{Usage: ai.Usage{Requests: 1}},
			}
		}
		return []ai.StreamEvent{
			ai.TextDeltaEvent{Delta: "Sunny."},
			ai.FinishEvent{Usage: ai.Usage{Requests: 1}},
		}
	})
	agent := ai.NewAgent[deps, string](model)
	var gotCity string
	ai.AddTool(agent, "get_weather", func(_ context.Context, _ *ai.RunContext[deps], args weatherArgs) (string, error) {
		gotCity = args.City
		return "sunny", nil
	})

	stream := agent.RunStream(t.Context(), "weather?", deps{})
	var kinds []string
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		switch event.(type) {
		case ai.ThinkingDeltaEvent:
			kinds = append(kinds, "thinking")
		case ai.ToolCallStartEvent:
			kinds = append(kinds, "tool-start")
		case ai.ToolCallDeltaEvent:
			kinds = append(kinds, "tool-delta")
		case ai.TextDeltaEvent:
			kinds = append(kinds, "text")
		case ai.FinishEvent:
			kinds = append(kinds, "finish")
		}
	}
	if gotCity != "SF" {
		t.Fatalf("tool args not accumulated: %q", gotCity)
	}
	want := []string{"thinking", "tool-start", "tool-delta", "tool-delta", "finish", "text", "finish"}
	if len(kinds) != len(want) {
		t.Fatalf("unexpected events %v", kinds)
	}
	for i := range want {
		if kinds[i] != want[i] {
			t.Fatalf("event %d: expected %s, got %s (%v)", i, want[i], kinds[i], kinds)
		}
	}
	if stream.Result().Output != "Sunny." {
		t.Fatalf("unexpected output %q", stream.Result().Output)
	}
}

func TestRunStreamNonStreamingFallback(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	ai.AddSimpleTool(agent, "noop", func(context.Context, struct{}) (string, error) { return "", nil })

	stream := agent.RunStream(t.Context(), "go", deps{})
	sawToolStart, sawText := false, false
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		switch event.(type) {
		case ai.ToolCallStartEvent:
			sawToolStart = true
		case ai.TextDeltaEvent:
			sawText = true
		}
	}
	if !sawToolStart || !sawText {
		t.Fatalf("fallback replay incomplete: tool=%v text=%v", sawToolStart, sawText)
	}
	if stream.Result().Output != "success" {
		t.Fatalf("unexpected output %q", stream.Result().Output)
	}
}

func TestRunStreamEarlyBreak(t *testing.T) {
	agent := ai.NewAgent[deps, string](fakes.NewTestModel())
	stream := agent.RunStream(t.Context(), "go", deps{})
	for range stream.Events() {
		break
	}
	if stream.Result() != nil {
		t.Fatal("result should be nil after early break")
	}
}

func TestRunStreamModelError(t *testing.T) {
	model := &streamingModel{Model: fakes.NewTestModel(), fail: errors.New("stream down")}
	agent := ai.NewAgent[deps, string](model)
	var got error
	for _, err := range agent.RunStream(t.Context(), "go", deps{}).Events() {
		if err != nil {
			got = err
		}
	}
	if got == nil || got.Error() != "stream down" {
		t.Fatalf("unexpected error %v", got)
	}
}

func TestRunStreamSetupError(t *testing.T) {
	agent := ai.NewAgent[deps, chan int](fakes.NewTestModel())
	var got error
	for _, err := range agent.RunStream(t.Context(), "go", deps{}).Events() {
		if err != nil {
			got = err
		}
	}
	if got == nil {
		t.Fatal("expected setup error for unsupported output type")
	}
}

func TestRunStreamStructuredOutput(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.StreamEvent {
		return []ai.StreamEvent{
			ai.ToolCallStartEvent{ToolName: "final_result", ToolCallID: "c1"},
			ai.ToolCallDeltaEvent{ArgsDelta: `{"city":"SF","temp_c":18}`},
			ai.FinishEvent{Usage: ai.Usage{Requests: 1}},
		}
	})
	agent := ai.NewAgent[deps, weather](model)
	stream := agent.RunStream(t.Context(), "go", deps{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	if stream.Result().Output.City != "SF" {
		t.Fatalf("unexpected output %+v", stream.Result().Output)
	}
}

func TestAccumulateErrors(t *testing.T) {
	// Delta before start.
	model := newStreamingModel(func([]ai.ModelMessage) []ai.StreamEvent {
		return []ai.StreamEvent{ai.ToolCallDeltaEvent{ArgsDelta: "{}"}}
	})
	agent := ai.NewAgent[deps, string](model)
	var got error
	for _, err := range agent.RunStream(t.Context(), "go", deps{}).Events() {
		if err != nil {
			got = err
		}
	}
	var ube *ai.UnexpectedModelBehaviorError
	if !errors.As(got, &ube) {
		t.Fatalf("expected UnexpectedModelBehaviorError, got %v", got)
	}

	// Stream ends without finish.
	model = newStreamingModel(func([]ai.ModelMessage) []ai.StreamEvent {
		return []ai.StreamEvent{ai.TextDeltaEvent{Delta: "hi"}}
	})
	agent = ai.NewAgent[deps, string](model)
	got = nil
	for _, err := range agent.RunStream(t.Context(), "go", deps{}).Events() {
		if err != nil {
			got = err
		}
	}
	if !errors.As(got, &ube) {
		t.Fatalf("expected UnexpectedModelBehaviorError, got %v", got)
	}
}

func TestStreamErrorMidStream(t *testing.T) {
	model := &streamingErrModel{Model: fakes.NewTestModel()}
	agent := ai.NewAgent[deps, string](model)
	var got error
	for _, err := range agent.RunStream(t.Context(), "go", deps{}).Events() {
		if err != nil {
			got = err
		}
	}
	if got == nil || got.Error() != "mid-stream failure" {
		t.Fatalf("unexpected error %v", got)
	}
}

type streamingErrModel struct{ ai.Model }

func (m *streamingErrModel) StreamRequest(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.StreamEvent, error], error) {
	return func(yield func(ai.StreamEvent, error) bool) {
		if !yield(ai.TextDeltaEvent{Delta: "partial"}, nil) {
			return
		}
		yield(nil, errors.New("mid-stream failure"))
	}, nil
}

func TestReplayPreservesRawArgs(t *testing.T) {
	// The fallback path must round-trip tool call args exactly.
	raw := `{"city":"SF"}`
	model := fakes.NewFunctionModel(func(_ context.Context, msgs []ai.ModelMessage, _ ai.ModelRequestParams) (*ai.ModelResponse, error) {
		if len(msgs) == 1 {
			return &ai.ModelResponse{Parts: []ai.ResponsePart{
				ai.ToolCallPart{ToolName: "echo", Args: json.RawMessage(raw), ToolCallID: "c1"},
			}}, nil
		}
		last := msgs[len(msgs)-1].(ai.ModelRequest)
		content := last.Parts[0].(ai.ToolReturnPart).Content.(string)
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: content}}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	ai.AddSimpleTool(agent, "echo", func(_ context.Context, args weatherArgs) (string, error) {
		return args.City, nil
	})
	stream := agent.RunStream(t.Context(), "go", deps{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	if stream.Result().Output != "SF" {
		t.Fatalf("args lost in replay: %q", stream.Result().Output)
	}
}

func TestRunStreamEarlyBreakDuringStreaming(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.StreamEvent {
		return []ai.StreamEvent{
			ai.TextDeltaEvent{Delta: "a"},
			ai.TextDeltaEvent{Delta: "b"},
			ai.FinishEvent{Usage: ai.Usage{Requests: 1}},
		}
	})
	agent := ai.NewAgent[deps, string](model)
	stream := agent.RunStream(t.Context(), "go", deps{})
	count := 0
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		count++
		if count == 1 {
			break
		}
	}
	if stream.Result() != nil {
		t.Fatal("result should be nil after early break")
	}
}

func TestRunStreamFallbackEarlyBreakOnReplay(t *testing.T) {
	// Break while the fallback replays a multi-part response, covering
	// yield-returned-false branches in replayAsEvents.
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ThinkingPart{Content: "hmm"},
			ai.TextPart{Content: "hi"},
			ai.ToolCallPart{ToolName: "t", Args: json.RawMessage(`{}`)},
		}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	for breakAt := 1; breakAt <= 3; breakAt++ {
		stream := agent.RunStream(t.Context(), "go", deps{})
		count := 0
		for _, err := range stream.Events() {
			if err != nil {
				t.Fatal(err)
			}
			count++
			if count == breakAt {
				break
			}
		}
		if stream.Result() != nil {
			t.Fatalf("breakAt %d: result should be nil", breakAt)
		}
	}
}

func TestRunStreamFallbackModelError(t *testing.T) {
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return nil, errors.New("down")
	})
	agent := ai.NewAgent[deps, string](model)
	var got error
	for _, err := range agent.RunStream(t.Context(), "go", deps{}).Events() {
		if err != nil {
			got = err
		}
	}
	if got == nil {
		t.Fatal("expected error")
	}
}

func TestRunStreamFallbackBreakOnToolCallDelta(t *testing.T) {
	// Break on the fourth replayed event: the ToolCallDeltaEvent of a
	// second tool call, covering the delta yield-false branch.
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "a", Args: json.RawMessage(`{}`)},
			ai.ToolCallPart{ToolName: "b", Args: json.RawMessage(`{}`)},
		}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	stream := agent.RunStream(t.Context(), "go", deps{})
	count := 0
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		count++
		if count == 4 {
			break
		}
	}
	if stream.Result() != nil {
		t.Fatal("result should be nil")
	}
}
