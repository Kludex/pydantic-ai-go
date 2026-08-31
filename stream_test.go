package ai_test

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"slices"
	"strings"
	"sync"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

// streamingModel wraps a FunctionModel and streams scripted events.
type streamingModel struct {
	ai.Model
	script func(msgs []ai.ModelMessage) []ai.ModelStreamEvent
	fail   error
}

func (m *streamingModel) StreamRequest(_ context.Context, msgs []ai.ModelMessage, _ ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	if m.fail != nil {
		return nil, m.fail
	}
	events := m.script(msgs)
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		for _, e := range events {
			if !yield(e, nil) {
				return
			}
		}
	}, nil
}

func newStreamingModel(script func(msgs []ai.ModelMessage) []ai.ModelStreamEvent) *streamingModel {
	return &streamingModel{Model: fakes.NewTestModel(), script: script}
}

func TestRunStreamFillsToolCallIDFromDelta(t *testing.T) {
	request := 0
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		request++
		if request == 1 {
			return []ai.ModelStreamEvent{
				ai.ToolCallStartEvent{PartID: "work", ToolName: "work"},
				ai.ToolCallDeltaEvent{PartID: "work", ToolCallID: "final-id", ArgsDelta: `{}`},
				ai.FinishEvent{},
			}
		}
		return []ai.ModelStreamEvent{ai.TextDeltaEvent{Delta: "done"}, ai.FinishEvent{}}
	})
	agent := ai.NewAgent[deps, string](model)
	seenID := ""
	ai.AddTool(agent, "work", func(_ context.Context, rc *ai.RunContext[deps], _ struct{}) (string, error) {
		seenID = rc.ToolCallID
		return "worked", nil
	})
	stream := agent.RunStream(t.Context(), "go", deps{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	if stream.Result() == nil || stream.Result().Output != "done" || seenID != "final-id" {
		t.Fatalf("delta tool call ID was not retained: result=%+v id=%q", stream.Result(), seenID)
	}
}

func TestRunStreamRejectsChangedDeltaToolCallID(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.ToolCallStartEvent{PartID: "work", ToolName: "work", ToolCallID: "first"},
			ai.ToolCallDeltaEvent{PartID: "work", ToolCallID: "second", ArgsDelta: `{}`},
		}
	})
	agent := ai.NewAgent[deps, string](model)
	stream := agent.RunStream(t.Context(), "go", deps{})
	for _, err := range stream.Events() {
		if err != nil {
			if !strings.Contains(err.Error(), "changed") {
				t.Fatalf("unexpected changed ID error: %v", err)
			}
			return
		}
	}
	t.Fatal("expected changed tool call ID error")
}

func TestRunStreamPreservesPartProviderMetadata(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.ThinkingDeltaEvent{
				PartID: "thinking", Delta: "plan", ID: "reasoning-1", ProviderName: "provider",
				ProviderDetails: map[string]any{"first": true},
			},
			ai.ThinkingDeltaEvent{
				PartID: "thinking", SignatureDelta: "signature", ProviderName: "provider",
				ProviderDetails: map[string]any{"second": true},
			},
			ai.TextDeltaEvent{
				PartID: "text", Delta: "do", ID: "message-1", ProviderName: "provider",
				ProviderDetails: map[string]any{"first": true},
			},
			ai.TextDeltaEvent{
				PartID: "text", Delta: "ne", ProviderName: "provider",
				ProviderDetails: map[string]any{"second": true},
			},
			ai.FinishEvent{ProviderDetails: map[string]any{"finish": true}},
		}
	})
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		switch event := event.(type) {
		case ai.PartStartEvent:
			switch part := event.Part.(type) {
			case ai.TextPart:
				part.ProviderDetails["consumer"] = true
			case ai.ThinkingPart:
				part.ProviderDetails["consumer"] = true
			}
		case ai.FinishEvent:
			event.ProviderDetails["consumer"] = true
		}
	}
	response := stream.Result().Messages()[1].(ai.ModelResponse)
	thinking := response.Parts[0].(ai.ThinkingPart)
	text := response.Parts[1].(ai.TextPart)
	if thinking.Content != "plan" || thinking.ID != "reasoning-1" || thinking.Signature != "signature" ||
		thinking.ProviderName != "provider" || thinking.ProviderDetails["first"] != true ||
		thinking.ProviderDetails["second"] != true || thinking.ProviderDetails["consumer"] != nil {
		t.Fatalf("unexpected thinking metadata: %+v", thinking)
	}
	if text.Content != "done" || text.ID != "message-1" || text.ProviderName != "provider" ||
		text.ProviderDetails["first"] != true || text.ProviderDetails["second"] != true ||
		text.ProviderDetails["consumer"] != nil || response.ProviderDetails["consumer"] != nil {
		t.Fatalf("unexpected text/response metadata: text=%+v response=%+v", text, response)
	}
}

func TestRunStreamTextDeltas(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.TextDeltaEvent{Delta: "Hel"},
			ai.TextDeltaEvent{Delta: "lo!"},
			ai.FinishEvent{Usage: ai.Usage{Requests: 1, OutputTokens: 2}, ModelName: "scripted"},
		}
	})
	agent := ai.NewAgent[deps, string](model)
	stream := agent.RunStream(t.Context(), "hi", deps{})

	var text string
	var finalResults int
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		text += streamText(event)
		if _, ok := event.(ai.FinalResultEvent); ok {
			finalResults++
		}
	}
	if text != "Hello!" || finalResults != 1 {
		t.Fatalf("unexpected streamed text %q or final-result count %d", text, finalResults)
	}
	result := stream.Result()
	if result == nil || result.Output != "Hello!" {
		t.Fatalf("unexpected result %+v", result)
	}
	if result.Usage().OutputTokens != 2 {
		t.Fatalf("unexpected usage %+v", result.Usage())
	}
}

func TestRunStreamPartLifecycle(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.TextDeltaEvent{PartID: "answer", Delta: "a"},
			ai.TextDeltaEvent{PartID: "answer", Delta: "b"},
			ai.ThinkingDeltaEvent{PartID: "reasoning", Delta: "why"},
			ai.FinishEvent{},
		}
	})
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	var events []ai.StreamEvent
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, event)
	}
	if len(events) != 7 {
		t.Fatalf("unexpected lifecycle events: %v", events)
	}
	textStart := events[0].(ai.PartStartEvent)
	if textStart.Index != 0 || textStart.PartID != "answer" || textStart.PreviousPartKind != "" ||
		textStart.Part.(ai.TextPart).Content != "a" {
		t.Fatalf("unexpected text start: %+v", textStart)
	}
	if _, ok := events[1].(ai.FinalResultEvent); !ok {
		t.Fatalf("final result did not follow matching start: %T", events[1])
	}
	textDelta := events[2].(ai.PartDeltaEvent)
	if textDelta.Index != 0 || textDelta.PartID != "answer" ||
		textDelta.Delta.(ai.TextPartDelta).ContentDelta != "b" {
		t.Fatalf("unexpected text delta: %+v", textDelta)
	}
	textEnd := events[3].(ai.PartEndEvent)
	if textEnd.NextPartKind != ai.ResponsePartKindThinking || textEnd.Part.(ai.TextPart).Content != "ab" {
		t.Fatalf("unexpected text end: %+v", textEnd)
	}
	thinkingStart := events[4].(ai.PartStartEvent)
	if thinkingStart.Index != 1 || thinkingStart.PreviousPartKind != ai.ResponsePartKindText {
		t.Fatalf("unexpected thinking start: %+v", thinkingStart)
	}
	thinkingEnd := events[5].(ai.PartEndEvent)
	if thinkingEnd.NextPartKind != "" || thinkingEnd.Part.(ai.ThinkingPart).Content != "why" {
		t.Fatalf("unexpected thinking end: %+v", thinkingEnd)
	}
	if _, ok := events[6].(ai.FinishEvent); !ok {
		t.Fatalf("finish event missing: %T", events[6])
	}
	if stream.Result().Output != "ab" {
		t.Fatalf("unexpected output %q", stream.Result().Output)
	}
}

func TestResponsePartDeltasApply(t *testing.T) {
	text, err := (ai.TextPartDelta{ContentDelta: "b", ProviderName: "provider"}).Apply(ai.TextPart{Content: "a"})
	if err != nil || text.(ai.TextPart).Content != "ab" || text.(ai.TextPart).ProviderName != "provider" {
		t.Fatalf("unexpected text delta result=%v err=%v", text, err)
	}
	thinking, err := (ai.ThinkingPartDelta{
		ContentDelta: "b", SignatureDelta: "signature", ProviderName: "provider",
	}).Apply(ai.ThinkingPart{Content: "a"})
	if err != nil || thinking.(ai.ThinkingPart).Content != "ab" ||
		thinking.(ai.ThinkingPart).Signature != "signature" || thinking.(ai.ThinkingPart).ProviderName != "provider" {
		t.Fatalf("unexpected thinking delta result=%v err=%v", thinking, err)
	}
	call, err := (ai.ToolCallPartDelta{
		ToolNameDelta: "ther", ArgsDelta: "1}", ToolCallID: "call", ProviderName: "provider",
	}).Apply(ai.ToolCallPart{ToolName: "wea", Args: json.RawMessage(`{"x":`), ToolCallID: "call"})
	if err != nil {
		t.Fatal(err)
	}
	toolCall := call.(ai.ToolCallPart)
	if toolCall.ToolName != "weather" || string(toolCall.Args) != `{"x":1}` || toolCall.ToolCallID != "call" ||
		toolCall.ProviderName != "provider" {
		t.Fatalf("unexpected tool-call delta result: %+v", toolCall)
	}
	call, err = (ai.ToolCallPartDelta{ToolCallID: "assigned"}).Apply(ai.ToolCallPart{})
	if err != nil || call.(ai.ToolCallPart).ToolCallID != "assigned" {
		t.Fatalf("tool-call ID was not assigned: result=%v err=%v", call, err)
	}

	for name, apply := range map[string]func() error{
		"text mismatch": func() error {
			_, err := (ai.TextPartDelta{}).Apply(ai.ThinkingPart{})
			return err
		},
		"thinking mismatch": func() error {
			_, err := (ai.ThinkingPartDelta{}).Apply(ai.TextPart{})
			return err
		},
		"tool mismatch": func() error {
			_, err := (ai.ToolCallPartDelta{}).Apply(ai.TextPart{})
			return err
		},
		"tool ID change": func() error {
			_, err := (ai.ToolCallPartDelta{ToolCallID: "new"}).Apply(ai.ToolCallPart{ToolCallID: "old"})
			return err
		},
	} {
		t.Run(name, func(t *testing.T) {
			if err := apply(); err == nil {
				t.Fatal("expected incompatible delta error")
			}
		})
	}
}

func TestRunStreamWithToolCalls(t *testing.T) {
	model := newStreamingModel(func(msgs []ai.ModelMessage) []ai.ModelStreamEvent {
		if len(msgs) == 1 {
			return []ai.ModelStreamEvent{
				ai.ThinkingDeltaEvent{Delta: "let me check"},
				ai.ToolCallStartEvent{ToolName: "get_weather", ToolCallID: "c1"},
				ai.ToolCallDeltaEvent{ArgsDelta: `{"city":`},
				ai.ToolCallDeltaEvent{ArgsDelta: `"SF"}`},
				ai.FinishEvent{Usage: ai.Usage{Requests: 1}},
			}
		}
		return []ai.ModelStreamEvent{
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
		switch event := event.(type) {
		case ai.PartStartEvent:
			kinds = append(kinds, "start:"+string(responsePartKind(event.Part)))
		case ai.PartDeltaEvent:
			kinds = append(kinds, "delta:"+string(deltaKind(event.Delta)))
		case ai.PartEndEvent:
			kinds = append(kinds, "end:"+string(responsePartKind(event.Part)))
		case ai.FinalResultEvent:
			kinds = append(kinds, "final")
		case ai.FinishEvent:
			kinds = append(kinds, "finish")
		}
	}
	if gotCity != "SF" {
		t.Fatalf("tool args not accumulated: %q", gotCity)
	}
	want := []string{
		"start:thinking", "end:thinking", "start:tool-call", "delta:tool-call", "delta:tool-call",
		"end:tool-call", "finish", "start:text", "final", "end:text", "finish",
	}
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
		if start, ok := event.(ai.PartStartEvent); ok {
			switch start.Part.(type) {
			case ai.ToolCallPart:
				sawToolStart = true
			case ai.TextPart:
				sawText = true
			}
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
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.ToolCallStartEvent{ToolName: "final_result", ToolCallID: "c1"},
			ai.ToolCallDeltaEvent{ArgsDelta: `{"city":"SF","temp_c":18}`},
			ai.FinishEvent{Usage: ai.Usage{Requests: 1}},
		}
	})
	agent := ai.NewAgent[deps, weather](model)
	stream := agent.RunStream(t.Context(), "go", deps{})
	var final ai.FinalResultEvent
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if event, ok := event.(ai.FinalResultEvent); ok {
			final = event
		}
	}
	if final.ToolName != "final_result" || final.ToolCallID != "c1" {
		t.Fatalf("unexpected final-result event: %+v", final)
	}
	if stream.Result().Output.City != "SF" {
		t.Fatalf("unexpected output %+v", stream.Result().Output)
	}
}

func TestRunStreamInterleavesKeyedToolCalls(t *testing.T) {
	model := newStreamingModel(func(msgs []ai.ModelMessage) []ai.ModelStreamEvent {
		if len(msgs) == 1 {
			return []ai.ModelStreamEvent{
				ai.ToolCallStartEvent{PartID: "first", ToolName: "lookup", ToolCallID: "a"},
				ai.ToolCallStartEvent{PartID: "second", ToolName: "lookup", ToolCallID: "b"},
				ai.ToolCallDeltaEvent{PartID: "first", ArgsDelta: `{"city":"S`},
				ai.ToolCallDeltaEvent{PartID: "second", ArgsDelta: `{"city":"N`},
				ai.ToolCallDeltaEvent{PartID: "first", ArgsDelta: `F"}`},
				ai.ToolCallDeltaEvent{PartID: "second", ArgsDelta: `Y"}`},
				ai.FinishEvent{Usage: ai.Usage{Requests: 1}},
			}
		}
		return []ai.ModelStreamEvent{ai.TextDeltaEvent{PartID: "answer", Delta: "done"}, ai.FinishEvent{}}
	})
	agent := ai.NewAgent[deps, string](model)
	var mutex sync.Mutex
	var cities []string
	ai.AddSimpleTool(agent, "lookup", func(_ context.Context, args weatherArgs) (string, error) {
		mutex.Lock()
		defer mutex.Unlock()
		cities = append(cities, args.City)
		return args.City, nil
	})
	stream := agent.RunStream(t.Context(), "go", deps{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	slices.Sort(cities)
	if !slices.Equal(cities, []string{"NY", "SF"}) {
		t.Fatalf("interleaved arguments were routed incorrectly: %v", cities)
	}
	response := stream.Result().Messages()[1].(ai.ModelResponse)
	calls := response.ToolCalls()
	if calls[0].ToolCallID != "a" || calls[1].ToolCallID != "b" {
		t.Fatalf("first-seen part order was not preserved: %v", calls)
	}
}

func TestRunStreamInterleavesKeyedTextAndThinking(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.TextDeltaEvent{PartID: "answer", Delta: "A"},
			ai.ThinkingDeltaEvent{PartID: "reasoning", Delta: "why"},
			ai.TextDeltaEvent{PartID: "answer", Delta: "B"},
			ai.FinishEvent{},
		}
	})
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	var IDs []string
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		switch event := event.(type) {
		case ai.PartStartEvent:
			IDs = append(IDs, event.PartID)
		case ai.PartDeltaEvent:
			IDs = append(IDs, event.PartID)
		}
	}
	if !slices.Equal(IDs, []string{"answer", "reasoning", "answer"}) {
		t.Fatalf("part IDs changed: %v", IDs)
	}
	result := stream.Result()
	if result.Output != "AB" {
		t.Fatalf("unexpected interleaved output %q", result.Output)
	}
	parts := result.Messages()[1].(ai.ModelResponse).Parts
	if parts[0].(ai.TextPart).Content != "AB" || parts[1].(ai.ThinkingPart).Content != "why" {
		t.Fatalf("unexpected materialized parts: %v", parts)
	}
}

func TestRunStreamRejectsInvalidPartIDs(t *testing.T) {
	tests := map[string][]ai.ModelStreamEvent{
		"text to thinking": {
			ai.TextDeltaEvent{PartID: "same", Delta: "text"},
			ai.ThinkingDeltaEvent{PartID: "same", Delta: "thinking"},
		},
		"thinking to text": {
			ai.ThinkingDeltaEvent{PartID: "same", Delta: "thinking"},
			ai.TextDeltaEvent{PartID: "same", Delta: "text"},
		},
		"duplicate tool start": {
			ai.ToolCallStartEvent{PartID: "same", ToolName: "one"},
			ai.ToolCallStartEvent{PartID: "same", ToolName: "two"},
		},
		"unknown tool delta": {ai.ToolCallDeltaEvent{PartID: "missing", ArgsDelta: `{}`}},
	}
	for name, events := range tests {
		t.Run(name, func(t *testing.T) {
			model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent { return events })
			stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
			var got error
			for _, err := range stream.Events() {
				if err != nil {
					got = err
				}
			}
			var unexpected *ai.UnexpectedModelBehaviorError
			if !errors.As(got, &unexpected) {
				t.Fatalf("expected UnexpectedModelBehaviorError, got %v", got)
			}
		})
	}
}

func TestAccumulateErrors(t *testing.T) {
	// Delta before start.
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{ai.ToolCallDeltaEvent{ArgsDelta: "{}"}}
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
	model = newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{ai.TextDeltaEvent{Delta: "hi"}}
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

func streamText(event ai.StreamEvent) string {
	switch event := event.(type) {
	case ai.PartStartEvent:
		if part, ok := event.Part.(ai.TextPart); ok {
			return part.Content
		}
	case ai.PartDeltaEvent:
		if delta, ok := event.Delta.(ai.TextPartDelta); ok {
			return delta.ContentDelta
		}
	}
	return ""
}

func responsePartKind(part ai.ResponsePart) ai.ResponsePartKind {
	switch part.(type) {
	case ai.TextPart:
		return ai.ResponsePartKindText
	case ai.ThinkingPart:
		return ai.ResponsePartKindThinking
	case ai.ToolCallPart:
		return ai.ResponsePartKindToolCall
	default:
		panic("unknown response part")
	}
}

func deltaKind(delta ai.ResponsePartDelta) ai.ResponsePartKind {
	switch delta.(type) {
	case ai.TextPartDelta:
		return ai.ResponsePartKindText
	case ai.ThinkingPartDelta:
		return ai.ResponsePartKindThinking
	case ai.ToolCallPartDelta:
		return ai.ResponsePartKindToolCall
	default:
		panic("unknown response delta")
	}
}

type streamingErrModel struct{ ai.Model }

func (m *streamingErrModel) StreamRequest(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
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
	tests := map[string]struct {
		events     []ai.ModelStreamEvent
		eventCount int
	}{
		"text": {
			events: []ai.ModelStreamEvent{
				ai.TextDeltaEvent{Delta: "a"}, ai.TextDeltaEvent{Delta: "b"},
				ai.FinishEvent{Usage: ai.Usage{Requests: 1}},
			},
			eventCount: 5,
		},
		"thinking": {
			events: []ai.ModelStreamEvent{
				ai.ThinkingDeltaEvent{Delta: "a"}, ai.ThinkingDeltaEvent{Delta: "b"}, ai.FinishEvent{},
			},
			eventCount: 4,
		},
		"tool": {
			events: []ai.ModelStreamEvent{
				ai.ToolCallStartEvent{ToolName: "work"}, ai.ToolCallDeltaEvent{ArgsDelta: `{}`}, ai.FinishEvent{},
			},
			eventCount: 4,
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			for breakAt := 1; breakAt <= test.eventCount; breakAt++ {
				model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent { return test.events })
				stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
				count := 0
				for range stream.Events() {
					count++
					if count == breakAt {
						break
					}
				}
				if stream.Result() != nil {
					t.Fatalf("breakAt %d: result should be nil after early break", breakAt)
				}
			}
		})
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

func TestRunStreamFallbackBreakOnToolCallEvents(t *testing.T) {
	// Break on the normalized start and delta so replayAsEvents observes
	// each provider-facing tool event being rejected.
	model := fakes.NewFunctionModel(func(context.Context, []ai.ModelMessage, ai.ModelRequestParams) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "a", Args: json.RawMessage(`{}`)},
			ai.ToolCallPart{ToolName: "b", Args: json.RawMessage(`{}`)},
		}}, nil
	})
	agent := ai.NewAgent[deps, string](model)
	for breakAt := 1; breakAt <= 2; breakAt++ {
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
