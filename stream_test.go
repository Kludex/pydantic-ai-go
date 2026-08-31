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
	script      func(msgs []ai.ModelMessage) []ai.ModelStreamEvent
	fail        error
	cancelCount int
}

func (m *streamingModel) CancelSuspendedResponse(context.Context, ai.ModelResponse) error {
	m.cancelCount++
	return nil
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

func TestRunStreamDetachPreservesResumableSuspendedSnapshot(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.ResponseMetadataEvent{
				Usage: ai.Usage{Requests: 1, InputTokens: 3}, ModelName: "background",
				ProviderName: "provider", ProviderResponseID: "job",
				ProviderDetails: map[string]any{"background": true}, State: ai.ModelResponseStateSuspended,
			},
			ai.TextDeltaEvent{PartID: "message", Delta: "partial", ProviderName: "provider"},
			ai.FinishEvent{State: ai.ModelResponseStateSuspended},
		}
	})
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	if stream.Suspended() != nil {
		t.Fatal("stream was suspended before it started")
	}
	for range stream.Events() {
		break
	}
	if stream.Result() != nil || model.cancelCount != 0 {
		t.Fatalf("detached stream completed or canceled its job: result=%+v cancels=%d", stream.Result(), model.cancelCount)
	}
	suspended := stream.Suspended()
	if suspended == nil || suspended.Response().State != ai.ModelResponseStateSuspended ||
		suspended.Response().ProviderResponseID != "job" || suspended.Response().Text() != "partial" ||
		suspended.Usage().Requests != 1 || suspended.Usage().InputTokens != 3 {
		t.Fatalf("unexpected suspended snapshot: %+v", suspended)
	}
	messages := suspended.Messages()
	if len(messages) != 2 || messages[1].(ai.ModelResponse).State != ai.ModelResponseStateSuspended {
		t.Fatalf("unexpected suspended history: %+v", messages)
	}
	messages[1] = ai.ModelRequest{}
	if _, ok := stream.Suspended().Messages()[1].(ai.ModelResponse); !ok {
		t.Fatal("suspended history was not detached")
	}

	resumeModel := &continuationModel{responses: []*ai.ModelResponse{{
		Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}, Usage: ai.Usage{Requests: 1, OutputTokens: 1},
		ModelName: "background", ProviderResponseID: "job", State: ai.ModelResponseStateComplete,
	}}}
	result, err := ai.NewAgent[deps, string](resumeModel).Resume(t.Context(), suspended.Messages(), deps{})
	if err != nil || result.Output != "done" || len(result.NewMessages()) != 1 {
		t.Fatalf("detached snapshot did not resume: result=%+v err=%v", result, err)
	}
}

func TestRunStreamUsesAuthoritativeFinishParts(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.TextDeltaEvent{Delta: "suffix"},
			ai.FinishEvent{Parts: []ai.ResponsePart{ai.TextPart{Content: "full output"}}},
		}
	})
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	if result := stream.Result(); result == nil || result.Output != "full output" {
		t.Fatalf("finish snapshot was not authoritative: %+v", result)
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
	textDetails := map[string]any{"phase": "final_answer"}
	text, err := (ai.TextPartDelta{
		ContentDelta: "b", ProviderName: "provider", ProviderDetails: textDetails,
	}).Apply(ai.TextPart{Content: "a", ProviderDetails: map[string]any{"old": true}})
	textDetails["phase"] = "changed"
	textPart := text.(ai.TextPart)
	if err != nil || textPart.Content != "ab" || textPart.ProviderName != "provider" ||
		textPart.ProviderDetails["old"] != true || textPart.ProviderDetails["phase"] != "final_answer" {
		t.Fatalf("unexpected text delta result=%v err=%v", text, err)
	}
	thinkingDetails := map[string]any{"encrypted": true}
	thinking, err := (ai.ThinkingPartDelta{
		ContentDelta: "b", SignatureDelta: "signature", ProviderName: "provider",
		ProviderDetails: thinkingDetails,
	}).Apply(ai.ThinkingPart{Content: "a"})
	thinkingDetails["encrypted"] = false
	thinkingPart := thinking.(ai.ThinkingPart)
	if err != nil || thinkingPart.Content != "ab" || thinkingPart.Signature != "signature" ||
		thinkingPart.ProviderName != "provider" || thinkingPart.ProviderDetails["encrypted"] != true {
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

func TestRunStreamPreservesProviderNativeToolLifecycle(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.ToolCallStartEvent{
				PartID: "search", ToolName: ai.ToolSearchName, ToolCallID: "search-1",
				ToolKind: ai.ToolPartKindToolSearch, ProviderName: "provider", Native: true,
			},
			ai.ToolCallDeltaEvent{PartID: "search", ArgsDelta: `{"queries":["weather"]}`},
			ai.NativeToolReturnEvent{PartID: "search-result", Part: ai.NativeToolReturnPart{
				ToolName: ai.ToolSearchName, ToolCallID: "search-1", ToolKind: ai.ToolPartKindToolSearch,
				Content:      ai.ToolSearchResult{DiscoveredTools: []ai.ToolSearchMatch{{Name: "weather"}}},
				ProviderName: "provider",
			}},
			ai.FileEvent{PartID: "file", Part: ai.FilePart{
				Content: ai.BinaryContent{Data: []byte("image"), MediaType: "image/png"}, ProviderName: "provider",
			}},
			ai.TextDeltaEvent{PartID: "answer", Delta: "done"},
			ai.FinishEvent{},
		}
	})
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	var starts []ai.PartStartEvent
	var deltas []ai.PartDeltaEvent
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		switch event := event.(type) {
		case ai.PartStartEvent:
			starts = append(starts, event)
		case ai.PartDeltaEvent:
			deltas = append(deltas, event)
		}
	}
	result := stream.Result()
	if result == nil || result.Output != "done" || len(starts) != 4 || len(deltas) != 1 {
		t.Fatalf("unexpected native stream lifecycle: result=%+v starts=%+v deltas=%+v", result, starts, deltas)
	}
	messages := result.Messages()
	response := messages[len(messages)-1].(ai.ModelResponse)
	call := response.Parts[0].(ai.NativeToolCallPart)
	returned := response.Parts[1].(ai.NativeToolReturnPart)
	file := response.Parts[2].(ai.FilePart)
	if string(call.Args) != `{"queries":["weather"]}` || call.ToolCallID != returned.ToolCallID ||
		returned.ProviderName != "provider" || string(file.Content.Data) != "image" {
		t.Fatalf("unexpected native stream parts: %+v", response.Parts)
	}
	if _, ok := deltas[0].Delta.(ai.NativeToolCallPartDelta); !ok {
		t.Fatalf("unexpected native delta type %T", deltas[0].Delta)
	}
}

func TestFilePartDeltaReplacesGeneratedFile(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.FileEvent{PartID: "file", Part: ai.FilePart{
				Content: ai.BinaryContent{Data: []byte("partial"), MediaType: "image/png"},
			}},
			ai.FileEvent{PartID: "file", Replace: true, Part: ai.FilePart{
				Content: ai.BinaryContent{Data: []byte("final"), MediaType: "image/webp"},
			}},
			ai.TextDeltaEvent{PartID: "answer", Delta: "done"},
			ai.FinishEvent{},
		}
	})
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	var delta ai.FilePartDelta
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if partDelta, ok := event.(ai.PartDeltaEvent); ok {
			delta, _ = partDelta.Delta.(ai.FilePartDelta)
		}
	}
	result := stream.Result()
	file := result.Messages()[1].(ai.ModelResponse).Parts[0].(ai.FilePart)
	if string(file.Content.Data) != "final" || file.Content.MediaType != "image/webp" ||
		string(delta.Part.Content.Data) != "final" {
		t.Fatalf("file was not replaced: file=%+v delta=%+v", file, delta)
	}
	applied, err := delta.Apply(ai.FilePart{})
	if err != nil {
		t.Fatal(err)
	}
	delta.Part.Content.Data[0] = 'X'
	if string(applied.(ai.FilePart).Content.Data) != "final" {
		t.Fatal("applied file delta was not detached")
	}
	if _, err := delta.Apply(ai.TextPart{}); err == nil {
		t.Fatal("expected file delta type error")
	}
}

func TestNativeToolCallPartDeltaApply(t *testing.T) {
	delta := ai.NativeToolCallPartDelta{
		ToolNameDelta: "_search", ArgsDelta: `{"queries":[]}`, ToolCallID: "call", ProviderName: "provider",
	}
	part, err := delta.Apply(ai.NativeToolCallPart{ToolName: "tool", ToolCallID: "call"})
	if err != nil {
		t.Fatal(err)
	}
	call := part.(ai.NativeToolCallPart)
	if call.ToolName != "tool_search" || string(call.Args) != `{"queries":[]}` || call.ProviderName != "provider" {
		t.Fatalf("unexpected applied native delta: %+v", call)
	}
	if _, err := delta.Apply(ai.TextPart{}); err == nil {
		t.Fatal("expected native delta type error")
	}
	if _, err := (ai.NativeToolCallPartDelta{ToolCallID: "changed"}).Apply(call); err == nil ||
		!strings.Contains(err.Error(), `tool call ID changed from "call" to "changed"`) {
		t.Fatalf("unexpected native call ID error: %v", err)
	}
}

func TestRunStreamFallbackReplaysNativeToolParts(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.NativeToolCallPart{
				ToolName: ai.ToolSearchName, ToolCallID: "search", ToolKind: ai.ToolPartKindToolSearch,
				Args: []byte(`{"queries":["x"]}`), ProviderName: "provider",
			},
			ai.NativeToolReturnPart{
				ToolName: ai.ToolSearchName, ToolCallID: "search", ToolKind: ai.ToolPartKindToolSearch,
				Content: ai.ToolSearchResult{DiscoveredTools: []ai.ToolSearchMatch{}}, ProviderName: "provider",
			},
			ai.FilePart{Content: ai.BinaryContent{Data: []byte("file"), MediaType: "text/plain"}},
			ai.TextPart{Content: "done"},
		}}, nil
	})
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
	}
	messages := stream.Result().Messages()
	parts := messages[len(messages)-1].(ai.ModelResponse).Parts
	if len(parts) != 4 {
		t.Fatalf("native fallback parts were lost: %+v", parts)
	}
	if _, ok := parts[0].(ai.NativeToolCallPart); !ok {
		t.Fatalf("unexpected native fallback call %T", parts[0])
	}
	if _, ok := parts[1].(ai.NativeToolReturnPart); !ok {
		t.Fatalf("unexpected native fallback return %T", parts[1])
	}
}

func TestRunStreamCanStopFilePart(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.FileEvent{PartID: "file", Part: ai.FilePart{
				Content: ai.BinaryContent{Data: []byte("file"), MediaType: "text/plain"},
			}},
			ai.FinishEvent{},
		}
	})
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	for _, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		break
	}
	if stream.Result() != nil {
		t.Fatal("stopped file stream produced a result")
	}
}

func TestRunStreamCanStopFileReplacement(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		part := ai.FilePart{Content: ai.BinaryContent{Data: []byte("file"), MediaType: "text/plain"}}
		return []ai.ModelStreamEvent{
			ai.FileEvent{PartID: "file", Part: part},
			ai.FileEvent{PartID: "file", Part: part, Replace: true},
			ai.FinishEvent{},
		}
	})
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := event.(ai.PartDeltaEvent); ok {
			break
		}
	}
	if stream.Result() != nil {
		t.Fatal("stopped file replacement stream produced a result")
	}
}

func TestRunStreamRejectsDuplicateFilePartID(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		part := ai.FilePart{Content: ai.BinaryContent{Data: []byte("file"), MediaType: "text/plain"}}
		return []ai.ModelStreamEvent{
			ai.FileEvent{PartID: "file", Part: part},
			ai.FileEvent{PartID: "file", Part: part},
		}
	})
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	for _, err := range stream.Events() {
		if err == nil {
			continue
		}
		if !strings.Contains(err.Error(), `duplicate file stream part "file"`) {
			t.Fatalf("unexpected duplicate file error: %v", err)
		}
		return
	}
	t.Fatal("expected duplicate file error")
}

func TestRunStreamRejectsDuplicateNativeToolReturnPartID(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		part := ai.NativeToolReturnPart{ToolName: "native", Content: map[string]any{}}
		return []ai.ModelStreamEvent{
			ai.NativeToolReturnEvent{PartID: "result", Part: part},
			ai.NativeToolReturnEvent{PartID: "result", Part: part},
		}
	})
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	for _, err := range stream.Events() {
		if err == nil {
			continue
		}
		if !strings.Contains(err.Error(), `duplicate native tool return stream part "result"`) {
			t.Fatalf("unexpected duplicate native return error: %v", err)
		}
		return
	}
	t.Fatal("expected duplicate native return error")
}

func TestRunStreamCanStopNativeFallbackLifecycle(t *testing.T) {
	for name, stopAfter := range map[string]int{"call start": 1, "call delta": 2, "return start": 4} {
		t.Run(name, func(t *testing.T) {
			model := fakes.NewFunctionModel(func(
				context.Context, []ai.ModelMessage, ai.ModelRequestParams,
			) (*ai.ModelResponse, error) {
				return &ai.ModelResponse{Parts: []ai.ResponsePart{
					ai.NativeToolCallPart{ToolName: "native", ToolCallID: "call", Args: []byte(`{}`)},
					ai.NativeToolReturnPart{ToolName: "native", ToolCallID: "call", Content: "done"},
					ai.TextPart{Content: "unused"},
				}}, nil
			})
			stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
			seen := 0
			for range stream.Events() {
				seen++
				if seen == stopAfter {
					break
				}
			}
			if stream.Result() != nil {
				t.Fatalf("stopped native fallback unexpectedly completed: %+v", stream.Result())
			}
		})
	}
}

func TestStreamedRunUsageIsLiveAndDetached(t *testing.T) {
	model := newStreamingModel(func([]ai.ModelMessage) []ai.ModelStreamEvent {
		return []ai.ModelStreamEvent{
			ai.ResponseMetadataEvent{
				Usage:     ai.Usage{Requests: 1, InputTokens: 5, Details: map[string]int{"provider": 1}},
				ModelName: "gpt-5", ProviderName: "openai",
			},
			ai.TextDeltaEvent{Delta: "done"},
			ai.FinishEvent{
				Usage:     ai.Usage{Requests: 1, InputTokens: 5, OutputTokens: 2, Details: map[string]int{"provider": 1}},
				ModelName: "gpt-5", ProviderName: "openai",
			},
		}
	})
	stream := ai.NewAgent[deps, string](model).RunStream(t.Context(), "go", deps{})
	if !stream.Usage().IsZero() {
		t.Fatalf("usage was nonzero before stream consumption: %+v", stream.Usage())
	}
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		usage := stream.Usage()
		if usage.InputTokens != 5 || usage.CostUSD == nil || *usage.CostUSD <= 0 {
			t.Fatalf("missing live usage for %T: %+v", event, usage)
		}
		usage.Details["provider"] = 99
		if stream.Usage().Details["provider"] != 1 {
			t.Fatalf("live usage was not detached: %+v", stream.Usage())
		}
		if _, ok := event.(ai.FinishEvent); ok && usage.OutputTokens != 2 {
			t.Fatalf("finish usage was not visible while handling finish: %+v", usage)
		}
	}
	if stream.Result() == nil || stream.Usage().OutputTokens != 2 ||
		*stream.Usage().CostUSD != *stream.Result().Usage().CostUSD {
		t.Fatalf("terminal live usage differs from result: live=%+v result=%+v", stream.Usage(), stream.Result())
	}
}
