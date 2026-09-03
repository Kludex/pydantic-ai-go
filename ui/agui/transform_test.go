package agui_test

import (
	"errors"
	"iter"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/ui/agui"
)

func TestTransformStreamEventVariants(t *testing.T) {
	valid := true
	stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		events := []ai.StreamEvent{
			ai.PartStartEvent{Index: 0, PartID: "text", Part: ai.TextPart{Content: "a"}},
			ai.PartDeltaEvent{Index: 0, PartID: "text", Delta: ai.TextPartDelta{ContentDelta: "b"}},
			ai.PartStartEvent{Index: 1, PartID: "call", Part: ai.ToolCallPart{
				ToolName: "one", ToolCallID: "call-1", Args: []byte(`{"x":`),
			}},
			ai.PartStartEvent{Index: 1, PartID: "call", Part: ai.ToolCallPart{
				ToolName: "one", ToolCallID: "call-1",
			}},
			ai.PartDeltaEvent{Index: 1, PartID: "call", Delta: ai.ToolCallPartDelta{ArgsDelta: "1}"}},
			ai.FunctionToolCallEvent{Part: ai.ToolCallPart{ToolName: "one", ToolCallID: "call-1"}, ArgsValid: &valid},
			ai.FunctionToolCallEvent{Part: ai.ToolCallPart{ToolName: "one", ToolCallID: "call-1"}, ArgsValid: &valid},
			ai.PartStartEvent{Index: 1, PartID: "call", Part: ai.ToolCallPart{
				ToolName: "one", ToolCallID: "call-1",
			}},
			ai.FunctionToolResultEvent{Part: ai.ToolReturnPart{ToolCallID: "call-1", Content: map[string]any{"ok": true}}},
			ai.FunctionToolResultEvent{Part: ai.ToolReturnPart{ToolCallID: "call-1", Content: "done"}},
			ai.OutputToolCallEvent{Part: ai.ToolCallPart{ToolName: "final", ToolCallID: "call-2", Args: []byte(`{}`)}},
			ai.OutputToolResultEvent{Part: ai.RetryPromptPart{ToolCallID: "call-2", Content: "retry"}},
			ai.PartStartEvent{Index: 2, PartID: "native", Part: ai.NativeToolCallPart{
				ToolName: "search", ToolCallID: "native-1", Args: []byte(`{}`),
			}},
			ai.PartDeltaEvent{Index: 2, PartID: "native", Delta: ai.NativeToolCallPartDelta(
				ai.ToolCallPartDelta{ArgsDelta: " "},
			)},
			ai.PartStartEvent{Index: 3, PartID: "return", Part: ai.NativeToolReturnPart{
				ToolCallID: "native-1", Content: map[string]any{"result": "found"},
			}},
			ai.PartStartEvent{Index: 4, PartID: "return-text", Part: ai.NativeToolReturnPart{
				ToolCallID: "native-2", Content: "found",
			}},
			ai.PartStartEvent{Index: 4, PartID: "ignored", Part: ai.FilePart{}},
			ai.FinishEvent{},
		}
		for _, event := range events {
			if !yield(event, nil) {
				return
			}
		}
	})
	var events []agui.Event
	for event, err := range agui.TransformStream(stream, "thread", "run") {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, event)
	}
	if events[0].Type != agui.EventRunStarted || events[len(events)-1].Type != agui.EventRunFinished {
		t.Fatalf("unexpected lifecycle: %#v", events)
	}
	if eventIndex(events, agui.EventToolCallArgs) < 0 || eventIndex(events, agui.EventToolCallResult) < 0 ||
		eventIndex(events, agui.EventTextMessageEnd) < 0 {
		t.Fatalf("missing transformed events: %#v", events)
	}
}

func TestTransformStreamErrors(t *testing.T) {
	tests := []struct {
		name   string
		events []ai.StreamEvent
		err    error
		match  string
	}{
		{name: "stream error", events: []ai.StreamEvent{
			ai.PartStartEvent{PartID: "text", Part: ai.TextPart{Content: "open"}},
		}, err: errors.New("stream failed"), match: "stream failed"},
		{name: "unsupported result", events: []ai.StreamEvent{
			ai.FunctionToolResultEvent{Part: ai.UserPromptPart{Content: "bad"}},
		}, match: "unsupported tool result"},
		{name: "unencodable result", events: []ai.StreamEvent{
			ai.FunctionToolResultEvent{Part: ai.ToolReturnPart{Content: make(chan int)}},
		}, match: "encode tool result"},
		{name: "unencodable native result", events: []ai.StreamEvent{
			ai.PartStartEvent{Part: ai.NativeToolReturnPart{Content: make(chan int)}},
		}, match: "encode native tool result"},
		{name: "unencodable output result", events: []ai.StreamEvent{
			ai.OutputToolResultEvent{Part: ai.ToolReturnPart{Content: make(chan int)}},
		}, match: "encode tool result"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
				for _, event := range test.events {
					yield(event, nil)
				}
				if test.err != nil {
					yield(nil, test.err)
				}
			})
			var got error
			for event, err := range agui.TransformStream(stream, "", "") {
				if err != nil {
					got = err
					if event.Type != agui.EventRunError {
						t.Fatalf("unexpected error event: %#v", event)
					}
				}
			}
			if got == nil || !strings.Contains(got.Error(), test.match) {
				t.Fatalf("unexpected error: %v", got)
			}
		})
	}
}

func TestTransformStopsAtEveryMessageBoundary(t *testing.T) {
	tests := []struct {
		name   string
		events []ai.StreamEvent
		stopAt agui.EventType
	}{
		{name: "text start", events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.TextPart{}}}, stopAt: agui.EventTextMessageStart},
		{name: "text content", events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.TextPart{Content: "x"}}}, stopAt: agui.EventTextMessageContent},
		{name: "delta start", events: []ai.StreamEvent{ai.PartDeltaEvent{Delta: ai.TextPartDelta{}}}, stopAt: agui.EventTextMessageStart},
		{name: "tool message start", events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.ToolCallPart{ToolCallID: "c"}}}, stopAt: agui.EventTextMessageStart},
		{name: "tool start", events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.ToolCallPart{ToolCallID: "c"}}}, stopAt: agui.EventToolCallStart},
		{name: "tool args", events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.ToolCallPart{ToolCallID: "c", Args: []byte(`{}`)}}}, stopAt: agui.EventToolCallArgs},
		{name: "finish close", events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.TextPart{}}, ai.FinishEvent{}}, stopAt: agui.EventTextMessageEnd},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
				for _, event := range test.events {
					if !yield(event, nil) {
						return
					}
				}
			})
			calls := 0
			agui.TransformStream(stream, "thread", "run")(func(event agui.Event, _ error) bool {
				calls++
				return event.Type != test.stopAt
			})
			if calls < 2 {
				t.Fatalf("stream did not reach %s", test.stopAt)
			}
		})
	}
}

func TestTransformStopsWhileClosingErrors(t *testing.T) {
	stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.PartStartEvent{Part: ai.TextPart{}}, nil)
		yield(nil, errors.New("failed"))
	})
	agui.TransformStream(stream, "thread", "run")(func(event agui.Event, _ error) bool {
		return event.Type != agui.EventTextMessageEnd
	})

	stream = ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.PartStartEvent{Part: ai.TextPart{}}, nil)
	})
	agui.TransformStream(stream, "thread", "run")(func(event agui.Event, _ error) bool {
		return event.Type != agui.EventTextMessageEnd
	})
}

func TestTransformConsumerStops(t *testing.T) {
	stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.PartStartEvent{Part: ai.TextPart{Content: "text"}}, nil)
	})
	calls := 0
	agui.TransformStream(stream, "thread", "run")(func(agui.Event, error) bool {
		calls++
		return false
	})
	if calls != 1 {
		t.Fatalf("unexpected calls: %d", calls)
	}
}

var _ iter.Seq2[agui.Event, error] = agui.TransformStream(nil, "thread", "run")
