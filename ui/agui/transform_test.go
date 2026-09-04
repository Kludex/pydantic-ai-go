package agui_test

import (
	"encoding/json"
	"errors"
	"fmt"
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
	for _, event := range events {
		if event.Timestamp == 0 {
			t.Fatalf("event has no timestamp: %#v", event)
		}
	}
	if eventIndex(events, agui.EventToolCallArgs) < 0 || eventIndex(events, agui.EventToolCallResult) < 0 ||
		eventIndex(events, agui.EventTextMessageEnd) < 0 {
		t.Fatalf("missing transformed events: %#v", events)
	}
}

func TestTransformReasoningVersions(t *testing.T) {
	for _, test := range []struct {
		version string
		start   agui.EventType
		content agui.EventType
		end     agui.EventType
		role    string
		modern  bool
	}{
		{version: "0.1.10", start: agui.EventThinkingStart, content: agui.EventThinkingTextMessageContent, end: agui.EventThinkingEnd},
		{version: "0.1.13rc1", start: agui.EventReasoningStart, content: agui.EventReasoningMessageContent, end: agui.EventReasoningEnd, role: "assistant", modern: true},
		{version: "0.1.14", start: agui.EventReasoningStart, content: agui.EventReasoningMessageContent, end: agui.EventReasoningEnd, role: "reasoning", modern: true},
		{version: "0.2", start: agui.EventReasoningStart, content: agui.EventReasoningMessageContent, end: agui.EventReasoningEnd, role: "reasoning", modern: true},
		{version: "1", start: agui.EventReasoningStart, content: agui.EventReasoningMessageContent, end: agui.EventReasoningEnd, role: "reasoning", modern: true},
	} {
		stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
			yield(ai.PartStartEvent{PartID: "thinking", Part: ai.ThinkingPart{
				Content: "a", ID: "thinking-id", Signature: "signature", ProviderName: "provider",
				ProviderDetails: map[string]any{"key": "value"},
			}}, nil)
			yield(ai.PartDeltaEvent{PartID: "thinking", Delta: ai.ThinkingPartDelta{ContentDelta: "b"}}, nil)
			yield(ai.PartEndEvent{PartID: "thinking", Part: ai.ThinkingPart{
				ID: "thinking-id", Signature: "signature", ProviderName: "provider",
				ProviderDetails: map[string]any{"key": "value"},
			}}, nil)
		})
		var events []agui.Event
		for event, err := range agui.TransformStreamWithConfig(stream, agui.StreamConfig{Version: test.version}) {
			if err != nil {
				t.Fatal(err)
			}
			events = append(events, event)
		}
		if eventIndex(events, test.start) < 0 || eventIndex(events, test.content) < 0 ||
			eventIndex(events, test.end) < 0 {
			t.Fatalf("version=%s: missing reasoning lifecycle: %#v", test.version, events)
		}
		if test.modern {
			start := events[eventIndex(events, agui.EventReasoningMessageStart)]
			encrypted := events[eventIndex(events, agui.EventReasoningEncryptedValue)]
			if start.Role != test.role || start.MessageID == "" || encrypted.Subtype != "message" ||
				encrypted.EntityID != start.MessageID || !strings.Contains(encrypted.EncryptedValue, `"signature":"signature"`) {
				t.Fatalf("version=%s: unexpected reasoning metadata: %#v", test.version, events)
			}
		} else if events[eventIndex(events, test.start)].MessageID != "" {
			t.Fatalf("legacy reasoning included a message ID: %#v", events)
		}
	}
}

func TestTransformLazyReasoning(t *testing.T) {
	stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.PartStartEvent{PartID: "first", Part: ai.ThinkingPart{}}, nil)
		yield(ai.PartEndEvent{PartID: "first", Part: ai.ThinkingPart{}}, nil)
		yield(ai.PartDeltaEvent{PartID: "second", Delta: ai.ThinkingPartDelta{ContentDelta: "late"}}, nil)
		yield(ai.PartEndEvent{PartID: "second", Part: ai.ThinkingPart{}}, nil)
		yield(ai.PartStartEvent{PartID: "metadata", Part: ai.ThinkingPart{}}, nil)
		yield(ai.PartEndEvent{PartID: "metadata", Part: ai.ThinkingPart{Signature: "opaque"}}, nil)
	})
	var starts, encrypted int
	for event, err := range agui.TransformStreamWithConfig(stream, agui.StreamConfig{Version: "0.1.19"}) {
		if err != nil {
			t.Fatal(err)
		}
		if event.Type == agui.EventReasoningStart {
			starts++
		}
		if event.Type == agui.EventReasoningEncryptedValue {
			encrypted++
		}
	}
	if starts != 2 || encrypted != 1 {
		t.Fatalf("unexpected lazy reasoning lifecycles: starts=%d encrypted=%d", starts, encrypted)
	}
}

func TestTransformTypedToolMetadata(t *testing.T) {
	stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		call := ai.ToolCallPart{
			ToolName: "load", ToolCallID: "call", Args: json.RawMessage(`{}`),
			ToolKind: ai.ToolPartKindCapabilityLoad,
		}
		yield(ai.PartStartEvent{PartID: "call", Part: call}, nil)
		yield(ai.FunctionToolCallEvent{Part: call}, nil)
		yield(ai.FunctionToolResultEvent{Part: ai.ToolReturnPart{
			ToolName: "load", ToolCallID: "call", Content: "denied",
			ToolKind: ai.ToolPartKindCapabilityLoad, Outcome: ai.ToolReturnOutcomeDenied,
		}}, nil)
		yield(ai.PartStartEvent{PartID: "native-result", Part: ai.NativeToolReturnPart{
			ToolName: "search", ToolCallID: "native", ProviderName: "openai", Content: "found",
		}}, nil)
	})
	var events []agui.Event
	for event, err := range agui.TransformStream(stream, "thread", "run") {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, event)
	}
	var toolMetadata, resultMetadata *agui.Event
	for index := range events {
		event := &events[index]
		if event.Type == agui.EventReasoningEncryptedValue && event.Subtype == "tool-call" {
			toolMetadata = event
		}
		if event.Type == agui.EventReasoningEncryptedValue && event.Subtype == "message" {
			resultMetadata = event
		}
	}
	if toolMetadata == nil || toolMetadata.EntityID != "call" ||
		!strings.Contains(toolMetadata.EncryptedValue, `"tool_kind":"capability-load"`) ||
		resultMetadata == nil || !strings.Contains(resultMetadata.EncryptedValue, `"outcome":"denied"`) {
		t.Fatalf("unexpected typed tool metadata: %#v", events)
	}
	native := events[eventIndex(events, agui.EventToolCallResult)]
	for _, event := range events {
		if event.Type == agui.EventToolCallResult && strings.HasPrefix(event.ToolCallID, "pyd_ai_builtin|") {
			native = event
		}
	}
	if native.ToolCallID != "pyd_ai_builtin|openai|native" {
		t.Fatalf("unexpected native result identity: %#v", events)
	}
}

func TestTransformFiles(t *testing.T) {
	first := ai.FilePart{
		Content: ai.BinaryContent{Data: []byte("first"), MediaType: "image/png", VendorMetadata: map[string]any{"v": "one"}},
		ID:      "file", ProviderName: "openai", ProviderDetails: map[string]any{"p": "one"},
	}
	second := first
	second.Content.Data = []byte("second")
	stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.PartStartEvent{PartID: "file", Part: first}, nil)
		yield(ai.PartDeltaEvent{PartID: "file", Delta: ai.FilePartDelta{Part: second}}, nil)
		yield(ai.PartDeltaEvent{PartID: "late-file", Delta: ai.FilePartDelta{Part: second}}, nil)
	})
	for _, config := range []agui.StreamConfig{
		{Version: "0.1.19"},
		{Version: "0.1.18", PreserveFileData: true},
		{Version: "0.1.19", PreserveFileData: true},
	} {
		var files []agui.Event
		for event, err := range agui.TransformStreamWithConfig(stream, config) {
			if err != nil {
				t.Fatal(err)
			}
			if event.ActivityType == "pydantic_ai_file" {
				files = append(files, event)
			}
		}
		if !config.PreserveFileData || config.Version == "0.1.18" {
			if len(files) != 0 {
				t.Fatalf("files were not omitted: %#v", files)
			}
			continue
		}
		if len(files) != 3 || files[0].MessageID != files[1].MessageID || files[1].MessageID == files[2].MessageID ||
			files[0].Content.(map[string]any)["url"] != "data:image/png;base64,Zmlyc3Q=" ||
			files[1].Content.(map[string]any)["url"] != "data:image/png;base64,c2Vjb25k" ||
			files[0].Content.(map[string]any)["id"] != "file" ||
			files[0].Content.(map[string]any)["provider_name"] != "openai" {
			t.Fatalf("unexpected file snapshots: %#v", files)
		}
	}
}

func TestTransformActivities(t *testing.T) {
	stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.PartStartEvent{PartID: "compaction", Part: ai.CompactionPart{
			Content: "summary", ID: "compact", ProviderName: "openai",
			ProviderDetails: map[string]any{"encrypted": "value"},
		}}, nil)
		yield(ai.ToolAvailabilityDeltaEvent{Part: ai.ToolAvailabilityDeltaPart{
			ToolsAdded: []string{"search"}, ToolCallID: "call",
		}}, nil)
	})
	for _, version := range []string{"0.1.18", "0.1.19"} {
		var activities []agui.Event
		for event, err := range agui.TransformStreamWithConfig(stream, agui.StreamConfig{Version: version}) {
			if err != nil {
				t.Fatal(err)
			}
			if event.Type == agui.EventActivitySnapshot {
				activities = append(activities, event)
			}
		}
		if version == "0.1.18" && len(activities) != 0 {
			t.Fatalf("old version received activities: %#v", activities)
		}
		if version == "0.1.19" {
			if len(activities) != 2 || activities[0].ActivityType != "pydantic_ai_compaction" ||
				activities[0].Content.(map[string]any)["content"] != "summary" ||
				activities[1].ActivityType != "pydantic_ai_tool_availability_delta" ||
				activities[1].Content.(map[string]any)["tool_call_id"] != "call" ||
				activities[0].Replace == nil || !*activities[0].Replace {
				t.Fatalf("unexpected activities: %#v", activities)
			}
		}
	}
}

func TestTransformStreamErrors(t *testing.T) {
	var versionErr error
	for _, err := range agui.TransformStreamWithConfig(nil, agui.StreamConfig{Version: "0.1.2.3"}) {
		versionErr = err
	}
	if versionErr == nil {
		t.Fatal("expected invalid version error")
	}
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
	cancelled := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.PartStartEvent{PartID: "text", Part: ai.TextPart{Content: "partial"}}, nil)
		yield(nil, fmt.Errorf("cancelled: %w", ai.ErrRunCancelled))
	})
	var events []agui.Event
	for event, err := range agui.TransformStream(cancelled, "thread", "run") {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, event)
	}
	if events[len(events)-1].Type != agui.EventRunFinished || events[len(events)-1].Outcome != nil ||
		eventIndex(events, agui.EventRunError) >= 0 {
		t.Fatalf("unexpected cancellation lifecycle: %#v", events)
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
