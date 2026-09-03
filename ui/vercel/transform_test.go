package vercel_test

import (
	"errors"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/ui/vercel"
)

func TestTransformStreamVariants(t *testing.T) {
	stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		events := []ai.StreamEvent{
			ai.PartStartEvent{PartID: "text", Part: ai.TextPart{Content: "a"}},
			ai.PartDeltaEvent{PartID: "text", Delta: ai.TextPartDelta{ContentDelta: "b"}},
			ai.PartEndEvent{PartID: "text", Part: ai.TextPart{}},
			ai.PartStartEvent{PartID: "thinking", Part: ai.ThinkingPart{Content: "c"}},
			ai.PartDeltaEvent{PartID: "thinking", Delta: ai.ThinkingPartDelta{ContentDelta: "d"}},
			ai.PartEndEvent{PartID: "thinking", Part: ai.ThinkingPart{}},
			ai.PartStartEvent{PartID: "call", Part: ai.ToolCallPart{ToolName: "tool", ToolCallID: "call-1", Args: []byte(`{"x":`)}},
			ai.PartDeltaEvent{PartID: "call", Delta: ai.ToolCallPartDelta{ArgsDelta: "1}"}},
			ai.PartDeltaEvent{PartID: "call", Delta: ai.ToolCallPartDelta{ToolCallID: "call-1"}},
			ai.FunctionToolCallEvent{Part: ai.ToolCallPart{ToolName: "tool", ToolCallID: "call-1", Args: []byte(`{"x":1}`)}},
			ai.FunctionToolResultEvent{Part: ai.ToolReturnPart{ToolCallID: "call-1", Content: map[string]any{"ok": true}}},
			ai.OutputToolCallEvent{Part: ai.ToolCallPart{ToolName: "final", ToolCallID: "call-2"}},
			ai.OutputToolResultEvent{Part: ai.RetryPromptPart{ToolCallID: "call-2", Content: "retry"}},
			ai.PartStartEvent{PartID: "native", Part: ai.NativeToolCallPart{ToolName: "search", ToolCallID: "native-1"}},
			ai.PartDeltaEvent{PartID: "native", Delta: ai.NativeToolCallPartDelta(ai.ToolCallPartDelta{ArgsDelta: `{}`})},
			ai.PartStartEvent{PartID: "native-return", Part: ai.NativeToolReturnPart{ToolCallID: "native-1", Content: "found"}},
			ai.PartStartEvent{PartID: "ignored", Part: ai.FilePart{}},
			ai.PartDeltaEvent{PartID: "ignored", Delta: ai.FilePartDelta{}},
			ai.PartEndEvent{PartID: "ignored", Part: ai.FilePart{}},
			ai.FinishEvent{FinishReason: ai.FinishReasonToolCall},
		}
		for _, event := range events {
			if !yield(event, nil) {
				return
			}
		}
	})
	var chunks []vercel.Chunk
	for chunk, err := range vercel.TransformStream(stream, "") {
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}
	if chunks[0].Type != vercel.ChunkStart || chunks[0].MessageID == "" ||
		chunks[len(chunks)-1].Type != vercel.ChunkDone {
		t.Fatalf("unexpected lifecycle: %#v", chunks)
	}
	for _, kind := range []vercel.ChunkType{
		vercel.ChunkTextStart, vercel.ChunkTextDelta, vercel.ChunkTextEnd,
		vercel.ChunkReasoningStart, vercel.ChunkReasoningDelta, vercel.ChunkReasoningEnd,
		vercel.ChunkToolInputStart, vercel.ChunkToolInputDelta, vercel.ChunkToolInputAvailable,
		vercel.ChunkToolOutputAvailable, vercel.ChunkToolOutputError,
	} {
		if chunkIndex(chunks, kind) < 0 {
			t.Fatalf("missing %s: %#v", kind, chunks)
		}
	}
	finish := chunks[chunkIndex(chunks, vercel.ChunkFinish)]
	if finish.FinishReason != "tool-calls" {
		t.Fatalf("unexpected finish: %#v", finish)
	}
}

func TestTransformApprovalVersions(t *testing.T) {
	stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.DeferredToolRequestsEvent{Requests: ai.DeferredToolRequests{Approvals: []ai.ToolCallPart{{
			ToolCallID: "call",
		}}}}, nil)
	})
	for _, version := range []int{5, 6, 7} {
		seen := false
		for chunk, err := range vercel.TransformStreamWithConfig(stream, vercel.StreamConfig{SDKVersion: version}) {
			if err != nil {
				t.Fatal(err)
			}
			if chunk.Type == vercel.ChunkToolApprovalRequest {
				seen = true
			}
		}
		if seen != (version >= 6) {
			t.Fatalf("version=%d approval=%v", version, seen)
		}
	}
	vercel.TransformStreamWithConfig(stream, vercel.StreamConfig{SDKVersion: 6, ServerMessageID: "message"})(
		func(chunk vercel.Chunk, _ error) bool { return chunk.Type != vercel.ChunkToolApprovalRequest },
	)
	for _, version := range []int{4, 8} {
		var got error
		for _, err := range vercel.TransformStreamWithConfig(nil, vercel.StreamConfig{SDKVersion: version}) {
			got = err
		}
		if got == nil {
			t.Fatalf("version=%d: expected error", version)
		}
	}
	empty := ai.EventStream(func(func(ai.StreamEvent, error) bool) {})
	for _, err := range vercel.TransformStreamWithConfig(empty, vercel.StreamConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestTransformFinishReasons(t *testing.T) {
	tests := []struct {
		reason ai.FinishReason
		want   string
	}{
		{ai.FinishReasonStop, "stop"},
		{ai.FinishReasonLength, "length"},
		{ai.FinishReasonContentFilter, "content-filter"},
		{ai.FinishReasonToolCall, "tool-calls"},
		{ai.FinishReasonError, "error"},
		{"future", "other"},
	}
	for _, test := range tests {
		stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
			yield(ai.FinishEvent{FinishReason: test.reason}, nil)
		})
		var finish vercel.Chunk
		for chunk, err := range vercel.TransformStream(stream, "message") {
			if err != nil {
				t.Fatal(err)
			}
			if chunk.Type == vercel.ChunkFinish {
				finish = chunk
			}
		}
		if finish.FinishReason != test.want {
			t.Fatalf("reason=%q: got %q", test.reason, finish.FinishReason)
		}
	}
}

func TestTransformErrors(t *testing.T) {
	tests := []struct {
		name   string
		events []ai.StreamEvent
		err    error
		match  string
	}{
		{name: "stream", events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.TextPart{}}}, err: errors.New("stream failed"), match: "stream failed"},
		{name: "tool input", events: []ai.StreamEvent{ai.FunctionToolCallEvent{Part: ai.ToolCallPart{Args: []byte("{")}}}, match: "unexpected end"},
		{name: "result", events: []ai.StreamEvent{ai.FunctionToolResultEvent{Part: ai.UserPromptPart{}}}, match: "unsupported tool result"},
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
			for chunk, err := range vercel.TransformStream(stream, "message") {
				if err != nil {
					got = err
					if chunk.Type != vercel.ChunkError {
						t.Fatalf("unexpected error chunk: %#v", chunk)
					}
				}
			}
			if got == nil || !strings.Contains(got.Error(), test.match) {
				t.Fatalf("unexpected error: %v", got)
			}
		})
	}
}

func TestTransformConsumerStops(t *testing.T) {
	streams := []ai.EventStream{
		func(yield func(ai.StreamEvent, error) bool) { yield(ai.PartStartEvent{Part: ai.TextPart{}}, nil) },
		func(yield func(ai.StreamEvent, error) bool) { yield(ai.PartStartEvent{Part: ai.ThinkingPart{}}, nil) },
		func(yield func(ai.StreamEvent, error) bool) {
			yield(ai.PartStartEvent{Part: ai.ToolCallPart{Args: []byte(`{}`)}}, nil)
		},
		func(yield func(ai.StreamEvent, error) bool) { yield(ai.FinishEvent{}, nil) },
	}
	for _, stream := range streams {
		calls := 0
		vercel.TransformStream(stream, "message")(func(vercel.Chunk, error) bool {
			calls++
			return calls < 2
		})
		if calls != 2 {
			t.Fatalf("unexpected callback count: %d", calls)
		}
	}

	stopCases := []struct {
		events []ai.StreamEvent
		stopAt vercel.ChunkType
	}{
		{events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.TextPart{}}}, stopAt: vercel.ChunkTextStart},
		{events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.TextPart{Content: "x"}}}, stopAt: vercel.ChunkTextDelta},
		{events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.ThinkingPart{}}}, stopAt: vercel.ChunkReasoningStart},
		{events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.ThinkingPart{Content: "x"}}}, stopAt: vercel.ChunkReasoningDelta},
		{events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.ToolCallPart{}}}, stopAt: vercel.ChunkToolInputStart},
		{events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.NativeToolCallPart{}}}, stopAt: vercel.ChunkToolInputStart},
		{events: []ai.StreamEvent{ai.PartStartEvent{Part: ai.ToolCallPart{Args: []byte(`{}`)}}}, stopAt: vercel.ChunkToolInputDelta},
	}
	for _, test := range stopCases {
		stream := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
			for _, event := range test.events {
				yield(event, nil)
			}
		})
		vercel.TransformStream(stream, "message")(func(chunk vercel.Chunk, _ error) bool {
			return chunk.Type != test.stopAt
		})
	}

	empty := ai.EventStream(func(func(ai.StreamEvent, error) bool) {})
	vercel.TransformStream(empty, "message")(func(chunk vercel.Chunk, _ error) bool {
		return chunk.Type != vercel.ChunkFinish
	})
	vercel.TransformStream(empty, "message")(func(vercel.Chunk, error) bool { return false })

	open := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.PartStartEvent{Part: ai.TextPart{}}, nil)
	})
	vercel.TransformStream(open, "message")(func(chunk vercel.Chunk, _ error) bool {
		return chunk.Type != vercel.ChunkFinishStep
	})
	failed := ai.EventStream(func(yield func(ai.StreamEvent, error) bool) {
		yield(ai.PartStartEvent{Part: ai.TextPart{}}, nil)
		yield(nil, errors.New("failed"))
	})
	vercel.TransformStream(failed, "message")(func(chunk vercel.Chunk, _ error) bool {
		return chunk.Type != vercel.ChunkFinishStep
	})
}
