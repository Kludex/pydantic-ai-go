package ai

import "testing"

// The kind marker methods exist to seal the message and part interfaces.
// They are exercised here so coverage reflects reality rather than pragmas.
func TestKindMarkers(t *testing.T) {
	if (ModelRequest{}).messageKind() != "request" || (ModelResponse{}).messageKind() != "response" {
		t.Fatal("unexpected message kinds")
	}
	requestKinds := []struct {
		part RequestPart
		want string
	}{
		{SystemPromptPart{}, "system-prompt"},
		{UserPromptPart{}, "user-prompt"},
		{ToolReturnPart{}, "tool-return"},
		{ToolAvailabilityDeltaPart{}, "tool-availability-delta"},
		{RetryPromptPart{}, "retry-prompt"},
	}
	for _, tc := range requestKinds {
		if tc.part.requestPartKind() != tc.want {
			t.Fatalf("expected %q, got %q", tc.want, tc.part.requestPartKind())
		}
	}
	responseKinds := []struct {
		part ResponsePart
		want string
	}{
		{TextPart{}, "text"},
		{ToolCallPart{}, "tool-call"},
		{NativeToolCallPart{}, "builtin-tool-call"},
		{NativeToolReturnPart{}, "builtin-tool-return"},
		{ThinkingPart{}, "thinking"},
		{CompactionPart{}, "compaction"},
	}
	for _, tc := range responseKinds {
		if tc.part.responsePartKind() != tc.want {
			t.Fatalf("expected %q, got %q", tc.want, tc.part.responsePartKind())
		}
	}
}

func TestStreamEventKinds(t *testing.T) {
	modelKinds := []struct {
		event ModelStreamEvent
		want  string
	}{
		{TextDeltaEvent{}, "text-delta"},
		{ThinkingDeltaEvent{}, "thinking-delta"},
		{CompactionEvent{}, "compaction"},
		{ToolCallStartEvent{}, "tool-call-start"},
		{ToolCallDeltaEvent{}, "tool-call-delta"},
		{NativeToolReturnEvent{}, "builtin-tool-return"},
		{FinishEvent{}, "finish"},
	}
	for _, test := range modelKinds {
		if test.event.modelStreamEventKind() != test.want {
			t.Fatalf("expected %q, got %q", test.want, test.event.modelStreamEventKind())
		}
	}
	streamKinds := []struct {
		event StreamEvent
		want  string
	}{
		{PartStartEvent{}, "part-start"},
		{PartDeltaEvent{}, "part-delta"},
		{PartEndEvent{}, "part-end"},
		{FinalResultEvent{}, "final-result"},
		{FunctionToolCallEvent{}, "function-tool-call"},
		{OutputToolCallEvent{}, "output-tool-call"},
		{FunctionToolResultEvent{}, "function-tool-result"},
		{OutputToolResultEvent{}, "output-tool-result"},
		{FinishEvent{}, "finish"},
	}
	for _, test := range streamKinds {
		if test.event.streamEventKind() != test.want {
			t.Fatalf("expected %q, got %q", test.want, test.event.streamEventKind())
		}
	}
	deltaKinds := []struct {
		delta ResponsePartDelta
		want  ResponsePartKind
	}{
		{TextPartDelta{}, ResponsePartKindText},
		{ThinkingPartDelta{}, ResponsePartKindThinking},
		{ToolCallPartDelta{}, ResponsePartKindToolCall},
		{NativeToolCallPartDelta{}, ResponsePartKindNativeToolCall},
	}
	for _, test := range deltaKinds {
		if test.delta.responsePartDeltaKind() != test.want {
			t.Fatalf("expected %q, got %q", test.want, test.delta.responsePartDeltaKind())
		}
	}
}

func TestUserContentKinds(t *testing.T) {
	kinds := []struct {
		c    UserContent
		want string
	}{
		{TextContent{}, "text-content"},
		{ImageURL{}, "image-url"},
		{BinaryContent{}, "binary"},
		{UploadedFile{}, "uploaded-file"},
	}
	for _, tc := range kinds {
		if tc.c.userContentKind() != tc.want {
			t.Fatalf("expected %q, got %q", tc.want, tc.c.userContentKind())
		}
		if tc.c.enqueueItemKind() != "user-content" {
			t.Fatalf("unexpected enqueue item kind %q", tc.c.enqueueItemKind())
		}
	}
}
