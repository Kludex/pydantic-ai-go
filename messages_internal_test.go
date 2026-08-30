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
		{ThinkingPart{}, "thinking"},
	}
	for _, tc := range responseKinds {
		if tc.part.responsePartKind() != tc.want {
			t.Fatalf("expected %q, got %q", tc.want, tc.part.responsePartKind())
		}
	}
}
