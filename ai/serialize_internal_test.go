package ai

import "testing"

type alienMessage struct{}

func (alienMessage) messageKind() string     { return "alien" }
func (alienMessage) enqueueItemKind() string { return "alien" }

type alienRequestPart struct{}

func (alienRequestPart) requestPartKind() string { return "alien" }
func (alienRequestPart) enqueueItemKind() string { return "alien" }

type alienResponsePart struct{}

func (alienResponsePart) responsePartKind() string { return "alien" }

func TestMarshalUnknownTypes(t *testing.T) {
	if _, err := MarshalMessages([]ModelMessage{alienMessage{}}); err == nil {
		t.Fatal("expected error for unknown message type")
	}
	if _, err := MarshalMessages([]ModelMessage{ModelRequest{Parts: []RequestPart{alienRequestPart{}}}}); err == nil {
		t.Fatal("expected error for unknown request part type")
	}
	if _, err := MarshalMessages([]ModelMessage{ModelResponse{Parts: []ResponsePart{alienResponsePart{}}}}); err == nil {
		t.Fatal("expected error for unknown response part type")
	}
}

type alienUserContent struct{}

func (alienUserContent) userContentKind() string { return "alien" }
func (alienUserContent) enqueueItemKind() string { return "alien" }

func TestMarshalUnknownUserContent(t *testing.T) {
	msgs := []ModelMessage{ModelRequest{Parts: []RequestPart{
		UserPromptPart{Contents: []UserContent{alienUserContent{}}},
	}}}
	if _, err := MarshalMessages(msgs); err == nil {
		t.Fatal("expected error for unknown user content type")
	}
}

func TestSanitizeUnknownPartTypes(t *testing.T) {
	messages := []ModelMessage{
		ModelRequest{Parts: []RequestPart{alienRequestPart{}}},
		ModelResponse{Parts: []ResponsePart{alienResponsePart{}}},
	}
	sanitized, report, err := SanitizeMessages(messages, MessageSanitizationOptions{})
	if err != nil || report.Changed() || len(sanitized) != 2 {
		t.Fatalf("unexpected sanitization: messages=%+v report=%+v err=%v", sanitized, report, err)
	}
}

func TestUnmarshalMalformedNativeToolReturn(t *testing.T) {
	if _, err := unmarshalResponsePart(wirePart{
		PartKind: "builtin-tool-return", Content: []byte(`{`),
	}); err == nil {
		t.Fatal("expected malformed native tool return error")
	}
}

func TestStringContentFallback(t *testing.T) {
	// Non-string raw content falls back to the raw bytes, e.g. legacy
	// histories where system prompt content was a number.
	if got := stringContent([]byte(`42`)); got != "42" {
		t.Fatalf("expected raw fallback, got %q", got)
	}
}
