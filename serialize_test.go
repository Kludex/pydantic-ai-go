package ai_test

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go"
)

func TestMessagesRoundTrip(t *testing.T) {
	msgs := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SystemPromptPart{Content: "be nice"},
			ai.UserPromptPart{Content: "weather in SF?"},
		}},
		ai.ModelResponse{
			Parts: []ai.ResponsePart{
				ai.ThinkingPart{Content: "hmm"},
				ai.ToolCallPart{ToolName: "get_weather", Args: json.RawMessage(`{"city":"SF"}`), ToolCallID: "c1"},
			},
			Usage:     ai.Usage{Requests: 1, InputTokens: 10, OutputTokens: 5},
			ModelName: "test-model",
			Timestamp: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
		},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "get_weather", Content: "sunny", ToolCallID: "c1"},
			ai.RetryPromptPart{Content: "try again", ToolName: "get_weather", ToolCallID: "c1"},
		}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "It is sunny."}}},
	}

	data, err := ai.MarshalMessages(msgs)
	if err != nil {
		t.Fatal(err)
	}
	for _, expected := range []string{`"kind":"request"`, `"kind":"response"`, `"part_kind":"system-prompt"`,
		`"part_kind":"user-prompt"`, `"part_kind":"tool-call"`, `"part_kind":"tool-return"`,
		`"part_kind":"retry-prompt"`, `"part_kind":"thinking"`, `"part_kind":"text"`, `"model_name":"test-model"`} {
		if !strings.Contains(string(data), expected) {
			t.Fatalf("expected %s in %s", expected, data)
		}
	}

	back, err := ai.UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(back) != len(msgs) {
		t.Fatalf("expected %d messages, got %d", len(msgs), len(back))
	}
	resp := back[1].(ai.ModelResponse)
	if resp.ModelName != "test-model" || resp.Usage.InputTokens != 10 {
		t.Fatalf("unexpected response %+v", resp)
	}
	if resp.ToolCalls()[0].ToolName != "get_weather" {
		t.Fatal("tool call lost in round trip")
	}
	req := back[2].(ai.ModelRequest)
	if req.Parts[0].(ai.ToolReturnPart).Content != "sunny" {
		t.Fatal("tool return lost in round trip")
	}
	if back[3].(ai.ModelResponse).Text() != "It is sunny." {
		t.Fatal("text lost in round trip")
	}
}

func TestUnmarshalUnknownKind(t *testing.T) {
	if _, err := ai.UnmarshalMessages([]byte(`[{"kind":"mystery"}]`)); err == nil {
		t.Fatal("expected error")
	}
}

func TestMarshalUnmarshalEdgeCases(t *testing.T) {
	if _, err := ai.UnmarshalMessages([]byte(`not json`)); err == nil {
		t.Fatal("expected error for invalid JSON")
	}
	if _, err := ai.UnmarshalMessages([]byte(`[42]`)); err == nil {
		t.Fatal("expected error for non-object message")
	}
	if _, err := ai.UnmarshalMessages([]byte(`[{"kind":"request","parts":[{"part_kind":"mystery"}]}]`)); err == nil {
		t.Fatal("expected error for unknown request part kind")
	}
	if _, err := ai.UnmarshalMessages([]byte(`[{"kind":"response","parts":[{"part_kind":"mystery"}]}]`)); err == nil {
		t.Fatal("expected error for unknown response part kind")
	}
	if _, err := ai.UnmarshalMessages([]byte(`[{"kind":"request","parts":42}]`)); err == nil {
		t.Fatal("expected error for malformed request")
	}
	if _, err := ai.UnmarshalMessages([]byte(`[{"kind":"response","parts":42}]`)); err == nil {
		t.Fatal("expected error for malformed response")
	}
	if _, err := ai.UnmarshalMessages([]byte(`[{"kind":"request","parts":[{"part_kind":"tool-return","content":42,"tool_name":"t"}]}]`)); err != nil {
		t.Fatalf("non-string tool return content should unmarshal: %v", err)
	}

	if _, err := ai.MarshalMessages([]ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
		ai.ToolReturnPart{ToolName: "t", Content: make(chan int)},
	}}}); err == nil {
		t.Fatal("expected error for unmarshallable tool return content")
	}
}

func TestUnmarshalNonStringContentFallsBack(t *testing.T) {
	msgs, err := ai.UnmarshalMessages([]byte(`[{"kind":"request","parts":[{"part_kind":"user-prompt","content":42}]}]`))
	if err != nil {
		t.Fatal(err)
	}
	part := msgs[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart)
	if part.Content != "42" {
		t.Fatalf("expected raw fallback, got %q", part.Content)
	}
}
