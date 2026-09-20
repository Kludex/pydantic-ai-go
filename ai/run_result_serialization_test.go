package ai_test

import (
	"encoding/json"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func TestRunResultJSONRoundTrip(t *testing.T) {
	result, err := ai.NewAgent[struct{}, string](fakes.NewTestModel()).Run(
		t.Context(), "hello", struct{}{}, ai.WithRunID("run"), ai.WithConversationID("conversation"),
	)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	var restored ai.RunResult[string]
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatal(err)
	}
	if restored.Output != result.Output || restored.RunID() != "run" || restored.ConversationID() != "conversation" ||
		len(restored.Messages()) != len(result.Messages()) || len(restored.NewMessages()) != len(result.NewMessages()) {
		t.Fatalf("unexpected restored result: %+v", restored)
	}
	if err := json.Unmarshal([]byte(`{"messages":[],"new_message_index":1}`), &restored); err == nil {
		t.Fatal("expected invalid new message index")
	}
	if err := json.Unmarshal([]byte(`{"messages":{}}`), &restored); err == nil {
		t.Fatal("expected invalid messages")
	}
	if err := restored.UnmarshalJSON([]byte(`{`)); err == nil {
		t.Fatal("expected invalid JSON")
	}
	result.Messages()[0] = nil
	if _, err := json.Marshal(result); err == nil {
		t.Fatal("expected invalid result messages")
	}
}
