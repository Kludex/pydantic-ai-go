package ai_test

import (
	"iter"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func TestCollectModelStream(t *testing.T) {
	events := func(yield func(ai.ModelStreamEvent, error) bool) {
		yield(ai.TextDeltaEvent{PartID: "text", Delta: "hello"}, nil)
		yield(ai.FinishEvent{State: ai.ModelResponseStateComplete}, nil)
	}
	response, err := ai.CollectModelStream(iter.Seq2[ai.ModelStreamEvent, error](events), ai.ModelRequestParams{})
	if err != nil || response.Text() != "hello" {
		t.Fatalf("unexpected collected response: %+v %v", response, err)
	}
}
