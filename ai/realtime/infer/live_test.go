package infer_test

import (
	"testing"

	"github.com/Kludex/pydantic-ai-go/ai/realtime/infer"
	openairt "github.com/Kludex/pydantic-ai-go/ai/realtime/openai"
)

func TestLiveFamilyInference(t *testing.T) {
	model, err := infer.Model("openai:gpt-live-1+gpt-5")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := model.(*openairt.LiveModel); !ok || model.Name() != "gpt-live-1" {
		t.Fatalf("model: %#v", model)
	}
}
