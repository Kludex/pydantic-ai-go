package infer_test

import (
	"testing"

	"github.com/Kludex/pydantic-ai-go/realtime/infer"
)

func TestModel(t *testing.T) {
	t.Setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
	t.Setenv("AZURE_OPENAI_API_KEY", "key")
	for _, name := range []string{
		"azure:gpt-realtime", "openai:gpt-realtime", "xai:grok-voice-latest",
		"google:gemini-live", "google-cloud:gemini-live",
	} {
		model, err := infer.Model(name)
		if err != nil || model.Name() == "" {
			t.Fatalf("infer %q: model=%v err=%v", name, model, err)
		}
	}
	for _, name := range []string{"missing", "openai:", "future:model"} {
		if _, err := infer.Model(name); err == nil {
			t.Fatalf("expected inference error for %q", name)
		}
	}
}
