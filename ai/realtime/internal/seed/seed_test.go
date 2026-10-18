package seed_test

import (
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime/internal/seed"
)

func TestAssistantSpeech(t *testing.T) {
	text := func(value string) *string { return &value }
	heard := 1200
	cases := []struct {
		name    string
		message ai.ModelResponse
		index   int
		want    string
	}{
		{"complete", ai.ModelResponse{Parts: []ai.ResponsePart{ai.SpeechPart{Transcript: text("Hello.")}}}, 0, "Hello."},
		{"cut at offset", ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.SpeechPart{Transcript: text("Hello there.")}, ai.SpeechPart{Transcript: text("Bye."), InterruptedAtMS: &heard},
		}}, 1, "Bye.\n[Interrupted after 1200 ms]"},
		{"interrupted last", ai.ModelResponse{State: ai.ModelResponseStateInterrupted, Parts: []ai.ResponsePart{
			ai.SpeechPart{Transcript: text("One.")}, ai.SpeechPart{Transcript: text("Two.")}, ai.TextPart{Content: "x"},
		}}, 1, "Two.\n[Interrupted]"},
		{"interrupted earlier", ai.ModelResponse{State: ai.ModelResponseStateInterrupted, Parts: []ai.ResponsePart{
			ai.SpeechPart{Transcript: text("One.")}, ai.SpeechPart{Transcript: text("Two.")},
		}}, 0, "One."},
	}
	for _, test := range cases {
		if got := seed.AssistantSpeech(test.message, test.index); got != test.want {
			t.Errorf("%s: %q", test.name, got)
		}
	}
}
