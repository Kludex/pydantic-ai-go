// Package seed renders portable history the way a realtime provider replays it.
package seed

import (
	"strconv"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// AssistantSpeech returns the text the transcribed assistant speech part at index of message seeds as: its
// transcript, marked where a barge-in cut it off, because the model never heard what came after. The markers
// match the ones ai.PrepareModelMessages renders for a text model.
func AssistantSpeech(message ai.ModelResponse, index int) string {
	speech := message.Parts[index].(ai.SpeechPart)
	if speech.InterruptedAtMS != nil {
		return speech.Content() + "\n[Interrupted after " + strconv.Itoa(*speech.InterruptedAtMS) + " ms]"
	}
	if message.State == ai.ModelResponseStateInterrupted && index == lastSpeech(message.Parts) {
		return speech.Content() + "\n[Interrupted]"
	}
	return speech.Content()
}

func lastSpeech(parts []ai.ResponsePart) int {
	last := -1
	for index, part := range parts {
		if _, ok := part.(ai.SpeechPart); ok {
			last = index
		}
	}
	return last
}
