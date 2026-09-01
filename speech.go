package ai

import (
	"fmt"
	"strconv"
	"time"
)

// PrepareModelMessages converts realtime speech history into content a standard model can consume.
// It returns a detached history and leaves durable SpeechPart values unchanged. Composite
// dispatchers may defer conversion so each selected child applies its own profile.
func PrepareModelMessages(model Model, messages []ModelMessage) ([]ModelMessage, error) {
	messages = cloneModelMessages(messages)
	if dispatcher, ok := model.(ModelMessageProfileDispatcher); ok && dispatcher.DispatchesMessageProfile() {
		return messages, nil
	}
	return convertSpeechMessages(messages, modelProfile(model).SupportsAudioInput)
}

func convertSpeechMessages(messages []ModelMessage, includeAudio bool) ([]ModelMessage, error) {
	hasSpeech := false
	for _, message := range messages {
		switch message := message.(type) {
		case ModelRequest:
			for _, part := range message.Parts {
				if _, ok := part.(SpeechPart); ok {
					hasSpeech = true
					break
				}
			}
		case ModelResponse:
			for _, part := range message.Parts {
				if _, ok := part.(SpeechPart); ok {
					hasSpeech = true
					break
				}
			}
		}
		if hasSpeech {
			break
		}
	}
	if !hasSpeech {
		return messages, nil
	}

	prepared := make([]ModelMessage, 0, len(messages))
	for _, message := range messages {
		switch message := message.(type) {
		case ModelRequest:
			parts := make([]RequestPart, 0, len(message.Parts))
			promptTimestamp := message.Timestamp
			if promptTimestamp.IsZero() {
				promptTimestamp = time.Now().UTC()
			}
			for _, requestPart := range message.Parts {
				speech, ok := requestPart.(SpeechPart)
				if !ok {
					parts = append(parts, requestPart)
					continue
				}
				if speech.Speaker != SpeechSpeakerUser {
					return nil, speechSpeakerError("ModelRequest", SpeechSpeakerUser, speech.Speaker)
				}
				switch {
				case includeAudio && speech.Audio != nil:
					audio := cloneUserContents([]UserContent{*speech.Audio})[0].(BinaryContent)
					parts = append(parts, UserPromptPart{Contents: []UserContent{audio}, Timestamp: promptTimestamp})
				case speech.Content() != "":
					parts = append(parts, UserPromptPart{Content: speech.Content(), Timestamp: promptTimestamp})
				}
			}
			if len(parts) > 0 {
				message.Parts = parts
				prepared = append(prepared, message)
			}
		case ModelResponse:
			lastSpeech := -1
			for index, part := range message.Parts {
				if _, ok := part.(SpeechPart); ok {
					lastSpeech = index
				}
			}
			parts := make([]ResponsePart, 0, len(message.Parts))
			for index, responsePart := range message.Parts {
				speech, ok := responsePart.(SpeechPart)
				if !ok {
					parts = append(parts, responsePart)
					continue
				}
				if speech.Speaker != SpeechSpeakerAssistant {
					return nil, speechSpeakerError("ModelResponse", SpeechSpeakerAssistant, speech.Speaker)
				}
				content := speech.Content()
				if speech.InterruptedAtMS != nil {
					content = appendSpeechLine(content, "[Interrupted after "+strconv.Itoa(*speech.InterruptedAtMS)+" ms]")
				} else if message.State == ModelResponseStateInterrupted && index == lastSpeech {
					content = appendSpeechLine(content, "[Interrupted]")
				}
				if content != "" {
					parts = append(parts, TextPart{Content: content})
				}
			}
			if len(parts) > 0 {
				message.Parts = parts
				prepared = append(prepared, message)
			}
		}
	}
	return prepared, nil
}

func validateResponseSpeech(response *ModelResponse) error {
	if response == nil {
		return nil
	}
	for _, responsePart := range response.Parts {
		if speech, ok := responsePart.(SpeechPart); ok && speech.Speaker != SpeechSpeakerAssistant {
			return speechSpeakerError("ModelResponse", SpeechSpeakerAssistant, speech.Speaker)
		}
	}
	return nil
}

func appendSpeechLine(content, line string) string {
	if content == "" {
		return line
	}
	return content + "\n" + line
}

func speechSpeakerError(message string, expected, actual SpeechSpeaker) error {
	return fmt.Errorf(
		"ai: SpeechPart in %s.Parts must have speaker %q, got %q", message, expected, actual,
	)
}

func cloneSpeechPart(part SpeechPart) SpeechPart {
	part.Transcript = clonePointer(part.Transcript)
	part.InterruptedAtMS = clonePointer(part.InterruptedAtMS)
	if part.Audio != nil {
		audio := cloneUserContents([]UserContent{*part.Audio})[0].(BinaryContent)
		part.Audio = &audio
	}
	part.ProviderDetails = cloneSchemaMap(part.ProviderDetails)
	return part
}
