package realtime

import (
	"slices"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func (session *Session) finishUserLocked(active *activeSpeech) {
	active.finished = true
	if active.audio == nil {
		active.audio = session.inputAudio
		session.inputAudio = nil
	}
	part := session.userSpeechLocked(active)
	session.publishLocked(ai.PartEndEvent{Index: active.index, PartID: active.partID, Part: part})
	if active.itemID == "" {
		session.anonymousUser = nil
	} else {
		delete(session.userTurns, active.itemID)
	}
	session.flushUsersLocked(false)
	if active.afterResponse {
		time.AfterFunc(5*time.Second, func() {
			session.mu.Lock()
			session.flushUsersLocked(true)
			session.mu.Unlock()
		})
	}
}

func (session *Session) userSpeechLocked(active *activeSpeech) ai.SpeechPart {
	part := ai.SpeechPart{Speaker: ai.SpeechSpeakerUser}
	if active.transcript != "" {
		text := active.transcript
		part.Transcript = &text
	}
	if len(active.audio) > 0 && session.config.retainAudioMax != 0 {
		part.Audio = &ai.BinaryContent{Data: pcmToWAV(active.audio, session.profile.AudioInputSampleRate), MediaType: "audio/wav"}
	}
	return part
}

func (session *Session) flushUsersLocked(force bool) {
	for len(session.userOrder) > 0 {
		active := session.userOrder[0]
		if !active.finished || (active.afterResponse || active.commitHeld) && !force {
			return
		}
		session.userOrder = session.userOrder[1:]
		position := min(active.position, len(session.history))
		request := ai.ModelRequest{Parts: []ai.RequestPart{session.userSpeechLocked(active)},
			Timestamp: active.timestamp, ConversationID: session.conversationID}
		session.history = slices.Insert(session.history, position, ai.ModelMessage(request))
		for _, pending := range session.userOrder {
			if pending.position >= position {
				pending.position++
			}
		}
	}
}

func requestsAsMessages(requests []ai.ModelRequest) []ai.ModelMessage {
	messages := make([]ai.ModelMessage, len(requests))
	for index, request := range requests {
		messages[index] = request
	}
	return messages
}

func (session *Session) boundAudioLocked() {
	if session.config.retainAudioMax < 0 {
		return
	}
	inputRate := int64(session.profile.AudioInputSampleRate)
	outputRate := int64(session.profile.AudioOutputSampleRate)
	maximum := int64(session.config.retainAudioMax.Seconds() * float64(2*inputRate*outputRate))
	weight := int64(len(session.inputAudio)) * outputRate
	if session.activeAssistant != nil {
		weight += int64(len(session.activeAssistant.audio)) * inputRate
	}
	for _, active := range session.userOrder {
		weight += int64(len(active.audio)) * outputRate
	}
	count := func(part ai.SpeechPart) int64 {
		if part.Audio == nil {
			return 0
		}
		rate := inputRate
		if part.Speaker == ai.SpeechSpeakerAssistant {
			rate = outputRate
		}
		return int64(max(0, len(part.Audio.Data)-44)) * (inputRate * outputRate / rate)
	}
	for _, message := range session.history {
		switch message := message.(type) {
		case ai.ModelRequest:
			for _, part := range message.Parts {
				if speech, ok := part.(ai.SpeechPart); ok {
					weight += count(speech)
				}
			}
		case ai.ModelResponse:
			for _, part := range message.Parts {
				if speech, ok := part.(ai.SpeechPart); ok {
					weight += count(speech)
				}
			}
		}
	}
	for _, part := range session.responseParts {
		if speech, ok := part.(ai.SpeechPart); ok {
			weight += count(speech)
		}
	}
	if weight <= maximum {
		return
	}
	strip := func(part ai.SpeechPart) ai.SpeechPart {
		if weight > maximum {
			weight -= count(part)
			part.Audio = nil
		}
		return part
	}
	for index, message := range session.history {
		switch message := message.(type) {
		case ai.ModelRequest:
			message.Parts = slices.Clone(message.Parts)
			for index, part := range message.Parts {
				if speech, ok := part.(ai.SpeechPart); ok {
					message.Parts[index] = strip(speech)
				}
			}
			session.history[index] = message
		case ai.ModelResponse:
			message.Parts = slices.Clone(message.Parts)
			for index, part := range message.Parts {
				if speech, ok := part.(ai.SpeechPart); ok {
					message.Parts[index] = strip(speech)
				}
			}
			session.history[index] = message
		}
	}
	for index, part := range session.responseParts {
		if speech, ok := part.(ai.SpeechPart); ok {
			session.responseParts[index] = strip(speech)
		}
	}
	trim := func(buffer []byte, byteWeight int64) []byte {
		excess := weight - maximum
		if excess <= 0 {
			return buffer
		}
		drop := min(int((excess+2*byteWeight-1)/(2*byteWeight))*2, len(buffer))
		weight -= int64(drop) * byteWeight
		return slices.Clone(buffer[drop:])
	}
	for _, active := range session.userOrder {
		active.audio = trim(active.audio, outputRate)
	}
	session.inputAudio = trim(session.inputAudio, outputRate)
	if session.activeAssistant != nil {
		session.activeAssistant.audio = trim(session.activeAssistant.audio, inputRate)
	}
}
