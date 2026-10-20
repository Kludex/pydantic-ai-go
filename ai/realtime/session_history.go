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
		speech := session.userSpeechLocked(active)
		if speech.Audio != nil {
			session.retainAudioLocked(speech.Audio, int64(len(active.audio))*int64(session.profile.AudioOutputSampleRate))
		}
		request := ai.ModelRequest{Parts: []ai.RequestPart{speech},
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

// retainedAudio is a SpeechPart's audio counted against the retention budget.
// Its weight is in a unit both sample rates divide: an input byte weighs the
// output rate and an output byte the input rate.
type retainedAudio struct {
	audio  *ai.BinaryContent
	weight int64
}

func (session *Session) retainAudioLocked(audio *ai.BinaryContent, weight int64) {
	if session.config.retainAudioMax < 0 {
		return
	}
	session.retainedAudio = append(session.retainedAudio, retainedAudio{audio: audio, weight: weight})
	session.retainedAudioWeight += weight
}

// boundAudioLocked keeps retained audio within the budget. A running total of
// recorded audio keeps a frame within budget from walking the history; over
// budget, recorded audio goes oldest first, then the oldest buffered audio.
func (session *Session) boundAudioLocked() {
	if session.config.retainAudioMax < 0 {
		return
	}
	inputRate := int64(session.profile.AudioInputSampleRate)
	outputRate := int64(session.profile.AudioOutputSampleRate)
	maximum := int64(session.config.retainAudioMax.Seconds() * float64(2*inputRate*outputRate))
	weight := session.retainedAudioWeight + int64(len(session.inputAudio))*outputRate
	if session.activeAssistant != nil {
		weight += int64(len(session.activeAssistant.audio)) * inputRate
	}
	for _, active := range session.userOrder {
		weight += int64(len(active.audio)) * outputRate
	}
	for weight > maximum && len(session.retainedAudio) > 0 {
		oldest := session.retainedAudio[0]
		session.retainedAudio = session.retainedAudio[1:]
		session.retainedAudioWeight -= oldest.weight
		weight -= oldest.weight
		session.stripAudioLocked(oldest.audio)
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

// stripAudioLocked drops audio from the part that holds it, keeping its
// transcript. Messages are replaced rather than edited, so snapshots already
// handed out do not change.
func (session *Session) stripAudioLocked(audio *ai.BinaryContent) {
	if index := strippedAudioIndex(session.responseParts, audio); index >= 0 {
		session.responseParts[index] = stripSpeechAudio(session.responseParts[index])
		return
	}
	for index, message := range session.history {
		switch message := message.(type) {
		case ai.ModelRequest:
			if position := strippedAudioIndex(message.Parts, audio); position >= 0 {
				message.Parts = slices.Clone(message.Parts)
				message.Parts[position] = stripSpeechAudio(message.Parts[position])
				session.history[index] = message
				return
			}
		case ai.ModelResponse:
			if position := strippedAudioIndex(message.Parts, audio); position >= 0 {
				message.Parts = slices.Clone(message.Parts)
				message.Parts[position] = stripSpeechAudio(message.Parts[position])
				session.history[index] = message
				return
			}
		}
	}
}

func strippedAudioIndex[Part any](parts []Part, audio *ai.BinaryContent) int {
	return slices.IndexFunc(parts, func(part Part) bool {
		speech, ok := any(part).(ai.SpeechPart)
		return ok && speech.Audio == audio
	})
}

func stripSpeechAudio[Part any](part Part) Part {
	speech := any(part).(ai.SpeechPart)
	speech.Audio = nil
	return any(speech).(Part)
}
