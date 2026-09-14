package ai

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// PrepareModelMessages converts realtime speech history into content a standard model can consume.
// It returns a detached history and leaves durable SpeechPart values unchanged. Composite
// dispatchers may defer conversion so each selected child applies its own profile.
func PrepareModelMessages(model Model, messages []ModelMessage) ([]ModelMessage, error) {
	return prepareModelMessages(model, messages, nil)
}

func prepareModelMessages(
	model Model, messages []ModelMessage, params *ModelRequestParams,
) ([]ModelMessage, error) {
	messages = cloneModelMessages(messages)
	if dispatcher, ok := model.(ModelMessageProfileDispatcher); ok && dispatcher.DispatchesMessageProfile() {
		return messages, nil
	}
	profile := modelProfile(model)
	messages, err := convertSpeechMessages(messages, profile.SupportsAudioInput)
	if err != nil {
		return nil, err
	}
	supportsToolAvailability := profile.SupportsToolAvailabilityDelta
	if params != nil {
		if provider, ok := model.(ToolAvailabilityDeltaModel); ok {
			supportsToolAvailability = provider.SupportsToolAvailabilityDelta(*params)
		}
	}
	if supportsToolAvailability {
		return messages, nil
	}
	return synthesizeToolAvailabilityMessages(messages, params)
}

func synthesizeToolAvailabilityMessages(
	messages []ModelMessage, params *ModelRequestParams,
) ([]ModelMessage, error) {
	var allowed map[string]struct{}
	if params != nil {
		allowed = make(map[string]struct{}, len(params.DeferredTools))
		for _, tool := range params.DeferredTools {
			if tool.DeferLoading {
				allowed[tool.Name] = struct{}{}
			}
		}
	}
	used := make(map[string]struct{})
	for _, message := range messages {
		switch message := message.(type) {
		case ModelRequest:
			for _, part := range message.Parts {
				switch part := part.(type) {
				case ToolReturnPart:
					used[part.ToolCallID] = struct{}{}
				case RetryPromptPart:
					used[part.ToolCallID] = struct{}{}
				}
			}
		case ModelResponse:
			for _, part := range message.Parts {
				switch part := part.(type) {
				case ToolCallPart:
					used[part.ToolCallID] = struct{}{}
				case NativeToolCallPart:
					used[part.ToolCallID] = struct{}{}
				case NativeToolReturnPart:
					used[part.ToolCallID] = struct{}{}
				}
			}
		}
	}

	changed := false
	ordinal := 0
	prepared := make([]ModelMessage, 0, len(messages))
	for _, message := range messages {
		request, ok := message.(ModelRequest)
		if !ok || !requestHasToolAvailabilityDelta(request) {
			prepared = append(prepared, message)
			continue
		}
		changed = true
		boundary := len(request.Parts)
		for index, part := range request.Parts {
			if _, delta := part.(ToolAvailabilityDeltaPart); delta || isToolResultPart(part) {
				continue
			}
			boundary = index
			break
		}
		parts := make([]RequestPart, 0, len(request.Parts))
		for _, part := range request.Parts[:boundary] {
			if _, delta := part.(ToolAvailabilityDeltaPart); !delta {
				parts = append(parts, part)
			}
		}
		for _, part := range request.Parts[:boundary] {
			if _, delta := part.(ToolAvailabilityDeltaPart); delta {
				parts = append(parts, part)
			}
		}
		parts = append(parts, request.Parts[boundary:]...)

		var pending []RequestPart
		flush := func() {
			if len(pending) == 0 {
				return
			}
			copy := request
			copy.Parts = append([]RequestPart(nil), pending...)
			prepared = append(prepared, copy)
			pending = nil
		}
		for _, part := range parts {
			delta, ok := part.(ToolAvailabilityDeltaPart)
			if !ok {
				pending = append(pending, part)
				continue
			}
			names := make([]string, 0, len(delta.ToolsAdded))
			for _, name := range delta.ToolsAdded {
				if allowed != nil {
					if _, ok := allowed[name]; !ok {
						continue
					}
				}
				names = append(names, name)
			}
			if len(names) == 0 {
				continue
			}
			callID := delta.ToolCallID
			if _, duplicate := used[callID]; callID == "" || duplicate {
				for {
					hash := sha256.Sum256([]byte(strconv.Itoa(ordinal) + "\x00" + strings.Join(names, "\x00")))
					ordinal++
					callID = "tool-call-" + hex.EncodeToString(hash[:8])
					if _, exists := used[callID]; !exists {
						break
					}
				}
			}
			used[callID] = struct{}{}
			arguments, _ := json.Marshal(map[string]any{"queries": names})
			flush()
			prepared = append(prepared, ModelResponse{Parts: []ResponsePart{ToolCallPart{
				ToolName: ToolSearchName, ToolCallID: callID, ToolKind: ToolPartKindToolSearch, Args: arguments,
			}}})
			matches := make([]ToolSearchMatch, len(names))
			for index, name := range names {
				matches[index] = ToolSearchMatch{Name: name}
			}
			pending = append(pending, ToolReturnPart{
				ToolName: ToolSearchName, ToolCallID: callID, ToolKind: ToolPartKindToolSearch,
				Content: ToolSearchResult{DiscoveredTools: matches},
			})
		}
		flush()
	}
	if !changed {
		return messages, nil
	}
	return prepared, nil
}

func requestHasToolAvailabilityDelta(request ModelRequest) bool {
	for _, part := range request.Parts {
		if _, ok := part.(ToolAvailabilityDeltaPart); ok {
			return true
		}
	}
	return false
}

func isToolResultPart(part RequestPart) bool {
	switch part := part.(type) {
	case ToolReturnPart:
		return true
	case RetryPromptPart:
		return part.ToolName != ""
	default:
		return false
	}
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
