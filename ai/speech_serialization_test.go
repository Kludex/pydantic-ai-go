package ai_test

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func TestSpeechMessageSerialization(t *testing.T) {
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.SpeechPart{
			Speaker: ai.SpeechSpeakerUser, Transcript: speechPointer("Hello"),
			Audio: &ai.BinaryContent{Data: []byte{1, 2}, MediaType: "audio/pcm", Identifier: "0ca623"},
			ID:    "item-1", ProviderName: "openai", ProviderDetails: map[string]any{"sequence": float64(1)},
		}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.SpeechPart{
			Speaker: ai.SpeechSpeakerAssistant, Transcript: speechPointer("Hi!"),
			InterruptedAtMS: speechPointer(420),
		}}},
	}
	encoded, err := ai.MarshalMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := ai.UnmarshalMessages(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, messages) {
		t.Fatalf("speech round trip changed messages:\n%s\n%#v", encoded, decoded)
	}
	decodedSpeech := decoded[0].(ai.ModelRequest).Parts[0].(ai.SpeechPart)
	decodedSpeech.Audio.Data[0] = 9
	decodedSpeech.ProviderDetails["sequence"] = float64(2)
	original := messages[0].(ai.ModelRequest).Parts[0].(ai.SpeechPart)
	if original.Audio.Data[0] != 1 || original.ProviderDetails["sequence"] != float64(1) {
		t.Fatal("decoded speech shared mutable state")
	}

	upstream, err := os.ReadFile("testdata/messages/upstream_speech.json")
	if err != nil {
		t.Fatal(err)
	}
	fromUpstream, err := ai.UnmarshalMessages(upstream)
	if err != nil {
		t.Fatal(err)
	}
	if len(fromUpstream) != 2 || fromUpstream[1].(ai.ModelResponse).Text() != "Hi!" {
		t.Fatalf("unexpected upstream speech fixture: %#v", fromUpstream)
	}
	goJSON, err := ai.MarshalMessages(fromUpstream)
	if err != nil {
		t.Fatal(err)
	}
	var normalized any
	if err := json.Unmarshal(goJSON, &normalized); err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(goJSON, []byte(`"part_kind":"speech"`)) ||
		!bytes.Contains(goJSON, []byte(`"data":"AQI="`)) {
		t.Fatalf("Go speech encoding is incompatible: %s", goJSON)
	}
}

func TestSpeechMessageSerializationValidatesSpeakers(t *testing.T) {
	tests := []struct {
		message ai.ModelMessage
		want    string
	}{
		{ai.ModelRequest{Parts: []ai.RequestPart{ai.SpeechPart{
			Speaker: ai.SpeechSpeakerAssistant,
		}}}, "ModelRequest"},
		{ai.ModelResponse{Parts: []ai.ResponsePart{ai.SpeechPart{
			Speaker: ai.SpeechSpeakerUser,
		}}}, "ModelResponse"},
	}
	for _, test := range tests {
		if _, err := ai.MarshalMessages([]ai.ModelMessage{test.message}); err == nil ||
			!strings.Contains(err.Error(), test.want) {
			t.Fatalf("unexpected marshal speaker error: %v", err)
		}
	}
	for _, input := range []string{
		`[{"kind":"request","parts":[{"part_kind":"speech","speaker":"assistant"}]}]`,
		`[{"kind":"response","parts":[{"part_kind":"speech","speaker":"user"}]}]`,
	} {
		if _, err := ai.UnmarshalMessages([]byte(input)); err == nil || !strings.Contains(err.Error(), "SpeechPart") {
			t.Fatalf("unexpected unmarshal speaker error: %v", err)
		}
	}
}

func TestSanitizeSpeechMessagesDetachesAudio(t *testing.T) {
	audio := ai.BinaryContent{
		Data: []byte{1}, MediaType: "audio/pcm", VendorMetadata: map[string]any{"items": []any{"original"}},
	}
	providerDetails := map[string]any{"items": []any{"original"}}
	messages := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.SpeechPart{
			Speaker: ai.SpeechSpeakerUser, Audio: &audio, ProviderDetails: providerDetails,
		}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.SpeechPart{
			Speaker: ai.SpeechSpeakerAssistant, Audio: &audio, ProviderDetails: providerDetails,
		}}},
	}
	sanitized, _, err := ai.SanitizeMessages(messages, ai.MessageSanitizationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	request := sanitized[0].(ai.ModelRequest).Parts[0].(ai.SpeechPart)
	response := sanitized[1].(ai.ModelResponse).Parts[0].(ai.SpeechPart)
	request.Audio.Data[0], response.Audio.Data[0] = 2, 3
	request.ProviderDetails["items"].([]any)[0] = "request"
	response.ProviderDetails["items"].([]any)[0] = "response"
	if audio.Data[0] != 1 || providerDetails["items"].([]any)[0] != "original" {
		t.Fatal("sanitized speech shared mutable state")
	}
}
