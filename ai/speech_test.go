package ai_test

import (
	"context"
	"strings"
	"testing"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/models/fakes"
)

func speechPointer[T any](value T) *T { return &value }

func TestSpeechPartContentAndResponseText(t *testing.T) {
	audio := ai.BinaryContent{Data: []byte{1}, MediaType: "audio/pcm"}
	if (ai.SpeechPart{Speaker: ai.SpeechSpeakerUser}).HasContent() {
		t.Fatal("empty speech reported content")
	}
	if !(ai.SpeechPart{Speaker: ai.SpeechSpeakerUser, Audio: &audio}).HasContent() {
		t.Fatal("retained audio did not report content")
	}
	empty := ""
	if (ai.SpeechPart{Speaker: ai.SpeechSpeakerUser, Transcript: &empty}).HasContent() {
		t.Fatal("empty transcript reported content")
	}
	response := ai.ModelResponse{Parts: []ai.ResponsePart{
		ai.SpeechPart{Speaker: ai.SpeechSpeakerAssistant, Transcript: speechPointer("Hello")},
		ai.TextPart{Content: " world"},
		ai.ThinkingPart{Content: "private"},
		ai.SpeechPart{Speaker: ai.SpeechSpeakerAssistant},
		ai.TextPart{Content: "Again"},
	}}
	if response.Text() != "Hello world\n\nAgain" {
		t.Fatalf("unexpected speech text %q", response.Text())
	}
}

func TestPrepareModelMessagesConvertsSpeech(t *testing.T) {
	timestamp := time.Date(2026, 4, 1, 12, 0, 0, 0, time.UTC)
	audio := ai.BinaryContent{
		Data: []byte{1}, MediaType: "audio/pcm", VendorMetadata: map[string]any{"nested": []any{"value"}},
	}
	history := []ai.ModelMessage{
		ai.ModelRequest{Timestamp: timestamp, Parts: []ai.RequestPart{
			ai.SpeechPart{
				Speaker: ai.SpeechSpeakerUser, Transcript: speechPointer("What time is it?"), Audio: &audio,
			},
		}},
		ai.ModelResponse{State: ai.ModelResponseStateInterrupted, Parts: []ai.ResponsePart{
			ai.SpeechPart{Speaker: ai.SpeechSpeakerAssistant, Transcript: speechPointer("It is")},
			ai.TextPart{Content: " noon."},
			ai.SpeechPart{Speaker: ai.SpeechSpeakerAssistant, Transcript: speechPointer("Next")},
		}},
	}
	prepared, err := ai.PrepareModelMessages(fakes.NewTestModel(), history)
	if err != nil {
		t.Fatal(err)
	}
	prompt := prepared[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart)
	if prompt.Content != "What time is it?" || prompt.Timestamp != timestamp {
		t.Fatalf("unexpected speech prompt: %#v", prompt)
	}
	response := prepared[1].(ai.ModelResponse)
	if len(response.Parts) != 3 || response.Parts[0].(ai.TextPart).Content != "It is" ||
		response.Parts[2].(ai.TextPart).Content != "Next\n[Interrupted]" {
		t.Fatalf("unexpected speech response: %#v", response.Parts)
	}
	if _, ok := history[0].(ai.ModelRequest).Parts[0].(ai.SpeechPart); !ok {
		t.Fatal("speech preparation mutated durable history")
	}

	profiled := ai.NewProfiledModel(fakes.NewTestModel(), ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool, SupportsAudioInput: true,
	})
	withAudio, err := ai.PrepareModelMessages(profiled, history)
	if err != nil {
		t.Fatal(err)
	}
	content := withAudio[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Contents[0].(ai.BinaryContent)
	content.Data[0] = 9
	content.VendorMetadata["nested"].([]any)[0] = "changed"
	if audio.Data[0] != 1 || audio.VendorMetadata["nested"].([]any)[0] != "value" {
		t.Fatal("prepared speech audio was not detached")
	}
}

func TestPrepareModelMessagesRendersKnownInterruption(t *testing.T) {
	prepared, err := ai.PrepareModelMessages(fakes.NewTestModel(), []ai.ModelMessage{ai.ModelResponse{
		Parts: []ai.ResponsePart{ai.SpeechPart{
			Speaker: ai.SpeechSpeakerAssistant, InterruptedAtMS: speechPointer(420),
		}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	content := prepared[0].(ai.ModelResponse).Parts[0].(ai.TextPart).Content
	if content != "[Interrupted after 420 ms]" {
		t.Fatalf("unexpected interruption marker %q", content)
	}
}

func TestFallbackPreparesSpeechForSelectedCandidate(t *testing.T) {
	audio := ai.BinaryContent{Data: []byte{1}, MediaType: "audio/pcm"}
	history := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.SpeechPart{
		Speaker: ai.SpeechSpeakerUser, Transcript: speechPointer("fallback transcript"), Audio: &audio,
	}}}}
	var received []ai.ModelMessage
	primary := ai.NewProfiledModel(fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		received = messages
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	}), ai.ModelProfile{DefaultOutputMode: ai.OutputModeTool, SupportsAudioInput: true})
	model := ai.WrapModel(ai.NewFallbackModel(primary))
	if _, err := ai.RequestModel(t.Context(), model, history, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	prompt := received[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart)
	if len(prompt.Contents) != 1 || prompt.Contents[0].(ai.BinaryContent).MediaType != "audio/pcm" {
		t.Fatalf("fallback candidate did not apply its speech profile: %#v", received)
	}
}

func TestPrepareModelMessagesDropsEmptySpeech(t *testing.T) {
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.SpeechPart{Speaker: ai.SpeechSpeakerUser}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.SpeechPart{Speaker: ai.SpeechSpeakerAssistant}}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.UserPromptPart{Content: "keep"},
			ai.SpeechPart{Speaker: ai.SpeechSpeakerUser, Audio: &ai.BinaryContent{
				Data: []byte{1}, MediaType: "audio/pcm",
			}},
		}},
	}
	prepared, err := ai.PrepareModelMessages(fakes.NewTestModel(), history)
	if err != nil {
		t.Fatal(err)
	}
	if len(prepared) != 1 || len(prepared[0].(ai.ModelRequest).Parts) != 1 {
		t.Fatalf("empty speech messages were retained: %#v", prepared)
	}

	plain := []ai.ModelMessage{ai.ModelRequest{Metadata: map[string]any{"values": []any{"original"}}}}
	detached, err := ai.PrepareModelMessages(fakes.NewTestModel(), plain)
	if err != nil {
		t.Fatal(err)
	}
	detached[0].(ai.ModelRequest).Metadata["values"].([]any)[0] = "changed"
	if plain[0].(ai.ModelRequest).Metadata["values"].([]any)[0] != "original" {
		t.Fatal("speech-free history was not detached")
	}
}

func TestPrepareModelMessagesValidatesSpeakers(t *testing.T) {
	tests := []struct {
		name    string
		history []ai.ModelMessage
		want    string
	}{
		{"request", []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{
			ai.SpeechPart{Speaker: ai.SpeechSpeakerAssistant},
		}}}, `ModelRequest.Parts must have speaker "user"`},
		{"response", []ai.ModelMessage{ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.SpeechPart{Speaker: ai.SpeechSpeakerUser},
		}}}, `ModelResponse.Parts must have speaker "assistant"`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := ai.PrepareModelMessages(fakes.NewTestModel(), test.history)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("unexpected speaker error: %v", err)
			}
			if _, err := ai.RequestModel(t.Context(), fakes.NewTestModel(), test.history, ai.ModelRequestParams{}); err == nil {
				t.Fatal("direct request accepted invalid speech history")
			}
			agent := ai.NewAgent[struct{}, string](fakes.NewTestModel())
			if _, err := agent.Run(
				t.Context(), "continue", struct{}{}, ai.WithMessageHistory(test.history),
			); err == nil {
				t.Fatal("agent accepted invalid speech history")
			}
		})
	}
}

func TestSpeechPreparationAppliesToTokenCountingAndCompaction(t *testing.T) {
	history := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.SpeechPart{
		Speaker: ai.SpeechSpeakerUser, Transcript: speechPointer("count this"),
	}}}}
	counter := &countingModel{countUsage: ai.Usage{InputTokens: 2}}
	if _, err := ai.CountModelTokens(t.Context(), counter, history, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	if counter.countMessages[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content != "count this" {
		t.Fatalf("token counting received speech directly: %#v", counter.countMessages)
	}
	compacted := false
	compactor := &explicitCompactionModel{compact: func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		compacted = messages[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content == "count this"
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.CompactionPart{Content: "summary"}}}, nil
	}}
	if _, err := ai.CompactModelMessages(t.Context(), compactor, history, ai.ModelRequestParams{}); err != nil {
		t.Fatal(err)
	}
	if !compacted {
		t.Fatal("compaction received speech directly")
	}

	invalid := []ai.ModelMessage{ai.ModelRequest{Parts: []ai.RequestPart{ai.SpeechPart{
		Speaker: ai.SpeechSpeakerAssistant,
	}}}}
	if _, err := ai.CountModelTokens(t.Context(), counter, invalid, ai.ModelRequestParams{}); err == nil {
		t.Fatal("token counting accepted invalid speech")
	}
	if _, err := ai.CompactModelMessages(t.Context(), compactor, invalid, ai.ModelRequestParams{}); err == nil {
		t.Fatal("compaction accepted invalid speech")
	}
}

func TestAgentConvertsSpeechHistory(t *testing.T) {
	var received []ai.ModelMessage
	model := fakes.NewFunctionModel(func(
		_ context.Context, messages []ai.ModelMessage, _ ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		received = messages
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.TextPart{Content: "done"}}}, nil
	})
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.SpeechPart{
			Speaker: ai.SpeechSpeakerUser, Transcript: speechPointer("Hello"),
		}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{ai.SpeechPart{
			Speaker: ai.SpeechSpeakerAssistant, Transcript: speechPointer("Hi!"),
		}}},
	}
	agent := ai.NewAgent[struct{}, string](model)
	result, err := agent.Run(t.Context(), "Continue", struct{}{}, ai.WithMessageHistory(history))
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "done" || len(received) != 3 ||
		received[0].(ai.ModelRequest).Parts[0].(ai.UserPromptPart).Content != "Hello" ||
		received[1].(ai.ModelResponse).Parts[0].(ai.TextPart).Content != "Hi!" {
		t.Fatalf("speech history was not prepared: %#v", received)
	}
	if _, ok := result.Messages()[0].(ai.ModelRequest).Parts[0].(ai.SpeechPart); !ok {
		t.Fatal("durable result history lost speech part")
	}
}
