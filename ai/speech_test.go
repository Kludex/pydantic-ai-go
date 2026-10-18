package ai_test

import (
	"context"
	"strconv"
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

func preparedToolCallIDs(messages []ai.ModelMessage) []string {
	var ids []string
	for _, message := range messages {
		switch message := message.(type) {
		case ai.ModelRequest:
			for _, part := range message.Parts {
				switch part := part.(type) {
				case ai.ToolReturnPart:
					ids = append(ids, "return:"+part.ToolCallID)
				case ai.RetryPromptPart:
					ids = append(ids, "retry:"+part.ToolCallID)
				}
			}
		case ai.ModelResponse:
			for _, part := range message.Parts {
				if call, ok := part.(ai.ToolCallPart); ok {
					ids = append(ids, "call:"+call.ToolCallID)
				}
			}
		}
	}
	return ids
}

func TestPrepareModelMessagesOrdersToolResultsByCall(t *testing.T) {
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "go"}}},
		ai.ModelResponse{Parts: []ai.ResponsePart{
			ai.ToolCallPart{ToolName: "a", ToolCallID: "ca"},
			ai.ToolCallPart{ToolName: "b", ToolCallID: "cb"},
			ai.ToolCallPart{ToolName: "c", ToolCallID: "cc"},
		}},
		ai.ModelRequest{Parts: []ai.RequestPart{
			ai.ToolReturnPart{ToolName: "c", ToolCallID: "cc", Content: "C"},
			ai.RetryPromptPart{Content: "output invalid"},
			ai.RetryPromptPart{ToolName: "b", ToolCallID: "cb", Content: "bad args"},
			ai.ToolReturnPart{ToolName: "z", ToolCallID: "not-a-call", Content: "Z"},
			ai.ToolReturnPart{ToolName: "a", ToolCallID: "ca", Content: "A"},
			ai.UserPromptPart{Content: "file from c"},
		}},
	}

	prepared, err := ai.PrepareModelMessages(fakes.NewTestModel(), history)
	if err != nil {
		t.Fatal(err)
	}

	var got []string
	for _, part := range prepared[2].(ai.ModelRequest).Parts {
		switch part := part.(type) {
		case ai.ToolReturnPart:
			got = append(got, part.Content.(string))
		case ai.RetryPromptPart:
			got = append(got, part.Content)
		case ai.UserPromptPart:
			got = append(got, part.Content)
		}
	}
	want := []string{"A", "output invalid", "bad args", "Z", "C", "file from c"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("prepared order = %q, want %q", got, want)
	}
	if first := history[2].(ai.ModelRequest).Parts[0].(ai.ToolReturnPart); first.ToolCallID != "cc" {
		t.Fatalf("stored history was reordered: %#v", first)
	}
}

func TestPrepareModelMessagesFitsToolCallIDsToProfileLength(t *testing.T) {
	ids := []string{"aB3dE5gH9", "aB3dE5gH", "ab_cd_efg", "aB3dE5gH9X", "aB3dE5gHé", "call_abc123def456ghi789jkl012"}
	response := ai.ModelResponse{}
	request := ai.ModelRequest{}
	for index, id := range ids {
		response.Parts = append(response.Parts, ai.ToolCallPart{ToolName: "t", ToolCallID: id})
		if index == 1 {
			request.Parts = append(request.Parts, ai.RetryPromptPart{ToolName: "t", ToolCallID: id, Content: "bad"})
			continue
		}
		request.Parts = append(request.Parts, ai.ToolReturnPart{ToolName: "t", ToolCallID: id})
	}
	request.Parts = append(request.Parts, ai.RetryPromptPart{Content: "output invalid"})
	history := []ai.ModelMessage{
		ai.ModelRequest{Parts: []ai.RequestPart{ai.UserPromptPart{Content: "go"}}}, response, request,
	}
	model := ai.NewProfiledModel(fakes.NewTestModel(), ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool, ToolCallIDLength: 9,
	})

	prepared, err := ai.PrepareModelMessages(model, history)
	if err != nil {
		t.Fatal(err)
	}

	want := []string{
		"call:aB3dE5gH9", "call:BsnFb66F9", "call:5ZoXBIVHV", "call:BgySIGZn7", "call:pFwy2hHZJ", "call:2pESUE4jw",
		"return:aB3dE5gH9", "retry:BsnFb66F9", "return:5ZoXBIVHV", "return:BgySIGZn7", "return:pFwy2hHZJ",
		"return:2pESUE4jw", "retry:",
	}
	if got := preparedToolCallIDs(prepared); strings.Join(got, " ") != strings.Join(want, " ") {
		t.Fatalf("prepared IDs = %q, want %q", got, want)
	}
	again, err := ai.PrepareModelMessages(model, history)
	if err != nil || strings.Join(preparedToolCallIDs(again), " ") != strings.Join(want, " ") {
		t.Fatalf("second preparation = %q, %v", preparedToolCallIDs(again), err)
	}
	if got := preparedToolCallIDs(history)[0]; got != "call:aB3dE5gH9" {
		t.Fatalf("stored history changed: %q", preparedToolCallIDs(history))
	}
	if got := preparedToolCallIDs(history)[1]; got != "call:aB3dE5gH" {
		t.Fatalf("stored history changed: %q", preparedToolCallIDs(history))
	}
}

func TestPrepareModelMessagesResolvesToolCallIDCollisionsInHistoryOrder(t *testing.T) {
	exchange := func(name, id string) []ai.ModelMessage {
		return []ai.ModelMessage{
			ai.ModelResponse{Parts: []ai.ResponsePart{ai.ToolCallPart{ToolName: name, ToolCallID: id}}},
			ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{ToolName: name, ToolCallID: id}}},
		}
	}
	model := ai.NewProfiledModel(fakes.NewTestModel(), ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool, ToolCallIDLength: 9,
	})
	for name, test := range map[string]struct {
		history []ai.ModelMessage
		want    string
	}{
		"foreign first": {
			append(exchange("a", "call_x"), exchange("b", "nVlA23dIQ")...),
			"call:nVlA23dIQ return:nVlA23dIQ call:Q30Sez398 return:Q30Sez398",
		},
		"fitting first": {
			append(exchange("b", "nVlA23dIQ"), exchange("a", "call_x")...),
			"call:nVlA23dIQ return:nVlA23dIQ call:heCJTQ2G1 return:heCJTQ2G1",
		},
	} {
		t.Run(name, func(t *testing.T) {
			prepared, err := ai.PrepareModelMessages(model, test.history)
			if err != nil {
				t.Fatal(err)
			}
			if got := strings.Join(preparedToolCallIDs(prepared), " "); got != test.want {
				t.Fatalf("prepared IDs = %q, want %q", got, test.want)
			}
		})
	}
}

func TestPrepareModelMessagesRejectsToolCallIDLengthWithoutRoom(t *testing.T) {
	calls := func(count int) ai.ModelResponse {
		response := ai.ModelResponse{}
		for index := range count {
			response.Parts = append(response.Parts, ai.ToolCallPart{ToolName: "a", ToolCallID: "call_" + strconv.Itoa(index)})
		}
		return response
	}
	model := ai.NewProfiledModel(fakes.NewTestModel(), ai.ModelProfile{
		DefaultOutputMode: ai.OutputModeTool, ToolCallIDLength: 1,
	})
	for name, history := range map[string][]ai.ModelMessage{
		"call":   {calls(63)},
		"result": {calls(62), ai.ModelRequest{Parts: []ai.RequestPart{ai.ToolReturnPart{ToolName: "a", ToolCallID: "x_"}}}},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := ai.PrepareModelMessages(model, history)
			if err == nil || !strings.Contains(err.Error(), "more distinct tool call IDs than fit in 1 alphanumeric") {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
