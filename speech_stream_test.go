package ai_test

import (
	"bytes"
	"context"
	"iter"
	"strings"
	"testing"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/models/fakes"
)

func TestSpeechPartDelta(t *testing.T) {
	audio := ai.BinaryContent{Data: []byte{1}, MediaType: "audio/pcm"}
	part := ai.SpeechPart{
		Speaker: ai.SpeechSpeakerAssistant, Transcript: speechPointer("Hello"), Audio: &audio,
	}
	appliedPart, err := (ai.SpeechPartDelta{
		Speaker: ai.SpeechSpeakerAssistant, TranscriptDelta: " there", AudioChunk: []byte{2},
	}).Apply(part)
	if err != nil {
		t.Fatal(err)
	}
	applied := appliedPart.(ai.SpeechPart)
	if applied.Content() != "Hello there" || !bytes.Equal(applied.Audio.Data, []byte{1, 2}) {
		t.Fatalf("unexpected applied speech: %#v", applied)
	}
	if part.Content() != "Hello" || !bytes.Equal(part.Audio.Data, []byte{1}) {
		t.Fatal("speech delta mutated input")
	}

	revisedPart, err := (ai.SpeechPartDelta{Transcript: speechPointer("Hello, my name is")}).Apply(applied)
	if err != nil || revisedPart.(ai.SpeechPart).Content() != "Hello, my name is" {
		t.Fatalf("whole transcript did not replace: %#v, %v", revisedPart, err)
	}
	startedPart, err := (ai.SpeechPartDelta{TranscriptDelta: "Start", AudioChunk: []byte{9}}).Apply(
		ai.SpeechPart{Speaker: ai.SpeechSpeakerUser},
	)
	if err != nil || startedPart.(ai.SpeechPart).Content() != "Start" || startedPart.(ai.SpeechPart).Audio != nil {
		t.Fatalf("unexpected unretained speech delta: %#v, %v", startedPart, err)
	}
	if _, err := (ai.SpeechPartDelta{}).Apply(ai.TextPart{}); err == nil ||
		!strings.Contains(err.Error(), "cannot apply SpeechPartDelta") {
		t.Fatalf("unexpected speech delta type error: %v", err)
	}
}

func TestStreamModelAccumulatesSpeech(t *testing.T) {
	audio := ai.BinaryContent{MediaType: "audio/pcm", VendorMetadata: map[string]any{"source": "test"}}
	model := speechTestStreamingModel(&audio)
	stream := ai.StreamModel(t.Context(), model, nil, ai.ModelRequestParams{AllowText: true})
	var starts, deltas, finals int
	for event, err := range stream.Events() {
		if err != nil {
			t.Fatal(err)
		}
		switch event := event.(type) {
		case ai.PartStartEvent:
			starts++
			speech := event.Part.(ai.SpeechPart)
			if speech.Content() != "Hel" || !bytes.Equal(speech.Audio.Data, []byte{1}) {
				t.Fatalf("unexpected speech start: %#v", speech)
			}
		case ai.PartDeltaEvent:
			deltas++
			if _, ok := event.Delta.(ai.SpeechPartDelta); !ok {
				t.Fatalf("unexpected speech delta %T", event.Delta)
			}
		case ai.FinalResultEvent:
			finals++
		}
	}
	response := stream.Response()
	speech := response.Parts[0].(ai.SpeechPart)
	if starts != 1 || deltas != 1 || finals != 1 || speech.Content() != "Hello" ||
		!bytes.Equal(speech.Audio.Data, []byte{1, 2}) || speech.Audio.VendorMetadata["source"] != "test" {
		t.Fatalf(
			"unexpected accumulated speech: starts=%d deltas=%d finals=%d response=%#v",
			starts, deltas, finals, response,
		)
	}
}

func TestSpeechStreamErrorsAndConsumerStops(t *testing.T) {
	kindMismatch := streamingRequestModel{
		Model: requestModel{name: "mismatch", request: func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			return nil, nil
		}},
		stream: func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (iter.Seq2[ai.ModelStreamEvent, error], error) {
			return func(yield func(ai.ModelStreamEvent, error) bool) {
				yield(ai.TextDeltaEvent{PartID: "same", Delta: "text"}, nil)
				yield(ai.SpeechDeltaEvent{PartID: "same"}, nil)
			}, nil
		},
	}
	stream := ai.StreamModel(t.Context(), kindMismatch, nil, ai.ModelRequestParams{})
	var mismatchErr error
	for _, err := range stream.Events() {
		if err != nil {
			mismatchErr = err
		}
	}
	if mismatchErr == nil || !strings.Contains(mismatchErr.Error(), "changed from text to speech") {
		t.Fatalf("unexpected speech kind mismatch: %v", mismatchErr)
	}

	for _, stopAtDelta := range []bool{false, true} {
		audio := ai.BinaryContent{MediaType: "audio/pcm"}
		stream := ai.StreamModel(t.Context(), speechTestStreamingModel(&audio), nil, ai.ModelRequestParams{})
		for event, err := range stream.Events() {
			if err != nil {
				t.Fatal(err)
			}
			_, delta := event.(ai.PartDeltaEvent)
			if delta == stopAtDelta {
				break
			}
		}
		if stream.Err() != nil {
			t.Fatalf("stopped speech stream failed: %v", stream.Err())
		}
	}

	static := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.SpeechPart{
			Speaker: ai.SpeechSpeakerAssistant, Transcript: speechPointer("static"),
		}}}, nil
	})
	staticStream := ai.StreamModel(t.Context(), static, nil, ai.ModelRequestParams{})
	for range staticStream.Events() {
		break
	}
	if staticStream.Err() != nil {
		t.Fatalf("stopped replayed speech failed: %v", staticStream.Err())
	}
}

func speechTestStreamingModel(audio *ai.BinaryContent) streamingRequestModel {
	return streamingRequestModel{
		Model: requestModel{name: "speech", request: func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (*ai.ModelResponse, error) {
			return nil, nil
		}},
		stream: func(
			context.Context, []ai.ModelMessage, ai.ModelRequestParams,
		) (iter.Seq2[ai.ModelStreamEvent, error], error) {
			return func(yield func(ai.ModelStreamEvent, error) bool) {
				if !yield(ai.SpeechDeltaEvent{
					PartID: "speech", Part: ai.SpeechPart{
						Transcript: speechPointer("Hel"), Audio: audio,
					},
					Delta: ai.SpeechPartDelta{Speaker: ai.SpeechSpeakerAssistant, AudioChunk: []byte{1}},
				}, nil) {
					return
				}
				if !yield(ai.SpeechDeltaEvent{
					PartID: "speech", Delta: ai.SpeechPartDelta{
						Speaker: ai.SpeechSpeakerAssistant, Transcript: speechPointer("Hello"), AudioChunk: []byte{2},
					},
				}, nil) {
					return
				}
				yield(ai.FinishEvent{}, nil)
			}, nil
		},
	}
}

func TestSpeechResponseWorksAsAgentTextOutput(t *testing.T) {
	model := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.SpeechPart{
			Speaker: ai.SpeechSpeakerAssistant, Transcript: speechPointer("spoken answer"),
		}}}, nil
	})
	agent := ai.NewAgent[struct{}, string](model)
	result, err := agent.Run(t.Context(), "speak", struct{}{})
	if err != nil || result.Output != "spoken answer" {
		t.Fatalf("unexpected speech output: %#v, %v", result, err)
	}
	streamed := agent.RunStream(t.Context(), "speak", struct{}{})
	var outputs []string
	for output, err := range streamed.Outputs() {
		if err != nil {
			t.Fatal(err)
		}
		outputs = append(outputs, output)
	}
	if len(outputs) == 0 || outputs[0] != "spoken answer" {
		t.Fatalf("speech stream did not yield text output: %v", outputs)
	}

	invalid := fakes.NewFunctionModel(func(
		context.Context, []ai.ModelMessage, ai.ModelRequestParams,
	) (*ai.ModelResponse, error) {
		return &ai.ModelResponse{Parts: []ai.ResponsePart{ai.SpeechPart{Speaker: ai.SpeechSpeakerUser}}}, nil
	})
	if _, err := ai.RequestModel(t.Context(), invalid, nil, ai.ModelRequestParams{}); err == nil ||
		!strings.Contains(err.Error(), "ModelResponse.Parts") {
		t.Fatalf("unexpected invalid speech response error: %v", err)
	}
	if _, err := ai.NewAgent[struct{}, string](invalid).Run(t.Context(), "speak", struct{}{}); err == nil ||
		!strings.Contains(err.Error(), "ModelResponse.Parts") {
		t.Fatalf("agent accepted invalid speech response: %v", err)
	}
}
