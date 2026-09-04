package anthropic_test

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/models/anthropic"
)

type legacyEventStream struct {
	events     chan types.ResponseStream
	err        error
	closeCount atomic.Int32
}

func newLegacyEventStream(events ...types.ResponseStream) *legacyEventStream {
	stream := &legacyEventStream{events: make(chan types.ResponseStream, len(events))}
	for _, event := range events {
		stream.events <- event
	}
	close(stream.events)
	return stream
}

func (stream *legacyEventStream) Events() <-chan types.ResponseStream { return stream.events }
func (stream *legacyEventStream) Close() error {
	stream.closeCount.Add(1)
	return nil
}
func (stream *legacyEventStream) Err() error { return stream.err }

type valueLegacyEventStream struct{ events <-chan types.ResponseStream }

func (stream valueLegacyEventStream) Events() <-chan types.ResponseStream { return stream.events }
func (valueLegacyEventStream) Close() error                               { return nil }
func (valueLegacyEventStream) Err() error                                 { return nil }

type streamingLegacyBedrockClient struct {
	*legacyBedrockClient
	stream func(*bedrockruntime.InvokeModelWithResponseStreamInput) (anthropic.LegacyBedrockEventStream, error)
}

func (client *streamingLegacyBedrockClient) InvokeModelWithResponseStream(
	_ context.Context, input *bedrockruntime.InvokeModelWithResponseStreamInput,
	options ...func(*bedrockruntime.Options),
) (anthropic.LegacyBedrockEventStream, error) {
	for _, option := range options {
		option(&bedrockruntime.Options{})
	}
	return client.stream(input)
}

func TestLegacyBedrockStreaming(t *testing.T) {
	stream := newLegacyEventStream(
		legacyChunk(`{"type":"message_start","message":{"id":"message-1","model":"claude","usage":{"input_tokens":3}}}`),
		legacyChunk(`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		legacyChunk(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}`),
		legacyChunk(`{"type":"content_block_stop","index":0}`),
		legacyChunk(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`),
		legacyChunk(`{"type":"message_stop"}`),
	)
	client := &streamingLegacyBedrockClient{legacyBedrockClient: &legacyBedrockClient{}, stream: func(
		input *bedrockruntime.InvokeModelWithResponseStreamInput,
	) (anthropic.LegacyBedrockEventStream, error) {
		if !strings.Contains(string(input.Body), `"stream":true`) {
			t.Fatalf("streaming request omitted stream flag: %s", input.Body)
		}
		return stream, nil
	}}
	model := anthropic.NewLegacyBedrockModel("model", anthropic.LegacyBedrockConfig{
		Client: client, ProviderURL: "https://bedrock.example",
	})
	sequence, err := model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	var text string
	var finish ai.FinishEvent
	for event, eventErr := range sequence {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
		switch value := event.(type) {
		case ai.TextDeltaEvent:
			text += value.Delta
		case ai.FinishEvent:
			finish = value
		}
	}
	if text != "hello" || finish.Usage.InputTokens != 3 || finish.Usage.OutputTokens != 2 ||
		finish.ProviderURL != "https://bedrock.example" || stream.closeCount.Load() != 1 {
		t.Fatalf("unexpected stream: text=%q finish=%#v closes=%d", text, finish, stream.closeCount.Load())
	}
}

func TestLegacyBedrockStreamingErrors(t *testing.T) {
	openErr := errors.New("open failed")
	client := &streamingLegacyBedrockClient{legacyBedrockClient: &legacyBedrockClient{}, stream: func(
		*bedrockruntime.InvokeModelWithResponseStreamInput,
	) (anthropic.LegacyBedrockEventStream, error) {
		return nil, openErr
	}}
	model := anthropic.NewLegacyBedrockModel("model", anthropic.LegacyBedrockConfig{Client: client})
	_, err := model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if !errors.Is(err, openErr) {
		t.Fatalf("unexpected open error: %v", err)
	}

	client.stream = func(*bedrockruntime.InvokeModelWithResponseStreamInput) (anthropic.LegacyBedrockEventStream, error) {
		return nil, nil
	}
	_, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), "returned nil stream") {
		t.Fatalf("unexpected nil stream error: %v", err)
	}
	var typedNil *legacyEventStream
	client.stream = func(*bedrockruntime.InvokeModelWithResponseStreamInput) (anthropic.LegacyBedrockEventStream, error) {
		return typedNil, nil
	}
	_, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err == nil || !strings.Contains(err.Error(), "returned nil stream") {
		t.Fatalf("unexpected typed nil stream error: %v", err)
	}
	_, err = model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{Settings: ai.ModelSettings{
		ExtraBody: map[string]any{"bad": make(chan int)},
	}})
	if err == nil || !strings.Contains(err.Error(), "marshal legacy Bedrock request") {
		t.Fatalf("unexpected stream marshal error: %v", err)
	}

	tests := []struct {
		name   string
		stream *legacyEventStream
		match  string
	}{
		{name: "unsupported event", stream: newLegacyEventStream(nil), match: "unsupported legacy Bedrock stream event"},
		{name: "reader error", stream: &legacyEventStream{
			events: closedLegacyEvents(), err: errors.New("read failed"),
		}, match: "read failed"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client.stream = func(*bedrockruntime.InvokeModelWithResponseStreamInput) (anthropic.LegacyBedrockEventStream, error) {
				return test.stream, nil
			}
			sequence, err := model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
			if err != nil {
				t.Fatal(err)
			}
			var got error
			for _, eventErr := range sequence {
				if eventErr != nil {
					got = eventErr
				}
			}
			if got == nil || !strings.Contains(got.Error(), test.match) {
				t.Fatalf("unexpected stream error: %v", got)
			}
		})
	}

	events := make(chan types.ResponseStream, 1)
	events <- legacyChunk(`{"type":"message_stop"}`)
	close(events)
	client.stream = func(*bedrockruntime.InvokeModelWithResponseStreamInput) (anthropic.LegacyBedrockEventStream, error) {
		return valueLegacyEventStream{events: events}, nil
	}
	sequence, err := model.StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	for _, eventErr := range sequence {
		if eventErr != nil {
			t.Fatal(eventErr)
		}
	}
}

func TestLegacyBedrockStreamConsumerStopsCopy(t *testing.T) {
	stream := newLegacyEventStream(
		legacyChunk(`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		legacyChunk(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"one"}}`),
		legacyChunk(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"two"}}`),
	)
	client := &streamingLegacyBedrockClient{legacyBedrockClient: &legacyBedrockClient{}, stream: func(
		*bedrockruntime.InvokeModelWithResponseStreamInput,
	) (anthropic.LegacyBedrockEventStream, error) {
		return stream, nil
	}}
	sequence, err := anthropic.NewLegacyBedrockModel(
		"model", anthropic.LegacyBedrockConfig{Client: client},
	).StreamRequest(context.Background(), nil, ai.ModelRequestParams{})
	if err != nil {
		t.Fatal(err)
	}
	sequence(func(event ai.ModelStreamEvent, eventErr error) bool {
		return eventErr == nil && event == nil
	})
	deadline := time.Now().Add(time.Second)
	for stream.closeCount.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if stream.closeCount.Load() != 1 {
		t.Fatalf("stream was not closed after consumer stop: %d", stream.closeCount.Load())
	}
}

func legacyChunk(value string) types.ResponseStream {
	return &types.ResponseStreamMemberChunk{Value: types.PayloadPart{Bytes: []byte(value)}}
}

func closedLegacyEvents() chan types.ResponseStream {
	events := make(chan types.ResponseStream)
	close(events)
	return events
}
