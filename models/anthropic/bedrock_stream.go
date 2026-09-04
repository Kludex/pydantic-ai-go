package anthropic

import (
	"context"
	"fmt"
	"io"
	"iter"
	"reflect"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

func (model *Model) streamLegacyBedrock(
	ctx context.Context, payload *messagesRequest, headers map[string]string,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	streamer, ok := model.legacyBedrockClient.(LegacyBedrockStreamingClient)
	if !ok {
		response, err := model.requestLegacyBedrock(ctx, payload, headers)
		if err != nil {
			return nil, err
		}
		return func(yield func(ai.ModelStreamEvent, error) bool) {
			yield(ai.FinishEvent{
				Parts: response.Parts, Usage: response.Usage, ModelName: response.ModelName,
				Timestamp: response.Timestamp, ProviderName: response.ProviderName, ProviderURL: response.ProviderURL,
				ProviderDetails: response.ProviderDetails, ProviderResponseID: response.ProviderResponseID,
				FinishReason: response.FinishReason, State: response.State,
			}, nil)
		}, nil
	}
	payload.Stream = true
	body, err := legacyBedrockBody(payload)
	if err != nil {
		return nil, err
	}
	stream, err := streamer.InvokeModelWithResponseStream(ctx, &bedrockruntime.InvokeModelWithResponseStreamInput{
		ModelId: aws.String(model.name), Body: body,
		ContentType: aws.String("application/json"), Accept: aws.String("application/json"),
	}, legacyBedrockOptions(headers))
	if err != nil {
		return nil, model.legacyBedrockError(ctx, "stream request", err)
	}
	if legacyBedrockStreamIsNil(stream) {
		return nil, fmt.Errorf("anthropic: legacy Bedrock streaming client returned nil stream")
	}
	reader, writer := io.Pipe()
	done := make(chan struct{})
	go func() {
		copyLegacyBedrockStream(stream, writer)
		close(done)
	}()
	events := model.eventStream(ctx, reader)
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		for event, eventErr := range events {
			if !yield(event, eventErr) {
				_ = reader.Close()
				<-done
				return
			}
		}
		<-done
	}, nil
}

func copyLegacyBedrockStream(stream LegacyBedrockEventStream, writer *io.PipeWriter) {
	var streamErr error
	for event := range stream.Events() {
		chunk, ok := event.(*types.ResponseStreamMemberChunk)
		if !ok || chunk == nil {
			streamErr = fmt.Errorf("anthropic: unsupported legacy Bedrock stream event %T", event)
			break
		}
		if _, err := writer.Write(append(append([]byte("data: "), chunk.Value.Bytes...), '\n', '\n')); err != nil {
			streamErr = err
			break
		}
	}
	if streamErr == nil {
		streamErr = stream.Err()
	}
	_ = stream.Close()
	_ = writer.CloseWithError(streamErr)
}

func legacyBedrockStreamIsNil(stream LegacyBedrockEventStream) bool {
	if stream == nil {
		return true
	}
	value := reflect.ValueOf(stream)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}
