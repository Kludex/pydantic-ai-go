package bedrock

import (
	"context"
	"fmt"
	"iter"
	"reflect"
	"strconv"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	ai "github.com/Kludex/pydantic-ai-go"
)

// StreamRequest implements ai.StreamingModel through Bedrock ConverseStream.
func (model *Model) StreamRequest(
	ctx context.Context, messages []ai.ModelMessage, params ai.ModelRequestParams,
) (iter.Seq2[ai.ModelStreamEvent, error], error) {
	input, err := buildConverseInput(ctx, model.name, messages, params)
	if err != nil {
		return nil, err
	}
	client, err := model.resolveClient(ctx)
	if err != nil {
		return nil, err
	}
	streamer, ok := client.(StreamingClient)
	if !ok {
		response, err := model.Request(ctx, messages, params)
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
	stream, err := streamer.ConverseStream(ctx, converseStreamInput(input), requestOptions(params.Settings.ExtraHeaders))
	if err != nil {
		return nil, modelError(ctx, model, "stream request", err)
	}
	if streamIsNil(stream) {
		return nil, fmt.Errorf("bedrock: streaming client returned nil stream")
	}
	return model.streamEvents(ctx, stream), nil
}

func (model *Model) streamEvents(ctx context.Context, stream EventStream) iter.Seq2[ai.ModelStreamEvent, error] {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		defer func() { _ = stream.Close() }()
		usage := ai.Usage{Requests: 1}
		finish := ai.FinishReason("")
		stopReason := types.StopReason("")
		providerDetails := map[string]any{}
		stopped := false
		for event := range stream.Events() {
			switch value := event.(type) {
			case *types.ConverseStreamOutputMemberMessageStart:
				continue
			case *types.ConverseStreamOutputMemberContentBlockStart:
				if value == nil || value.Value.ContentBlockIndex == nil {
					yield(nil, fmt.Errorf("bedrock: stream content start omitted index"))
					return
				}
				partID := strconv.Itoa(int(*value.Value.ContentBlockIndex))
				switch start := value.Value.Start.(type) {
				case *types.ContentBlockStartMemberToolUse:
					if start == nil {
						yield(nil, fmt.Errorf("bedrock: stream contains nil tool-use start"))
						return
					}
					if !yield(ai.ToolCallStartEvent{
						PartID: partID, ToolName: stringValue(start.Value.Name), ToolCallID: stringValue(start.Value.ToolUseId),
						ProviderName: "bedrock",
					}, nil) {
						return
					}
				default:
					yield(nil, fmt.Errorf("bedrock: unsupported stream content start type %T", value.Value.Start))
					return
				}
			case *types.ConverseStreamOutputMemberContentBlockDelta:
				if value == nil || value.Value.ContentBlockIndex == nil {
					yield(nil, fmt.Errorf("bedrock: stream content delta omitted index"))
					return
				}
				partID := strconv.Itoa(int(*value.Value.ContentBlockIndex))
				switch delta := value.Value.Delta.(type) {
				case *types.ContentBlockDeltaMemberText:
					if delta == nil {
						yield(nil, fmt.Errorf("bedrock: stream contains nil text delta"))
						return
					}
					if !yield(ai.TextDeltaEvent{PartID: partID, Delta: delta.Value, ProviderName: "bedrock"}, nil) {
						return
					}
				case *types.ContentBlockDeltaMemberToolUse:
					if delta == nil || delta.Value.Input == nil {
						yield(nil, fmt.Errorf("bedrock: stream tool-use delta omitted input"))
						return
					}
					if !yield(ai.ToolCallDeltaEvent{PartID: partID, ArgsDelta: *delta.Value.Input}, nil) {
						return
					}
				case *types.ContentBlockDeltaMemberReasoningContent:
					if delta == nil {
						yield(nil, fmt.Errorf("bedrock: stream contains nil reasoning delta"))
						return
					}
					var event ai.ThinkingDeltaEvent
					event.PartID = partID
					event.ProviderName = "bedrock"
					switch reasoning := delta.Value.(type) {
					case *types.ReasoningContentBlockDeltaMemberText:
						if reasoning == nil {
							yield(nil, fmt.Errorf("bedrock: stream contains nil reasoning-text delta"))
							return
						}
						event.Delta = reasoning.Value
					case *types.ReasoningContentBlockDeltaMemberSignature:
						if reasoning == nil {
							yield(nil, fmt.Errorf("bedrock: stream contains nil reasoning-signature delta"))
							return
						}
						event.SignatureDelta = reasoning.Value
					case *types.ReasoningContentBlockDeltaMemberRedactedContent:
						if reasoning == nil {
							yield(nil, fmt.Errorf("bedrock: stream contains nil redacted-reasoning delta"))
							return
						}
						event.ProviderDetails = map[string]any{"redacted_content": append([]byte(nil), reasoning.Value...)}
					default:
						yield(nil, fmt.Errorf("bedrock: unsupported stream reasoning delta type %T", delta.Value))
						return
					}
					if !yield(event, nil) {
						return
					}
				default:
					yield(nil, fmt.Errorf("bedrock: unsupported stream content delta type %T", value.Value.Delta))
					return
				}
			case *types.ConverseStreamOutputMemberContentBlockStop:
				continue
			case *types.ConverseStreamOutputMemberMessageStop:
				if value == nil {
					yield(nil, fmt.Errorf("bedrock: stream contains nil message stop"))
					return
				}
				stopped = true
				stopReason = value.Value.StopReason
				finish = finishReason(stopReason)
				providerDetails["stop_reason"] = string(stopReason)
			case *types.ConverseStreamOutputMemberMetadata:
				if value == nil {
					yield(nil, fmt.Errorf("bedrock: stream contains nil metadata"))
					return
				}
				usage = responseUsage(value.Value.Usage)
				if value.Value.Metrics != nil && value.Value.Metrics.LatencyMs != nil {
					providerDetails["latency_ms"] = *value.Value.Metrics.LatencyMs
				}
				if value.Value.ServiceTier != nil {
					providerDetails["service_tier"] = string(value.Value.ServiceTier.Type)
				}
			default:
				yield(nil, fmt.Errorf("bedrock: unsupported stream event type %T", event))
				return
			}
		}
		if err := stream.Err(); err != nil {
			yield(nil, ai.NewModelTransportError(ctx, model, "read stream", err))
			return
		}
		if !stopped {
			yield(nil, fmt.Errorf("bedrock: stream ended without message stop"))
			return
		}
		yield(ai.FinishEvent{
			Usage: usage, ModelName: model.name, Timestamp: time.Now().UTC(), ProviderName: "bedrock",
			ProviderURL: model.ProviderURL(), ProviderDetails: providerDetails, FinishReason: finish,
		}, nil)
	}
}

func converseStreamInput(input *bedrockruntime.ConverseInput) *bedrockruntime.ConverseStreamInput {
	return &bedrockruntime.ConverseStreamInput{
		ModelId: input.ModelId, AdditionalModelRequestFields: input.AdditionalModelRequestFields,
		AdditionalModelResponseFieldPaths: input.AdditionalModelResponseFieldPaths,
		InferenceConfig:                   input.InferenceConfig, Messages: input.Messages, OutputConfig: input.OutputConfig,
		PerformanceConfig: input.PerformanceConfig, PromptVariables: input.PromptVariables,
		RequestMetadata: input.RequestMetadata, ServiceTier: input.ServiceTier, System: input.System,
		ToolConfig: input.ToolConfig,
	}
}

func streamIsNil(stream EventStream) bool {
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
