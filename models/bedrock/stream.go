package bedrock

import (
	"context"
	"encoding/json"
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
		results := map[string]*streamNativeToolResult{}
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
					toolName := stringValue(start.Value.Name)
					toolKind := ai.ToolPartKind("")
					native := false
					var details map[string]any
					if start.Value.Type == types.ToolUseTypeServerToolUse && toolName == "nova_code_interpreter" {
						toolName = "code_execution"
						toolKind = ai.ToolPartKindCodeExecution
						native = true
						details = map[string]any{"code_arg_name": "snippet", "code_arg_language": "python"}
					}
					if !yield(ai.ToolCallStartEvent{
						PartID: partID, ToolName: toolName, ToolCallID: stringValue(start.Value.ToolUseId),
						ToolKind: toolKind, ProviderName: "bedrock", ProviderDetails: details, Native: native,
					}, nil) {
						return
					}
				case *types.ContentBlockStartMemberToolResult:
					if start == nil || stringValue(start.Value.Type) != "nova_code_interpreter_result" {
						yield(nil, fmt.Errorf("bedrock: unsupported stream native tool result start"))
						return
					}
					results[partID] = &streamNativeToolResult{
						toolCallID: stringValue(start.Value.ToolUseId), status: start.Value.Status,
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
				case *types.ContentBlockDeltaMemberToolResult:
					result := results[partID]
					if result == nil || delta == nil {
						yield(nil, fmt.Errorf("bedrock: stream native tool result delta has no matching start"))
						return
					}
					for _, block := range delta.Value {
						content, err := streamToolResultDelta(block)
						if err != nil {
							yield(nil, err)
							return
						}
						result.content = append(result.content, content)
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
				if value == nil || value.Value.ContentBlockIndex == nil {
					yield(nil, fmt.Errorf("bedrock: stream content stop omitted index"))
					return
				}
				partID := strconv.Itoa(int(*value.Value.ContentBlockIndex))
				result := results[partID]
				if result == nil {
					continue
				}
				content := any(result.content)
				if len(result.content) == 1 {
					content = result.content[0]
				}
				outcome := ai.ToolReturnOutcomeSuccess
				if result.status == types.ToolResultStatusError {
					outcome = ai.ToolReturnOutcomeFailed
				}
				if !yield(ai.NativeToolReturnEvent{PartID: partID, Part: ai.NativeToolReturnPart{
					ToolName: "code_execution", ToolCallID: result.toolCallID, ToolKind: ai.ToolPartKindCodeExecution,
					Content: content, Outcome: outcome, ProviderName: "bedrock",
					ProviderDetails: map[string]any{"status": string(result.status)},
				}}, nil) {
					return
				}
				delete(results, partID)
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

type streamNativeToolResult struct {
	toolCallID string
	status     types.ToolResultStatus
	content    []any
}

func streamToolResultDelta(delta types.ToolResultBlockDelta) (any, error) {
	switch value := delta.(type) {
	case *types.ToolResultBlockDeltaMemberText:
		if value == nil {
			return nil, fmt.Errorf("bedrock: stream native tool result contains nil text")
		}
		return value.Value, nil
	case *types.ToolResultBlockDeltaMemberJson:
		if value == nil || value.Value == nil {
			return nil, fmt.Errorf("bedrock: stream native tool result contains nil JSON")
		}
		encoded, err := value.Value.MarshalSmithyDocument()
		if err != nil {
			return nil, fmt.Errorf("bedrock: encode streamed native tool result: %w", err)
		}
		var decoded any
		if err := json.Unmarshal(encoded, &decoded); err != nil {
			return nil, fmt.Errorf("bedrock: decode streamed native tool result: %w", err)
		}
		return decoded, nil
	default:
		return nil, fmt.Errorf("bedrock: unsupported stream native tool result content %T", delta)
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
