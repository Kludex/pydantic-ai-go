package bedrock

import (
	"encoding/json"
	"fmt"
	"time"

	awsmiddleware "github.com/aws/aws-sdk-go-v2/aws/middleware"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	ai "github.com/Kludex/pydantic-ai-go"
)

func convertResponse(model *Model, output *bedrockruntime.ConverseOutput) (*ai.ModelResponse, error) {
	if output == nil {
		return nil, fmt.Errorf("bedrock: response is nil")
	}
	message, ok := output.Output.(*types.ConverseOutputMemberMessage)
	if !ok || message == nil {
		return nil, fmt.Errorf("bedrock: response omitted output message")
	}
	parts := make([]ai.ResponsePart, 0, len(message.Value.Content))
	for _, block := range message.Value.Content {
		switch value := block.(type) {
		case *types.ContentBlockMemberText:
			if value == nil {
				return nil, fmt.Errorf("bedrock: response contains a nil text block")
			}
			parts = append(parts, ai.TextPart{Content: value.Value, ProviderName: "bedrock"})
		case *types.ContentBlockMemberToolUse:
			if value == nil {
				return nil, fmt.Errorf("bedrock: response contains a nil tool-use block")
			}
			if value.Value.Input == nil {
				return nil, fmt.Errorf("bedrock: tool use omitted input")
			}
			encoded, err := value.Value.Input.MarshalSmithyDocument()
			if err != nil {
				return nil, fmt.Errorf("bedrock: encode tool use input: %w", err)
			}
			if !json.Valid(encoded) {
				return nil, fmt.Errorf("bedrock: tool use input is not valid JSON")
			}
			if value.Value.Type == types.ToolUseTypeServerToolUse && stringValue(value.Value.Name) == "nova_code_interpreter" {
				parts = append(parts, ai.NativeToolCallPart{
					ToolName: "code_execution", ToolCallID: stringValue(value.Value.ToolUseId),
					ToolKind: ai.ToolPartKindCodeExecution, Args: append(json.RawMessage(nil), encoded...),
					ProviderName: "bedrock", ProviderDetails: map[string]any{
						"code_arg_name": "snippet", "code_arg_language": "python",
					},
				})
			} else {
				parts = append(parts, ai.ToolCallPart{
					ToolName: stringValue(value.Value.Name), ToolCallID: stringValue(value.Value.ToolUseId),
					Args: append(json.RawMessage(nil), encoded...), ProviderName: "bedrock",
				})
			}
		case *types.ContentBlockMemberToolResult:
			if value == nil || stringValue(value.Value.Type) != "nova_code_interpreter_result" {
				return nil, fmt.Errorf("bedrock: unsupported native tool result")
			}
			content, err := bedrockToolResultContent(value.Value.Content)
			if err != nil {
				return nil, err
			}
			outcome := ai.ToolReturnOutcomeSuccess
			if value.Value.Status == types.ToolResultStatusError {
				outcome = ai.ToolReturnOutcomeFailed
			}
			parts = append(parts, ai.NativeToolReturnPart{
				ToolName: "code_execution", ToolCallID: stringValue(value.Value.ToolUseId),
				ToolKind: ai.ToolPartKindCodeExecution, Content: content, Outcome: outcome,
				ProviderName: "bedrock", ProviderDetails: map[string]any{"status": string(value.Value.Status)},
			})
		case *types.ContentBlockMemberReasoningContent:
			if value == nil || value.Value == nil {
				return nil, fmt.Errorf("bedrock: response contains a nil reasoning block")
			}
			switch reasoning := value.Value.(type) {
			case *types.ReasoningContentBlockMemberReasoningText:
				if reasoning == nil {
					return nil, fmt.Errorf("bedrock: response contains a nil reasoning-text block")
				}
				parts = append(parts, ai.ThinkingPart{
					Content: stringValue(reasoning.Value.Text), Signature: stringValue(reasoning.Value.Signature),
					ProviderName: "bedrock",
				})
			case *types.ReasoningContentBlockMemberRedactedContent:
				if reasoning == nil {
					return nil, fmt.Errorf("bedrock: response contains a nil redacted-reasoning block")
				}
				parts = append(parts, ai.ThinkingPart{ProviderName: "bedrock", ProviderDetails: map[string]any{
					"redacted_content": append([]byte(nil), reasoning.Value...),
				}})
			}
		default:
			return nil, fmt.Errorf("bedrock: unsupported response content type %T", block)
		}
	}
	response := &ai.ModelResponse{
		Parts: parts, Usage: responseUsage(output.Usage), ModelName: model.name, Timestamp: time.Now().UTC(),
		ProviderName: "bedrock", ProviderURL: model.ProviderURL(), FinishReason: finishReason(output.StopReason),
		ProviderDetails: map[string]any{"stop_reason": string(output.StopReason)},
	}
	if requestID, ok := awsmiddleware.GetRequestIDMetadata(output.ResultMetadata); ok {
		response.ProviderResponseID = requestID
	}
	if output.Metrics != nil && output.Metrics.LatencyMs != nil {
		response.ProviderDetails["latency_ms"] = *output.Metrics.LatencyMs
	}
	if output.ServiceTier != nil {
		response.ProviderDetails["service_tier"] = string(output.ServiceTier.Type)
	}
	if output.PerformanceConfig != nil {
		response.ProviderDetails["performance_latency"] = string(output.PerformanceConfig.Latency)
	}
	if output.AdditionalModelResponseFields != nil {
		encoded, err := output.AdditionalModelResponseFields.MarshalSmithyDocument()
		if err != nil {
			return nil, fmt.Errorf("bedrock: encode additional response fields: %w", err)
		}
		var fields any
		_ = json.Unmarshal(encoded, &fields)
		response.ProviderDetails["additional_model_response_fields"] = fields
	}
	if output.Trace != nil {
		response.ProviderDetails["trace"] = bedrockMetadataValue(output.Trace)
	}
	return response, nil
}

func bedrockToolResultContent(blocks []types.ToolResultContentBlock) (any, error) {
	content := make([]any, 0, len(blocks))
	for _, block := range blocks {
		switch value := block.(type) {
		case *types.ToolResultContentBlockMemberText:
			if value == nil {
				return nil, fmt.Errorf("bedrock: native tool result contains nil text")
			}
			content = append(content, value.Value)
		case *types.ToolResultContentBlockMemberJson:
			if value == nil || value.Value == nil {
				return nil, fmt.Errorf("bedrock: native tool result contains nil JSON")
			}
			encoded, err := value.Value.MarshalSmithyDocument()
			if err != nil {
				return nil, fmt.Errorf("bedrock: encode native tool result: %w", err)
			}
			var decoded any
			if err := json.Unmarshal(encoded, &decoded); err != nil {
				return nil, fmt.Errorf("bedrock: decode native tool result: %w", err)
			}
			content = append(content, decoded)
		default:
			return nil, fmt.Errorf("bedrock: unsupported native tool result content %T", block)
		}
	}
	if len(content) == 1 {
		return content[0], nil
	}
	return content, nil
}

func responseUsage(usage *types.TokenUsage) ai.Usage {
	if usage == nil {
		return ai.Usage{Requests: 1}
	}
	return ai.Usage{
		Requests: 1, InputTokens: int32Value(usage.InputTokens), OutputTokens: int32Value(usage.OutputTokens),
		CacheReadTokens: int32Value(usage.CacheReadInputTokens), CacheWriteTokens: int32Value(usage.CacheWriteInputTokens),
	}
}

func finishReason(reason types.StopReason) ai.FinishReason {
	switch reason {
	case types.StopReasonEndTurn, types.StopReasonStopSequence:
		return ai.FinishReasonStop
	case types.StopReasonToolUse:
		return ai.FinishReasonToolCall
	case types.StopReasonMaxTokens, types.StopReasonModelContextWindowExceeded:
		return ai.FinishReasonLength
	case types.StopReasonGuardrailIntervened, types.StopReasonContentFiltered:
		return ai.FinishReasonContentFilter
	default:
		return ai.FinishReasonError
	}
}
