package openai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	ai "github.com/Kludex/pydantic-ai-go"
	"github.com/Kludex/pydantic-ai-go/realtime"
)

// MapEvent translates one OpenAI Realtime JSON frame into provider-neutral codec events.
func MapEvent(data []byte) ([]realtime.CodecEvent, error) {
	var frame map[string]any
	if err := json.Unmarshal(data, &frame); err != nil {
		return nil, fmt.Errorf("openai realtime: decode event: %w", err)
	}
	kind, _ := frame["type"].(string)
	switch kind {
	case "response.created":
		response := object(frame["response"])
		return []realtime.CodecEvent{realtime.ResponseStarted{ResponseID: stringValue(response["id"])}}, nil
	case "response.output_audio.delta", "response.audio.delta":
		decoded, err := base64.StdEncoding.DecodeString(stringValue(frame["delta"]))
		if err != nil {
			return nil, fmt.Errorf("openai realtime: decode audio delta: %w", err)
		}
		return []realtime.CodecEvent{realtime.AudioDelta{Data: decoded, ItemID: stringValue(frame["item_id"])}}, nil
	case "response.output_audio_transcript.delta", "response.audio_transcript.delta":
		return []realtime.CodecEvent{realtime.OutputTranscript{
			Text: stringValue(frame["delta"]), ItemID: stringValue(frame["item_id"]),
		}}, nil
	case "response.output_audio_transcript.done", "response.audio_transcript.done":
		return []realtime.CodecEvent{realtime.OutputTranscript{
			Text: stringValue(frame["transcript"]), Final: true, ItemID: stringValue(frame["item_id"]),
		}}, nil
	case "response.output_text.delta", "response.text.delta":
		return []realtime.CodecEvent{realtime.OutputTranscript{
			Text: stringValue(frame["delta"]), OutputText: true, ItemID: stringValue(frame["item_id"]),
		}}, nil
	case "response.output_text.done", "response.text.done":
		return []realtime.CodecEvent{realtime.OutputTranscript{
			Text: stringValue(frame["text"]), Final: true, OutputText: true, ItemID: stringValue(frame["item_id"]),
		}}, nil
	case "conversation.item.input_audio_transcription.delta":
		return []realtime.CodecEvent{realtime.InputTranscript{
			Text: stringValue(frame["delta"]), ItemID: stringValue(frame["item_id"]),
		}}, nil
	case "conversation.item.input_audio_transcription.completed":
		return []realtime.CodecEvent{realtime.InputTranscript{
			Text: stringValue(frame["transcript"]), Final: true, Cumulative: true,
			ItemID: stringValue(frame["item_id"]),
		}}, nil
	case "conversation.item.input_audio_transcription.failed":
		details := object(frame["error"])
		return []realtime.CodecEvent{realtime.InputTranscriptionError{
			ItemID: stringValue(frame["item_id"]), Err: fmt.Errorf("%s", stringValue(details["message"])),
		}}, nil
	case "input_audio_buffer.speech_started":
		return []realtime.CodecEvent{realtime.InputSpeechStarted{ItemID: stringValue(frame["item_id"])}}, nil
	case "input_audio_buffer.speech_stopped":
		return []realtime.CodecEvent{realtime.InputSpeechEnded{ItemID: stringValue(frame["item_id"])}}, nil
	case "output_audio_buffer.started":
		return []realtime.CodecEvent{realtime.OutputSpeechStarted{}}, nil
	case "output_audio_buffer.stopped", "output_audio_buffer.cleared":
		return []realtime.CodecEvent{realtime.OutputSpeechEnded{}}, nil
	case "response.function_call_arguments.done":
		return []realtime.CodecEvent{realtime.ToolCall{
			ToolCallID: stringValue(frame["call_id"]), ToolName: stringValue(frame["name"]),
			Arguments: stringValue(frame["arguments"]), ItemID: stringValue(frame["item_id"]),
			ResponseUsageFollows: true,
		}}, nil
	case "response.done":
		return mapResponseDone(frame), nil
	case "conversation.created":
		conversation := object(frame["conversation"])
		return []realtime.CodecEvent{realtime.ConversationCreated{ConversationID: stringValue(conversation["id"])}}, nil
	case "conversation.item.created", "conversation.item.added":
		item := object(frame["item"])
		return []realtime.CodecEvent{realtime.ConversationItemCreated{
			ItemID: stringValue(item["id"]), ToolCallID: stringValue(item["call_id"]),
		}}, nil
	case "error":
		details := object(frame["error"])
		message := stringValue(details["message"])
		if message == "" {
			message = "provider reported an unknown error"
		}
		return []realtime.CodecEvent{realtime.SessionError{Err: fmt.Errorf("openai realtime: %s", message)}}, nil
	case "session.created", "session.updated", "rate_limits.updated", "response.output_item.added",
		"response.output_item.done", "response.content_part.added", "response.content_part.done",
		"input_audio_buffer.committed", "input_audio_buffer.cleared":
		return nil, nil
	default:
		return nil, nil
	}
}

func mapResponseDone(frame map[string]any) []realtime.CodecEvent {
	response := object(frame["response"])
	result := make([]realtime.CodecEvent, 0, 3)
	for _, raw := range array(response["output"]) {
		item := object(raw)
		if stringValue(item["type"]) == "function_call" {
			result = append(result, realtime.ToolCall{
				ToolCallID: stringValue(item["call_id"]), ToolName: stringValue(item["name"]),
				Arguments: stringValue(item["arguments"]), ItemID: stringValue(item["id"]),
				ResponseUsageFollows: true,
			})
		}
	}
	responseID := stringValue(response["id"])
	finish := finishReason(response)
	if usageData := object(response["usage"]); len(usageData) > 0 {
		result = append(result, realtime.SessionUsage{
			Usage: mapUsage(usageData), ProviderResponseID: responseID, FinishReason: finish, ResponseScoped: true,
		})
	}
	status := stringValue(response["status"])
	result = append(result, realtime.ResponseDone{
		Interrupted: status == "cancelled", ProviderResponseID: responseID, FinishReason: finish,
		ProviderDetails: map[string]any{
			"status": status, "status_details": response["status_details"],
		},
	})
	return result
}

func finishReason(response map[string]any) ai.FinishReason {
	status := stringValue(response["status"])
	if status == "cancelled" {
		return ai.FinishReasonStop
	}
	details := object(response["status_details"])
	reason := stringValue(details["reason"])
	switch reason {
	case "max_output_tokens", "max_tokens":
		return ai.FinishReasonLength
	case "content_filter":
		return ai.FinishReasonContentFilter
	case "tool_call", "tool_calls":
		return ai.FinishReasonToolCall
	default:
		return ai.FinishReasonStop
	}
}

func mapUsage(usage map[string]any) ai.Usage {
	input := object(usage["input_token_details"])
	output := object(usage["output_token_details"])
	cached := object(input["cached_tokens_details"])
	result := ai.Usage{
		Requests: 1, InputTokens: integer(usage["input_tokens"]), OutputTokens: integer(usage["output_tokens"]),
		InputAudioTokens: integer(input["audio_tokens"]), CacheReadTokens: integer(input["cached_tokens"]),
		CacheAudioReadTokens: integer(cached["audio_tokens"]), OutputAudioTokens: integer(output["audio_tokens"]),
		ReasoningTokens: integer(output["reasoning_tokens"]), Details: map[string]int{},
	}
	for key, value := range map[string]int{
		"input_text_tokens": integer(input["text_tokens"]), "input_image_tokens": integer(input["image_tokens"]),
		"output_text_tokens": integer(output["text_tokens"]), "audio_tokens": integer(output["audio_tokens"]),
		"reasoning_tokens": integer(output["reasoning_tokens"]),
	} {
		if value != 0 {
			result.Details[key] = value
		}
	}
	return result
}

func object(value any) map[string]any {
	object, _ := value.(map[string]any)
	return object
}

func array(value any) []any {
	array, _ := value.([]any)
	return array
}

func stringValue(value any) string {
	text, _ := value.(string)
	return text
}

func integer(value any) int {
	number, _ := value.(float64)
	return int(number)
}
