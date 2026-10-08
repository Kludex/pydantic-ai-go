package openai

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	ai "github.com/Kludex/pydantic-ai-go/ai"
	"github.com/Kludex/pydantic-ai-go/ai/realtime"
)

func (connection *LiveConnection) mapLiveFrame(data []byte) ([]realtime.CodecEvent, error) {
	var frame map[string]any
	if err := json.Unmarshal(data, &frame); err != nil {
		return nil, fmt.Errorf("openai GPT-Live: parse event: %w", err)
	}
	if frame == nil {
		return nil, fmt.Errorf("openai GPT-Live: event must be a JSON object")
	}
	kind := stringValue(frame["type"])
	malformed := fmt.Errorf("openai GPT-Live: malformed %s event", kind)
	switch kind {
	case "session.output_audio.delta":
		delta, ok := frame["delta"].(string)
		if !ok {
			return nil, malformed
		}
		pcm, err := base64.StdEncoding.DecodeString(delta)
		if err != nil || len(pcm)%2 != 0 {
			return nil, malformed
		}
		voiced := false
		for index := 0; index+1 < len(pcm); index += 2 {
			sample := int(int16(binary.LittleEndian.Uint16(pcm[index:])))
			if sample > 64 || sample < -64 {
				voiced = true
				break
			}
		}
		if voiced {
			connection.hasPause = true
			connection.pauseBytes = 0
			events := connection.openResponse()
			if !connection.sideband {
				events = append(events, realtime.AudioDelta{Data: pcm})
			}
			return events, nil
		}
		if !connection.hasPause || connection.sideband {
			return nil, nil
		}
		allowed := max(0, connection.rate-connection.pauseBytes)
		connection.pauseBytes += len(pcm)
		if !connection.responseOpen || connection.inputOpen || allowed == 0 {
			return nil, nil
		}
		return []realtime.CodecEvent{realtime.AudioDelta{Data: pcm[:min(len(pcm), allowed)]}}, nil
	case "session.output_transcript.delta", "session.input_transcript.delta":
		text, ok := frame["delta"].(string)
		if !ok {
			return nil, malformed
		}
		if kind == "session.output_transcript.delta" {
			return append(connection.openResponse(), realtime.OutputTranscript{Text: connection.fragment(1, text)}), nil
		}
		start, startOK := frame["start_ms"].(float64)
		end, endOK := frame["end_ms"].(float64)
		if !startOK || !endOK {
			return nil, malformed
		}
		continues := connection.responseOpen && !connection.inputOpen && connection.hasInputEnd && start <= connection.lastInputEnd
		connection.hasInputEnd = true
		connection.lastInputEnd = end
		if continues && strings.IndexFunc(text, func(char rune) bool { return unicode.IsLetter(char) || unicode.IsNumber(char) }) < 0 {
			return nil, nil
		}
		connection.inputOpen = true
		connection.lastVoice = time.Now()
		return []realtime.CodecEvent{realtime.InputTranscript{Text: connection.fragment(0, text)}}, nil
	case "session.delegation.created":
		delegation := object(frame["delegation"])
		id, target := stringValue(delegation["id"]), stringValue(delegation["target"])
		if id == "" || target != "responses" && target != "client" {
			return nil, malformed
		}
		if target == "client" {
			return []realtime.CodecEvent{liveError("live_client_delegation", "client delegation cannot be answered; configure Responses delegation", true)}, nil
		}
		connection.delegations[id] = &liveDelegation{pending: map[string]bool{}, inFlight: true}
		return connection.openResponse(), nil
	case "response.event":
		nested := object(frame["event"])
		if nested == nil {
			return nil, malformed
		}
		return connection.mapBackend(nested, stringValue(frame["delegation_id"]))
	case "session.usage.updated":
		seconds, ok := object(frame["usage"])["seconds"].(float64)
		if !ok {
			return nil, malformed
		}
		var ratio *float64
		if contextWindow := object(frame["context_window"]); contextWindow != nil {
			fraction, ok := contextWindow["usage_ratio"].(float64)
			if !ok {
				return nil, malformed
			}
			ratio = &fraction
		}
		return connection.audioUsage(seconds, ratio), nil
	case "session.closed":
		reason, ok := frame["reason"].(string)
		if !ok {
			return nil, malformed
		}
		seconds, usageOK := object(frame["usage"])["seconds"].(float64)
		var events []realtime.CodecEvent
		if usageOK {
			events = connection.audioUsage(seconds, nil)
		} else {
			events = append(events, realtime.SessionError{Err: malformed, Recoverable: true})
		}
		connection.ended = true
		if reason == "close_requested" || reason == "remote_hangup" {
			return append(events, connection.settleTurns(false)...), nil
		}
		events = append(events, connection.settleTurns(true)...)
		if (reason == "expired" || reason == "connection_lost") && connection.canReconnect() {
			connection.redial = true
			return events, nil
		}
		return append(events, liveError("live_session_"+reason, "session ended: "+reason, false)), nil
	case "error":
		details := object(frame["error"])
		message, ok := details["message"].(string)
		if !ok {
			return nil, malformed
		}
		code := stringValue(details["code"])
		if connection.backend != "" && strings.Contains(message, "`"+connection.backend+"`") {
			return []realtime.CodecEvent{liveError("live_backend_model_unavailable", message, false)}, nil
		}
		var events []realtime.CodecEvent
		if strings.HasPrefix(message, "Responses handoff") {
			for id, delegation := range connection.delegations {
				if delegation.inFlight {
					events = append(events, realtime.SessionUsage{Usage: ai.Usage{Requests: 1}, ResponseScoped: true})
					connection.settleDelegation(id, true)
				}
			}
			code = "live_delegation_failed"
		}
		return append(events, liveError(code, message, true)), nil
	default:
		return nil, nil
	}
}

func liveError(code, message string, recoverable bool) realtime.SessionError {
	return realtime.SessionError{Err: &LiveError{Code: code, Message: message}, Recoverable: recoverable}
}

// LiveError describes an inspectable GPT-Live provider or delegation failure.
type LiveError struct {
	Code    string
	Message string
}

// Error describes the provider failure.
func (err *LiveError) Error() string { return "openai GPT-Live: " + err.Code + ": " + err.Message }

func (connection *LiveConnection) fragment(direction int, text string) string {
	previous := connection.fragments[direction]
	connection.fragments[direction] = text
	first, _ := utf8.DecodeRuneInString(text)
	if previous != "" && strings.ContainsAny(previous[len(previous)-1:], ".!?") && unicode.IsUpper(first) {
		return " " + text
	}
	return text
}

func (connection *LiveConnection) closeInput() []realtime.CodecEvent {
	if !connection.inputOpen {
		return nil
	}
	connection.inputOpen = false
	connection.fragments[0] = ""
	return []realtime.CodecEvent{realtime.InputTranscript{Final: true}}
}

func (connection *LiveConnection) openResponse() []realtime.CodecEvent {
	connection.lastVoice = time.Now()
	events := connection.closeInput()
	if !connection.responseOpen {
		events = append(events, realtime.ResponseStarted{})
	}
	connection.responseOpen = true
	return events
}

func (connection *LiveConnection) settleTurns(interrupted bool) []realtime.CodecEvent {
	events := connection.closeInput()
	if connection.responseOpen {
		connection.responseOpen = false
		connection.hasPause = false
		connection.fragments[1] = ""
		events = append(events, realtime.ResponseDone{Interrupted: interrupted})
	}
	return events
}

func (connection *LiveConnection) audioUsage(seconds float64, ratio *float64) []realtime.CodecEvent {
	increment := max(0, seconds-connection.seconds)
	if increment == 0 && ratio == nil {
		return nil
	}
	if increment > 0 {
		connection.seconds = seconds
	}
	usage := ai.Usage{AudioSeconds: increment}
	connection.priceUsage(&usage, connection.model)
	return []realtime.CodecEvent{realtime.SessionUsage{Usage: usage, ContextWindowUsed: ratio}}
}

func (connection *LiveConnection) priceUsage(usage *ai.Usage, model string) {
	if price, err := (ai.ModelResponse{Usage: *usage, ModelName: model, ProviderName: "openai", ProviderURL: connection.baseURL}).Price(); err == nil {
		usage.CostUSD = &price.TotalPrice
	}
}

func (connection *LiveConnection) mapBackend(frame map[string]any, id string) ([]realtime.CodecEvent, error) {
	kind := stringValue(frame["type"])
	malformed := fmt.Errorf("openai GPT-Live: malformed delegated %s event", kind)
	switch kind {
	case "error":
		message, ok := frame["message"].(string)
		if !ok {
			return nil, malformed
		}
		return []realtime.CodecEvent{liveError(stringValue(frame["code"]), message, true)}, nil
	case "response.completed", "response.failed", "response.incomplete":
		response := object(frame["response"])
		if response == nil {
			return nil, malformed
		}
		data := object(response["usage"])
		usage := ai.Usage{Requests: 1}
		if data != nil {
			input, output := object(data["input_tokens_details"]), object(data["output_tokens_details"])
			usage.InputTokens = integer(data["input_tokens"])
			usage.OutputTokens = integer(data["output_tokens"])
			usage.CacheReadTokens = integer(input["cached_tokens"])
			usage.CacheWriteTokens = integer(input["cache_write_tokens"])
			usage.ReasoningTokens = integer(output["reasoning_tokens"])
			usage.Details = map[string]int{"reasoning_tokens": usage.ReasoningTokens}
		}
		connection.priceUsage(&usage, stringValue(response["model"]))
		events := []realtime.CodecEvent{realtime.SessionUsage{Usage: usage, ResponseScoped: true,
			ProviderDetails: map[string]any{"delegated_model": response["model"], "delegated_response_id": response["id"]}}}
		connection.settleDelegation(id, kind != "response.completed")
		delete(connection.reasoning, id)
		if kind != "response.completed" {
			reason := render(response["error"])
			if response["error"] == nil {
				reason = render(response["incomplete_details"])
			}
			events = append(events, liveError("live_delegation_"+strings.TrimPrefix(kind, "response."), reason, true))
		}
		return events, nil
	case "response.output_item.done":
		item := object(frame["item"])
		if item == nil {
			return nil, malformed
		}
		itemKind := stringValue(item["type"])
		if itemKind == "reasoning" {
			signature := stringValue(item["encrypted_content"])
			summaries := array(item["summary"])
			if len(summaries) == 0 && signature != "" {
				connection.reasoning[id] = append(connection.reasoning[id], ai.ThinkingPart{ID: stringValue(item["id"]), Signature: signature, ProviderName: "openai"})
			}
			for _, raw := range summaries {
				connection.reasoning[id] = append(connection.reasoning[id], ai.ThinkingPart{Content: stringValue(object(raw)["text"]), ID: stringValue(item["id"]), Signature: signature, ProviderName: "openai"})
				signature = ""
			}
			return nil, nil
		}
		reasoning := connection.reasoning[id]
		delete(connection.reasoning, id)
		if itemKind == "web_search_call" {
			itemID := stringValue(item["id"])
			if itemID == "" {
				return nil, malformed
			}
			arguments, _ := json.Marshal(item["action"])
			if string(arguments) == "null" {
				arguments = []byte("{}")
			}
			reasoning = append(reasoning, ai.NativeToolCallPart{ToolName: "web_search", ToolCallID: itemID, ID: itemID, ToolKind: ai.ToolPartKindWebSearch, Args: arguments, ProviderName: "openai", ProviderDetails: map[string]any{"status": item["status"]}},
				ai.NativeToolReturnPart{ToolName: "web_search", ToolCallID: itemID, ToolKind: ai.ToolPartKindWebSearch, Content: map[string]any{"status": item["status"]}, ProviderName: "openai", Timestamp: time.Now().UTC()})
			var events []realtime.CodecEvent
			for _, part := range reasoning {
				partID := fmt.Sprintf("live-native-%d", connection.nativeIndex)
				connection.nativeIndex++
				events = append(events, realtime.PartStarted{Event: ai.PartStartEvent{PartID: partID, Part: part}}, realtime.PartEnded{Event: ai.PartEndEvent{PartID: partID, Part: part}})
			}
			return events, nil
		}
		if itemKind != "function_call" {
			return nil, nil
		}
		callID, name := stringValue(item["call_id"]), stringValue(item["name"])
		arguments, ok := item["arguments"].(string)
		if callID == "" || name == "" || !ok {
			return nil, malformed
		}
		if connection.seenCalls[callID] {
			return nil, nil
		}
		connection.seenCalls[callID] = true
		if arguments == "" {
			arguments = "{}"
		}
		delegation := connection.delegations[id]
		connection.calls[callID] = id
		if delegation != nil {
			delegation.pending[callID] = true
		}
		events := connection.openResponse()
		connection.hasPause = false
		return append(events, realtime.ToolCall{ToolCallID: callID, ToolName: name, Arguments: arguments, ResponseUsageFollows: delegation != nil}), nil
	default:
		return nil, nil
	}
}

func (connection *LiveConnection) settleDelegation(id string, failed bool) {
	delegation := connection.delegations[id]
	if delegation == nil {
		return
	}
	delegation.inFlight = false
	if failed {
		for call := range delegation.pending {
			connection.abandoned[call] = true
			delete(connection.calls, call)
		}
		clear(delegation.pending)
		delegation.continuation = false
	}
	if len(delegation.pending) == 0 && !delegation.continuation {
		delete(connection.delegations, id)
		connection.lastVoice = time.Now()
	}
}
