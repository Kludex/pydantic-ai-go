package bedrockmantle

import (
	"crypto/sha256"
	"iter"
	"math/big"
	"slices"
	"strings"
	"time"

	ai "github.com/Kludex/pydantic-ai-go/ai"
)

// GPT-5 on /openai/v1/responses numbers tool calls per response (call_0 again in the next one), so stored IDs carry
// a hash of their response's ID, or of its timestamp without one, in the [a-zA-Z0-9_-] set Anthropic and Bedrock take.
func qualifyToolCallID(id string, scope string) string {
	return id + "_" + hashedScope(scope)
}

func qualifyResponseToolCallIDs(response *ai.ModelResponse) {
	if response.Timestamp.IsZero() {
		response.Timestamp = time.Now().UTC()
	}
	scope := toolCallIDScopes(response.ProviderResponseID, response.Timestamp)[0]
	for index, part := range response.Parts {
		if call, ok := part.(ai.ToolCallPart); ok && call.ToolCallID != "" {
			call.ToolCallID = qualifyToolCallID(call.ToolCallID, scope)
			response.Parts[index] = call
		}
	}
}

// A call streamed before the response ID is kept as sent: a request continuing the response by
// previous_response_id could not restore it from the timestamp.
func qualifyStreamToolCallIDs(
	stream iter.Seq2[ai.ModelStreamEvent, error],
) iter.Seq2[ai.ModelStreamEvent, error] {
	return func(yield func(ai.ModelStreamEvent, error) bool) {
		responseID := ""
		unscoped := make(map[string]bool)
		for event, err := range stream {
			switch typed := event.(type) {
			case ai.ResponseMetadataEvent:
				if typed.ProviderResponseID != "" {
					responseID = typed.ProviderResponseID
				}
			case ai.ToolCallStartEvent:
				if !typed.Native && typed.ToolCallID != "" {
					if responseID == "" {
						unscoped[typed.ToolCallID] = true
					} else {
						typed.ToolCallID = qualifyToolCallID(typed.ToolCallID, responseID)
						event = typed
					}
				}
			case ai.FinishEvent:
				if responseID == "" {
					responseID = typed.ProviderResponseID
				}
				typed.Parts = slices.Clone(typed.Parts)
				for index, part := range typed.Parts {
					if call, ok := part.(ai.ToolCallPart); ok && call.ToolCallID != "" && responseID != "" &&
						!unscoped[call.ToolCallID] {
						call.ToolCallID = qualifyToolCallID(call.ToolCallID, responseID)
						typed.Parts[index] = call
					}
				}
				event = typed
			}
			if !yield(event, err) {
				return
			}
		}
	}
}

// Only this endpoint issued the IDs of its own responses; other providers and endpoints get the stored, unique ones.
func restoreProviderToolCallIDs(messages []ai.ModelMessage, providerURL string) []ai.ModelMessage {
	restored := slices.Clone(messages)
	var scopes []string
	for messageIndex, message := range messages {
		switch message := message.(type) {
		case ai.ModelResponse:
			scopes = nil
			if message.ProviderName == "bedrock-mantle" &&
				(message.ProviderURL == "" || sameEndpoint(message.ProviderURL, providerURL)) {
				scopes = toolCallIDScopes(message.ProviderResponseID, message.Timestamp)
			}
			message.Parts = slices.Clone(message.Parts)
			for partIndex, part := range message.Parts {
				if call, ok := part.(ai.ToolCallPart); ok {
					call.ToolCallID = unqualifyToolCallID(call.ToolCallID, scopes)
					message.Parts[partIndex] = call
				}
			}
			restored[messageIndex] = message
		case ai.ModelRequest:
			message.Parts = slices.Clone(message.Parts)
			for partIndex, part := range message.Parts {
				switch part := part.(type) {
				case ai.ToolReturnPart:
					part.ToolCallID = unqualifyToolCallID(part.ToolCallID, scopes)
					message.Parts[partIndex] = part
				case ai.RetryPromptPart:
					part.ToolCallID = unqualifyToolCallID(part.ToolCallID, scopes)
					message.Parts[partIndex] = part
				}
			}
			restored[messageIndex] = message
		}
	}
	return restored
}

// A streamed response can hold IDs scoped by either its response ID or its timestamp, so both are tried.
func toolCallIDScopes(responseID string, timestamp time.Time) []string {
	timestamp = timestamp.UTC().Truncate(time.Microsecond)
	layout := "2006-01-02T15:04:05+00:00"
	if timestamp.Nanosecond() != 0 {
		layout = "2006-01-02T15:04:05.000000+00:00"
	}
	timestampScope := timestamp.Format(layout)
	if responseID == "" {
		return []string{timestampScope}
	}
	return []string{responseID, timestampScope}
}

func unqualifyToolCallID(id string, scopes []string) string {
	for _, scope := range scopes {
		if restored, ok := strings.CutSuffix(id, "_"+hashedScope(scope)); ok {
			return restored
		}
		if restored, ok := strings.CutPrefix(id, scope+":"); ok {
			return restored
		}
	}
	return id
}

func sameEndpoint(a, b string) bool {
	return strings.EqualFold(strings.TrimSuffix(a, "/"), strings.TrimSuffix(b, "/"))
}

const base62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

// The low-order eight base62 digits of the scope's SHA-256, matching PydanticAI's stored IDs.
func hashedScope(scope string) string {
	sum := sha256.Sum256([]byte(scope))
	number, base, remainder := new(big.Int).SetBytes(sum[:]), big.NewInt(62), new(big.Int)
	digits := make([]byte, 8)
	for index := range digits {
		number.DivMod(number, base, remainder)
		digits[index] = base62[remainder.Int64()]
	}
	return string(digits)
}
