package ai

import (
	"crypto/sha256"
	"fmt"
	"math"
	"math/big"
	"slices"
	"strconv"
	"strings"
)

// The prompt cache is keyed on request bytes, so results go out in call order however history stored them.
func orderToolResultsByCall(messages []ModelMessage) {
	for index := 1; index < len(messages); index++ {
		request, isRequest := messages[index].(ModelRequest)
		response, followsResponse := messages[index-1].(ModelResponse)
		if !isRequest || !followsResponse {
			continue
		}
		order := make(map[string]int)
		for _, part := range response.Parts {
			if call, ok := part.(ToolCallPart); ok && call.ToolCallID != "" {
				if _, seen := order[call.ToolCallID]; !seen {
					order[call.ToolCallID] = len(order)
				}
			}
		}
		var slots []int
		var results []RequestPart
		for partIndex, part := range request.Parts {
			if _, answers := order[resultToolCallID(part)]; answers {
				slots = append(slots, partIndex)
				results = append(results, part)
			}
		}
		slices.SortStableFunc(results, func(a, b RequestPart) int {
			return order[resultToolCallID(a)] - order[resultToolCallID(b)]
		})
		for resultIndex, slot := range slots {
			request.Parts[slot] = results[resultIndex]
		}
	}
}

func resultToolCallID(part RequestPart) string {
	switch part := part.(type) {
	case ToolReturnPart:
		return part.ToolCallID
	case RetryPromptPart:
		if part.ToolName != "" {
			return part.ToolCallID
		}
	}
	return ""
}

// Replacements are assigned in history order, so an earlier ID never changes as the history grows.
func fitToolCallIDs(messages []ModelMessage, length int) error {
	replacements := make(map[string]string)
	taken := make(map[string]struct{})
	capacity := math.Pow(62, float64(length))
	fit := func(id string) (string, error) {
		if id == "" {
			return id, nil
		}
		if replacement, ok := replacements[id]; ok {
			return replacement, nil
		}
		if float64(len(taken)) >= capacity {
			return "", fmt.Errorf(
				"ai: the message history has more distinct tool call IDs than fit in %d alphanumeric characters "+
					"(the profile's ToolCallIDLength)", length,
			)
		}
		replacement := id
		if !fitsToolCallID(id, length) {
			replacement = hashedToolCallID(id, length)
		}
		for attempt := 1; ; attempt++ {
			if _, used := taken[replacement]; !used {
				break
			}
			replacement = hashedToolCallID(id+"#"+strconv.Itoa(attempt), length)
		}
		replacements[id] = replacement
		taken[replacement] = struct{}{}
		return replacement, nil
	}
	var err error
	for _, message := range messages {
		switch message := message.(type) {
		case ModelRequest:
			for index, part := range message.Parts {
				switch part := part.(type) {
				case ToolReturnPart:
					part.ToolCallID, err = fit(part.ToolCallID)
					message.Parts[index] = part
				case RetryPromptPart:
					part.ToolCallID, err = fit(part.ToolCallID)
					message.Parts[index] = part
				}
				if err != nil {
					return err
				}
			}
		case ModelResponse:
			for index, part := range message.Parts {
				if call, ok := part.(ToolCallPart); ok {
					if call.ToolCallID, err = fit(call.ToolCallID); err != nil {
						return err
					}
					message.Parts[index] = call
				}
			}
		}
	}
	return nil
}

const toolCallIDAlphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

func fitsToolCallID(id string, length int) bool {
	return len(id) == length && strings.Trim(id, toolCallIDAlphabet) == ""
}

func hashedToolCallID(id string, length int) string {
	sum := sha256.Sum256([]byte(id))
	number, base, remainder := new(big.Int).SetBytes(sum[:]), big.NewInt(62), new(big.Int)
	digits := make([]byte, length)
	for index := range digits {
		number.DivMod(number, base, remainder)
		digits[index] = toolCallIDAlphabet[remainder.Int64()]
	}
	return string(digits)
}
