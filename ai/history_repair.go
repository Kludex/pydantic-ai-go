package ai

// MessageRepairOptions controls explicit history repair.
type MessageRepairOptions struct {
	// PreserveLastResponse leaves unanswered calls in the latest response open.
	PreserveLastResponse bool
}

// RepairMessages returns a detached, provider-valid history. It removes
// orphaned tool results, synthesizes interrupted results for dangling calls,
// and merges adjacent messages without changing the input slice.
func RepairMessages(messages []ModelMessage, options MessageRepairOptions) []ModelMessage {
	repaired := dropOrphanedToolResults(cloneModelMessages(messages))
	if options.PreserveLastResponse {
		lastResponse := -1
		for index := len(repaired) - 1; index >= 0; index-- {
			if _, ok := repaired[index].(ModelResponse); ok {
				lastResponse = index
				break
			}
		}
		if lastResponse >= 0 {
			prefix, pending := repairDanglingToolCalls(repaired[:lastResponse], false)
			if len(pending) > 0 {
				prefix = append(prefix, synthesizedToolReturnRequest(prefix, pending))
			}
			repaired = append(prefix, repaired[lastResponse:]...)
			return mergeConsecutiveMessages(repaired)
		}
	}
	repaired, pending := repairDanglingToolCalls(repaired, false)
	if len(pending) > 0 {
		repaired = append(repaired, synthesizedToolReturnRequest(repaired, pending))
	}
	return mergeConsecutiveMessages(repaired)
}

func synthesizedToolReturnRequest(messages []ModelMessage, parts []RequestPart) ModelRequest {
	request := ModelRequest{Parts: parts, State: RequestStateInterrupted}
	for index := len(messages) - 1; index >= 0; index-- {
		if response, ok := messages[index].(ModelResponse); ok {
			request.Timestamp = response.Timestamp
			request.RunID = response.RunID
			request.ConversationID = response.ConversationID
			break
		}
	}
	return request
}
